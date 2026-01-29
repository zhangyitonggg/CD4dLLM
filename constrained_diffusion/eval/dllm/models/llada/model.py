import time

import stopit
import torch
from transformers import AutoTokenizer, AutoModel

from constrained_diffusion.constrain_utils import (
    EOS,
    autocomplete_valid,
    partial_output_from_tokens,
    generated_language,
    derive_supertokens,
    LexMap,
    CompiledLexMap,
)
from constrained_diffusion.eval.dllm.datasets.generic import Instance
from constrained_diffusion.eval.dllm.models.generic import Model
from constrained_diffusion.eval.dllm.models.llada.generate_constrained import (
    generate as generate_constrained,
    generate_ours,
)
from constrained_diffusion.eval.dllm.models.llada.generate_our import generate as generate_ours
from rustformlang.cfg import CFG

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
# MODEL_NAME = "GSAI-ML/LLaDA-1.5"

class LLaDAModel(Model):
    """
    Model for LLaDA-8B-Instruct.
    """

    def tokenizer(self, device):
        return AutoTokenizer.from_pretrained(MODEL_NAME)

    def model(self, device):
        kwargs = (
            {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
            }
            if device == "cuda"
            else {"device_map": device}
        )

        model = AutoModel.from_pretrained(
            MODEL_NAME, **kwargs, trust_remote_code=True
        ).eval()
        return model

    def prepare_prompt(self, instance: Instance, tokenizer, model, trace: bool):
        system_message_content = instance.system_message_content()
        system_messages = [
            {
                "role": "system",
                "content": system_message_content,
            },
        ]

        messages = system_messages + [
            {
                "role": "user",
                "content": instance.user_prompt_content(),
            },
        ]
        try:
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
            messages.pop(0)
        user_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        suffix = f"```{instance.language_short_name()}\n"
        start_line = instance.assistant_start_line()
        user_input += suffix
        input_ids_no_line = tokenizer(user_input)["input_ids"]
        user_input += start_line
        input_ids = tokenizer(user_input)["input_ids"]
        prompt_len = len(input_ids_no_line)
        input_ids = torch.tensor(input_ids).to(model.device).unsqueeze(0)
        prompt = input_ids

        if trace:
            print(user_input)
        return prompt, prompt_len, suffix, start_line, user_input

    def generate_unconstrained(
        self,
        instance: Instance,
        model,
        tokenizer,
        steps: int,
        gen_length: int,
        temperature: int,
        alg: str = "low_confidence",
        trace: bool = False,
        block_length: int = 32,
    ):
        prompt, prompt_len, suffix, start_line, prompt_raw = self.prepare_prompt(
            instance, tokenizer, model, trace
        )

        out = None
        for out, resamples, valid, start_time in generate_constrained(
            model,
            prompt,
            tokenizer,
            prelex=None,
            constraint_lang=None,
            lex_map=None,
            prompt_len=prompt_len,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=0.0,
            remasking="low_confidence",
            trace=trace,
            subtokens={},
            additional_stuff=None,
            strip_chars=instance.strip_chars(),
            max_total_injections=0,
            inject_gap_size=0,
            constrain=False,
        ):
            pass
        if out is None:
            code = "TIMEOUT"
        else:
            code = tokenizer.batch_decode(
                out[:, prompt.shape[1] :], skip_special_tokens=True
            )[0]
        extracted = instance.extract_result(suffix + start_line + code)
        return prompt_raw, code, extracted, False, time.monotonic() - start_time

    def generate_constrained(
        self,
        instance: Instance,
        model,
        tokenizer,
        steps: int,
        gen_length: int,
        temperature: int,
        lang: CFG,
        lex_map: CompiledLexMap,
        orig_lex_map: LexMap,
        subtokens,
        additional_stuff,
        max_total_injections: int = 0,
        inject_gap_size: int = 0,
        prelex: str | None = None,
        alg: str = "low_confidence",
        timeout: int = 60,
        trace: bool = False,
        block_length: int = 32,
    ):
        prompt, prompt_len, suffix, start_line, prompt_raw = self.prepare_prompt(
            instance, tokenizer, model, trace
        )

        out = None
        resamples = []
        valid = False
        start_time = time.monotonic()
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            for out, resamples, valid, init_start_time in generate_constrained(
                model,
                prompt,
                tokenizer,
                prelex=prelex,
                constraint_lang=lang,
                lex_map=lex_map,
                prompt_len=prompt_len,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=0.0,
                remasking="low_confidence",
                trace=trace,
                subtokens=subtokens,
                additional_stuff=additional_stuff,
                strip_chars=instance.strip_chars(),
                max_total_injections=max_total_injections,
                inject_gap_size=inject_gap_size,
            ):
                pass
        if out is None:
            code = "TIMEOUT"
            code_raw = "TIMEOUT"
        else:
            code = tokenizer.batch_decode(
                out[:, prompt.shape[1] :], skip_special_tokens=True
            )[0]
            code_raw = tokenizer.batch_decode(out.squeeze(), skip_special_tokens=False)
        extracted = instance.extract_result(suffix + start_line + code)

        # extract a valid completion
        if not valid:
            start_time = time.monotonic()
            generated_words = tokenizer.batch_decode(out.squeeze())
            mask_id = 126336
            mask_decoded = tokenizer.decode(mask_id)
            generated_words = [
                None
                if x == mask_decoded
                else EOS
                if x in ("<|endoftext|>", "<|eot_id|>")
                else x
                for x in generated_words[prompt_len:]
            ]
            partial_output, first_token_gap, last_token_eos_adj = (
                partial_output_from_tokens(generated_words, prelex)
            )
            if trace:
                print("Generated words:", generated_words)
                print("Partial output:", partial_output)
            valid_completion = autocomplete_valid(
                partial_output=partial_output,
                first_token_gap=first_token_gap,
                last_token_eos_adj=last_token_eos_adj,
                generated_lang=generated_language(
                    generated_words,
                    lex_map,
                    lang.get_terminals(),
                    trace=trace,
                    prelex=prelex,
                    subtokens=subtokens,
                    supertokens=derive_supertokens(subtokens),
                    strip_chars=instance.strip_chars(),
                ),
                subtokens=subtokens,
                lex_map=orig_lex_map,
                constraint_lang=lang,
                trace=trace,
            )
            if trace:
                print(f"Valid completion: {valid_completion}")
            completion_extracted = (
                instance.extract_result(suffix + valid_completion)
                if valid_completion
                else None
            )
            end_time = time.monotonic()
            intersection_time = end_time - start_time
            if trace:
                print(
                    f"Time taken to extract valid completion: {intersection_time:.2f} seconds"
                )
        else:
            valid_completion = None
            completion_extracted = None
            intersection_time = 0.0
        return (
            prompt_raw,
            code,
            code_raw,
            extracted,
            not bool(to_ctx_mgr),
            resamples,
            valid_completion,
            completion_extracted,
            intersection_time,
            time.monotonic() - init_start_time,
        )

    def generate_ours(
        self,
        instance: Instance,

        model,
        tokenizer,
        
        steps: int,
        gen_length: int,
        block_length: int,
        temperature: int,
        timeout: int = 120,
        trace: bool = False,
        change_logits: bool = False,

        top_k_per_mask: int = 5,
        top_n_beam: int = 30,
        random_n_beam: int = 20,
        max_retry_num_total: int = 1000, 
    ):
        prompt_ids, input_len, suffix, start_line, prompt_raw = self.prepare_prompt(
            instance, tokenizer, model, trace
        )
        cfg_lang = instance.cfg()

        out = None
        total_retry_num = 0
        start_time = time.monotonic()
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            out, total_retry_num, start_time = generate_ours(
                model,
                tokenizer,
                prompt_ids,
                input_len=input_len,
                grammar=cfg_lang,

                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                remasking="low_confidence",
                trace=trace,
                change_logits=change_logits,
                top_k_per_mask=top_k_per_mask,
                top_n_beam=top_n_beam,
                random_n_beam=random_n_beam,
                max_retry_num_total=max_retry_num_total,
            )
        if out is None:
            code = "TIMEOUT"
            code_raw = "TIMEOUT"
        else:
            code = tokenizer.batch_decode(
                out[:, prompt_ids.shape[1] :], skip_special_tokens=True
            )[0]
            code_raw = tokenizer.batch_decode(out.squeeze(), skip_special_tokens=False)
        extracted = instance.extract_result(suffix + start_line + code)

        # valid_completion = None
        # completion_extracted = None
        # intersection_time = 0.0
        return (
            prompt_raw,
            code,
            code_raw,
            extracted,
            not bool(to_ctx_mgr),
            time.monotonic() - start_time,
            total_retry_num,
            # resamples,
            # valid_completion,
            # completion_extracted,
            # intersection_time,
        )