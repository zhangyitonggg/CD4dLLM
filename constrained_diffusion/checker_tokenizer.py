from llguidance import LLMatcher, LLTokenizer, LLParserLimits
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import torch
from llguidance import TokenizerWrapper

model_name_map = {
    "LLaDA": "GSAI-ML/LLaDA-8B-Instruct",
    "Dream": "Qwen/Qwen2.5-7B",
}


class Checker():
    tokenizer: LLTokenizer
    matcher: LLMatcher
    tokens: list[int]

    def __init__(self, grammar: str, model_name=None) -> None:
        def get_tokenizer(model_name: str | None) -> LLTokenizer:
            if model_name is None:
                _tokenizer = LLTokenizer("byte")
            else:
                if model_name_map.get(model_name):
                    model_name = model_name_map[model_name]
                tokenizer_path = hf_hub_download(
                    repo_id=f"{model_name}",
                    filename="tokenizer.json",
                )
                _tokenizer = LLTokenizer(tokenizer_path)
            return _tokenizer

        def get_matcher(tokenizer: LLTokenizer, grammar: str) -> LLMatcher:
            def is_json_loadable(s: str) -> bool:
                import json
                try:
                    json.loads(s)
                    return True
                except (json.JSONDecodeError, TypeError):
                    return False
            if is_json_loadable(grammar):
                grm = LLMatcher.grammar_from_json_schema(grammar)
            else:
                grm = LLMatcher.grammar_from_lark(grammar)
            is_err, _ = LLMatcher.validate_grammar_with_warnings(grm)

            assert not is_err, "Grammar is not valid"

            limits = LLParserLimits(
                max_items_in_row=20000,
                step_max_items=600000,
            )
            return LLMatcher(tokenizer, grm, log_level=1, limits=limits)

        self.tokenizer = get_tokenizer(model_name=model_name)
        self.matcher = get_matcher(self.tokenizer, grammar)
        self.tokens = []

    def validate_str(self, next_str):
        next_tokens = self.tokenizer.tokenize_str(next_str)        
        return self.matcher.validate_tokens(next_tokens) == len(next_tokens)

    def validate_tokens(self, next_tokens):
        assert not self.matcher.is_error(), "Matcher is in error state"
        if next_tokens is None or len(next_tokens) == 0:
            return True
        if self.matcher.is_stopped():
            return False
        res = self.matcher.validate_tokens(next_tokens) == len(next_tokens)
        if self.matcher.is_error():
            print(f"Cannot validate tokens: {self.tokenizer.dbg_tokens(next_tokens)}")
            print(next_tokens)
            print(self.matcher.validate_tokens(next_tokens))
            print("Error state:")
            print(self.matcher.get_error())
            print("-------------------")
            assert False
        return res

    def end_with_str(self, next_str):
        next_tokens = self.tokenizer.tokenize_str(next_str)        
        count = self.matcher.try_consume_tokens(next_tokens)
        if count != len(next_tokens):
            self.matcher.rollback(count)
            return False
        if self.matcher.is_accepting():
            self.matcher.rollback(count)
            return True
        else:
            self.matcher.rollback(count)
            return False

    def end_with_tokens(self, next_tokens):
        if next_tokens is None or len(next_tokens) == 0:
            return self.matcher.is_accepting()
        count = self.matcher.try_consume_tokens(next_tokens)
        if count != len(next_tokens):
            self.matcher.rollback(count)
            return False
        if self.matcher.is_accepting():
            self.matcher.rollback(count)
            return True
        else:
            self.matcher.rollback(count)
            return False
        
    def reset(self):
        self.matcher.reset()

    def consume_str(self, next_str):
        next_tokens = self.tokenizer.tokenize_str(next_str)
        count = self.matcher.try_consume_tokens(next_tokens)
        if count != len(next_tokens):
            print(f"Cannot consume {next_str}")
            self.matcher.rollback(count)
            return False
        return True

    def consume_tokens(self, next_tokens):
        assert not self.matcher.is_error(), "Matcher is in error state"
        if next_tokens is None or len(next_tokens) == 0:
            return True
        count = self.matcher.try_consume_tokens(next_tokens)
        if count != len(next_tokens):
            print(f"Cannot consume {next_tokens}")
            next_str = self.tokenizer.dbg_tokens(next_tokens)
            print(f"count: {count}, len: {len(next_tokens)}")
            print(f"Cannot consume {next_str}")
            self.matcher.rollback(count)
            return False
        assert not self.matcher.is_error()
        self.tokens.extend(next_tokens)
        return True

    def dbg_tokens(self, tokens):
        res =  self.tokenizer.dbg_tokens(tokens)
        # assert not self.matcher.is_error()
        return res

    def compute_mask(self) -> torch.Tensor:
        """
        Compute the token mask, with one byte per tokenizer word, for the next parsing step.
        Entries are either 0 (not allowed) or 1 (allowed).
        """
        assert not self.matcher.is_error(), "Cannot compute mask in error state"
        masks = self.matcher.compute_logit_bias()
        masks = torch.tensor([1 if m > 0 else 0 for m in masks], dtype=torch.float32)
        assert not self.matcher.is_error(), f"Error occurred during mask computation: {self.matcher.get_error()}, tokens: {self.tokens}"
        return masks
    
    def rollback(self, count: int):
        assert not self.matcher.is_error(), "Cannot rollback in error state"
        if count <= 0:
            return True
        res = self.matcher.rollback(count)
        # assert res
        assert not self.matcher.is_error()
        self.tokens = self.tokens[:-count]
        return res

    def is_stoped(self) -> bool:
        # assert not self.matcher.is_error()
        return self.matcher.is_stopped()

    def is_accepting(self) -> bool:
        # assert not self.matcher.is_error()
        return self.matcher.is_accepting()

    def is_error(self) -> bool:
        return self.matcher.is_error()
    
    def reset(self):
        self.matcher.reset()