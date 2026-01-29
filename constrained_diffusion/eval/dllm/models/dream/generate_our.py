# import json
# import time
# import warnings
# from collections import defaultdict
# from dataclasses import dataclass
# from typing import Dict, Optional, Tuple, Union, List

# import numpy as np
# import torch
# import torch.distributions as dists
# from torch.nn import functional as F
# from transformers import __version__
# from transformers.generation.configuration_utils import GenerationConfig
# from transformers.utils import (
#     ModelOutput,
#     is_torchdynamo_compiling,
#     logging,
# )

# from rustformlang.cfg import CFG

# from constrained_diffusion.checker_tokenizer import Checker
# import heapq
# import random

# logger = logging.get_logger(__name__)

# mask_id = 151666
# eos_pool = set()

# cache_seq: List[int] = []

# def top_p_logits(logits, top_p=None):
#     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#     sorted_indices_to_remove = cumulative_probs > top_p
#     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#     sorted_indices_to_remove[..., 0] = 0
#     mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
#     mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
#     logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
#     return logits


# def top_k_logits(logits, top_k=None):
#     top_k = min(top_k, logits.size(-1))  # Safety check
#     # Remove all tokens with a probability less than the last token of the top-k
#     indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#     logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
#     return logits


# def get_num_transfer_tokens(mask_index, steps):
#     """
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.
#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     """
#     mask_num = mask_index.sum(dim=1, keepdim=True)
#     if steps >= mask_num.max().item():
#         num_transfer_tokens = torch.zeros(
#             mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
#         )
#         for i in range(mask_num.size(0)):
#             for j in range(mask_num[i].item()):
#                 num_transfer_tokens[i, j] = 1
#         return num_transfer_tokens
#     base = mask_num // steps
#     remainder = mask_num % steps
#     num_transfer_tokens = (
#         torch.zeros(
#             mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
#         )
#         + base
#     )
#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, : remainder[i]] += 1
#     return num_transfer_tokens


# def sample_tokens(
#     logits,
#     temperature=0.0,
#     top_p=None,
#     top_k=None,
#     margin_confidence=False,
#     neg_entropy=False,
# ):
#     if temperature > 0:
#         logits = logits / temperature
#     if top_p is not None and top_p < 1:
#         logits = top_p_logits(logits, top_p)
#     if top_k is not None:
#         logits = top_k_logits(logits, top_k)
#     probs = torch.softmax(logits, dim=-1)

#     if temperature > 0:
#         try:
#             x0 = dists.Categorical(probs=probs).sample()
#             confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
#         except Exception:
#             confidence, x0 = probs.max(dim=-1)
#     else:
#         confidence, x0 = probs.max(dim=-1)

#     if margin_confidence:
#         sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
#         # Extract top1 and top2 probabilities
#         top1_probs = sorted_probs[:, 0]
#         top2_probs = sorted_probs[:, 1]
#         # Calculate confidence as top1 - top2
#         confidence = top1_probs - top2_probs

#     if neg_entropy:
#         epsilon = 1e-10
#         log_probs = torch.log(probs + epsilon)
#         confidence = torch.sum(probs * log_probs, dim=-1)

#     return confidence, x0


# @dataclass
# class DreamModelOutput(ModelOutput):
#     sequences: torch.LongTensor = None
#     history: Optional[Tuple[torch.FloatTensor]] = None


# class DreamGenerationConfig(GenerationConfig):
#     def __init__(self, **kwargs):
#         self.temperature: float = kwargs.pop("temperature", 0.0)
#         self.top_p: Optional[float] = kwargs.pop("top_p", None)
#         self.top_k: Optional[int] = kwargs.pop("top_k", None)
#         self.max_length = kwargs.pop("max_length", 20)
#         self.max_new_tokens = kwargs.pop("max_new_tokens", None)
#         # diffusion specific params
#         self.eps: float = kwargs.pop("eps", 1e-3)
#         self.steps: int = kwargs.pop("steps", 512)
#         self.alg: str = kwargs.pop("alg", "origin")
#         self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

#         # Parameters that define the output variables of `generate`
#         self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
#         self.return_dict_in_generate: bool = kwargs.pop(
#             "return_dict_in_generate", False
#         )
#         self.output_history: bool = kwargs.pop("output_history", False)

#         # Special tokens that can be used at generation time
#         self.mask_token_id = kwargs.pop("mask_token_id", None)
#         self.pad_token_id = kwargs.pop("pad_token_id", None)
#         self.bos_token_id = kwargs.pop("bos_token_id", None)
#         self.eos_token_id = kwargs.pop("eos_token_id", None)

#         # Wild card
#         self.generation_kwargs = kwargs.pop("generation_kwargs", {})

#         # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
#         # interface.
#         self._from_model_config = kwargs.pop("_from_model_config", False)
#         self._commit_hash = kwargs.pop("_commit_hash", None)
#         self.transformers_version = kwargs.pop("transformers_version", __version__)

#         # Additional attributes without default values
#         if not self._from_model_config:
#             # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
#             # model's default configuration file
#             for key, value in kwargs.items():
#                 try:
#                     setattr(self, key, value)
#                 except AttributeError as err:
#                     logger.error(f"Can't set {key} with value {value} for {self}")
#                     raise err

#         # Validate the values of the attributes
#         self.validate(is_init=True)

#     def validate(self, is_init=False):
#         pass


# def get_end_index(all_token_ids, num_block: int, block_length: int, prompt):
#     """
#     返回最后一个既不是 EOS 也不是 None 的位置索引。
#     """
#     end_index = 0
#     for i in range(prompt.shape[1] + (num_block + 1) * block_length - 1, -1, -1):
#         w = all_token_ids[i]
#         if w not in eos_pool and w != mask_id:
#             end_index = i
#             break
#     return end_index


# def get_min_eos_index(all_token_ids, num_block: int, block_length: int, prompt):
#     """
#     在 token_ids 中找到第一个 EOS 的位置索引，若不存在则返回 -1。
#     """
#     exist_mask = False
#     for i in range(prompt.shape[1] + num_block * block_length, prompt.shape[1] + (num_block + 1) * block_length):
#         w = all_token_ids[i]
#         if w in eos_pool:
#             return i, exist_mask
#         if w == mask_id:
#             exist_mask = True
#     return -1, exist_mask


# def compute_logits_mask(
#     checker: Checker,
#     logits: Optional[torch.Tensor],
# ):
#     logits_mask = checker.compute_mask()

#     logits_mask_long = torch.zeros(len(logits[0][0]))
#     logits_mask_long[0:len(logits_mask)] = logits_mask
#     logits_mask = logits_mask_long
    
#     logits_mask[mask_id] = 0
#     logits_mask[151667] = 0   
#     for id in eos_pool:
#         if logits_mask[id] == 1:
#             for other_id in eos_pool:
#                 logits_mask[other_id] = 1
#             break
#     assert logits_mask.sum() > 0, "logits_mask should have valid tokens."
#     return logits_mask


# def pre_check(
#     checker: Checker,
#     tokenizer,
#     all_token_ids: List[int],
#     p: torch.Tensor,
#     logits: torch.Tensor,
#     index_to_consume: int,
#     last_token_index: int,
#     min_eos_index: int,          
#     trace: bool = False,
#     top_k_per_mask: int = 5,
#     top_n_beam: int = 3,
#     random_n_beam: int = 3,
# ):
#     import heapq
#     import random
#     global cache_seq
    
#     beams = [(all_token_ids.copy(), 0.0)]
#     # temp_consume_index = index_to_consume
#     if top_n_beam == 1 and random_n_beam == 0:
#         for pos in range(index_to_consume, last_token_index + 1):
#             if all_token_ids[pos] == mask_id:
#                 new_beams: List[Tuple[List[int], float]] = []
#                 for seq, score in beams:
#                     p[0, pos][mask_id] = 0
#                     for id in eos_pool:
#                         p[0, id] = 0
#                     top_probs, top_idxs = torch.topk(p[0, pos], 1)
#                     for prob, tid in zip(top_probs.detach().to("cpu").tolist(), top_idxs.detach().to("cpu").tolist()):
#                         new_seq = seq.copy()
#                         new_seq[pos] = tid
#                         new_score = score + prob
#                         new_beams.append((new_seq, new_score))
#                 top_beams = heapq.nlargest(top_n_beam, new_beams, key=lambda x: x[1])
#                 beams = top_beams
#             else:
#                 for seq, score in beams:
#                     if checker.validate_tokens(seq[index_to_consume:pos+1]) == len(seq[index_to_consume:pos+1]):
#                         continue
#                     else:
#                         beams.remove((seq, score))
#                 if len(beams) == 0:
#                     return False
#     else:
#         for pos in range(index_to_consume, last_token_index + 1):
#             if all_token_ids[pos] == mask_id:
#                 new_beams: List[Tuple[List[int], float]] = []
#                 for seq, score in beams:
#                     p[0, pos][mask_id] = 0
#                     p[0, pos][151667] = 0
#                     top_probs, top_idxs = torch.topk(p[0, pos], top_k_per_mask)
#                     for prob, tid in zip(top_probs.detach().to("cpu").tolist(), top_idxs.detach().to("cpu").tolist()):
#                         new_seq = seq.copy()
#                         new_seq[pos] = tid
#                         new_score = score + prob
#                         new_beams.append((new_seq, new_score))
#                 top_beams = heapq.nlargest(top_n_beam, new_beams, key=lambda x: x[1])
#                 remaining = [b for b in new_beams if b not in top_beams]
#                 random_beams = random.sample(remaining, min(random_n_beam, len(remaining)))
#                 beams = top_beams + random_beams
#             else:
#                 beams = [
#                     (seq, score)
#                     for seq, score in beams
#                     if checker.validate_tokens(seq[index_to_consume:pos+1])
#                 ]
#                 if len(beams) == 0:
#                     return False
        
#     assert len(beams) > 0, "No valid beams after processing tokens."
#     largest_seq = max(beams, key=lambda x: x[1])[0]
#     if min_eos_index == -1:
#         if trace:
#             print(f"Pre-check passed. No EOS/EOT.")
#             print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: last_token_index+1])}")
#             print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: last_token_index+1])}")
#         cache_seq = largest_seq.copy()
#         return True

#     assert checker.validate_tokens(largest_seq[index_to_consume:last_token_index + 1]), f"Should be valid sequence, {checker.dbg_tokens(largest_seq[index_to_consume:last_token_index + 1])}, {last_token_index - index_to_consume + 1}"
#     checker.consume_tokens(largest_seq[index_to_consume:last_token_index + 1])
#     assert checker.is_error() == False, "Should not be error here."

#     for pos in range(last_token_index + 1, len(largest_seq)):
#         largest_seq[pos] = tokenizer.eos_token_id

#     for pos in range(last_token_index + 1, min_eos_index):
#         logits_mask = compute_logits_mask(checker, logits)
#         assert logits_mask.sum() > 0, "There should be valid tokens to select."
#         flag = False
#         for id in eos_pool:
#             if logits_mask[id] == 1:
#                 flag = True
#                 break
#         if flag:
#             checker.rollback(pos - index_to_consume)
#             if trace:
#                 print(f"Pre-check passed. EOS/EOT found)).")
#             if trace:
#                 print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: pos])}")
#                 print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: pos])}")
#             cache_seq = largest_seq.copy()
#             return True

#         probs = p[0, pos] * logits_mask.to(p.device)
#         top_prob, top_idx = torch.topk(probs, 1)
#         top_prob = top_prob.detach().to("cpu").tolist()
#         top_idx = top_idx.detach().to("cpu").tolist()
#         prob, tid = top_prob[0], top_idx[0]
#         assert prob > 1e-9, "Should not reach here."
#         checker.consume_tokens([tid])
#         largest_seq[pos] = tid
#     if trace:
#         print(f"is_stoped: {checker.is_stoped()}")
#     if checker.is_accepting():
#         if trace:
#             print(f"Pre-check passed. Accepting state reached.")
#         assert checker.rollback(min_eos_index - index_to_consume)
#         if trace:
#             print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: min_eos_index])}")
#             print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: min_eos_index])}")
#         cache_seq = largest_seq.copy()
#         return True

#     assert checker.rollback(min_eos_index - index_to_consume), "Should rollback"
#     return False


# def check(
#     checker: Checker,
#     tokenizer,
#     all_token_ids: List[int],
#     prompt_ids,
#     p: torch.Tensor,
#     logits: torch.Tensor,

#     index_to_consume: int,
#     block_num: int,
#     block_length: int,
#     top_k_per_mask: int = 5,
#     top_n_beam: int = 30,
#     random_n_beam: int = 20,
#     trace: bool = False,    
# ):
#     last_token_index = get_end_index(all_token_ids, block_num, block_length, prompt_ids)
#     min_eos_index, _ = get_min_eos_index(all_token_ids, block_num, block_length, prompt_ids)

#     if min_eos_index != -1 and min_eos_index < last_token_index:
#         return False

#     pre_accept = pre_check(
#         checker,
#         tokenizer,
#         all_token_ids,
#         p,
#         logits,
#         index_to_consume,
#         last_token_index,
#         min_eos_index,
#         trace=trace,
#         top_k_per_mask=top_k_per_mask,
#         top_n_beam=top_n_beam,
#         random_n_beam=random_n_beam,
#     )
#     if pre_accept:
#         return True
#     else:
#         return False


# @torch.no_grad()
# def diffusion_generate(
#     model,
#     tokenizer,
#     inputs: Optional[torch.Tensor],
#     input_len: int,
#     grammar: str,
#     generation_config: Optional[DreamGenerationConfig] = None,
#     block_length: int = 32,
#     trace: bool = False,
#     top_k_per_mask: int = 5,
#     top_n_beam: int = 3,
#     random_n_beam: int = 3,
#     max_retry_num_total: int = 5,
#     **kwargs,
# ):
#     checker = Checker(grammar=grammar, model_name="Dream")
#     if trace:
#         print("Initial checker state:")
#         print(tokenizer.batch_decode(inputs))
#         print(tokenizer.batch_decode(inputs[0][input_len:]))
#     if checker.consume_tokens(inputs[0][input_len:].tolist()) == False:
#         raise ValueError("Prompt does not conform to grammar.")
#     # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
#     generation_config = model._prepare_generation_config(generation_config, **kwargs)
#     generation_tokens_hook_func = kwargs.pop(
#         "generation_tokens_hook_func", lambda step, x, logits: x
#     )
#     generation_logits_hook_func = kwargs.pop(
#         "generation_logits_hook_func", lambda step, x, logits: logits
#     )

#     # 2. Define model inputs
#     assert inputs is not None
#     prompt_ids = inputs
#     device = prompt_ids.device
#     attention_mask = kwargs.pop("attention_mask", None)
#     model._prepare_special_tokens(generation_config, device=device)

#     # 3. Prepare `max_length`.
#     prompt_ids_length = prompt_ids.shape[-1]
#     has_default_max_length = (
#         kwargs.get("max_length") is None and generation_config.max_length is not None
#     )
#     generation_config = model._prepare_generated_length(
#         generation_config=generation_config,
#         has_default_max_length=has_default_max_length,
#         input_ids_length=prompt_ids_length,
#     )

#     model._validate_generated_length(
#         generation_config, prompt_ids_length, has_default_max_length
#     )

#     # 4. Check prompt_ids
#     if not is_torchdynamo_compiling() and model.device.type != prompt_ids.device.type:
#         warnings.warn(
#             "You are calling .generate() with the `prompt_ids` being on a device type different"
#             f" than your model's device. `prompt_ids` is on {prompt_ids.device.type}, whereas the model"
#             f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
#             " Please make sure that you have put `prompt_ids` to the"
#             f" correct device by calling for example prompt_ids = prompt_ids.to('{model.device.type}') before"
#             " running `.generate()`.",
#             UserWarning,
#         )
#     if (
#         hasattr(generation_config, "pad_token_id")
#         and torch.any(prompt_ids == generation_config.pad_token_id)
#         and attention_mask is None
#     ):
#         warnings.warn(
#             "Padding was detected but no attention mask is passed here. For correct "
#             "generation results, please set `attention_mask` when batch-padding inputs.",
#             UserWarning,
#         )

#     prompt_ids, attention_mask = model._expand_inputs_for_generation(
#         expand_size=generation_config.num_return_sequences,
#         input_ids=prompt_ids,
#         attention_mask=attention_mask,
#     )

#     result = _sample(
#         model,
#         checker,
#         prompt_ids,
#         tokenizer=tokenizer,
#         input_len=input_len,
#         attention_mask=attention_mask,
#         generation_config=generation_config,
#         generation_tokens_hook_func=generation_tokens_hook_func,
#         generation_logits_hook_func=generation_logits_hook_func,
#         trace=trace,
#         block_length=block_length,
#         top_k_per_mask=top_k_per_mask,
#         top_n_beam=top_n_beam,
#         random_n_beam=random_n_beam,
#         max_retry_num_total=max_retry_num_total,
#     )
#     return result


# def _sample(
#     model,
#     checker: Checker,
#     prompt_ids: torch.LongTensor,
#     tokenizer,
#     input_len: int,
#     attention_mask: Optional[torch.LongTensor],
#     generation_config: DreamGenerationConfig,
#     generation_tokens_hook_func,
#     generation_logits_hook_func,
#     trace: bool = False,
#     block_length: int = 32,
#     top_k_per_mask: int = 5,
#     top_n_beam: int = 3,
#     random_n_beam: int = 3,
#     max_retry_num_total: int = 5,
# ):
#     index_to_consume = prompt_ids.shape[1]
#     start_time = time.monotonic()

#     # --- sampling code
#     # init values
#     output_history = generation_config.output_history
#     return_dict_in_generate = generation_config.return_dict_in_generate
#     max_length = generation_config.max_length
#     mask_token_id = generation_config.mask_token_id
#     steps = generation_config.steps
#     eps = generation_config.eps
#     alg = generation_config.alg
#     alg_temp = generation_config.alg_temp
#     temperature = generation_config.temperature
#     top_p = generation_config.top_p
#     top_k = generation_config.top_k

#     # mask_id = mask_token_id
#     eos_pool.add(tokenizer.eos_token_id)
#     if tokenizer.convert_tokens_to_ids("<|im_end|>") is not None:
#         eos_pool.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
#     if tokenizer.convert_tokens_to_ids("<|dlm_pad|>") is not None:
#         eos_pool.add(tokenizer.convert_tokens_to_ids("<|dlm_pad|>"))

#     histories = [] if (return_dict_in_generate and output_history) else None

#     # pad prompt_ids to max_length
#     x = F.pad(prompt_ids, (0, max_length - prompt_ids.shape[1]), value=mask_id)

#     if attention_mask is not None and torch.any(attention_mask == 0.0):
#         # we do not mask the [MASK] tokens so value = 1.0
#         attention_mask = F.pad(
#             attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0
#         )
#         tok_idx = attention_mask.long().cumsum(-1) - 1
#         tok_idx.masked_fill_(attention_mask == 0, 1)
#         # attention_mask is of shape [B, N]
#         # broadcast to [B, 1, N, N]
#         attention_mask = torch.logical_and(
#             attention_mask.unsqueeze(1).unsqueeze(-2),
#             attention_mask.unsqueeze(1).unsqueeze(-1),
#         )
#     else:
#         tok_idx = None
#         attention_mask = "full"

#     # timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

#     # this allows user-defined token control of the intermediate steps
#     x = generation_tokens_hook_func(None, x, None)
    
#     all_token_ids = x[0].detach().cpu().tolist()
#     global cache_seq
#     cache_seq = all_token_ids.copy()

#     total_retry_num = 0
#     complete = False
    
#     gen_length = max_length - prompt_ids.shape[1]
#     assert gen_length % block_length == 0
#     num_blocks = gen_length // block_length
#     assert steps % num_blocks == 0
#     steps_per_block = steps // num_blocks

#     has_constrain = True
#     for block_idx in range(num_blocks):
#         if complete:
#             break
#         block_mask_index = (
#             x[
#                 :,
#                 prompt_ids.shape[1] + block_idx * block_length : prompt_ids.shape[1]
#                 + (block_idx + 1) * block_length :,
#             ]
#             == mask_id
#         )
#         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
#         # timesteps = torch.linspace(1, eps, steps_per_block + 1, device=x.device)
        
#         # start_ar = False
#         for step_idx in range(steps_per_block):
#             if complete:
#                 break
#             # if start_ar:
#                 # break
#             start_ar = False

#             # ----- sampling code
#             logits = model(x, attention_mask, tok_idx).logits
#             logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

#             # this allows user-defined logits control of the intermediate steps
#             logits = generation_logits_hook_func(step_idx, x, logits)

#             # t = timesteps[step_idx]
#             # s = timesteps[step_idx + 1]
#             # mask_index = x == mask_token_id
#             # block_mask = torch.zeros_like(x, dtype=torch.bool)
#             # start = prompt_ids.shape[1] + block_idx * block_length
#             # end = prompt_ids.shape[1] + (block_idx + 1) * block_length
#             # block_mask[:, start:end] = mask_index[:, start:end]
#             # num_mask_token = block_mask.sum() / block_mask.shape[0]
#             # number_transfer_tokens_overall = (
#                 # int(num_mask_token * (1 - s / t)) if step_idx < steps_per_block - 1 else int(num_mask_token)
#             # )
#             # if trace:
#                 # print(f"number_transfer_tokens_overall: {number_transfer_tokens_overall}, t: {t}, s: {s}, num_mask_token: {num_mask_token}")
#             number_transfer_tokens_overall = num_transfer_tokens[0, step_idx]
#             transfer_num = 0 
#             while transfer_num < number_transfer_tokens_overall:
#                 if complete:
#                     break
#                 if start_ar:
#                     break

#                 if trace:
#                     print(f"\033[38;2;165;42;42m[Block {block_idx} / {num_blocks}, step {step_idx} / {steps}, token {transfer_num} / {number_transfer_tokens_overall}]\033[0m")

#                 # number_transfer_tokens = 1
#                 mask_index = x == mask_token_id

#                 # ----- constraining code
#                 tokens_found = False
#                 one_token_retry_num = 0

#                 while not tokens_found:
#                     # --------- sampling code
#                     mask_logits = logits[mask_index]

#                     if alg == "entropy":
#                         confidence, x0 = sample_tokens(
#                             mask_logits,
#                             temperature,
#                             top_p=top_p,
#                             top_k=top_k,
#                             neg_entropy=True,
#                         )
#                     else:
#                         raise RuntimeError(f"Unknown alg: {alg}")
#                     full_confidence = torch.full_like(
#                         x, -torch.inf, device=model.device, dtype=logits.dtype
#                     )
#                     full_confidence[mask_index] = confidence

#                     full_confidence[:, prompt_ids.shape[1] + (block_idx + 1) * block_length:] = -torch.inf
    
#                     if alg_temp is None or alg_temp == 0:
#                         conf, select_index = torch.topk(
#                             full_confidence[0], 1
#                         )
#                     else:
#                         raise RuntimeError("alg_temp other than 0 not implemented yet.")
                    
#                     x_ = (
#                         torch.zeros_like(x, device=model.device, dtype=torch.long)
#                         + mask_token_id
#                     )
#                     x_[mask_index] = x0.clone()

#                     index_of_new_token = select_index.item()
#                     new_token_vocab_index = x_[0][index_of_new_token]
#                     assert logits[0][index_of_new_token][new_token_vocab_index] != -np.inf, "No valid token sampled."

#                     all_token_ids[index_of_new_token] = new_token_vocab_index.item()

#                     if has_constrain:
#                         if cache_seq[index_of_new_token] == new_token_vocab_index.item():
#                             is_accept = True
#                             if trace:
#                                 print("cache_seq hit, accept directly.")
#                         else:
#                             p = F.softmax(logits.to(torch.float64), dim=-1)
#                             is_accept = check(
#                                 checker,
#                                 tokenizer,
#                                 all_token_ids,
#                                 prompt_ids,
#                                 p,
#                                 logits,
#                                 index_to_consume,
#                                 block_idx,
#                                 block_length,
#                                 top_k_per_mask,
#                                 top_n_beam,
#                                 random_n_beam,
#                                 trace=trace,
#                             )

#                     if trace:
#                         new_word = tokenizer.decode(new_token_vocab_index)
#                         if is_accept:
#                             print(
#                                 f"+++ Accept New word at {index_of_new_token}: {json.dumps(new_word)} ({new_token_vocab_index}), confidence={conf.item():.6f}"
#                             )
#                         else:
#                             print(
#                                 f"--- Reject {index_of_new_token}: {json.dumps(new_word)} ({new_token_vocab_index}), confidence={conf.item():.6f}"
#                             )
#                     if has_constrain:
#                         if not is_accept:
#                             logits[0][index_of_new_token][
#                                 new_token_vocab_index
#                             ] = -np.inf
#                             all_token_ids[index_of_new_token] = mask_id
#                             one_token_retry_num += 1
#                             total_retry_num += 1
#                             if one_token_retry_num >= max_retry_num_total:
#                                 if trace: 
#                                     print("Too many retries for one token, start autoregressive generation.")
#                                 # start_ar = True
#                                 has_constrain = False
#                                 # r = checker.rollback(index_to_consume - (block_idx * block_length + prompt_ids.shape[1]))
#                                 if trace:
#                                     print(f"index_to_consume={index_to_consume},{block_idx * block_length + prompt_ids.shape[1]}")
#                                 break
#                         else:
#                             tokens_found = True
#                             transfer_num += 1
#                             one_token_retry_num = 0
#                             transfer_index = torch.zeros_like(
#                                 x_, dtype=torch.bool, device=x_.device
#                             )
#                             transfer_index[0, select_index] = True
#                             if new_token_vocab_index in eos_pool:
#                                 transfer_index[0, select_index:] = True
#                                 val = x_[0, select_index].clone()
#                                 x_[0, select_index:] = val
#                                 for idx in range(select_index, len(all_token_ids)):
#                                     all_token_ids[idx] = val.item()
#                             x[transfer_index] = x_[transfer_index]
#                             # if EOS in generated_words
#                             min_eos_index, exist_mask = get_min_eos_index(
#                                 all_token_ids, block_idx, block_length, prompt_ids
#                             )
#                             if min_eos_index != -1 and not exist_mask:
#                                 complete = True
#                                 break

#                             # if has_constrain:
#                             for idx in range(index_to_consume, prompt_ids.shape[1] + (block_idx + 1) * block_length):
#                                 if all_token_ids[idx] == mask_id or all_token_ids[idx] in eos_pool:
#                                     if idx > index_to_consume:
#                                         if trace:
#                                             print(f"Consume tokens: {checker.dbg_tokens(all_token_ids[index_to_consume:idx])}, index_to_consume={index_to_consume} -> {idx}")        
#                                         tokens_to_consume = all_token_ids[index_to_consume:idx]
#                                         assert checker.consume_tokens(tokens_to_consume), f"Tokens {checker.dbg_tokens(tokens_to_consume)} do not conform to grammar."
#                                         index_to_consume = idx
#                                         if checker.is_stoped():
#                                             x[0, index_to_consume] = tokenizer.eos_token_id
#                                             complete = True
#                                             if trace:
#                                                 print("Grammar matched complete sequence. Inserting EOS.")
#                                             break
#                                     break
#                     else:
#                         tokens_found = True
#                         transfer_num += 1
#                         one_token_retry_num = 0
#                         transfer_index = torch.zeros_like(
#                             x_, dtype=torch.bool, device=x_.device
#                         )
#                         transfer_index[0, select_index] = True
#                         if new_token_vocab_index in eos_pool:
#                             transfer_index[0, select_index:] = True
#                             val = x_[0, select_index].clone()
#                             x_[0, select_index:] = val
#                             for idx in range(select_index, len(all_token_ids)):
#                                 all_token_ids[idx] = val.item()
#                         x[transfer_index] = x_[transfer_index]
#                         # if EOS in generated_words
#                         min_eos_index, exist_mask = get_min_eos_index(
#                             all_token_ids, block_idx, block_length, prompt_ids
#                         )
#                         if min_eos_index != -1 and not exist_mask:
#                             complete = True
#                             break

#             if not start_ar:
#                 continue
#             if trace:
#                 print("Start autoregressive generation for the rest tokens in this block.")

#             # init
#             a = prompt_ids.shape[1] + block_idx * block_length
#             b = prompt_ids.shape[1] + (block_idx + 1) * block_length
#             for i in range(b, len(cache_seq)):
#                 cache_seq[i] = mask_id
#             index_cache = b
#             for i in range(a, b):
#                 if cache_seq[i] == mask_id or cache_seq[i] in eos_pool:
#                     if trace:
#                         print(f"All cache_seq from {i} to {b} are mask_id, set index_cache to {i}")
#                     index_cache = i
#                     break
#             free_num = 0
#             for i in range(a, index_cache):
#                 if x[0, i] == mask_id:
#                     free_num += 1
            
#             x[:, a:b] = mask_id
#             if b != x.shape[1]:
#                 x[0, b:b+2] = mask_id
#                 all_token_ids[b] = mask_id
#                 all_token_ids[b+1] = mask_id
#             for pos in range(a, b):
#                 all_token_ids[pos] = mask_id
            
#             for i in range(a, index_cache):
#                 assert cache_seq[i] != mask_id, "Cache seq should have value here."
#                 for id in eos_pool:
#                     assert cache_seq[i] != id, "Should not be EOS/EOT in cache seq here."
#                 x[0, i] = cache_seq[i]
#                 all_token_ids[i] = cache_seq[i]
#             tokens_to_consume = all_token_ids[index_to_consume:index_cache]

#             assert checker.consume_tokens(tokens_to_consume), f"Should be valid sequence in AR generation, {checker.dbg_tokens(tokens_to_consume)}"         
            
#             if trace:
#                 print(f"index_to_consume: {index_to_consume} -> index_cache: {index_cache}")
#             index_to_consume = index_cache

#             if checker.is_stoped():
#                 x[0, index_to_consume] = tokenizer.eos_token_id
#                 return x, total_retry_num, start_time

#             min_decode = num_transfer_tokens[0, step_idx] - transfer_num
            
#             if trace:
#                 print(f"min_decode before adjustment: {min_decode}, free_num: {free_num}, num_transfer_tokens: {num_transfer_tokens[0, step_idx]}, token_num: {transfer_num}")
#             if min_decode > free_num:
#                 min_decode = min_decode - free_num
#                 logits = model(x, attention_mask, tok_idx).logits
#                 logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
#                 logits = generation_logits_hook_func(step_idx, x, logits)

#                 for i in range(min_decode):
#                     index_of_new_token = index_cache + i
#                     one_logits = logits[0, index_of_new_token]
#                     logits_mask = compute_logits_mask(checker, logits)
#                     one_logits[logits_mask == 0] = -float('inf')
#                     token_id = torch.argmax(one_logits)
#                     assert logits_mask[token_id] == 1, "Selected token should be valid."
#                     if trace:
#                         new_word = tokenizer.decode(token_id.item())
#                         if token_id.item() in eos_pool:
#                             new_word = "<EOS>"
#                         print(
#                             f"index_cache:{index_cache} +++ Accept New word at {index_of_new_token}: {json.dumps(new_word)} ({token_id.item()})"
#                         )
#                     if token_id.item() in eos_pool:
#                         x[0, index_of_new_token] = token_id
#                         return x, total_retry_num, start_time
#                     else:
#                         x[0, index_of_new_token] = token_id
#                         all_token_ids[index_of_new_token] = token_id.item()
#                         cache_seq[index_of_new_token] = token_id.item()
#                         checker.consume_tokens([token_id.item()])
#                         if trace:
#                             print(f"Consume token: {checker.dbg_tokens([token_id.item()])}, index_to_consume={index_to_consume} -> {index_of_new_token + 1}")
#                         index_to_consume = index_of_new_token + 1
#                         if checker.is_stoped():
#                             x[0, index_of_new_token + 1] = tokenizer.eos_token_id
#                             return x, total_retry_num, start_time
#             else:
#                 total_decode = 0
#                 for i in range(step_idx + 1, steps_per_block):
#                     total_decode += num_transfer_tokens[0, i]
#                 sub_num = free_num - min_decode
#                 if trace:
#                     print(f"total_decode: {total_decode}, sub_num: {sub_num}, steps: {steps_per_block}, step_num: {step_idx}")
#                 if total_decode <= sub_num:
#                     for i in range(step_idx + 1, steps_per_block):
#                         num_transfer_tokens[0, i] = 0
#                 else:
#                     total_decode_new = total_decode - sub_num
#                     number = steps_per_block - step_idx - 1
#                     for i in range(step_idx + 1, steps_per_block):
#                         num_transfer_tokens[0, i] = total_decode_new // number
#                     remainder = total_decode_new % number
#                     for i in range(step_idx + 1, step_idx + 1 + remainder):
#                         num_transfer_tokens[0, i] += 1

#     return x, total_retry_num, start_time

import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

from rustformlang.cfg import CFG

from constrained_diffusion.checker_tokenizer import Checker
import heapq
import random

logger = logging.get_logger(__name__)

mask_id = 151666
eos_pool = set()

cache_seq: List[int] = []

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.
    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    if steps >= mask_num.max().item():
        num_transfer_tokens = torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        for i in range(mask_num.size(0)):
            for j in range(mask_num[i].item()):
                num_transfer_tokens[i, j] = 1
        return num_transfer_tokens
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def sample_tokens(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    margin_confidence=False,
    neg_entropy=False,
):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", "origin")
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop(
            "return_dict_in_generate", False
        )
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass


def get_end_index(all_token_ids, num_block: int, block_length: int, prompt):
    """
    返回最后一个既不是 EOS 也不是 None 的位置索引。
    """
    end_index = 0
    for i in range(prompt.shape[1] + (num_block + 1) * block_length - 1, -1, -1):
        w = all_token_ids[i]
        if w not in eos_pool and w != mask_id:
            end_index = i
            break
    return end_index


def get_min_eos_index(all_token_ids, num_block: int, block_length: int, prompt):
    """
    在 token_ids 中找到第一个 EOS 的位置索引，若不存在则返回 -1。
    """
    exist_mask = False
    for i in range(prompt.shape[1] + num_block * block_length, prompt.shape[1] + (num_block + 1) * block_length):
        w = all_token_ids[i]
        if w in eos_pool:
            return i, exist_mask
        if w == mask_id:
            exist_mask = True
    return -1, exist_mask


def compute_logits_mask(
    checker: Checker,
    logits: Optional[torch.Tensor],
):
    logits_mask = checker.compute_mask()

    logits_mask_long = torch.zeros(len(logits[0][0]))
    logits_mask_long[0:len(logits_mask)] = logits_mask
    logits_mask = logits_mask_long
    
    logits_mask[mask_id] = 0
    logits_mask[151667] = 0   
    for id in eos_pool:
        if logits_mask[id] == 1:
            for other_id in eos_pool:
                logits_mask[other_id] = 1
            break
    assert logits_mask.sum() > 0, "logits_mask should have valid tokens."
    return logits_mask


def pre_check(
    checker: Checker,
    tokenizer,
    all_token_ids: List[int],
    p: torch.Tensor,
    logits: torch.Tensor,
    index_to_consume: int,
    last_token_index: int,
    min_eos_index: int,          
    trace: bool = False,
    top_k_per_mask: int = 5,
    top_n_beam: int = 3,
    random_n_beam: int = 3,
):
    import heapq
    import random
    global cache_seq
    
    beams = [(all_token_ids.copy(), 0.0)]
    # temp_consume_index = index_to_consume
    if top_n_beam == 1 and random_n_beam == 0:
        for pos in range(index_to_consume, last_token_index + 1):
            if all_token_ids[pos] == mask_id:
                new_beams: List[Tuple[List[int], float]] = []
                for seq, score in beams:
                    p[0, pos][mask_id] = 0
                    for id in eos_pool:
                        p[0, id] = 0
                    top_probs, top_idxs = torch.topk(p[0, pos], 1)
                    for prob, tid in zip(top_probs.detach().to("cpu").tolist(), top_idxs.detach().to("cpu").tolist()):
                        new_seq = seq.copy()
                        new_seq[pos] = tid
                        new_score = score + prob
                        new_beams.append((new_seq, new_score))
                top_beams = heapq.nlargest(top_n_beam, new_beams, key=lambda x: x[1])
                beams = top_beams
            else:
                for seq, score in beams:
                    if checker.validate_tokens(seq[index_to_consume:pos+1]) == len(seq[index_to_consume:pos+1]):
                        continue
                    else:
                        beams.remove((seq, score))
                if len(beams) == 0:
                    return False
    else:
        for pos in range(index_to_consume, last_token_index + 1):
            if all_token_ids[pos] == mask_id:
                new_beams: List[Tuple[List[int], float]] = []
                for seq, score in beams:
                    p[0, pos][mask_id] = 0
                    p[0, pos][151667] = 0
                    top_probs, top_idxs = torch.topk(p[0, pos], top_k_per_mask)
                    for prob, tid in zip(top_probs.detach().to("cpu").tolist(), top_idxs.detach().to("cpu").tolist()):
                        new_seq = seq.copy()
                        new_seq[pos] = tid
                        new_score = score + prob
                        new_beams.append((new_seq, new_score))
                top_beams = heapq.nlargest(top_n_beam, new_beams, key=lambda x: x[1])
                remaining = [b for b in new_beams if b not in top_beams]
                random_beams = random.sample(remaining, min(random_n_beam, len(remaining)))
                beams = top_beams + random_beams
            else:
                beams = [
                    (seq, score)
                    for seq, score in beams
                    if checker.validate_tokens(seq[index_to_consume:pos+1])
                ]
                if len(beams) == 0:
                    return False
        
    assert len(beams) > 0, "No valid beams after processing tokens."
    largest_seq = max(beams, key=lambda x: x[1])[0]
    if min_eos_index == -1:
        if trace:
            print(f"Pre-check passed. No EOS/EOT.")
            print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: last_token_index+1])}")
            print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: last_token_index+1])}")
        cache_seq = largest_seq.copy()
        return True

    assert checker.validate_tokens(largest_seq[index_to_consume:last_token_index + 1]), f"Should be valid sequence, {checker.dbg_tokens(largest_seq[index_to_consume:last_token_index + 1])}, {last_token_index - index_to_consume + 1}"
    checker.consume_tokens(largest_seq[index_to_consume:last_token_index + 1])
    assert checker.is_error() == False, "Should not be error here."

    for pos in range(last_token_index + 1, len(largest_seq)):
        largest_seq[pos] = tokenizer.eos_token_id

    for pos in range(last_token_index + 1, min_eos_index):
        logits_mask = compute_logits_mask(checker, logits)
        assert logits_mask.sum() > 0, "There should be valid tokens to select."
        flag = False
        for id in eos_pool:
            if logits_mask[id] == 1:
                flag = True
                break
        if flag:
            checker.rollback(pos - index_to_consume)
            if trace:
                print(f"Pre-check passed. EOS/EOT found)).")
            if trace:
                print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: pos])}")
                print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: pos])}")
            cache_seq = largest_seq.copy()
            return True

        probs = p[0, pos] * logits_mask.to(p.device)
        top_prob, top_idx = torch.topk(probs, 1)
        top_prob = top_prob.detach().to("cpu").tolist()
        top_idx = top_idx.detach().to("cpu").tolist()
        prob, tid = top_prob[0], top_idx[0]
        assert prob > 1e-9, "Should not reach here."
        checker.consume_tokens([tid])
        largest_seq[pos] = tid
    if trace:
        print(f"is_stoped: {checker.is_stoped()}")
    if checker.is_accepting():
        if trace:
            print(f"Pre-check passed. Accepting state reached.")
        assert checker.rollback(min_eos_index - index_to_consume)
        if trace:
            print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: min_eos_index])}")
            print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: min_eos_index])}")
        cache_seq = largest_seq.copy()
        return True

    assert checker.rollback(min_eos_index - index_to_consume), "Should rollback"
    return False


def check(
    checker: Checker,
    tokenizer,
    all_token_ids: List[int],
    prompt_ids,
    p: torch.Tensor,
    logits: torch.Tensor,

    index_to_consume: int,
    block_num: int,
    block_length: int,
    top_k_per_mask: int = 5,
    top_n_beam: int = 30,
    random_n_beam: int = 20,
    trace: bool = False,    
):
    last_token_index = get_end_index(all_token_ids, block_num, block_length, prompt_ids)
    min_eos_index, _ = get_min_eos_index(all_token_ids, block_num, block_length, prompt_ids)

    if min_eos_index != -1 and min_eos_index < last_token_index:
        return False

    pre_accept = pre_check(
        checker,
        tokenizer,
        all_token_ids,
        p,
        logits,
        index_to_consume,
        last_token_index,
        min_eos_index,
        trace=trace,
        top_k_per_mask=top_k_per_mask,
        top_n_beam=top_n_beam,
        random_n_beam=random_n_beam,
    )
    if pre_accept:
        return True
    else:
        return False


@torch.no_grad()
def diffusion_generate(
    model,
    tokenizer,
    inputs: Optional[torch.Tensor],
    input_len: int,
    grammar: str,
    generation_config: Optional[DreamGenerationConfig] = None,
    block_length: int = 32,
    trace: bool = False,
    top_k_per_mask: int = 5,
    top_n_beam: int = 3,
    random_n_beam: int = 3,
    max_retry_num_total: int = 5,
    **kwargs,
):
    checker = Checker(grammar=grammar, model_name="Dream")
    if trace:
        print("Initial checker state:")
        print(tokenizer.batch_decode(inputs))
        print(tokenizer.batch_decode(inputs[0][input_len:]))
    if checker.consume_tokens(inputs[0][input_len:].tolist()) == False:
        raise ValueError("Prompt does not conform to grammar.")
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    generation_config = model._prepare_generation_config(generation_config, **kwargs)
    generation_tokens_hook_func = kwargs.pop(
        "generation_tokens_hook_func", lambda step, x, logits: x
    )
    generation_logits_hook_func = kwargs.pop(
        "generation_logits_hook_func", lambda step, x, logits: logits
    )

    # 2. Define model inputs
    assert inputs is not None
    prompt_ids = inputs
    device = prompt_ids.device
    attention_mask = kwargs.pop("attention_mask", None)
    model._prepare_special_tokens(generation_config, device=device)

    # 3. Prepare `max_length`.
    prompt_ids_length = prompt_ids.shape[-1]
    has_default_max_length = (
        kwargs.get("max_length") is None and generation_config.max_length is not None
    )
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        input_ids_length=prompt_ids_length,
    )

    model._validate_generated_length(
        generation_config, prompt_ids_length, has_default_max_length
    )

    # 4. Check prompt_ids
    if not is_torchdynamo_compiling() and model.device.type != prompt_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `prompt_ids` being on a device type different"
            f" than your model's device. `prompt_ids` is on {prompt_ids.device.type}, whereas the model"
            f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `prompt_ids` to the"
            f" correct device by calling for example prompt_ids = prompt_ids.to('{model.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )
    if (
        hasattr(generation_config, "pad_token_id")
        and torch.any(prompt_ids == generation_config.pad_token_id)
        and attention_mask is None
    ):
        warnings.warn(
            "Padding was detected but no attention mask is passed here. For correct "
            "generation results, please set `attention_mask` when batch-padding inputs.",
            UserWarning,
        )

    prompt_ids, attention_mask = model._expand_inputs_for_generation(
        expand_size=generation_config.num_return_sequences,
        input_ids=prompt_ids,
        attention_mask=attention_mask,
    )

    result = _sample(
        model,
        checker,
        prompt_ids,
        tokenizer=tokenizer,
        input_len=input_len,
        attention_mask=attention_mask,
        generation_config=generation_config,
        generation_tokens_hook_func=generation_tokens_hook_func,
        generation_logits_hook_func=generation_logits_hook_func,
        trace=trace,
        block_length=block_length,
        top_k_per_mask=top_k_per_mask,
        top_n_beam=top_n_beam,
        random_n_beam=random_n_beam,
        max_retry_num_total=max_retry_num_total,
    )
    return result


def _sample(
    model,
    checker: Checker,
    prompt_ids: torch.LongTensor,
    tokenizer,
    input_len: int,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    generation_tokens_hook_func,
    generation_logits_hook_func,
    trace: bool = False,
    block_length: int = 32,
    top_k_per_mask: int = 5,
    top_n_beam: int = 3,
    random_n_beam: int = 3,
    max_retry_num_total: int = 5,
):
    index_to_consume = prompt_ids.shape[1]
    start_time = time.monotonic()

    # --- sampling code
    # init values
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    steps = generation_config.steps
    eps = generation_config.eps
    alg = generation_config.alg
    alg_temp = generation_config.alg_temp
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k

    # mask_id = mask_token_id
    eos_pool.add(tokenizer.eos_token_id)
    if tokenizer.convert_tokens_to_ids("<|im_end|>") is not None:
        eos_pool.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    if tokenizer.convert_tokens_to_ids("<|dlm_pad|>") is not None:
        eos_pool.add(tokenizer.convert_tokens_to_ids("<|dlm_pad|>"))

    histories = [] if (return_dict_in_generate and output_history) else None

    # pad prompt_ids to max_length
    x = F.pad(prompt_ids, (0, max_length - prompt_ids.shape[1]), value=mask_id)

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # we do not mask the [MASK] tokens so value = 1.0
        attention_mask = F.pad(
            attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0
        )
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        # attention_mask is of shape [B, N]
        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
    else:
        tok_idx = None
        attention_mask = "full"

    # timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

    # this allows user-defined token control of the intermediate steps
    x = generation_tokens_hook_func(None, x, None)
    
    all_token_ids = x[0].detach().cpu().tolist()
    global cache_seq
    cache_seq = all_token_ids.copy()

    total_retry_num = 0
    complete = False
    
    gen_length = max_length - prompt_ids.shape[1]
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for block_idx in range(num_blocks):
        if complete:
            break
        block_mask_index = (
            x[
                :,
                prompt_ids.shape[1] + block_idx * block_length : prompt_ids.shape[1]
                + (block_idx + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        # timesteps = torch.linspace(1, eps, steps_per_block + 1, device=x.device)
        
        # start_ar = False
        for step_idx in range(steps_per_block):
            if complete:
                break
            # if start_ar:
                # break
            start_ar = False

            # ----- sampling code
            logits = model(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(step_idx, x, logits)

            # t = timesteps[step_idx]
            # s = timesteps[step_idx + 1]
            # mask_index = x == mask_token_id
            # block_mask = torch.zeros_like(x, dtype=torch.bool)
            # start = prompt_ids.shape[1] + block_idx * block_length
            # end = prompt_ids.shape[1] + (block_idx + 1) * block_length
            # block_mask[:, start:end] = mask_index[:, start:end]
            # num_mask_token = block_mask.sum() / block_mask.shape[0]
            # number_transfer_tokens_overall = (
                # int(num_mask_token * (1 - s / t)) if step_idx < steps_per_block - 1 else int(num_mask_token)
            # )
            # if trace:
                # print(f"number_transfer_tokens_overall: {number_transfer_tokens_overall}, t: {t}, s: {s}, num_mask_token: {num_mask_token}")
            number_transfer_tokens_overall = num_transfer_tokens[0, step_idx]
            transfer_num = 0 
            while transfer_num < number_transfer_tokens_overall:
                if complete:
                    break
                if start_ar:
                    break

                if trace:
                    print(f"\033[38;2;165;42;42m[Block {block_idx} / {num_blocks}, step {step_idx} / {steps}, token {transfer_num} / {number_transfer_tokens_overall}]\033[0m")

                # number_transfer_tokens = 1
                mask_index = x == mask_token_id

                # ----- constraining code
                tokens_found = False
                one_token_retry_num = 0

                while not tokens_found:
                    # --------- sampling code
                    mask_logits = logits[mask_index]

                    if alg == "entropy":
                        confidence, x0 = sample_tokens(
                            mask_logits,
                            temperature,
                            top_p=top_p,
                            top_k=top_k,
                            neg_entropy=True,
                        )
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")
                    full_confidence = torch.full_like(
                        x, -torch.inf, device=model.device, dtype=logits.dtype
                    )
                    full_confidence[mask_index] = confidence

                    full_confidence[:, prompt_ids.shape[1] + (block_idx + 1) * block_length:] = -torch.inf
    
                    if alg_temp is None or alg_temp == 0:
                        conf, select_index = torch.topk(
                            full_confidence[0], 1
                        )
                    else:
                        raise RuntimeError("alg_temp other than 0 not implemented yet.")
                    
                    x_ = (
                        torch.zeros_like(x, device=model.device, dtype=torch.long)
                        + mask_token_id
                    )
                    x_[mask_index] = x0.clone()

                    index_of_new_token = select_index.item()
                    new_token_vocab_index = x_[0][index_of_new_token]
                    assert logits[0][index_of_new_token][new_token_vocab_index] != -np.inf, "No valid token sampled."

                    all_token_ids[index_of_new_token] = new_token_vocab_index.item()

                    if cache_seq[index_of_new_token] == new_token_vocab_index.item():
                        is_accept = True
                        if trace:
                            print("cache_seq hit, accept directly.")
                    else:
                        p = F.softmax(logits.to(torch.float64), dim=-1)
                        is_accept = check(
                            checker,
                            tokenizer,
                            all_token_ids,
                            prompt_ids,
                            p,
                            logits,
                            index_to_consume,
                            block_idx,
                            block_length,
                            top_k_per_mask,
                            top_n_beam,
                            random_n_beam,
                            trace=trace,
                        )

                    if trace:
                        new_word = tokenizer.decode(new_token_vocab_index)
                        if is_accept:
                            print(
                                f"+++ Accept New word at {index_of_new_token}: {json.dumps(new_word)} ({new_token_vocab_index}), confidence={conf.item():.6f}"
                            )
                        else:
                            print(
                                f"--- Reject {index_of_new_token}: {json.dumps(new_word)} ({new_token_vocab_index}), confidence={conf.item():.6f}"
                            )
             
                    if not is_accept:
                        logits[0][index_of_new_token][
                            new_token_vocab_index
                        ] = -np.inf
                        all_token_ids[index_of_new_token] = mask_id
                        one_token_retry_num += 1
                        total_retry_num += 1
                        if one_token_retry_num >= max_retry_num_total:
                            if trace: 
                                print("Too many retries for one token, start autoregressive generation.")
                            start_ar = True
                            # r = checker.rollback(index_to_consume - (block_idx * block_length + prompt_ids.shape[1]))
                            if trace:
                                print(f"index_to_consume={index_to_consume},{block_idx * block_length + prompt_ids.shape[1]}")
                            break
                    else:
                        tokens_found = True
                        transfer_num += 1
                        one_token_retry_num = 0
                        transfer_index = torch.zeros_like(
                            x_, dtype=torch.bool, device=x_.device
                        )
                        transfer_index[0, select_index] = True
                        if new_token_vocab_index in eos_pool:
                            transfer_index[0, select_index:] = True
                            val = x_[0, select_index].clone()
                            x_[0, select_index:] = val
                            for idx in range(select_index, len(all_token_ids)):
                                all_token_ids[idx] = val.item()
                        x[transfer_index] = x_[transfer_index]
                        # if EOS in generated_words
                        min_eos_index, exist_mask = get_min_eos_index(
                            all_token_ids, block_idx, block_length, prompt_ids
                        )
                        if min_eos_index != -1 and not exist_mask:
                            complete = True
                            break

                        # if has_constrain:
                        for idx in range(index_to_consume, prompt_ids.shape[1] + (block_idx + 1) * block_length):
                            if all_token_ids[idx] == mask_id or all_token_ids[idx] in eos_pool:
                                if idx > index_to_consume:
                                    if trace:
                                        print(f"Consume tokens: {checker.dbg_tokens(all_token_ids[index_to_consume:idx])}, index_to_consume={index_to_consume} -> {idx}")        
                                    tokens_to_consume = all_token_ids[index_to_consume:idx]
                                    assert checker.consume_tokens(tokens_to_consume), f"Tokens {checker.dbg_tokens(tokens_to_consume)} do not conform to grammar."
                                    index_to_consume = idx
                                    if checker.is_stoped():
                                        x[0, index_to_consume] = tokenizer.eos_token_id
                                        complete = True
                                        if trace:
                                            print("Grammar matched complete sequence. Inserting EOS.")
                                        break
                                break

            if not start_ar:
                continue
            if trace:
                print("Start autoregressive generation for the rest tokens in this block.")

            # init
            a = prompt_ids.shape[1] + block_idx * block_length
            b = prompt_ids.shape[1] + (block_idx + 1) * block_length
            for i in range(b, len(cache_seq)):
                cache_seq[i] = mask_id
            index_cache = b
            for i in range(a, b):
                if cache_seq[i] == mask_id or cache_seq[i] in eos_pool:
                    if trace:
                        print(f"All cache_seq from {i} to {b} are mask_id, set index_cache to {i}")
                    index_cache = i
                    break
            free_num = 0
            for i in range(a, index_cache):
                if x[0, i] == mask_id:
                    free_num += 1
            
            x[:, a:b] = mask_id
            if b != x.shape[1]:
                x[0, b:b+2] = mask_id
                all_token_ids[b] = mask_id
                all_token_ids[b+1] = mask_id
            for pos in range(a, b):
                all_token_ids[pos] = mask_id
            
            for i in range(a, index_cache):
                assert cache_seq[i] != mask_id, "Cache seq should have value here."
                for id in eos_pool:
                    assert cache_seq[i] != id, "Should not be EOS/EOT in cache seq here."
                x[0, i] = cache_seq[i]
                all_token_ids[i] = cache_seq[i]
            tokens_to_consume = all_token_ids[index_to_consume:index_cache]

            assert checker.consume_tokens(tokens_to_consume), f"Should be valid sequence in AR generation, {checker.dbg_tokens(tokens_to_consume)}"         
            
            if trace:
                print(f"index_to_consume: {index_to_consume} -> index_cache: {index_cache}")
            index_to_consume = index_cache

            if checker.is_stoped():
                x[0, index_to_consume] = tokenizer.eos_token_id
                return x, total_retry_num, start_time

            min_decode = num_transfer_tokens[0, step_idx] - transfer_num
            
            if trace:
                print(f"min_decode before adjustment: {min_decode}, free_num: {free_num}, num_transfer_tokens: {num_transfer_tokens[0, step_idx]}, token_num: {transfer_num}")
            if min_decode > free_num:
                min_decode = min_decode - free_num
                logits = model(x, attention_mask, tok_idx).logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                logits = generation_logits_hook_func(step_idx, x, logits)

                for i in range(min_decode):
                    index_of_new_token = index_cache + i
                    one_logits = logits[0, index_of_new_token]
                    logits_mask = compute_logits_mask(checker, logits)
                    one_logits[logits_mask == 0] = -float('inf')
                    token_id = torch.argmax(one_logits)
                    assert logits_mask[token_id] == 1, "Selected token should be valid."
                    if trace:
                        new_word = tokenizer.decode(token_id.item())
                        if token_id.item() in eos_pool:
                            new_word = "<EOS>"
                        print(
                            f"index_cache:{index_cache} +++ Accept New word at {index_of_new_token}: {json.dumps(new_word)} ({token_id.item()})"
                        )
                    if token_id.item() in eos_pool:
                        x[0, index_of_new_token] = token_id
                        return x, total_retry_num, start_time
                    else:
                        x[0, index_of_new_token] = token_id
                        all_token_ids[index_of_new_token] = token_id.item()
                        cache_seq[index_of_new_token] = token_id.item()
                        checker.consume_tokens([token_id.item()])
                        if trace:
                            print(f"Consume token: {checker.dbg_tokens([token_id.item()])}, index_to_consume={index_to_consume} -> {index_of_new_token + 1}")
                        index_to_consume = index_of_new_token + 1
                        if checker.is_stoped():
                            x[0, index_of_new_token + 1] = tokenizer.eos_token_id
                            return x, total_retry_num, start_time
            else:
                total_decode = 0
                for i in range(step_idx + 1, steps_per_block):
                    total_decode += num_transfer_tokens[0, i]
                sub_num = free_num - min_decode
                if trace:
                    print(f"total_decode: {total_decode}, sub_num: {sub_num}, steps: {steps_per_block}, step_num: {step_idx}")
                if total_decode <= sub_num:
                    for i in range(step_idx + 1, steps_per_block):
                        num_transfer_tokens[0, i] = 0
                else:
                    total_decode_new = total_decode - sub_num
                    number = steps_per_block - step_idx - 1
                    for i in range(step_idx + 1, steps_per_block):
                        num_transfer_tokens[0, i] = total_decode_new // number
                    remainder = total_decode_new % number
                    for i in range(step_idx + 1, step_idx + 1 + remainder):
                        num_transfer_tokens[0, i] += 1

    return x, total_retry_num, start_time