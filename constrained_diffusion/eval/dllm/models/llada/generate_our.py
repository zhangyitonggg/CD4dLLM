import json
import time
from typing import Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import frozendict
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import concurrent.futures
from transformers import AutoTokenizer, AutoModel

from constrained_diffusion.constrain_utils import (
    EOS,
    EOSType,
)

from constrained_diffusion.checker_tokenizer import Checker
import heapq
import random

eos_id = 126081
eot_id = 126348
mask_id = 126336

cache_seq: List[int] = []

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


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_end_index(all_token_ids, num_block: int, block_length: int, prompt):
    end_index = 0
    for i in range(prompt.shape[1] + (num_block + 1) * block_length - 1, -1, -1):
        w = all_token_ids[i]
        if w not in (eos_id, eot_id, mask_id):
            end_index = i
            break
    return end_index


def min_eos_or_eot_index(all_token_ids, num_block: int, block_length: int, prompt):
    exist_mask = False
    for i in range(prompt.shape[1] + num_block * block_length, prompt.shape[1] + (num_block + 1) * block_length):
        w = all_token_ids[i]
        if w in (eos_id, eot_id):
            return i, exist_mask
        if w == mask_id:
            exist_mask = True
    return -1, exist_mask


def compute_logits_mask(
    checker: Checker,
):
    logits_mask = checker.compute_mask()
    logits_mask_long = torch.zeros(126464)
    logits_mask_long[0:len(logits_mask)] = logits_mask
    logits_mask = logits_mask_long
    logits_mask[mask_id] = 0
    if logits_mask[eos_id] == 1:
        logits_mask[eot_id] = 1
    if logits_mask[eot_id] == 1:
        logits_mask[eos_id] = 1
    return logits_mask


def validate(
    checker: Checker,
    all_token_ids: List[int],
    p: torch.Tensor,
    index_to_consume: int,
    last_token_index: int,
    min_eos_eot_index: int,          
    trace: bool = False,
    top_k_per_mask: int = 10,
    top_n_beam: int = 30,
    random_n_beam: int = 20,
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
                    p[0, pos][eos_id] = 0
                    p[0, pos][eot_id] = 0
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
    if min_eos_eot_index == -1:
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
        largest_seq[pos] = eos_id

    for pos in range(last_token_index + 1, min_eos_eot_index):
        logits_mask = compute_logits_mask(checker)
        assert logits_mask.sum() > 0, "There should be valid tokens to select."
        if logits_mask[eos_id] == 1 or logits_mask[eot_id] == 1:
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
        assert checker.rollback(min_eos_eot_index - index_to_consume)
        if trace:
            print(f"old_cache_seq: {checker.dbg_tokens(cache_seq[index_to_consume: min_eos_eot_index])}")
            print(f"new_cache_seq: {checker.dbg_tokens(largest_seq[index_to_consume: min_eos_eot_index])}")
        cache_seq = largest_seq.copy()
        return True

    assert checker.rollback(min_eos_eot_index - index_to_consume), "Should rollback"
    return False

    
def check(
    checker: Checker,
    
    all_token_ids: List[int],
    prompt_ids,
    p: torch.Tensor,

    index_to_consume: int,
    block_num: int,
    block_length: int,
    top_k_per_mask: int = 5,
    top_n_beam: int = 30,
    random_n_beam: int = 20,
    trace: bool = False,    
):
    last_token_index = get_end_index(all_token_ids, block_num, block_length, prompt_ids)
    min_eos_eot_index, _ = min_eos_or_eot_index(all_token_ids, block_num, block_length, prompt_ids)

    if min_eos_eot_index != -1 and min_eos_eot_index < last_token_index:
        return False
    
    accept = validate(
        checker=checker,
        all_token_ids=all_token_ids,
        p=p,
        index_to_consume=index_to_consume,
        last_token_index=last_token_index,
        min_eos_eot_index=min_eos_eot_index,
        trace=trace,
        top_k_per_mask=top_k_per_mask,
        top_n_beam=top_n_beam,
        random_n_beam=random_n_beam,
    )
    if accept:
        return True
    else:
        return False


@torch.no_grad()
def generate(
    model, 
    tokenizer,
    prompt_ids,
    input_len: int,
    grammar: str,
    
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    remasking: str = "low_confidence",
    trace: bool = False,
    change_logits: bool = False,

    top_k_per_mask: int = 5,
    top_n_beam: int = 3,
    random_n_beam: int = 3,
    max_retry_num_total: int = 5,
):    
    checker = Checker(grammar=grammar, model_name="LLaDA")
    if checker.consume_tokens(prompt_ids[0][input_len:].tolist()) == False:
        raise ValueError("Prompt does not conform to grammar.")
    index_to_consume = prompt_ids.shape[1]

    start_time = time.monotonic()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks  

    x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, : prompt_ids.shape[1]] = prompt_ids.clone()  

    all_token_ids = x[0].detach().cpu().tolist()
    global cache_seq
    cache_seq = all_token_ids.copy()

    total_retry_num = 0
    complete = False
    for block_num in range(num_blocks):
        if complete:
            break
        block_mask_index = (
            x[
                :,
                prompt_ids.shape[1] + block_num * block_length : prompt_ids.shape[1]
                + (block_num + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for step_num in range(steps):
            # If the generation is complete, we can stop early.
            if complete:
                break
            # if start_ar:
                # break
            start_ar = False

            logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature)

            # greedy constrained sampling for multi-step
            token_num_to_decode = num_transfer_tokens[0, step_num]
            token_num = 0
            while token_num < token_num_to_decode:
                # If the generation is complete, we can stop early.
                if complete:
                    # print("test1")
                    break
                if start_ar:
                    break

                if trace:
                    print(f"\033[38;2;165;42;42m[Block {block_num} / {num_blocks}, step {step_num} / {steps}, token {token_num} / {num_transfer_tokens[0, step_num]}]\033[0m")
                
                mask_indexs = x == mask_id

                token_found = False
                one_token_retry_num = 0

                while not token_found:            
                    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
                    
                    if remasking == "low_confidence":
                        p = F.softmax(logits.to(torch.float64), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                        )  # b, l
                    else:
                        raise NotImplementedError(remasking)

                    x0_p[
                        :, prompt_ids.shape[1] + (block_num + 1) * block_length :
                    ] = -np.inf
                    x0 = torch.where(mask_indexs, x0, x)
                    confidences = torch.where(mask_indexs, x0_p, -np.inf)
                    # TODO 
                    if False:
                        confs, select_indexs = torch.topk(
                                                        input=confidences[0], 
                                                        k=token_num_to_decode - token_num
                                                    )
                        assert select_indexs.shape[0] > 0, "No tokens to transfer"
                        pos_min = torch.argmin(select_indexs)
                        select_index = select_indexs[pos_min]
                        conf = confs[pos_min]
                    conf, select_index = torch.topk(
                                                    input=confidences[0], 
                                                    k=1
                                                )

                    index_of_new_token = select_index.item()
                    new_token_vocab_index = x0[0][index_of_new_token]
                    assert logits_with_noise[0][index_of_new_token][new_token_vocab_index] != -np.inf, "No valid token found"

                    all_token_ids[index_of_new_token] = new_token_vocab_index.item()
                    # if has_constrain:
                    if trace:
                        print(f"Checking token at position {index_of_new_token}, token id: {new_token_vocab_index.item()}")
                        print(f"cache_seq[index_of_new_token]: {cache_seq[index_of_new_token]}")
                        print(f"new_token_vocab_index.item(): {new_token_vocab_index.item()}")

                    if cache_seq[index_of_new_token] == new_token_vocab_index.item():
                        is_accept = True
                        if trace:
                            print("cache_seq hit, accept directly.")
                    else:
                        is_accept = check(
                            checker=checker,
                            all_token_ids=all_token_ids,
                            prompt_ids=prompt_ids,
                            p=p,
                            index_to_consume=index_to_consume,
                            block_num=block_num,
                            block_length=block_length,
                            top_k_per_mask=top_k_per_mask,
                            top_n_beam=top_n_beam,
                            random_n_beam=random_n_beam,
                            trace=trace
                        )

                    if trace:                
                        new_word = tokenizer.decode(new_token_vocab_index)
                        if new_word is EOS:
                            new_word = "<EOS>"
                        if is_accept:
                            print(
                                f"+++ Accept New word at {index_of_new_token}: {json.dumps(new_word)} ({new_token_vocab_index}), confidence={conf.item():.6f}"
                            )
                        else:
                            print(
                                f"--- Reject {index_of_new_token}: {json.dumps(new_word)} ({new_token_vocab_index}), confidence={conf.item():.6f}"
                            )
                    
                    if not is_accept:
                        logits_with_noise[0][index_of_new_token][
                            new_token_vocab_index
                        ] = -np.inf
                        if change_logits:
                            logits[0][index_of_new_token][
                                new_token_vocab_index
                            ] = -np.inf

                        # generated_words[index_of_new_token] = None
                        all_token_ids[index_of_new_token] = mask_id

                        one_token_retry_num += 1
                        total_retry_num += 1
                        # TODO
                        if one_token_retry_num >= max_retry_num_total:
                            if trace:
                                print("Too many retries for one token, start autoregressive generation.")
                            start_ar = True
                            # print("Rollback to index:", index_to_consume - (block_num * block_length + prompt_ids.shape[1]))
                            # r = checker.rollback(index_to_consume - (block_num * block_length + prompt_ids.shape[1]))
                            if trace:
                                print(f"index_to_consume={index_to_consume},{block_num * block_length + prompt_ids.shape[1]}")
                            # checker.reset()
                            break
                    else:
                        token_found = True
                        token_num += 1
                        one_token_retry_num = 0
                        transfer_index = torch.zeros_like(
                            x0, dtype=torch.bool, device=x0.device
                        )
                        transfer_index[0, select_index] = True
                        if new_token_vocab_index in (eos_id, eot_id):
                            transfer_index[0, select_index:] = True
                            val = x0[0, select_index].clone()
                            x0[0, select_index:] = val
                            # all_token_ids[select_index:] = [val.item()] * (len(all_token_ids) - select_index)
                            for idx in range(select_index, len(all_token_ids)):
                                all_token_ids[idx] = val.item()
                        x[transfer_index] = x0[transfer_index]
                        # if EOS in generated_words:
                        min_eos_eot_index, exist_mask = min_eos_or_eot_index(all_token_ids, block_num, block_length, prompt_ids)
                        if min_eos_eot_index != -1 and not exist_mask:
                            complete = True
                            break 

                        # if has_constrain:
                        for idx in range(index_to_consume, prompt_ids.shape[1] + (block_num + 1) * block_length):
                            if all_token_ids[idx] in (mask_id, eos_id, eot_id):
                                if idx > index_to_consume:
                                    if trace:
                                        print(f"Consume tokens: {checker.dbg_tokens(all_token_ids[index_to_consume:idx])}, index_to_consume={index_to_consume} -> {idx}")
                                    tokens_to_consume = all_token_ids[index_to_consume:idx]
                                    assert checker.consume_tokens(tokens_to_consume)
                                    index_to_consume = idx
                                    if checker.is_stoped():
                                        x[0, index_to_consume] = eos_id 
                                        complete = True
                                        if trace:
                                            print("Grammar matched complete sequence. Inserting EOS.")
                                        break
                                break

            if not start_ar:
                continue
            if trace:
                print("Start autoregressive generation.")
            
            # init
            a = prompt_ids.shape[1] + block_num * block_length
            b = prompt_ids.shape[1] + (block_num + 1) * block_length
            for i in range(b, len(cache_seq)):
                cache_seq[i] = mask_id
            index_cache = b
            for i in range(a, b):
                if cache_seq[i] in (eos_id, eot_id, mask_id):
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
            for i in range(a, b):
                all_token_ids[i] = mask_id

            
            for i in range(a, index_cache):
                assert cache_seq[i] != mask_id, "Cache seq should have value here."
                assert cache_seq[i] != eos_id and cache_seq[i] != eot_id, "Should not be EOS/EOT in cache seq here."
                x[0, i] = cache_seq[i]
                all_token_ids[i] = cache_seq[i]
            tokens_to_consume = all_token_ids[index_to_consume:index_cache]

            assert checker.consume_tokens(tokens_to_consume), f"Should be valid sequence in AR generation, {checker.dbg_tokens(tokens_to_consume)}"

            if trace:
                print(f"index_to_consume: {index_to_consume} -> index_cache: {index_cache}")
            index_to_consume = index_cache

            if checker.is_stoped():
                x[0, index_to_consume] = eos_id
                return x, total_retry_num, start_time

            min_decode = num_transfer_tokens[0, step_num] - token_num
            
            if trace:
                print(f"min_decode before adjustment: {min_decode}, free_num: {free_num}, num_transfer_tokens: {num_transfer_tokens[0, step_num]}, token_num: {token_num}")
            if min_decode > free_num:
                min_decode = min_decode - free_num
                logits = model(x).logits
                logits_with_noise = add_gumbel_noise(logits, temperature)
                for i in range(min_decode):
                    index_of_new_token = index_cache + i
                    one_logits = logits_with_noise[0, index_of_new_token]
                    logits_mask = compute_logits_mask(checker)
                    one_logits[logits_mask == 0] = -float('inf')
                    token_id = torch.argmax(one_logits)
                    assert logits_mask[token_id] == 1, "Selected token should be valid."
                    if trace:
                        conf = F.softmax(one_logits, dim=-1)[token_id]
                        new_word = tokenizer.decode(token_id.item())
                        if new_word is EOS:
                            new_word = "<EOS>"
                        print(
                            f"index_cache:{index_cache} +++ Accept New word at {index_of_new_token}: {json.dumps(new_word)} ({token_id.item()}), confidence={conf.item():.6f}"
                        )
                    if token_id.item() in (eos_id, eot_id):
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
                            x[0, index_of_new_token + 1] = eos_id
                            return x, total_retry_num, start_time
            else:
                total_decode = 0
                for i in range(step_num + 1, steps):
                    total_decode += num_transfer_tokens[0, i]
                sub_num = free_num - min_decode
                if trace:
                    print(f"total_decode: {total_decode}, sub_num: {sub_num}, steps: {steps}, step_num: {step_num}")
                if total_decode <= sub_num:
                    for i in range(step_num + 1, steps):
                        num_transfer_tokens[0, i] = 0
                else:
                    total_decode_new = total_decode - sub_num
                    number = steps - step_num - 1
                    for i in range(step_num + 1, steps):
                        num_transfer_tokens[0, i] = total_decode_new // number
                    remainder = total_decode_new % number
                    for i in range(step_num + 1, step_num + 1 + remainder):
                        num_transfer_tokens[0, i] += 1
            
    return x, total_retry_num, start_time