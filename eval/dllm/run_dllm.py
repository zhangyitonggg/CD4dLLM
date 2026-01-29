import subprocess
import time
from math import ceil
from pathlib import Path

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
N = 1  # Set the max number of allowed processes per GPU
GPUSIZE = 40
models = [
    ("Dream-org/Dream-v0-Instruct-7B", 7),
    ("GSAI-ML/LLaDA-8B-Instruct", 8),
]

def compute_needed_gpus(size_model, size_gpu):
    return (size_model * 2 * 1.2) / size_gpu

subsets = [
    "jsonschema",
    "THUDM/humaneval-x/cpp",
    "smiles",
]
step_sizes = [128]
block_lengths = [32]
temps = ["0.2"]

change_logitss = [True]
beam_sizes = [10]
# top_n_beams = [5]
# random_n_beams = [5]
max_retry_num_totals = [5]
top_k_per_masks = [5]

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
configs = [
    ("", ""),
]

NO_CONSTRAINED = 0
BASELINE_CONSTRAINED = 1
OUR_CONSTRAINED = 2

constraineds = [OUR_CONSTRAINED, BASELINE_CONSTRAINED, NO_CONSTRAINED]

def find_available_gpus(gpus, n):
    found_gpus = []
    for gpu in gpus:
        process_count = int(
            subprocess.check_output(
                [
                    "/bin/bash",
                    "-c",
                    f"nvidia-smi -i {gpu} --query-compute-apps=pid --format=csv,noheader | wc -l",
                ],
            ).strip()
        )
        if process_count < n:
            found_gpus.append(gpu)
    return found_gpus

total_configs = []
for seed in seeds:
    for subset in subsets:
        for step_size in step_sizes:
            for temp in temps:
                for block_length in block_lengths:
                    for config, name in configs:
                        for model, model_size in models:
                            for constrained in constraineds:
                                if constrained == OUR_CONSTRAINED:
                                    for top_k_per_mask in top_k_per_masks:
                                        for beam_size in beam_sizes:
                                            top_n_beam = beam_size // 2
                                            random_n_beam = beam_size - top_n_beam
                                            for change_logits in change_logitss:
                                                for max_retry_num_total in (
                                                    max_retry_num_totals
                                                ):
                                                    overall_config = (
                                                        seed,
                                                        temp,
                                                        block_length,
                                                        config,
                                                        name,
                                                        constrained,
                                                        model,
                                                        model_size,
                                                        subset,
                                                        step_size,
                                                        change_logits,
                                                        top_k_per_mask,
                                                        top_n_beam,
                                                        random_n_beam,
                                                        max_retry_num_total,
                                                    )
                                                    if overall_config not in total_configs:
                                                        total_configs.append(
                                                            overall_config
                                                        )
                                else:
                                    change_logits = None
                                    top_k_per_mask = None
                                    top_n_beam = None
                                    random_n_beam = None
                                    max_retry_num_total = None
                                    overall_config = (
                                        seed,
                                        temp,
                                        block_length,
                                        config,
                                        name,
                                        constrained,
                                        model,
                                        model_size,
                                        subset,
                                        step_size,
                                        change_logits,
                                        top_k_per_mask,
                                        top_n_beam,
                                        random_n_beam,
                                        max_retry_num_total,
                                    )
                                    if overall_config not in total_configs:
                                        total_configs.append(overall_config)

remaining_configs = total_configs.copy()
print(f"Total configs to run: {len(remaining_configs)}")
running_configs = list()
while remaining_configs or running_configs:
    # reinsert crashed programs
    for config, pipe in running_configs:
        if pipe.poll() is not None:
            running_configs.remove((config, pipe))
            if pipe.returncode != 0:
                remaining_configs.append(config)
    cuda_devices, needed_gpus = [], 1
    cuda_devices = find_available_gpus(GPUS, N)
    total_config = None
    for total_config in remaining_configs:
        (
            seed,
            temp,
            block_length,
            config,
            name,
            constrained,
            model,
            model_size,
            subset,
            step_size,
            change_logits,
            top_k_per_mask,
            top_n_beam,
            random_n_beam,
            max_retry_num_total,
        ) = total_config
        needed_gpus = compute_needed_gpus(model_size, GPUSIZE)
        # print(f"Model: {model}, size: {model_size}B, needed gpus: {needed_gpus}")
        # print(f"len(GPUS): {len(GPUS)}")
        if needed_gpus > len(GPUS):
            print(f"model {model} is too large, skipping")
            remaining_configs.remove(total_config)
            continue
        if len(cuda_devices) >= needed_gpus:
            break
    if len(cuda_devices) < needed_gpus:
        # print(f"Needed gpus: {needed_gpus}, available gpus: {cuda_devices}")
        print("No available GPU found. Waiting...")
        time.sleep(300)
        continue
    if total_config is None:
        print("No remaining config found. Exiting...")
        time.sleep(300)
        continue
    remaining_configs.remove(total_config)
    cuda_devices = cuda_devices[: int(ceil(needed_gpus))]

    if constrained == NO_CONSTRAINED:
        suffix = "nc"
    elif constrained == BASELINE_CONSTRAINED:
        suffix = "bc"
    elif constrained == OUR_CONSTRAINED:
        suffix = "oc"
    else:
        suffix = "unknown"
    
    if constrained != OUR_CONSTRAINED:
        command = (
            f"PYTHONPATH=. CUDA_VISIBLE_DEVICES={','.join(str(i) for i in cuda_devices)} python3 -m constrained_diffusion.eval.dllm.generic_inference "
            f"--max-tokens 256 --model_name {model} --seed {seed} --temp {temp} --trace False  --dataset-name '{subset}' --steps {step_size} "
            f"--block_length {block_length} "
            f"--constrained {constrained} --output_file 'results/{model}_{subset.replace('/', '_')}_s={seed}_t={temp}_bl={block_length}_sz={step_size}{name}_{suffix}.jsonl' {config}"
        )
    else:
        command = (
            f"PYTHONPATH=. CUDA_VISIBLE_DEVICES={','.join(str(i) for i in cuda_devices)} python3 -m constrained_diffusion.eval.dllm.generic_inference "
            f"--max-tokens 256 --model_name {model} --seed {seed} --temp {temp} --trace False  --dataset-name '{subset}' --steps {step_size} "
            f"--block_length {block_length} "
            f"--constrained {constrained} --change_logits {change_logits} --top_k_per_mask {top_k_per_mask} --top_n_beam {top_n_beam} --random_n_beam {random_n_beam} --max_retry_num_total {max_retry_num_total} "
            f"--output_file 'results/{model}_{subset.replace('/', '_')}_s={seed}_t={temp}_bl={block_length}_sz={step_size}{name}_tm={top_k_per_mask}_tn={top_n_beam}_rn={random_n_beam}_mr={max_retry_num_total}_{suffix}.jsonl' {config}"
        )
    print("+ " + command, flush=True)
    pipe = subprocess.Popen(
        ["/bin/bash", "-c", command],
        cwd=str(Path(__file__).parent.parent.parent),
    )
    running_configs.append((total_config, pipe))
    time.sleep(30)