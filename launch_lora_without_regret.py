#!/usr/bin/env python3
import argparse, subprocess, shlex, os, sys
from textwrap import dedent

# ----------------------------
# Helpers
# ----------------------------
def run(cmd, dry):
    print(cmd)
    if not dry:
        # run in a login shell so user env (like HF_TOKEN) is visible
        subprocess.run(cmd, shell=True, check=True)

def add_common_sft_flags(base, outdir, run_name, lr="2e-4", per_device=1, grad_accum=4, epochs=1,
                         lora_r=256, lora_alpha=16, lora_dropout=0.0, target="all-linear"):
    # TRL CLI: https://huggingface.co/docs/trl/main/en/clis
    # Use --use_peft for LoRA; keep effective batch < 32 as per doc
    extra = f"""\
        --use_peft \
        --lora_target_modules {target} \
        --lora_r {lora_r} \
        --lora_alpha {lora_alpha} \
        --lora_dropout {lora_dropout} \
        --learning_rate {lr} \
        --per_device_train_batch_size {per_device} \
        --gradient_accumulation_steps {grad_accum} \
        --num_train_epochs {epochs} \
        --report_to trackio \
        --output_dir {outdir} \
        --run_name {run_name}
    """
    return base + " " + " ".join(extra.split())

def mk_sft_cmd(model, dataset, outdir, run_name, **kw):
    base = f"trl sft --model_name_or_path {model} --dataset_name {dataset}"
    return add_common_sft_flags(base, outdir, run_name, **kw)

def mk_grpo_cmd_with_ref_script(grpo_py, model, dataset,
                                outdir, run_name,
                                use_lora=True,
                                lr=None,
                                lora_r=1, lora_alpha=32, lora_dropout=0.0,
                                max_prompt_len=None, max_completion_len=None,
                                per_device=1, grad_accum=4,
                                num_generations=8, gen_batch=8,
                                steps=None, push_to_hub=False):
    """
    Use the official reference GRPO script from the doc:
    https://huggingface.co/datasets/burtenshaw/lora-without-regrets/blob/main/grpo.py
    """
    pieces = [
        f"python {grpo_py}",
        f"--model_name_or_path {model}",
        f"--dataset_name {dataset}",
        f"--output_dir {outdir}",
        f"--run_name {run_name}",
        f"--per_device_train_batch_size {per_device}",
        f"--gradient_accumulation_steps {grad_accum}",
        f"--generation_batch_size {gen_batch}",
        f"--num_generations {num_generations}"
    ]
    if lr: pieces += [f"--learning_rate {lr}"]
    if max_prompt_len: pieces += [f"--max_prompt_length {max_prompt_len}"]
    if max_completion_len: pieces += [f"--max_completion_length {max_completion_len}"]
    if steps: pieces += [f"--max_steps {steps}"]
    if push_to_hub: pieces += ["--push_to_hub"]
    if use_lora:
        pieces += [
            "--use_peft",
            f"--lora_target_modules all-linear",
            f"--lora_r {lora_r}",
            f"--lora_alpha {lora_alpha}",
            f"--lora_dropout {lora_dropout}",
        ]
    return " ".join(pieces)

def all_experiments():
    """
    Returns a dict: {exp_name: [cmds]}
    Each exp_name maps to one command string (you can run them individually).
    """
    exps = {}

    # ----------------------------
    # SFT (doc-listed combos)
    # ----------------------------
    # Guidance: LoRA on all-linear, rank 256; example LR 2e-4; keep eff batch < 32.
    sft_defs = [
        # (name, model, dataset)
        ("sft-llama-1b-tulu3",       "meta-llama/Llama-3.2-1B-Instruct", "allenai/tulu-3-sft-mixture"),
        ("sft-llama-1b-openthoughts","meta-llama/Llama-3.2-1B-Instruct", "open-thoughts/OpenThoughts-114k"),
        ("sft-llama-8b-tulu3",       "meta-llama/Llama-3.1-8B-Instruct", "allenai/tulu-3-sft-mixture"),
        ("sft-llama-8b-openthoughts","meta-llama/Llama-3.1-8B-Instruct", "open-thoughts/OpenThoughts-114k"),
    ]
    for name, model, dataset in sft_defs:
        outdir = f"runs/{name}"
        cmd = mk_sft_cmd(
            model=model,
            dataset=dataset,
            outdir=outdir,
            run_name=name,
            lr="2e-4",              # example LR from doc snippet
            per_device=1,
            grad_accum=4,
            epochs=1,
            lora_r=256,             # recommended for SFT "post-training scale"
            lora_alpha=16,
            lora_dropout=0.0
        )
        exps[name] = [cmd]

    # ----------------------------
    # GRPO (doc-listed combos)
    # ----------------------------
    # Use official grpo.py with reward function wired in.
    # You can download it with:
    #   wget -O grpo.py https://huggingface.co/datasets/burtenshaw/lora-without-regrets/resolve/main/grpo.py
    grpo_py = os.environ.get("GRPO_PY", "grpo.py")

    grpo_defs = [
        ("grpo-llama8b-gsm8k",   "meta-llama/Llama-3.1-8B-Base", "openai/gsm8k"),
        ("grpo-llama8b-deepmath","meta-llama/Llama-3.1-8B-Base", "HuggingFaceH4/DeepMath-103K"),
        ("grpo-qwen8b-deepmath", "Qwen/Qwen3-8B-Base",           "HuggingFaceH4/DeepMath-103K"),
    ]
    for name, model, dataset in grpo_defs:
        outdir = f"runs/{name}"
        cmd = mk_grpo_cmd_with_ref_script(
            grpo_py=grpo_py,
            model=model,
            dataset=dataset,
            outdir=outdir,
            run_name=name,
            use_lora=True,
            lr="5e-5",             # doc example for GRPO snippet
            lora_r=1,              # RL typically low rank (1–32); default 1
            lora_alpha=32,
            lora_dropout=0.0,
            per_device=1,
            grad_accum=4,
            num_generations=8,
            gen_batch=8
        )
        exps[name] = [cmd]

    # ----------------------------
    # GRPO: SmolLM3 LoRA vs FullFT (doc’s reproduction table)
    # ----------------------------
    # Parameters taken from the table in the doc (same dataset & lengths).
    # LoRA:
    name = "grpo-smollm3-lora"
    exps[name] = [mk_grpo_cmd_with_ref_script(
        grpo_py=grpo_py,
        model="HuggingFaceTB/SmolLM3-3B",
        dataset="HuggingFaceH4/OpenR1-Math-220k-default-verified",
        outdir=f"runs/{name}",
        run_name=name,
        use_lora=True,
        lr="1e-5",
        lora_r=1,
        lora_alpha=32,
        lora_dropout=0.0,
        max_prompt_len=1024,
        max_completion_len=4096,
        per_device=1, grad_accum=4,
        num_generations=8, gen_batch=8
    )]
    # Full fine-tuning (no LoRA):
    name = "grpo-smollm3-fullft"
    exps[name] = [mk_grpo_cmd_with_ref_script(
        grpo_py=grpo_py,
        model="HuggingFaceTB/SmolLM3-3B",
        dataset="HuggingFaceH4/OpenR1-Math-220k-default-verified",
        outdir=f"runs/{name}",
        run_name=name,
        use_lora=False,
        lr="1e-6",
        max_prompt_len=1024,
        max_completion_len=4096,
        per_device=1, grad_accum=4,
        num_generations=8, gen_batch=8
    )]

    return exps

# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Central launcher for 'LoRA Without Regret' experiments (TRL).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # list
    sp_list = sub.add_parser("list", help="List all experiment commands.")
    sp_list.add_argument("--filter", type=str, default="", help="Substring to filter experiment names.")

    # run one
    sp_one = sub.add_parser("run", help="Run a single experiment by name.")
    sp_one.add_argument("name", type=str, help="Experiment name (see `list`).")
    sp_one.add_argument("--dry", action="store_true", help="Print command(s) only, do not execute.")

    # run many
    sp_many = sub.add_parser("run-many", help="Run many experiments by name patterns.")
    sp_many.add_argument("patterns", nargs="+", help="Name substrings (any experiment whose name contains one will be run).")
    sp_many.add_argument("--dry", action="store_true", help="Print command(s) only, do not execute.")

    # run all
    sp_all = sub.add_parser("run-all", help="Run ALL experiments.")
    sp_all.add_argument("--dry", action="store_true", help="Print command(s) only, do not execute.")

    args = p.parse_args()
    exps = all_experiments()

    if args.cmd == "list":
        filt = args.filter
        for k, v in exps.items():
            if filt and filt not in k:
                continue
            print(f"# {k}")
            for c in v:
                print(c)
            print()
        return

    if args.cmd == "run":
        if args.name not in exps:
            print(f"Unknown experiment: {args.name}")
            print("Use: list")
            sys.exit(2)
        for c in exps[args.name]:
            run(c, args.dry)
        return

    if args.cmd == "run-many":
        names = [k for k in exps if any(pat in k for pat in args.patterns)]
        if not names:
            print("No experiments matched.")
            sys.exit(2)
        for n in names:
            for c in exps[n]:
                run(c, args.dry)
        return

    if args.cmd == "run-all":
        for n in sorted(exps.keys()):
            for c in exps[n]:
                run(c, args.dry)
        return

if __name__ == "__main__":
    main()
