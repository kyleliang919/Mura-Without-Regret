# Mura-Without-Regret
Seamlessly incorporating Muon with Lora by modifying backpropagation

* **SFT experiments** (4 combos) suggested in the doc (Llama-3.2-1B / Llama-3.1-8B × Tulu-3-SFT-mixture / OpenThoughts-114k) with LoRA on **all linear layers** and recommended rank/capacity. ([Hugging Face][1])
* **GRPO experiments** (3 combos) from the doc (Llama-3.1-8B-Base on GSM8k/DeepMath-103K, Qwen3-8B-Base on DeepMath-103K) and the **SmolLM3** LoRA-vs-FullFT parameter table (so you can replicate the figure) using the official `grpo.py` reference script. ([Hugging Face][1])

> Notes this launcher bakes in from the sources:
> * Apply LoRA to **all-linear** weights (`--lora_target_modules all-linear`). ([Hugging Face][1])
> * SFT: LoRA rank **256** (“post-training scale” guidance), LR ≈ **2e-4** in the example. ([Hugging Face][1])
> * GRPO: small LoRA ranks (**1–32**, default 1), and **effective batch < 32**. ([Hugging Face][1])
> * SmolLM3 GRPO table (LoRA vs FullFT): specific LR and lengths used in the doc’s reproduction table. ([Hugging Face][1])
> * Official GRPO script (reward function & Trainer wiring) lives here: `burtenshaw/lora-without-regrets/grpo.py`. ([Hugging Face][2])

---

### How to use

1. **Install** TRL CLI (and deps used by the GRPO script):

```bash
pip install "trl>=0.24" peft trackio math-verify latex2sympy2-extended
# (optional) accelerate deepspeed bitsandbytes if you plan to scale
```

2. **Fetch the official GRPO reference script** (used for all GRPO runs):

```bash
wget -O grpo.py https://huggingface.co/datasets/burtenshaw/lora-without-regrets/resolve/main/grpo.py
```

3. **List every command** this launcher will run:

```bash
python launch_lora_without_regret.py list
# or filter:
python launch_lora_without_regret.py list --filter sft-
```

4. **Dry-run a single experiment** (print command only):

```bash
python launch_lora_without_regret.py run sft-llama-1b-tulu3 --dry
```

5. **Execute a single experiment**:

```bash
python launch_lora_without_regret.py run grpo-smollm3-lora
```

6. **Run everything** (or a subset):

```bash
python launch_lora_without_regret.py run-all --dry     # just show all commands
python launch_lora_without_regret.py run-all           # execute all
python launch_lora_without_regret.py run-many sft- grpo-llama8b-
```

---

### What’s included (exact commands)

Run `list` to see them on your machine; here they are for convenience:

#### SFT (LoRA on all-linear, r=256, LR=2e-4; eff batch size < 32) ([Hugging Face][1])

* `sft-llama-1b-tulu3`
  `trl sft --model_name_or_path meta-llama/Llama-3.2-1B-Instruct --dataset_name allenai/tulu-3-sft-mixture --use_peft --lora_target_modules all-linear --lora_r 256 --lora_alpha 16 --lora_dropout 0.0 --learning_rate 2e-4 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --num_train_epochs 1 --report_to trackio --output_dir runs/sft-llama-1b-tulu3 --run_name sft-llama-1b-tulu3`

* `sft-llama-1b-openthoughts`
  (same flags, dataset = `open-thoughts/OpenThoughts-114k`)

* `sft-llama-8b-tulu3`
  (model = `meta-llama/Llama-3.1-8B-Instruct`, dataset = `allenai/tulu-3-sft-mixture`)

* `sft-llama-8b-openthoughts`
  (model = `meta-llama/Llama-3.1-8B-Instruct`, dataset = `open-thoughts/OpenThoughts-114k`)

#### GRPO (LoRA small rank for RL; reward wired via reference script) ([Hugging Face][1])

* `grpo-llama8b-gsm8k`
  `python grpo.py --model_name_or_path meta-llama/Llama-3.1-8B-Base --dataset_name openai/gsm8k --output_dir runs/grpo-llama8b-gsm8k --run_name grpo-llama8b-gsm8k --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --generation_batch_size 8 --num_generations 8 --learning_rate 5e-5 --use_peft --lora_target_modules all-linear --lora_r 1 --lora_alpha 32 --lora_dropout 0.0`

* `grpo-llama8b-deepmath`
  (dataset = `HuggingFaceH4/DeepMath-103K`)

* `grpo-qwen8b-deepmath`
  (model = `Qwen/Qwen3-8B-Base`, dataset = `HuggingFaceH4/DeepMath-103K`)

#### GRPO SmolLM3 – **LoRA vs FullFT** (matches doc’s table) ([Hugging Face][1])

* `grpo-smollm3-lora`
  `python grpo.py --model_name_or_path HuggingFaceTB/SmolLM3-3B --dataset_name HuggingFaceH4/OpenR1-Math-220k-default-verified --output_dir runs/grpo-smollm3-lora --run_name grpo-smollm3-lora --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --generation_batch_size 8 --num_generations 8 --learning_rate 1e-5 --max_prompt_length 1024 --max_completion_length 4096 --use_peft --lora_target_modules all-linear --lora_r 1 --lora_alpha 32 --lora_dropout 0.0`

* `grpo-smollm3-fullft`
  `python grpo.py --model_name_or_path HuggingFaceTB/SmolLM3-3B --dataset_name HuggingFaceH4/OpenR1-Math-220k-default-verified --output_dir runs/grpo-smollm3-fullft --run_name grpo-smollm3-fullft --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --generation_batch_size 8 --num_generations 8 --learning_rate 1e-6 --max_prompt_length 1024 --max_completion_length 4096`

---

### References (what this launcher encodes)

* TRL doc “LoRA Without Regret”: models/datasets, “all-linear” target, SFT example hyperparams, RL rank guidance, batch-size note, and SmolLM3 LoRA vs FullFT table. ([Hugging Face][1])
* Official GRPO reference script (reward function + GRPOTrainer wiring). ([Hugging Face][2])
* TRL CLI usage & flags. ([Hugging Face][3])

If you want, I can also add toggles for **DeepSpeed** / **FSDP** profiles via `--accelerate_config` presets (e.g., `zero2`, `zero3`) and a `--hf_token` guard that checks env. ([Hugging Face][3])

[1]: https://huggingface.co/docs/trl/main/en/lora_without_regret?grpo=local&sft=python "LoRA Without Regret"
[2]: https://huggingface.co/datasets/burtenshaw/lora-without-regrets/blob/main/grpo.py "grpo.py · burtenshaw/lora-without-regrets at main"
[3]: https://huggingface.co/docs/trl/main/en/clis "Command Line Interfaces (CLIs)"
