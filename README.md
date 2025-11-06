# 🚀 LoRA (Mura) Without Regret — Multi-GPU Launcher

A unified launcher script to reproduce all experiments from the [**TRL “LoRA Without Regret” guide**](https://huggingface.co/docs/trl/main/en/lora_without_regret), including both **Supervised Fine-Tuning (SFT)** and **GRPO (RL)** runs.  
Fully configured for **8-GPU parallel training** and **Weights & Biases (wandb)** logging.

---

## ⚡️ Quickstart

```bash
# 1. Install dependencies
pip install "trl>=0.24" peft wandb accelerate

# 2. Log in to wandb
wandb login

# 3. Download the official GRPO script
wget -O grpo.py https://huggingface.co/datasets/burtenshaw/lora-without-regrets/resolve/main/grpo.py

# 4. Run any experiment (multi-GPU + wandb)
python launch_lora_without_regret.py run sft-llama-1b-tulu3
````

> 💡 To view all commands:
>
> ```bash
> python launch_lora_without_regret.py list
> ```

---

## 🧱 Requirements

Install core packages:

```bash
pip install "trl>=0.24" peft wandb accelerate
```

Authenticate Weights & Biases:

```bash
wandb login
```

Fetch the official GRPO reference script (for reward shaping and policy updates):

```bash
wget -O grpo.py https://huggingface.co/datasets/burtenshaw/lora-without-regrets/resolve/main/grpo.py
```

---

## ⚙️ Multi-GPU Configuration

The launcher automatically uses all 8 GPUs through `accelerate`:

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 --gpu_ids all --rdzv_backend c10d
```

To customize, override the prefix with an environment variable:

```bash
export ACCEL_PREFIX='accelerate launch --num_processes 8 --mixed_precision bf16 --deepspeed_config_file ds_zero2.json'
```

### (Optional) DeepSpeed ZeRO-2 Config Example

Save this as `ds_zero2.json`:

```json
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "reduce_scatter": true,
    "contiguous_gradients": true
  },
  "bf16": { "enabled": true },
  "gradient_accumulation_steps": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

---

## 📦 Running Experiments

### 1. List All Commands

```bash
python launch_lora_without_regret.py list
```

Filter by substring (e.g., only SFT):

```bash
python launch_lora_without_regret.py list --filter sft-
```

---

### 2. Run a Single Experiment

```bash
python launch_lora_without_regret.py run sft-llama-1b-tulu3
python launch_lora_without_regret.py run grpo-llama8b-gsm8k
```

Preview only (no execution):

```bash
python launch_lora_without_regret.py run sft-llama-1b-tulu3 --dry
```

---

### 3. Run Multiple or All Experiments

Run several by name pattern:

```bash
python launch_lora_without_regret.py run-many sft- grpo-llama8b-
```

Run the full suite:

```bash
python launch_lora_without_regret.py run-all
```

---

## 🧩 Experiment Matrix

| Type     | Model                 | Dataset            | Notes                      |
| -------- | --------------------- | ------------------ | -------------------------- |
| **SFT**  | Llama-3.2-1B-Instruct | Tulu-3-SFT-Mixture | LoRA rank = 256, LR = 2e-4 |
| **SFT**  | Llama-3.2-1B-Instruct | OpenThoughts-114k  | same setup                 |
| **SFT**  | Llama-3.1-8B-Instruct | Tulu-3-SFT-Mixture |                            |
| **SFT**  | Llama-3.1-8B-Instruct | OpenThoughts-114k  |                            |
| **GRPO** | Llama-3.1-8B-Base     | GSM8K              | LoRA rank = 1, LR = 5e-5   |
| **GRPO** | Llama-3.1-8B-Base     | DeepMath-103K      |                            |
| **GRPO** | Qwen3-8B-Base         | DeepMath-103K      |                            |
| **GRPO** | SmolLM3-3B            | OpenR1-Math-220k   | LoRA vs Full-FT comparison |

All SFT runs apply **LoRA on all linear layers**, and GRPO runs use the **official reference reward script**.

---

## 📊 Weights & Biases Logging

Each run reports automatically to **wandb**:

* Project: `lora_without_regret`
* Run name: experiment ID (e.g., `sft-llama-1b-tulu3`)

You can override defaults:

```bash
export WANDB_PROJECT=lora_without_regret
export WANDB_ENTITY=<your_username_or_team>
```

All outputs go to `runs/<experiment_name>` and logs to `wandb_logs/`.

---

## 🧠 Tips

* **Effective batch size** = `num_gpus × per_device_train_batch_size × gradient_accumulation_steps`
  (default = 8 × 1 × 4 = 32 samples per optimizer step)
* Increase `gradient_accumulation_steps` before enlarging micro-batch to keep LoRA stable.
* Use `--gradient_checkpointing` for memory savings on large models.
* `--dry` shows commands before launching distributed jobs.

---

## 🏁 Example: SFT Command

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 --gpu_ids all --rdzv_backend c10d \
trl sft \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --dataset_name allenai/tulu-3-sft-mixture \
  --use_peft --lora_target_modules all-linear --lora_r 256 --lora_alpha 16 --lora_dropout 0.0 \
  --learning_rate 2e-4 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --report_to wandb --logging_dir wandb_logs \
  --output_dir runs/sft-llama-1b-tulu3 --run_name sft-llama-1b-tulu3
```

---

## 🏁 Example: GRPO Command

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 --gpu_ids all --rdzv_backend c10d \
python grpo.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Base \
  --dataset_name openai/gsm8k \
  --output_dir runs/grpo-llama8b-gsm8k --run_name grpo-llama8b-gsm8k \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --generation_batch_size 8 --num_generations 8 \
  --learning_rate 5e-5 \
  --use_peft --lora_target_modules all-linear --lora_r 1 --lora_alpha 32 --lora_dropout 0.0 \
  --report_to wandb --wandb_project lora_without_regret --wandb_name grpo-llama8b-gsm8k --logging_dir wandb_logs
```

---

### 📚 References

* **TRL Documentation:** [LoRA Without Regret](https://huggingface.co/docs/trl/main/en/lora_without_regret)
* **Official GRPO Script:** [`burtenshaw/lora-without-regrets/grpo.py`](https://huggingface.co/datasets/burtenshaw/lora-without-regrets/blob/main/grpo.py)

---
