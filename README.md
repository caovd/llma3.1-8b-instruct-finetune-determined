# llma3.1-8b-instruct-finetune-determined
Finetune HF's Llama 3.1 8B instruct on a FSI consumer complaint dataset using Determined AI.

[Llama 3.1 8B Instruct HuggingFace model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
[Determined AI's banking customer complaint dataset](https://huggingface.co/datasets/determined-ai/customers-complaints)

## Quickly run an experiment

```sh
det -m <master address:8080> experiment create llama31_8b_instruct_ds3.yaml .
```
It should take around 10 mins to run this experiment but users can run the experiment longer by increasing the number of training steps if needed.

## How Determined AI integrates with DeepSpeed
We leverage [DeepSpeed](https://github.com/microsoft/DeepSpeed) ZeRO stage 3 for finetuning Llama 3.2 8B instruct on 02 A100 80GB GPUs via Determined's[DeepSpeed API](https://hpe-mlde.determined.ai/latest/model-dev-guide/api-guides/apis-howto/deepspeed/_index.html#deepspeed-api). DeepSpeed ZeRO stage 3 includes all optimizer state partitioning, gradient partitioning, and model parameter partitioning.

To enable DeepSpeed, users only need to run *python -m determined.launch.deepspeed* at the entry point before executing *run_clm.py* together with required arguments as in the below configuration file.

*llama31_8b_instruct_ds3.yaml*
```yaml
name: llama31_8b_instruct_ds3_customer_complaints
debug: false
environment:
  image: determinedai/genai-train:latest
  environment_variables:
    - NCCL_DEBUG=INFO
    - HF_HOME=/hf_cache
    - NCCL_SOCKET_IFNAME=ens,eth,ib
    - HF_TOKEN=<your token>
bind_mounts:
  - host_path: /path/to/hf_cache
    container_path: /hf_cache
resources:
  slots_per_trial: 2
  resource_pool: A100
searcher:
  name: single
  max_length:
    batches: 100
  metric: eval_loss
hyperparameters:
  deepspeed_config: ds_configs/ds_config_stage_3.json
  training_arguments:
    learning_rate: 1e-5
entrypoint: >-
  python -m determined.launch.deepspeed
  python run_clm.py
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct
  --dataset_name determined-ai/customers-complaints
  --dataset_config_name default
  --do_train
  --do_eval
  --use_lora
  --torch_dtype float16
  --max_steps 100  
  --logging_strategy steps
  --logging_steps 10
  --output_dir /tmp/test-clm
  --eval_steps 10
  --evaluation_strategy steps
  --save_total_limit 1
  --seed 1337
  --save_strategy steps
  --save_steps 20
  --deepspeed ds_configs/ds_config_stage_3.json
  --per_device_train_batch_size 2
  --per_device_eval_batch_size 2
  --trust_remote_code false
  --fp16
  --use_auth_token True
max_restarts: 0
workspace: <your workspace>
project: <your project>
```

## How Determined AI works with HuggingFace model
An explanation on the integration between HF Trainer and Determined and how the users can refactor the code to make it run on Determined AI can be found [here](https://github.com/caovd/det-finetune-swallow-tokyotechllm?tab=readme-ov-file#how-determined-ai-works-with-hf-trainer). 


## LoRA & how to checkpoint
We levearage Low-Rank Adaption ([LoRA](https://huggingface.co/docs/peft/en/package_reference/lora)) which is a [PEFT](https://huggingface.co/docs/peft/en/index) method that decomposes a large matrix into two smaller low-rank matrices in the attention layers in order to drastically reduce the number of parameters that need to be fine-tuned. We enable LoRA by setting *--use_lora* in the arguments. 

### Resume from a previous checkpoint with & without LoRA

Determined provides automated checkpointing capabilities as well as APIs for downloading [checkpoints](https://hpe-mlde.determined.ai/latest/model-dev-guide/model-management/checkpoints.html#use-trained-models) and loading them into memory in a Python process.

- Scenario #1: 

Run a finetuning job WITHOUT *--use_lora* option and save checkpoint A. Then do a following finetuning experiment but WITH *--use_lora* option, but starting from the checkpoint A saved in the point#1.

This is an example snippet of the configure file.
```sh
python -m determined.launch.deepspeed 
python run_clm.py
	--model_name_or_path meta-llama/Llama-3.1-8B-Instruct
	--resume_from_checkpoint /determined_shared_fs/determined-checkpoint/5fe9e93a-100d-4066-9ddc-d96f3f986fe3/checkpoint-30
    ...
	--use_lora
    ...
```

When using LoRA, the checkpoint artifact does not have a "config.json" but instead  "adapter_config.json" which PEFT requires. Also, we need to point the path to the "checkpoint-X" with X is the interval of saving checkpoints, ex., 20, 40, ... 100 steps. There's an [issue](https://github.com/huggingface/transformers/pull/24274) raised for resuming checkpoint when training with LORA on HF Trainer.

- Scenario #2 

Run a finetuning experiment WITH *--use_lora* argument and save one checkpoint B, then do a finetuning but WITH --use_lora option, but resuming from the checkpoint B.
```sh
python -m determined.launch.deepspeed 
python run_clm.py
	--model_name_or_path /determined_shared_fs/determined-checkpoint/ea9950df-0b1e-4bc2-946c-aa5721400dcb/checkpoint-30
    ...
	--use_lora
    ...
```