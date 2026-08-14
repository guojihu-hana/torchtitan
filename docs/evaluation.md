# Validation and Evaluation

TorchTitan supports two ways to evaluate a model during training:

- **Blocking validation** runs a `Validator` in the training process. It reuses the training model and parallelism setup, but pauses training until validation finishes.
- **Async evaluation** launches a separate eval process on a checkpoint. Training continues while evaluation runs on separate compute resources.

## Blocking Validation

For users who want to perform validation directly during the training loop, we provide the `Validator` class which can be conveniently configured via `Validator.Config` in your config_registry function. The validator class has access to and reuses many of the trainer's functions such as its parallelization, including pipelining.

Below is an example validation config:

```python
validator=Validator.Config(
    freq=500,
    dataset="c4_validation",
    steps=-1,  # consumes the entire validation set
),
```

## Async Evaluation

`AsyncEval` moves evaluation off the training loop's critical path. At each configured interval, the trainer saves or evaluator reuses a model checkpoint and launches an evaluator as a subprocess. The trainer continues training while the evaluator loads that checkpoint for evaluation.

Below is an example validation config:

```python
async_eval=AsyncEval.Config(
    freq=1000,
    extra_args=(
        "--validator.dataloader.dataset c4_validation "
        "--validator.steps 50"
    ),
    # Give evaluation devices that are not used by the training job.
    cuda_visible_devices="6,7",
),
```

Async evaluation requires `checkpoint.enable=True`. If a regular training checkpoint already exists for the eval step, `AsyncEval` reuses it. Otherwise, it writes a temporary model checkpoint under `{dump_folder}/{checkpoint.folder}/async_eval/` and deletes it after the eval job finishes. Set `async_eval.keep_checkpoint=True` to keep these temporary checkpoints.

If TensorBoard logging is enabled, numeric results are written to a separate `async_eval` run under `{dump_folder}/{metrics.save_tb_folder}/async_eval/`. Point TensorBoard at the dump folder to view training and evaluation curves together:

```bash
tensorboard --logdir outputs
```

At the end of training, the trainer waits up to `async_eval.exit_timeout` seconds for outstanding eval jobs. A failed eval job is logged and ignored by default. Set `async_eval.raise_on_failure=True` to make eval failures fatal to the training job.


## Third-Party Evaluation
With `./scripts/checkpoint_conversion/convert_to_hf.py`, `torchtitan` offers support for converting checkpoints from DCP to safetensors format. Using this script, users can perform efficient evaluation separate from their training using external libraries that support HuggingFace e.g. `lm_eval` with `vllm` backend.

### Example usage of `lm_eval` with `vllm`:
To use this specific setup make sure to include a HuggingFace `config.json` file which is not provided by conversion script or `last_save_in_hf` option. The HF config file can be downloaded by running `python ./scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets config`.

Note that pip installing `lm-eval` may result in breaking `torchtitan` dev environment so we recommend creating a separate env.
```bash
pip install "lm-eval[vllm]"
lm_eval --model vllm \
    --model_args pretrained=./outputs/checkpoint/step-1000,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8, \
    --tasks mmlu \
    --batch_size auto
```
|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6209|±  |0.0038|
| - humanities     |      2|none  |      |acc   |↑  |0.5481|±  |0.0066|
| - other          |      2|none  |      |acc   |↑  |0.7045|±  |0.0078|
| - social sciences|      2|none  |      |acc   |↑  |0.7351|±  |0.0078|
| - stem           |      2|none  |      |acc   |↑  |0.5357|±  |0.0085|
