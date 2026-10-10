# Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation

[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](https://arxiv.org/abs/TODO)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-DelegateCI--Bench-yellow)](https://huggingface.co/datasets/bellaXY/DelegateCI)

Official training code for our EMNLP 2026 paper **"Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation"**.

We train a query rewriter with GRPO so that a user query sent to a remote LLM keeps every
task-essential span and drops sensitive spans the task does not need. The code is built on
[verl](https://github.com/volcengine/verl).

The benchmark, **DelegateCI-Bench**, is released on [Hugging Face Datasets](https://huggingface.co/datasets/bellaXY/DelegateCI).

## Installation

```bash
conda create -n need2know python=3.10 -y
conda activate need2know
pip install -e .
pip install -r requirements.txt
pip install vllm bert-score swanlab
```

The reward uses BERTScore with `roberta-large` (layer 17, rescaled with baseline).
To load the model from a local copy, set:

```bash
export PRIV_BERTSCORE_MODEL_PATH=/path/to/roberta-large
```

## Data

The training data is in `data/med_priv_mix/`:

| File | Rows | Content |
| --- | ---: | --- |
| `train.parquet` | 1,222 | Train split of DelegateCI-Bench (872 general, 350 medical) |
| `test.parquet` | 300 | Validation set used during training (150 general, 150 medical) |

Each row is a verl-style record. These fields are used during training:

| Field | Content |
| --- | --- |
| `prompt` | System prompt with the rewriting instruction, plus the user query |
| `reward_model.ground_truth` | Reference rewrite |
| `extra_info.question` | Original user query |
| `extra_info.ne` | Non-essential sensitive spans |
| `extra_info.origin_ea` | Essential spans checked by the reward (general samples) |
| `extra_info.key_spans` | Essential spans checked by the reward (medical samples) |

## Reward

`verl/utils/reward_score/priv_batch.py` scores each rewrite `A` as follows:

1. **Privacy.** Drop any `ne` span that appears inside an `origin_ea` span. If `A` is empty,
   or any remaining `ne` span is fully covered by `A` (token coverage = 1.0) or has rescaled
   BERTScore precision > 0.3 against `A`, the score is **-1**.
2. **Essential spans.** Otherwise, a span counts as kept if its token coverage is ≥ 0.5
   or its BERTScore precision is ≥ 0.3.
   - If `origin_ea` is not empty, the score is `max(-1, -0.2 × #missed spans)`. If no span is missed, go to step 3.
   - Else, if `key_spans` is not empty, the score is `(kept / total) ^ 0.4 × brevity`,
     where `brevity = min(1, min(0.5·|question|, |ground_truth|) / |A|)` in characters.
3. **Length.** Otherwise, the score is `min(|A|, |ground_truth|) / max(|A|, |ground_truth|)` in characters.

## Training

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
bash examples/grpo_trainer/run_qwen2_5-3b_delegateci.sh
```

The script reads data from `data/med_priv_mix/` by default. Set `DATA_DIR` to use another
directory. The default setup runs on a single node with 4× A100 80GB, with 24 prompts per
batch, 8 rollouts per prompt, lr 1e-6, `low_var_kl` with coefficient 0.005 and up to 10
epochs. Any verl option can be overridden by adding it to the command line.

Each rollout's reward details are written to `trajectory/*.jsonl`. Logs are written to `logs/`.

## Citation

```bibtex
@inproceedings{huang2026needtoknow,
  title     = {Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation},
  author    = {Huang, Xinyue and Cao, Xiaochun and Yang, Wenyuan},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## Acknowledgements

This repository is built on [verl](https://github.com/volcengine/verl). We thank its authors.

## License

Apache-2.0, following verl.
