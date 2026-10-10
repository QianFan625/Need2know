# Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation

[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](https://arxiv.org/abs/TODO)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-DelegateCI--Bench-yellow)](https://huggingface.co/datasets/bellaXY/DelegateCI)

Official training code for our EMNLP 2026 paper **"Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation"**.

We train a query rewriter with GRPO so that a user query sent to a remote LLM keeps every
task-essential span and drops sensitive spans the task does not need. The reward combines
a hard privacy term, a concave essential-preservation term and a one-sided length penalty
(Sec. 3 of the paper). The code is built on [verl](https://github.com/volcengine/verl).

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

Training reads verl-style parquet files: `${DATA_DIR}/train.parquet` and `${DATA_DIR}/test.parquet`.
Build them from DelegateCI-Bench and put them in a directory of your choice. The default
`DATA_DIR` is `./data/delegateci`, which you create yourself and which git does not track.
Each row contains:

| Field | Content |
| --- | --- |
| `prompt` | Chat messages: the rewriting instruction and the user query |
| `data_source` | Dataset name |
| `reward_model.ground_truth` | Gold reformulation (`gold_reformulation`), used as the length target |
| `extra_info.question` | Original user query |
| `extra_info.key_spans` | Essential spans (`essential_spans`) |
| `extra_info.ne` | Non-essential sensitive spans (`non_essential_spans`) |

## Training

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
DATA_DIR=./data/delegateci \
bash examples/grpo_trainer/run_qwen2_5-3b_delegateci.sh
```

The default setup runs on a single node with 4× A100 80GB. Hyper-parameters match
Appendix A of the paper: 24 prompts per batch, 8 rollouts per prompt, lr 1e-6,
`low_var_kl` with coefficient 0.005, up to 10 epochs. Any verl option can be overridden
by adding it to the command line.

| File | Description |
| --- | --- |
| `verl/utils/reward_score/priv_batch.py` | Batched reward (privacy, essential preservation, length) |
| `examples/grpo_trainer/run_qwen2_5-3b_delegateci.sh` | Training script |

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
