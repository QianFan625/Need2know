# Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation

[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](https://arxiv.org/abs/TODO)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-DelegateCI--Bench-yellow)](https://huggingface.co/datasets/bellaXY/DelegateCI)

Official training code for our EMNLP 2026 paper **"Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation"**.

We train a query rewriter with GRPO so that a user query sent to a remote LLM keeps every
task-essential span and drops sensitive spans the task does not need. The code is built on [verl](https://github.com/volcengine/verl).

The benchmark, **DelegateCI-Bench**, is released on [Hugging Face Datasets](https://huggingface.co/datasets/Qianfan625/DelegateCI).

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

## Training

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
DATA_DIR=./data/delegateci \
bash examples/grpo_trainer/run_qwen2_5-3b_delegateci.sh
```

## Citation

```bibtex
@inproceedings{huang2026needtoknow,
  title     = {Need to Know: Contextual-Integrity-Grounded Query Rewriting for Privacy-Conscious LLM Delegation},
  author    = {Huang, Xinyue and Cao, Xiaochun and Yang, Wenyuan},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```
