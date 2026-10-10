# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Tokenizer / Jaccard 
# ---------------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "at", "for", "with", "by", "from",
    "as", "that", "this", "these", "those", "it", "its", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "they", "them", "his", "her", "their",
    "do", "does", "did", "have", "has", "had", "can", "could", "would", "should",
    "will", "shall", "may", "might", "not", "no", "so", "than", "then", "there",
    "some", "any", "all", "more", "most", "much", "many",
}


def _tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return [t for t in toks if t not in STOPWORDS]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def coverage(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sb:
        return 0.0
    return len(sa & sb) / len(sb)


def _filter_ne_by_oea(ne: list[str], oea: list[str]) -> tuple[list[str], list[str]]:
    if not ne or not oea:
        return list(ne), []
    oea_lower = [e.lower() for e in oea]
    kept: list[str] = []
    dropped: list[str] = []
    for c in ne:
        c_norm = c.strip().lower()
        if c_norm and any(c_norm in eo for eo in oea_lower):
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped



_BERT_SCORER = None
_BERT_LOAD_ERROR: Exception | None = None

DEFAULT_ROBERTA_LARGE_PATH = os.environ.get(
    "PRIV_BERTSCORE_MODEL_PATH",
    "/path/models--FacebookAI--roberta-large",
)


def _get_bert_scorer():
    global _BERT_SCORER, _BERT_LOAD_ERROR
    if _BERT_SCORER is not None or _BERT_LOAD_ERROR is not None:
        return _BERT_SCORER
    try:
        import bert_score

        baseline_path = os.path.join(
            os.path.dirname(bert_score.__file__),
            "rescale_baseline",
            "en",
            "roberta-large.tsv",
        )
        _BERT_SCORER = bert_score.BERTScorer(
            model_type=DEFAULT_ROBERTA_LARGE_PATH,
            num_layers=17,
            lang="en",
            rescale_with_baseline=True,
            baseline_path=baseline_path,
        )
    except Exception as e:
        _BERT_LOAD_ERROR = e
        _BERT_SCORER = None
    return _BERT_SCORER


def _bert_p_pairs(cands: list[str], refs: list[str], batch_size: int = 64) -> list[float] | None:
    assert len(cands) == len(refs), "cands / refs len same must"
    scorer = _get_bert_scorer()
    if scorer is None or not cands:
        return None
    try:
        P, _, _ = scorer.score(cands=list(cands), refs=list(refs), batch_size=batch_size)
        return [float(x) for x in P.tolist()]
    except Exception:
        return None



def _to_str_list(x: Any) -> list[str]:

    if x is None:
        return []
    if isinstance(x, str):
        return [x] if x else []
    if isinstance(x, Iterable):
        out = []
        for item in x:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return [str(x)]


PRIV_BERT_P_THRESHOLD = 0.3
PRIV_COVERAGE_THRESHOLD = 1.0
UTILITY_BERT_P_THRESHOLD = 0.3
UTILITY_COVERAGE_THRESHOLD = 0.5
UTILITY_MISS_PENALTY = -0.2
LENGTH_REWARD_WEIGHT = 1.0
KEY_SPANS_GAMMA = 0.4


def compute_score(
    data_sources: list[str] | None = None,
    solution_strs: list[str] | None = None,
    ground_truths: list[str | None] | None = None,
    extra_infos: list[dict | None] | None = None,
    bert_batch_size: int = 64,
    log_jsonl_path: str | None = "default_rollout_info_priv.jsonl",
    **kwargs: Any,
) -> list[dict]:
    if solution_strs is None:
        return []
    n = len(solution_strs)
    if extra_infos is None:
        extra_infos = [None] * n
    if ground_truths is None:
        ground_truths = [None] * n
    if data_sources is None:
        data_sources = ["priv"] * n
    assert len(extra_infos) == n and len(ground_truths) == n, (
        "solution_strs / extra_infos / ground_truths"
    )

    parsed: list[dict[str, Any]] = []
    for i in range(n):
        A = solution_strs[i] or ""
        ei = extra_infos[i] or {}
        ne_raw = _to_str_list(ei.get("ne"))
        oea = _to_str_list(ei.get("origin_ea"))
        question = ei.get("question")
        key_spans = _to_str_list(ei.get("key_spans"))
        ne, ne_skipped = _filter_ne_by_oea(ne_raw, oea)
        gt = ground_truths[i]
        gold = gt if isinstance(gt, str) and gt else str(ei.get("answer", "") or "")

        item: dict[str, Any] = {
            "A": A,
            "ne": ne,
            "oea": oea,
            "key_spans": key_spans,
            "gold": gold,
            "info": {
                "priv_ne_skipped_in_oea": ne_skipped,
                "priv_ne_skipped_count": len(ne_skipped),
            },
            "question": question,
            "score": None,  
        }

        if not A:
            item["score"] = -1.0
            parsed.append(item)
            continue

        priv_max_cov = 0.0
        priv_argmax_c: str | None = None
        for c in ne:
            cov = coverage(A, c)
            if cov > priv_max_cov:
                priv_max_cov, priv_argmax_c = cov, c
        item["info"]["priv_max_coverage"] = priv_max_cov
        item["info"]["priv_argmax_coverage"] = priv_argmax_c

        if ne and priv_max_cov >= PRIV_COVERAGE_THRESHOLD:
            item["score"] = -1.0

        parsed.append(item)

    priv_cands: list[str] = []
    priv_refs: list[str] = []
    priv_owner: list[tuple[int, int]] = []  
    for i, p in enumerate(parsed):
        if p["score"] is not None or not p["ne"]:
            continue
        for c_idx, c in enumerate(p["ne"]):
            priv_cands.append(c)
            priv_refs.append(p["A"])
            priv_owner.append((i, c_idx))

    priv_bert = _bert_p_pairs(priv_cands, priv_refs, batch_size=bert_batch_size)


    priv_scores_per_sample: dict[int, list[float]] = {i: [] for i in range(n)}
    if priv_bert is not None:
        for k, (i, c_idx) in enumerate(priv_owner):
            priv_scores_per_sample[i].append(priv_bert[k])

    bert_failed = priv_bert is None and len(priv_cands) > 0
    for i, p in enumerate(parsed):
        if p["score"] is not None:
            continue
        if not p["ne"]:
            continue
        ps = priv_scores_per_sample[i]
        if not ps:
            if bert_failed:
                print("[priv_batch][error] BERTScore loading error")
                p["info"]["priv_max_p_rescaled"] = None
                if _BERT_LOAD_ERROR is not None:
                    p["info"]["bertscore_error"] = repr(_BERT_LOAD_ERROR)
            continue
        max_p = max(ps)
        argmax_c = p["ne"][ps.index(max_p)]
        p["info"]["priv_max_p_rescaled"] = max_p
        p["info"]["priv_argmax_p_rescaled"] = argmax_c
        if max_p > PRIV_BERT_P_THRESHOLD:
            p["score"] = -1.0

    util_cands: list[str] = []
    util_refs: list[str] = []
    util_owner: list[tuple[int, int, str]] = []  
    for i, p in enumerate(parsed):
        if p["score"] is not None:
            continue
        if p["oea"]:
            for e_idx, ea in enumerate(p["oea"]):
                util_cands.append(ea)
                util_refs.append(p["A"])
                util_owner.append((i, e_idx, "oea"))
        elif p["key_spans"]:
            for e_idx, ks in enumerate(p["key_spans"]):
                util_cands.append(ks)
                util_refs.append(p["A"])
                util_owner.append((i, e_idx, "key_spans"))

    util_bert = _bert_p_pairs(util_cands, util_refs, batch_size=bert_batch_size)
    util_scores_per_sample: dict[int, dict[tuple[str, int], float]] = {i: {} for i in range(n)}
    if util_bert is not None:
        for k, (i, e_idx, kind) in enumerate(util_owner):
            util_scores_per_sample[i][(kind, e_idx)] = util_bert[k]

    for i, p in enumerate(parsed):
        if p["score"] is not None:
            continue

        if p["oea"]:
            bert_for_i = util_scores_per_sample[i]
            miss_items: list[str] = []
            ea_detail: list[dict[str, Any]] = []
            for e_idx, ea_item in enumerate(p["oea"]):
                cov = coverage(p["A"], ea_item)
                p_rs = bert_for_i.get(("oea", e_idx), 0.0)
                hit = (cov >= UTILITY_COVERAGE_THRESHOLD) or (p_rs >= UTILITY_BERT_P_THRESHOLD)
                ea_detail.append({"item": ea_item, "coverage": cov, "p_rescaled": p_rs, "hit": hit})
                if not hit:
                    miss_items.append(ea_item)
            p["info"]["utility_detail"] = ea_detail
            p["info"]["utility_miss_count"] = len(miss_items)
            p["info"]["utility_miss_items"] = miss_items
            if miss_items:
                p["score"] = max(-1.0, len(miss_items) * UTILITY_MISS_PENALTY)
        elif p["key_spans"]:
            bert_for_i = util_scores_per_sample[i]
            hit_count = 0
            ks_detail: list[dict[str, Any]] = []
            for e_idx, ks_item in enumerate(p["key_spans"]):
                cov = coverage(p["A"], ks_item)
                p_rs = bert_for_i.get(("key_spans", e_idx), 0.0)
                hit = (cov >= UTILITY_COVERAGE_THRESHOLD) or (p_rs >= UTILITY_BERT_P_THRESHOLD)
                ks_detail.append({"item": ks_item, "coverage": cov, "p_rescaled": p_rs, "hit": hit})
                if hit:
                    hit_count += 1
            ks_total = len(p["key_spans"])
            ratio = hit_count / ks_total if ks_total > 0 else 0.0
            ks_reward = float(ratio) ** KEY_SPANS_GAMMA
            brevity = min(1.0, min(len(p["question"])*0.5,len(p["gold"])) / max(len(p["A"]), 1))
            p["info"]["key_spans_detail"] = ks_detail
            p["info"]["key_spans_hit_count"] = hit_count
            p["info"]["key_spans_total"] = ks_total
            p["info"]["key_spans_reward"] = ks_reward
            p["info"]["key_spans_brevity"] = brevity

            p["info"]["utility_detail"] = []
            p["info"]["utility_miss_count"] = 0
            p["info"]["utility_miss_items"] = []
            p["score"] = ks_reward*brevity
        else:

            p["info"]["utility_detail"] = []
            p["info"]["utility_miss_count"] = 0
            p["info"]["utility_miss_items"] = []

    for p in parsed:
        if p["score"] is not None:
            continue
        A, gold = p["A"], p["gold"]
        if gold:
            ratio = min(len(A), len(gold)) / max(len(A), len(gold))
        else:
            ratio = 1.0 / (1.0 + abs(len(A) - 100) / 100.0)
        length_term = LENGTH_REWARD_WEIGHT * float(ratio)
        p["info"]["length_term"] = length_term
        p["score"] = length_term

    SCALAR_KEYS_DEFAULTS: dict[str, Any] = {
        "priv_max_coverage": 0.0,
        "priv_argmax_coverage": "",
        "priv_max_p_rescaled": 0.0,
        "priv_argmax_p_rescaled": "",
        "priv_ne_skipped_count": 0,
        "utility_miss_count": 0,
        "key_spans_total": 0,
        "key_spans_hit_count": 0,
        "key_spans_reward": 0.0,
        "length_term": 0.0,
    }

    results: list[dict[str, Any]] = []
    log_lines: list[str] | None = [] if log_jsonl_path else None
    for i, p in enumerate(parsed):
        score = float(p["score"])
        out: dict[str, Any] = {"score": score}
        info = p["info"]
        for k, default in SCALAR_KEYS_DEFAULTS.items():
            v = info.get(k, default)
            if v is None:
                v = default
            out[k] = v
        results.append(out)

        if log_lines is not None:
            ei = extra_infos[i] or {}
            log_lines.append(json.dumps({
                "reward_score": score,
                "solution_str": solution_strs[i],
                "ne": ei.get("ne"),
                "origin_ea": ei.get("origin_ea"),
                "key_spans": ei.get("key_spans"),
                "reward_info": info,
                "question": ei.get("question"),
                "ground_truth": ground_truths[i],
                "data_source": data_sources[i],
            }, ensure_ascii=False, default=str))

    if log_lines is not None and log_lines:
        try:
            with open(log_jsonl_path, "a") as f:
                f.write("\n".join(log_lines) + "\n")
        except Exception as e:
            # logging failures must not interrupt training
            print(f"[priv_batch][warn] failed to write {log_jsonl_path}: {e!r}")

    return results


