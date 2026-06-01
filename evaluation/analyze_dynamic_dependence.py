#!/usr/bin/env python
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_BASELINE = "/data/YangKe/project/dsr/relevant/4DThinker-experiment/evaluation/results/20260529_200527_4drl_flash_batch/dsr_eval_merged.jsonl"
DEFAULT_EXPERIMENTS = {
    "text_only": "/data/YangKe/project/dsr/relevant/4DThinker-experiment/evaluation/results/20260529_212313_4drl_text_only/dsr_eval_merged.jsonl",
    "repeat_first": "/data/YangKe/project/dsr/relevant/4DThinker-experiment/evaluation/results/20260529_225907_4drl_repeat_first/dsr_eval_merged.jsonl",
    "reversed_video": "/data/YangKe/project/dsr/relevant/4DThinker-experiment/evaluation/results/20260530_214415_4drl_reversed_video/dsr_eval_merged.jsonl",
}
DEFAULT_OUT_DIR = "/data/YangKe/project/dsr/relevant/4DThinker-experiment/evaluation/results/experiement_statistics"
OPTION_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def log(message: str) -> None:
    print(f"[dynamic-analysis] {message}", flush=True)


def normalize_option(value: Any) -> str:
    if value is None:
        return "MISSING"
    text = str(value).strip()
    if not text:
        return "MISSING"
    upper = text.upper()
    if upper in {"A", "B", "C", "D"}:
        return upper
    match = OPTION_RE.search(upper)
    return match.group(1) if match else "OTHER"


def truthy_hit(row: Dict[str, Any]) -> Optional[bool]:
    value = row.get("hit")
    if value is not None:
        try:
            return bool(int(value))
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in {"true", "yes", "y"}:
                return True
            if text in {"false", "no", "n"}:
                return False
    pred = normalize_option(row.get("pred", row.get("answer")))
    gt = normalize_option(row.get("gt", row.get("answer_gt")))
    if pred in {"A", "B", "C", "D"} and gt in {"A", "B", "C", "D"}:
        return pred == gt
    return None


def sample_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("video_id", "")),
        str(row.get("question", "")),
        normalize_option(row.get("gt", row.get("answer_gt"))),
        str(row.get("subtask_type", row.get("type", ""))),
    )


def load_jsonl(path: Path) -> Tuple[Dict[Tuple[str, str, str, str], Dict[str, Any]], Dict[str, Any]]:
    rows: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    duplicate_keys = 0
    bad_lines = 0
    log(f"loading {path}")
    if not path.exists():
        log(f"missing file: {path}")
        return rows, {"path": str(path), "rows": 0, "duplicate_keys": 0, "bad_lines": 0, "missing": True}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                log(f"bad json line {line_no} in {path}")
                continue
            key = sample_key(row)
            duplicate_keys += int(key in rows)
            rows[key] = row
    return rows, {
        "path": str(path),
        "rows": len(rows),
        "duplicate_keys": duplicate_keys,
        "bad_lines": bad_lines,
        "missing": False,
    }


def prediction(row: Dict[str, Any]) -> str:
    return normalize_option(row.get("pred", row.get("answer")))


def ground_truth(row: Dict[str, Any]) -> str:
    return normalize_option(row.get("gt", row.get("answer_gt")))


def answer_distribution(rows: Dict[Tuple[str, str, str, str], Dict[str, Any]]) -> Dict[str, Any]:
    pred_counts = Counter(prediction(row) for row in rows.values())
    gt_counts = Counter(ground_truth(row) for row in rows.values())
    by_subtask: Dict[str, Counter] = defaultdict(Counter)
    for row in rows.values():
        by_subtask[str(row.get("subtask_type", row.get("type", "")))] [prediction(row)] += 1
    return {
        "prediction_counts": ordered_counts(pred_counts),
        "prediction_frequency": frequencies(pred_counts),
        "ground_truth_counts": ordered_counts(gt_counts),
        "ground_truth_frequency": frequencies(gt_counts),
        "prediction_counts_by_subtask": {
            subtask: ordered_counts(counts)
            for subtask, counts in sorted(by_subtask.items())
        },
        "prediction_frequency_by_subtask": {
            subtask: frequencies(counts)
            for subtask, counts in sorted(by_subtask.items())
        },
    }


def ordered_counts(counts: Counter) -> Dict[str, int]:
    keys = ["A", "B", "C", "D", "OTHER", "MISSING"]
    return {key: int(counts.get(key, 0)) for key in keys if counts.get(key, 0)}


def frequencies(counts: Counter) -> Dict[str, float]:
    total = sum(counts.values())
    return {key: value / total for key, value in ordered_counts(counts).items()} if total else {}


def transition_matrix(
    baseline: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    experiment: Dict[Tuple[str, str, str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    common = sorted(set(baseline) & set(experiment))
    matrix = Counter({"CC": 0, "CW": 0, "WC": 0, "WW": 0, "unknown": 0})
    baseline_correct = 0
    baseline_wrong = 0
    for key in common:
        base_hit = truthy_hit(baseline[key])
        exp_hit = truthy_hit(experiment[key])
        if base_hit is None or exp_hit is None:
            matrix["unknown"] += 1
            continue
        baseline_correct += int(base_hit)
        baseline_wrong += int(not base_hit)
        if base_hit and exp_hit:
            matrix["CC"] += 1
        elif base_hit and not exp_hit:
            matrix["CW"] += 1
        elif not base_hit and exp_hit:
            matrix["WC"] += 1
        else:
            matrix["WW"] += 1
    lost = matrix["CW"]
    gained = matrix["WC"]
    return {
        "common_samples": len(common),
        "baseline_correct": baseline_correct,
        "baseline_wrong": baseline_wrong,
        "matrix": {
            "correct->correct": matrix["CC"],
            "correct->wrong": matrix["CW"],
            "wrong->correct": matrix["WC"],
            "wrong->wrong": matrix["WW"],
            "unknown": matrix["unknown"],
        },
        "lost_correct": lost,
        "gained_correct": gained,
        "net_change": gained - lost,
        "lost_correct_rate": lost / baseline_correct if baseline_correct else None,
        "gained_correct_rate": gained / baseline_wrong if baseline_wrong else None,
    }


def same_answer_stats(
    reference: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    experiment: Dict[Tuple[str, str, str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    common = sorted(set(reference) & set(experiment))
    same = [key for key in common if prediction(reference[key]) == prediction(experiment[key])]
    same_correct = [
        key for key in same
        if truthy_hit(reference[key]) is True and truthy_hit(experiment[key]) is True
    ]
    return {
        "common_with_reference": len(common),
        "same_answer_count": len(same),
        "same_answer_rate": len(same) / len(common) if common else None,
        "same_answer_correct_count": len(same_correct),
        "same_answer_accuracy": len(same_correct) / len(same) if same else None,
        "experiment_accuracy_on_common": accuracy(experiment, common),
        "reference_accuracy_on_common": accuracy(reference, common),
    }


def accuracy(rows: Dict[Tuple[str, str, str, str], Dict[str, Any]], keys: Iterable[Tuple[str, str, str, str]]) -> Optional[float]:
    hits: List[bool] = []
    for key in keys:
        hit = truthy_hit(rows[key])
        if hit is not None:
            hits.append(hit)
    return sum(hits) / len(hits) if hits else None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {path}")


def write_transition_csv(path: Path, transitions: Dict[str, Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "baseline", "experiment_correctness", "count"])
        for name, stats in sorted(transitions.items()):
            matrix = stats["matrix"]
            writer.writerow([name, "correct", "correct", matrix["correct->correct"]])
            writer.writerow([name, "correct", "wrong", matrix["correct->wrong"]])
            writer.writerow([name, "wrong", "correct", matrix["wrong->correct"]])
            writer.writerow([name, "wrong", "wrong", matrix["wrong->wrong"]])
    log(f"wrote {path}")


def write_sample_sensitivity(
    path: Path,
    baseline: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    experiments: Dict[str, Dict[Tuple[str, str, str, str], Dict[str, Any]]],
) -> Dict[str, Any]:
    common = set(baseline)
    for rows in experiments.values():
        common &= set(rows)
    unchanged_all = 0
    perturbations_agree = 0
    with path.open("w", encoding="utf-8") as f:
        for key in sorted(common):
            base_row = baseline[key]
            exp_preds = {name: prediction(rows[key]) for name, rows in sorted(experiments.items())}
            base_pred = prediction(base_row)
            all_unchanged = all(pred == base_pred for pred in exp_preds.values())
            pert_agree = len(set(exp_preds.values())) <= 1
            unchanged_all += int(all_unchanged)
            perturbations_agree += int(pert_agree)
            record = {
                "video_id": key[0],
                "question": key[1],
                "gt": key[2],
                "subtask_type": key[3],
                "baseline_pred": base_pred,
                "baseline_hit": truthy_hit(base_row),
                "experiment_preds": exp_preds,
                "experiment_hits": {name: truthy_hit(rows[key]) for name, rows in sorted(experiments.items())},
                "all_perturbations_same_as_baseline": all_unchanged,
                "all_perturbations_same_with_each_other": pert_agree,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log(f"wrote {path}")
    return {
        "common_across_baseline_and_all_experiments": len(common),
        "all_perturbations_same_as_baseline_count": unchanged_all,
        "all_perturbations_same_as_baseline_rate": unchanged_all / len(common) if common else None,
        "all_perturbations_same_with_each_other_count": perturbations_agree,
        "all_perturbations_same_with_each_other_rate": perturbations_agree / len(common) if common else None,
    }


def parse_experiment_arg(values: Optional[List[str]]) -> Dict[str, str]:
    if not values:
        return dict(DEFAULT_EXPERIMENTS)
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"experiment must be name=path, got {value}")
        name, path = value.split("=", 1)
        parsed[name] = path
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-jsonl", default=DEFAULT_BASELINE)
    parser.add_argument("--experiment", action="append", help="name=/path/to/merged.jsonl; defaults to the known perturbation runs")
    parser.add_argument("--baseline-repeat-jsonl", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"output directory: {out_dir}")

    baseline, baseline_meta = load_jsonl(Path(args.baseline_jsonl))
    experiment_paths = parse_experiment_arg(args.experiment)
    experiments = {}
    metadata = {"baseline": baseline_meta, "experiments": {}}
    for name, path in sorted(experiment_paths.items()):
        rows, meta = load_jsonl(Path(path))
        experiments[name] = rows
        metadata["experiments"][name] = meta

    distributions = {"baseline": answer_distribution(baseline)}
    for name, rows in sorted(experiments.items()):
        distributions[name] = answer_distribution(rows)

    transitions = {
        name: transition_matrix(baseline, rows)
        for name, rows in sorted(experiments.items())
    }
    sensitivity = write_sample_sensitivity(out_dir / "sample_perturbation_sensitivity.jsonl", baseline, experiments)

    same_answer = {
        "note": "Stats are keyed by (video_id, question, gt, subtask_type), not shard-local index.",
        "baseline_jsonl": str(args.baseline_jsonl),
        "experiments": {
            name: {
                "jsonl": experiment_paths[name],
                **same_answer_stats(baseline, rows),
            }
            for name, rows in sorted(experiments.items())
        },
    }

    if args.baseline_repeat_jsonl:
        repeat_rows, repeat_meta = load_jsonl(Path(args.baseline_repeat_jsonl))
        metadata["baseline_repeat"] = repeat_meta
        same_answer["baseline_vs_baseline_repeat"] = {
            "jsonl": str(args.baseline_repeat_jsonl),
            **same_answer_stats(baseline, repeat_rows),
        }
        transitions["baseline_repeat"] = transition_matrix(baseline, repeat_rows)
        distributions["baseline_repeat"] = answer_distribution(repeat_rows)
    else:
        same_answer["baseline_vs_baseline_repeat"] = {"status": "pending", "jsonl": None}

    summary = {
        "metadata": metadata,
        "correctness_transitions": transitions,
        "answer_distributions": distributions,
        "sample_sensitivity_summary": sensitivity,
        "same_answer_summary": same_answer,
    }

    write_json(out_dir / "dynamic_dependence_diagnostics.json", summary)
    write_json(out_dir / "correctness_transition_matrix.json", transitions)
    write_transition_csv(out_dir / "correctness_transition_matrix.csv", transitions)
    write_json(out_dir / "answer_distribution_analysis.json", distributions)
    write_json(out_dir / "sample_sensitivity_summary.json", sensitivity)
    write_json(out_dir / "ablation_same_answer_summary.json", same_answer)

    log("generated files:")
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            log(str(path))


if __name__ == "__main__":
    main()
