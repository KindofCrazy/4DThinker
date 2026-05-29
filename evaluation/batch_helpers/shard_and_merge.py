#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def shard(args):
    benchmark_path = Path(args.benchmark_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(benchmark_path)
    for shard_idx in range(args.num_shards):
        shard_df = df.iloc[shard_idx::args.num_shards].reset_index(drop=True)
        out_path = out_dir / f"benchmark_shard_{shard_idx}_of_{args.num_shards}.parquet"
        shard_df.to_parquet(out_path, index=False)
        print(f"wrote {out_path} rows={len(shard_df)}")


def merge(args):
    jsonl_dir = Path(args.jsonl_dir)
    out_jsonl = Path(args.out_jsonl)
    summary_json = Path(args.summary_json)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for shard_jsonl in sorted(jsonl_dir.glob("dsr_eval_shard_*_of_*.jsonl")):
        with shard_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    rows.sort(key=lambda r: int(r.get("index", -1)))
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    correct = sum(int(r.get("hit", 0)) for r in rows)
    by_type_total = Counter(str(r.get("subtask_type", "")) for r in rows)
    by_type_correct = Counter()
    for row in rows:
        by_type_correct[str(row.get("subtask_type", ""))] += int(row.get("hit", 0))

    summary = {
        "jsonl": str(out_jsonl),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "subtask_total": dict(by_type_total),
        "subtask_correct": dict(by_type_correct),
        "subtask_accuracy": {
            key: by_type_correct[key] / count if count else None
            for key, count in by_type_total.items()
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"merged_jsonl={out_jsonl}")
    print(f"merged_total={total}")
    print(f"merged_accuracy={summary['accuracy']:.4f}")
    print(f"summary_json={summary_json}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    shard_parser = subparsers.add_parser("shard")
    shard_parser.add_argument("--benchmark-path", required=True)
    shard_parser.add_argument("--out-dir", required=True)
    shard_parser.add_argument("--num-shards", type=int, required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--jsonl-dir", required=True)
    merge_parser.add_argument("--out-jsonl", required=True)
    merge_parser.add_argument("--summary-json", required=True)

    args = parser.parse_args()
    if args.cmd == "shard":
        shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
