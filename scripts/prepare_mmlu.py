"""Prepare MMLU and a lightweight replay buffer for Dohnuts training.

Generates the 4 required Dohnuts partitions:
- train.jsonl: Training mixture (MMLU + replay datasets)
- dev.jsonl: Development partition for macro-accuracy checkpoint selection
- calibration.jsonl: Calibration partition for fitting temperature scaling
- test.jsonl: Held-out test evaluation partition
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def train_split(group: str, *, has_dev: bool = False) -> str:
    bucket = int(digest("doh-split-2026:" + group)[:8], 16) % 100
    if bucket < 5:
        return "calibration"
    if not has_dev and bucket < 10:
        return "dev"
    return "train"


def format_example(
    dataset: str,
    uid: str,
    group: str,
    split: str,
    state: Any,
    question: dict[str, Any],
    target: int | list[float],
) -> dict[str, Any]:
    if isinstance(target, int):
        k = 2 if question["type"] == "noul" else len(question["criteria"])
        target = [float(i == target) for i in range(k)]
    total = sum(target)
    if not target or min(target) < 0 or abs(total - 1) > 1e-4:
        raise ValueError(f"Invalid target: {dataset}/{uid}")
    target = [v / total for v in target]
    return {
        "id": f"{dataset}:{uid}",
        "dataset": dataset,
        "group": group,
        "split": split,
        "state": state,
        "question": question,
        "target": target,
    }


def load_mmlu_records(smoke_test: bool = False):
    from datasets import load_dataset

    print("Loading cais/mmlu dataset...")
    # Load all subjects or subset
    mmlu = load_dataset("cais/mmlu", "all")
    records = []

    # Map MMLU splits
    # cais/mmlu splits: 'auxiliary_train' (~100k), 'test' (14042), 'validation' (1531), 'dev' (285)
    train_source = mmlu["auxiliary_train"] if "auxiliary_train" in mmlu else mmlu["validation"]
    test_source = mmlu["test"]
    val_source = mmlu["validation"]
    dev_source = mmlu["dev"]

    if smoke_test:
        train_rows = list(train_source.select(range(min(500, len(train_source)))))
        dev_rows = list(dev_source.select(range(min(100, len(dev_source)))))
        calib_rows = list(val_source.select(range(min(50, len(val_source)))))
        test_rows = list(test_source.select(range(min(200, len(test_source)))))
    else:
        # Full or balanced: take balanced subset or full
        train_rows = list(train_source.select(range(min(5000, len(train_source)))))
        dev_rows = list(dev_source)
        calib_rows = list(val_source)
        test_rows = list(test_source)

    for i, r in enumerate(train_rows):
        subject = r.get("subject", "general")
        records.append(
            format_example(
                dataset="mmlu",
                uid=f"train_{i}",
                group=f"mmlu:{subject}",
                split="train",
                state=f"Subject: {subject.replace('_', ' ')}",
                question={
                    "type": "choice",
                    "instructions": r["question"],
                    "criteria": r["choices"],
                },
                target=r["answer"],
            )
        )

    for i, r in enumerate(dev_rows):
        subject = r.get("subject", "general")
        records.append(
            format_example(
                dataset="mmlu",
                uid=f"dev_{i}",
                group=f"mmlu:{subject}",
                split="dev",
                state=f"Subject: {subject.replace('_', ' ')}",
                question={
                    "type": "choice",
                    "instructions": r["question"],
                    "criteria": r["choices"],
                },
                target=r["answer"],
            )
        )

    for i, r in enumerate(calib_rows):
        subject = r.get("subject", "general")
        records.append(
            format_example(
                dataset="mmlu",
                uid=f"calib_{i}",
                group=f"mmlu:{subject}",
                split="calibration",
                state=f"Subject: {subject.replace('_', ' ')}",
                question={
                    "type": "choice",
                    "instructions": r["question"],
                    "criteria": r["choices"],
                },
                target=r["answer"],
            )
        )

    for i, r in enumerate(test_rows):
        subject = r.get("subject", "general")
        records.append(
            format_example(
                dataset="mmlu",
                uid=f"test_{i}",
                group=f"mmlu:{subject}",
                split="test",
                state=f"Subject: {subject.replace('_', ' ')}",
                question={
                    "type": "choice",
                    "instructions": r["question"],
                    "criteria": r["choices"],
                },
                target=r["answer"],
            )
        )

    return records


def load_replay_records(smoke_test: bool = False):
    from datasets import load_dataset

    records = []
    # 1. BoolQ (boolean QA -> noul primitive)
    try:
        print("Loading BoolQ replay buffer...")
        boolq = load_dataset("google/boolq")
        n_train = 100 if smoke_test else 1000
        n_test = 50 if smoke_test else 300
        for i, r in enumerate(boolq["train"].select(range(min(n_train, len(boolq["train"]))))):
            group = "passage:" + digest(normalized(r["passage"]))
            split = train_split(group)
            records.append(
                format_example(
                    dataset="boolq",
                    uid=f"train_{i}",
                    group=group,
                    split=split,
                    state=r["passage"],
                    question={"type": "noul", "instructions": r["question"]},
                    target=int(r["answer"]),
                )
            )
        for i, r in enumerate(boolq["validation"].select(range(min(n_test, len(boolq["validation"]))))):
            group = "passage:" + digest(normalized(r["passage"]))
            records.append(
                format_example(
                    dataset="boolq",
                    uid=f"test_{i}",
                    group=group,
                    split="test",
                    state=r["passage"],
                    question={"type": "noul", "instructions": r["question"]},
                    target=int(r["answer"]),
                )
            )
    except Exception as e:
        print(f"Warning: Could not load BoolQ: {e}")

    # 2. AG News (topic classification -> choice primitive)
    try:
        print("Loading AG News replay buffer...")
        ag_news = load_dataset("fancyzhx/ag_news")
        labels = ["World", "Sports", "Business", "Science and Technology"]
        n_train = 100 if smoke_test else 1000
        n_test = 50 if smoke_test else 300
        for i, r in enumerate(ag_news["train"].select(range(min(n_train, len(ag_news["train"]))))):
            group = "utterance:" + digest(normalized(r["text"]))
            split = train_split(group)
            records.append(
                format_example(
                    dataset="ag_news",
                    uid=f"train_{i}",
                    group=group,
                    split=split,
                    state="Text classification",
                    question={
                        "type": "choice",
                        "instructions": f"What is the topic of the news article: {r['text']}",
                        "criteria": labels,
                    },
                    target=int(r["label"]),
                )
            )
        for i, r in enumerate(ag_news["test"].select(range(min(n_test, len(ag_news["test"]))))):
            group = "utterance:" + digest(normalized(r["text"]))
            records.append(
                format_example(
                    dataset="ag_news",
                    uid=f"test_{i}",
                    group=group,
                    split="test",
                    state="Text classification",
                    question={
                        "type": "choice",
                        "instructions": f"What is the topic of the news article: {r['text']}",
                        "criteria": labels,
                    },
                    target=int(r["label"]),
                )
            )
    except Exception as e:
        print(f"Warning: Could not load AG News: {e}")

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/mmlu_replay"), help="Output directory"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Prepare a small slice for fast verification"
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    mmlu_records = load_mmlu_records(smoke_test=args.smoke_test)
    replay_records = load_replay_records(smoke_test=args.smoke_test)
    all_records = mmlu_records + replay_records

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "calibration": [],
        "test": [],
    }

    for record in all_records:
        split = record.pop("split")
        splits[split].append(record)

    for split_name, rows in splits.items():
        out_path = args.output / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} records to {out_path}")


if __name__ == "__main__":
    main()
