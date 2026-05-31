#!/usr/bin/env python3
"""Write checkpoint provenance for the MG learned-prior run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def checkpoint_role(path: Path, train_json: dict[str, object], selected_model: Path) -> str:
    roles = []
    if path.resolve() == selected_model.resolve():
        roles.append("used_for_full_assembly")
    if str(path) == str(train_json.get("best_checkpoint")) or (
        path.name.endswith("_best.pt") and not path.name.endswith("_val_best.pt")
    ):
        roles.append("training_proxy_best")
    if path.name.endswith("_val_best.pt"):
        roles.append("validation_best")
    if str(path) == str(train_json.get("latest_checkpoint")) or path.name.endswith("_latest.pt"):
        roles.append("latest")
    if str(path) == str(train_json.get("checkpoint")):
        roles.append("final")
    return "+".join(roles) if roles else "candidate"


def load_checkpoint_metadata(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    train_instances = config.get("train_instances", []) if isinstance(config, dict) else []
    validation_instances = config.get("validation_instances", []) if isinstance(config, dict) else []
    return {
        "checkpoint": str(path),
        "epoch": checkpoint.get("epoch", "") if isinstance(checkpoint, dict) else "",
        "training_objective": checkpoint.get("training_objective", "") if isinstance(checkpoint, dict) else "",
        "best_energy": checkpoint.get("best_energy", "") if isinstance(checkpoint, dict) else "",
        "val_energy": checkpoint.get("val_energy", "") if isinstance(checkpoint, dict) else "",
        "val_objective": checkpoint.get("val_objective", "") if isinstance(checkpoint, dict) else "",
        "validation_objective": checkpoint.get("validation_objective", "") if isinstance(checkpoint, dict) else "",
        "training_seconds": checkpoint.get("training_seconds", "") if isinstance(checkpoint, dict) else "",
        "config_source": config.get("source", "") if isinstance(config, dict) else "",
        "config_training_objective": config.get("training_objective", "") if isinstance(config, dict) else "",
        "config_train_instances": len(train_instances),
        "config_validation_instances": len(validation_instances),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--selected-model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    train_json = json.loads(args.train_json.read_text())
    checkpoints = sorted(args.checkpoint_dir.glob("*.pt"))
    rows = []
    for path in checkpoints:
        row = load_checkpoint_metadata(path)
        row["role"] = checkpoint_role(path, train_json, args.selected_model)
        row["selected_for_full_assembly"] = int(path.resolve() == args.selected_model.resolve())
        rows.append(row)

    fields = [
        "checkpoint",
        "role",
        "selected_for_full_assembly",
        "epoch",
        "training_objective",
        "best_energy",
        "val_energy",
        "val_objective",
        "validation_objective",
        "training_seconds",
        "config_source",
        "config_training_objective",
        "config_train_instances",
        "config_validation_instances",
    ]
    write_csv(args.out_dir / "checkpoint_provenance.csv", rows, fields)

    split_rows = [
        {"split": "train", "instances": len(train_json.get("train_gfas", []))},
        {"split": "validation", "instances": len(train_json.get("test_gfas", []))},
        {"split": "final_eval_in_training_json", "instances": len(train_json.get("eval", []))},
    ]
    write_csv(args.out_dir / "training_split_provenance.csv", split_rows, ["split", "instances"])

    selected = [row for row in rows if row["selected_for_full_assembly"]]
    selected_role = selected[0]["role"] if selected else "missing"
    print(f"wrote {args.out_dir / 'checkpoint_provenance.csv'}")
    print(f"selected_model_role={selected_role}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
