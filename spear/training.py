"""Cohesive non-interactive CLI for readiness and frozen training bundles."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from training_bundle import (
    SourceModelProfile, TrainingBundleBuilder, TrainingBundleConfiguration,
    TrainingHardwareReport, validate_training_bundle,
)
from training_governance import TrainingDataGovernancePolicy
from training_readiness import TrainingReadinessEvaluator
from training_splits import SplitConfiguration


def _governance(args) -> TrainingDataGovernancePolicy:
    return TrainingDataGovernancePolicy(
        include_projects=tuple(sorted(set(getattr(args, "include_project", ())))),
        exclude_projects=tuple(sorted(set(getattr(args, "exclude_project", ())))),
        permanent_holdout_groups=tuple(sorted(set(getattr(args, "holdout_group", ())))),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training",
        description="Inspect readiness and freeze governed SPEAR training bundles.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("readiness", "freeze"):
        item = sub.add_parser(name)
        item.add_argument("--source", required=True)
        item.add_argument("--include-project", action="append", default=[])
        item.add_argument("--exclude-project", action="append", default=[])
        item.add_argument("--holdout-group", action="append", default=[])
        item.add_argument("--train-percent", type=int, default=90)
        item.add_argument("--validation-percent", type=int, default=5)
        item.add_argument("--test-percent", type=int, default=5)

        if name == "freeze":
            item.add_argument("--output", required=True)
            item.add_argument("--profile", default="qwen3_coder_next_sft",
                              choices=("qwen3_coder_next_sft", "qwen3_coder_next_base"))
            item.add_argument("--base-model")
            item.add_argument("--tokenizer-path")
            item.add_argument("--hardware-report", action="store_true")
            item.add_argument("--target-hardware-json")
            item.add_argument("--dry-run", action="store_true")
            item.add_argument("--report-only", action="store_true")

    inspect = sub.add_parser("inspect-bundle")
    inspect.add_argument("bundle")
    validate = sub.add_parser("validate-bundle")
    validate.add_argument("bundle")
    validate.add_argument("--source")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "readiness":
        split = SplitConfiguration(args.train_percent, args.validation_percent,
                                   args.test_percent)
        report = TrainingReadinessEvaluator(
            args.source, governance=_governance(args), split=split,
        ).evaluate()
        print(json.dumps(report.to_dict(), sort_keys=True, indent=2))

        return 0

    if args.command == "freeze":
        split = SplitConfiguration(args.train_percent, args.validation_percent,
                                   args.test_percent)
        model = SourceModelProfile.named(args.profile)

        if args.base_model:
            model = replace(model, base_model=args.base_model)

        hardware = None

        if args.target_hardware_json:
            hardware = TrainingHardwareReport.from_dict(json.loads(
                Path(args.target_hardware_json).read_text(encoding="utf-8"),
            )).classify()
        elif args.hardware_report:
            hardware = TrainingHardwareReport.detect(args.output).classify()

        configuration = TrainingBundleConfiguration(
            model=model, governance=_governance(args),
            split=split,
            tokenizer_path=args.tokenizer_path, hardware_report=hardware,
        )
        summary, directory = TrainingBundleBuilder(
            args.source, configuration=configuration,
        ).freeze(args.output, dry_run=args.dry_run or args.report_only)
        print(json.dumps({"bundle_directory": str(directory) if directory else None,
                          **summary}, sort_keys=True, indent=2))

        return 0

    if args.command == "inspect-bundle":
        root = Path(args.bundle).resolve()
        value = {name: json.loads((root / name).read_text())
                 for name in ("manifest.json", "readiness.json", "governance.json")}
        print(json.dumps(value, sort_keys=True, indent=2))

        return 0

    manifest = validate_training_bundle(args.bundle, args.source)
    print(json.dumps({"valid": True, "bundle_id": manifest["bundle_id"]},
                     sort_keys=True, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
