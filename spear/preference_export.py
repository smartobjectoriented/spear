"""Non-interactive export command for paired and unpaired preference data."""

from __future__ import annotations

import argparse
import json

from preference_dataset import (
    PreferenceConfiguration, PreferenceDatasetBuilder, PreferenceProfile,
)
from training_splits import SplitConfiguration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m preference_export",
        description="Mine grounded preference data from canonical SPEAR episodes.",
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default=PreferenceProfile.PAIRED_DPO.value,
                        choices=[item.value for item in PreferenceProfile])
    parser.add_argument("--train-percent", type=int, default=90)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--test-percent", type=int, default=5)
    parser.add_argument("--max-approximate-tokens", type=int, default=32768)
    parser.add_argument("--project", action="append", default=[])
    parser.add_argument("--split", action="append", default=[],
                        choices=("train", "validation", "test"))
    parser.add_argument("--superseded-memory-id", action="append", default=[])
    parser.add_argument("--allow-external-web", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configuration = PreferenceConfiguration(
        profile=args.profile,
        split=SplitConfiguration(args.train_percent, args.validation_percent,
                                 args.test_percent),
        allow_external_web=args.allow_external_web,
        max_approximate_tokens=args.max_approximate_tokens,
        project_filters=tuple(sorted(set(args.project))),
        split_filters=tuple(sorted(set(args.split))),
        superseded_memory_ids=tuple(sorted(set(args.superseded_memory_id))),
    )
    builder = PreferenceDatasetBuilder(args.source, configuration=configuration)

    if args.rebuild_index and not (args.dry_run or args.report_only):
        builder.rebuild_index()

    result, dataset_dir = builder.materialize(
        args.output, dry_run=args.dry_run, report_only=args.report_only,
    )
    print(json.dumps({
        "dataset_directory": str(dataset_dir) if dataset_dir else None,
        **result.report,
    }, sort_keys=True, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
