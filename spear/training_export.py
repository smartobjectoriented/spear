"""Non-interactive command line entry point for FT1 SFT exports."""

from __future__ import annotations

import argparse
import json

from sft_dataset import (
    ExportProfile, SFTDatasetBuilder, SFTExportConfiguration,
    SplitConfiguration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training_export",
        description="Curate canonical SPEAR episodes into OpenAI-message SFT data.",
    )
    parser.add_argument("--source", required=True,
                        help="Canonical TrainingStore root (contains episodes/).")
    parser.add_argument("--output", required=True,
                        help="Directory below which an immutable dataset ID is written.")
    parser.add_argument("--profile", default=ExportProfile.QWEN3_AXOLOTL.value,
                        choices=[item.value for item in ExportProfile])
    parser.add_argument("--train-percent", type=int, default=90)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--test-percent", type=int, default=5)
    parser.add_argument("--max-approximate-tokens", type=int, default=32768)
    parser.add_argument("--project", action="append", default=[],
                        help="Only include an exact project/corpus name; repeatable.")
    parser.add_argument("--superseded-memory-id", action="append", default=[],
                        help="Current superseded memory provenance ID; repeatable.")
    parser.add_argument("--allow-external-web", action="store_true",
                        help="Explicitly include reviewed external-web-dependent targets.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and validate in memory without writing trainer files.")
    parser.add_argument("--report-only", action="store_true",
                        help="Print aggregate quality data without writing any files.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configuration = SFTExportConfiguration(
        profile=args.profile,
        split=SplitConfiguration(args.train_percent, args.validation_percent,
                                 args.test_percent),
        allow_external_web=args.allow_external_web,
        max_approximate_tokens=args.max_approximate_tokens,
        project_filters=tuple(sorted(set(args.project))),
        superseded_memory_ids=tuple(sorted(set(args.superseded_memory_id))),
    )
    builder = SFTDatasetBuilder(args.source, configuration=configuration)
    result, dataset_dir = builder.materialize(
        args.output, dry_run=args.dry_run, report_only=args.report_only,
    )
    summary = {
        "dataset_directory": str(dataset_dir) if dataset_dir else None,
        "episodes": result.report["episodes"],
        "turn_candidates": result.report["turn_candidates"],
        "selected_sample_types": result.report["selected_sample_types"],
        "review_status": result.report["review_status"],
        "exclusion_reasons": result.report["exclusion_reasons"],
        "split_sizes": result.report["distribution"]["split_sizes"],
        "small_sample_warning": result.report["small_sample_warning"],
    }
    print(json.dumps(summary, sort_keys=True, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
