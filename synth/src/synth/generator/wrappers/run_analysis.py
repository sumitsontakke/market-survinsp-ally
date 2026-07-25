from __future__ import annotations

import argparse
import json
from pathlib import Path

from synth.generator.analysis.pipeline import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run correlation and network analysis on exported datasets.")
    parser.add_argument("--input", required=True, help="Input dataset directory.")
    parser.add_argument("--source", default="orders", choices=["orders", "trades"], help="Event source to analyze.")
    parser.add_argument("--bucket-minutes", type=int, default=1, help="Time bucket size in minutes.")
    parser.add_argument("--corr-method", default="pearson", choices=["pearson", "spearman"], help="Correlation method.")
    parser.add_argument("--corr-threshold", type=float, default=0.7, help="Correlation threshold for graph edges.")
    parser.add_argument("--min-active-buckets", type=int, default=2, help="Minimum non-zero buckets required per trader.")
    parser.add_argument("--instrument-id", help="Optional instrument filter.")
    parser.add_argument("--output", help="Output analysis directory. Defaults to <input>/analysis.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output or str(Path(args.input) / "analysis")
    result = run_analysis(
        input_dir=args.input,
        output_dir=output_dir,
        source=args.source,
        bucket_minutes=args.bucket_minutes,
        corr_method=args.corr_method,
        corr_threshold=args.corr_threshold,
        min_active_buckets=args.min_active_buckets,
        instrument_id=args.instrument_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
