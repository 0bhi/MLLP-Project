"""Command-line interface for ChartScribe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import ChartScribePipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chartscribe",
        description="Extract chart data from an image and generate a grounded summary.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one chart image.")
    analyze.add_argument("image", type=Path)
    analyze.add_argument(
        "--chart-type", choices=("auto", "bar", "pie", "line", "infographic"), default="auto"
    )
    analyze.add_argument("--backend", choices=("template", "llm"), default="template")
    analyze.add_argument("--output", type=Path, help="Write the JSON result to this file.")
    analyze.add_argument("--debug-image", type=Path, help="Write an OCR/extraction overlay image.")

    batch = subparsers.add_parser("batch", help="Analyze every image in a directory.")
    batch.add_argument("directory", type=Path)
    batch.add_argument("--output", type=Path, default=Path("outputs/batch_results.json"))
    batch.add_argument("--chart-type", choices=("auto", "bar", "pie", "line", "infographic"), default="auto")
    return parser


def _analyze(args: argparse.Namespace) -> int:
    pipeline = ChartScribePipeline()
    result, debug_image = pipeline.analyze(
        args.image,
        chart_type=args.chart_type,
        style="detailed",
        backend=args.backend,
        source_name=str(args.image),
    )
    payload = result.as_dict()
    print(result.summary.text)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.debug_image:
        args.debug_image.parent.mkdir(parents=True, exist_ok=True)
        import cv2

        cv2.imwrite(str(args.debug_image), debug_image)
    return 0


def _batch(args: argparse.Namespace) -> int:
    images = sorted(
        path for path in args.directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not images:
        raise SystemExit(f"No supported images found in {args.directory}")
    pipeline = ChartScribePipeline()
    results = []
    for image in images:
        result, _ = pipeline.analyze(image, chart_type=args.chart_type, source_name=str(image))
        results.append(result.as_dict())
        print(f"{image.name}: {result.summary.text}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "analyze":
        return _analyze(args)
    return _batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
