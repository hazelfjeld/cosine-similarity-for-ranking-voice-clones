"""Batch command-line interface."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .core import VoiceRanker, save_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank voice clips against a reference by pretrained WavLM speaker similarity."
    )
    parser.add_argument("--reference", "-r", required=True, help="Reference speech file")
    parser.add_argument("--candidates", "-c", nargs="+", required=True,
                        help="Candidate audio files, directories, or ZIP archives (includes subfolders)")
    parser.add_argument("--output", "-o", default="rankings.csv", help="Sorted CSV results")
    parser.add_argument("--json", dest="json_path", help="Optional JSON results path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-vad", action="store_true", help="Do not isolate voice-active regions")
    parser.add_argument("--no-cache", action="store_true", help="Recompute all embeddings")
    parser.add_argument("--cache-dir", default=".voice_ranker_cache")
    parser.add_argument("--max-segments", type=int, default=6)
    parser.add_argument("--max-files", type=int, default=5000)
    parser.add_argument("--model", default="microsoft/wavlm-base-plus-sv",
                        help="Hugging Face WavLM speaker verification model ID or local model dir")
    args = parser.parse_args(argv)

    print(f"Finding clips to compare with {args.reference}. Loading model if necessary...")
    scorer = VoiceRanker(model_name=args.model, device=args.device,
                         use_vad=not args.no_vad, use_cache=not args.no_cache,
                         cache_dir=args.cache_dir, max_segments=args.max_segments)
    try:
        reference, results = scorer.rank(args.reference, args.candidates, max_files=args.max_files)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    path = save_csv(results, args.output)
    if args.json_path:
        json_path = Path(args.json_path).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"reference": str(args.reference),
                                        "reference_speech_seconds": reference.speech_seconds,
                                        "results": [asdict(row) for row in results]}, indent=2),
                             encoding="utf-8")
    print(f"Reference: {reference.speech_seconds:.1f}s of speech, {reference.n_segments} windows")
    print("\nRank  Cosine    Filename")
    print("----  --------  -------------------------------------------")
    for row in results[:25]:
        print(f"{row.rank or '-':>4}  {row.similarity if row.similarity is not None else float('nan'):>8.4f}  {row.filename}")
        if row.error:
            print(f"               ERROR: {row.error[:115]}")
    n_failed = sum(1 for row in results if row.error)
    print(f"\nSaved {len(results)} rows -> {path} ({n_failed} failed)")
    print("Cosine is NOT a probability or a percentage. Rank is relative to this reference.")
    return 0 if n_failed < len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
