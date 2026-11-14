"""Run EB-NeRD preprocessing using the project's existing Python environment."""

import argparse
import json
from pathlib import Path

import yaml

from .preprocessing import dataset_config, prepare_ebnerd, validate_ebnerd
from .sampling import sample_ebnerd_test


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ebnerd"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/ebnerd/canonical-v1")
    )
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--sample-test-from", type=Path)
    parser.add_argument("--enrich-from", type=Path)
    parser.add_argument("--ids-from", type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--sampling-seed", type=int, default=2025)
    args = parser.parse_args(argv)
    if args.verify_only:
        manifest = validate_ebnerd(args.output_dir)
    elif args.ids_from:
        from .id_features import prepare_id_features

        manifest = prepare_id_features(args.ids_from, args.raw_dir, args.output_dir)
    elif args.enrich_from:
        from .enrichment import enrich_ebnerd

        manifest = enrich_ebnerd(args.enrich_from, args.raw_dir, args.output_dir)
    elif args.sample_test_from:
        manifest = sample_ebnerd_test(
            args.sample_test_from, args.output_dir, args.test_fraction, args.sampling_seed
        )
    else:
        manifest = prepare_ebnerd(args.raw_dir, args.output_dir, args.sequence_length)
    if args.config_output:
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.write_text(
            yaml.safe_dump({"ebnerd": dataset_config(manifest, args.output_dir)}, sort_keys=False),
            encoding="utf-8",
        )
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
