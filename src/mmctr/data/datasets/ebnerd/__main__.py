"""Run EB-NeRD preprocessing using the project's existing Python environment."""

import argparse
import json
from pathlib import Path

import yaml

from mmctr.data.manifest import DatasetManifest

from .preprocessing import dataset_config, prepare_ebnerd, validate_ebnerd
from .sampling import sample_ebnerd_test


def prepare_all(
    raw_dir: Path,
    output_dir: Path,
    work_dir: Path,
    sequence_length: int = 5,
    test_fraction: float = 0.1,
    sampling_seed: int = 2025,
) -> DatasetManifest:
    """Build canonical-v4 through the existing resumable preprocessing stages."""
    from .enrichment import enrich_ebnerd
    from .id_features import prepare_id_features

    impressions = work_dir / "impressions"
    sampled = work_dir / "sampled"
    attributes = work_dir / "attributes"
    prepare_ebnerd(raw_dir, impressions, sequence_length)
    sample_ebnerd_test(impressions, sampled, test_fraction, sampling_seed)
    enrich_ebnerd(sampled, raw_dir, attributes)
    prepare_id_features(attributes, raw_dir, output_dir)
    return validate_ebnerd(output_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ebnerd"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(".tmp/ebnerd-prepare"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare", action="store_true", help="Build and verify canonical-v4 through all stages"
    )
    parser.add_argument("--sequence-length", type=int, default=5)
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--config-output", type=Path)
    mode.add_argument("--sample-test-from", type=Path)
    mode.add_argument("--enrich-from", type=Path)
    mode.add_argument("--ids-from", type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--sampling-seed", type=int, default=2025)
    args = parser.parse_args(argv)
    if not 0 < args.test_fraction <= 1:
        parser.error("--test-fraction must be in (0, 1]")
    if not 0 <= args.sampling_seed < 2**32:
        parser.error("--sampling-seed must be in [0, 2**32)")
    if args.output_dir is None:
        args.output_dir = Path("data/processed/ebnerd") / (
            "canonical-v4" if args.prepare else "canonical-v1"
        )
    if args.prepare:
        manifest = prepare_all(
            args.raw_dir,
            args.output_dir,
            args.work_dir,
            args.sequence_length,
            args.test_fraction,
            args.sampling_seed,
        )
    elif args.verify_only:
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
