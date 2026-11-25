"""Check and prepare frozen quantizers before batch CTR training."""

import argparse
import fcntl
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Mapping, Optional, Sequence
from zipfile import BadZipFile

from mmctr.config import load_model_config, load_yaml_mapping, runtime_defaults
from mmctr.data import available_datasets
from mmctr.models import available_models


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def artifact_problem(model: str, config: Mapping, data: Mapping, root: Path) -> Optional[str]:
    """Return an artifact error, using the same checks as the CTR consumer."""
    from mmctr.core import ContractError
    from mmctr.quantization.artifacts import QuantizationArtifactError
    from mmctr.quantization.loading import load_model_quantization_dependencies

    try:
        load_model_quantization_dependencies(model, config, data, root)
    except (QuantizationArtifactError, ContractError, BadZipFile, EOFError) as error:
        return str(error)
    return None


def ensure_quantizers(
    dataset: str,
    models: Sequence[str],
    cuda: int = 0,
    use_local_data: bool = False,
    dry_run: bool = False,
) -> None:
    """Reuse valid artifacts; fit and revalidate only required missing/invalid ones.

    Real preparation is serialized per dataset across batch launchers. A failed
    pretrainer or failed post-fit validation propagates before CTR workers start.
    Dry runs read existing artifacts but never fit or create files.
    """
    if dataset not in available_datasets():
        raise ValueError("unknown dataset: " + dataset)
    for model in models:
        if model not in available_models():
            raise ValueError("unknown model: " + model)
    required = [name for name in ("qarm", "psrq") if name in models]
    if not required:
        return
    data = load_yaml_mapping(PROJECT_ROOT / "configs/datasets/catalog.yaml")[dataset]
    root = Path(runtime_defaults(PROJECT_ROOT)["quantization_artifact_dir"])
    parameters = {
        model: load_model_config(
            PROJECT_ROOT / "configs/models", dataset, model, data.get("fingerprint")
        )["model"]
        for model in required
    }

    def prepare() -> None:
        for model in required:
            config = parameters[model]
            missing: List[str] = []
            if model == "qarm":
                for modality in data["use_mm_features"]:
                    if modality == "id":
                        continue
                    problem = artifact_problem(
                        model, config, dict(data, use_mm_features=[modality]), root
                    )
                    if problem:
                        print("RQ/{} needs pretraining: {}".format(modality, problem), flush=True)
                        missing.append(modality)
                if not missing:
                    print("QARM quantizers ready; reusing existing artifacts.", flush=True)
                    continue
                rq_config = load_model_config(PROJECT_ROOT / "configs/models", dataset, "rq")[
                    "model"
                ]
                for key, default in (("n_levels", 3), ("codebook_size", 1024)):
                    if int(config.get(key, default)) != int(rq_config.get(key, default)):
                        raise ValueError(
                            "QARM and RQ {} must match in configs/models/{}".format(key, dataset)
                        )
                module = "mmctr.quantization.rq_entrypoint"
                options = ["--modalities", *missing]
            else:
                problem = artifact_problem(model, config, data, root)
                if problem is None:
                    print("PSRQ quantizer ready; reusing existing artifact.", flush=True)
                    continue
                print("PSRQ needs pretraining: {}".format(problem), flush=True)
                module = "mmctr.quantization.psrq_entrypoint"
                options = ["--cuda", str(cuda)]
            command = [sys.executable, "-m", module, "--dataset-name", dataset, *options]
            if use_local_data:
                command.append("--use-local-data")
            print(
                ("Would pretrain: " if dry_run else "Pretraining: ") + shlex.join(command),
                flush=True,
            )
            if dry_run:
                continue
            subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
            problem = artifact_problem(model, config, data, root)
            if problem:
                raise RuntimeError(
                    "{} pretraining did not produce usable artifacts: {}".format(model, problem)
                )
            print("{} quantizers ready after pretraining.".format(model.upper()), flush=True)

    if dry_run:
        prepare()
    else:
        root.mkdir(parents=True, exist_ok=True)
        with (root / (".prepare-" + dataset + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            prepare()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="antm2c")
    parser.add_argument("--models", default="qarm,psrq")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--use-local-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.num_threads)
    ensure_quantizers(
        args.dataset_name, args.models.split(","), args.cuda, args.use_local_data, args.dry_run
    )


if __name__ == "__main__":
    main()
