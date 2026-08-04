# Import migration

The public Python namespace is `mmctr`. New code should import through that namespace:

```python
from mmctr.models import DNN
from mmctr.data import get_data_loader
from mmctr.utils import helper
from mmctr.utils.run_context import create_run_context
from mmctr.experiments import ValidationOnlyTuner
```

The historical top-level `models` package and its `BaseModel`/`BaseSeqModel` implementations have
been removed from the distribution. Replace deep imports with `mmctr.models` exports or construct
by stable registry name:

```python
from mmctr.models.registry import create_model

model = create_model(model_name, model_config, data_config)
```

Model constructors no longer accept training config or logger arguments. Training state belongs to
`TrainingEngine`; RQ/PSRQ are obtained from `mmctr.quantization` or its registry.

Fusion studies no longer import from `analysis.fusion_analysis` or execute
`src/analysis/fusion_analysis.py`. Define the study in YAML, run `mmctr plan-fusion-study`, and pass
the verified tasks returned by `mmctr.analysis.load_fusion_study_matrix` to `ExperimentRunner`.

Modal robustness runs no longer execute `src/analysis/modal_robustness.py`. Use
`mmctr plan-robustness-study`, restore its tasks with
`mmctr.analysis.load_robustness_study_matrix`, and apply the declared `ModalityDropout` through
`TransformedDataLoader` in the shared experiment executor.

Alignment studies no longer use `src/analysis/alignment_analysis/`. Generate a matrix with
`mmctr plan-alignment-study`, load it through `mmctr.analysis.load_alignment_study_matrix`, and
apply the declared hooks/auxiliary objective without replacing the production model or Trainer.

Do not add experiment numbers or static figures under `src/`. Use `mmctr plot-results` for standard
metric figures, or call `load_standard_results` and `save_figure_provenance` from a specialized
report module when a paper figure needs a richer layout.

Cold-start jobs no longer use `Trainers_fenxi.py` or the old TFRecord filtering/batch scripts.
Create a `ColdStartProtocol` audit for the canonical split, then use
`mmctr plan-cold-start-study` and `load_cold_start_study_matrix`.
Do not add new external integrations against those paths. Public model classes and factories
should be obtained from `mmctr.models`.

The former validation helper in `mmctr.utils.tuning_protocol` was removed together with the legacy
tuners. New searches must use `ValidationOnlyTuner`, which freezes a validation-selected result
before any final-test task can be created.

The top-level `data` and `utils` packages were removed. New public imports must use `mmctr.data`
and `mmctr.utils`.
