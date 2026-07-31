# Import migration

The public Python namespace is `mmctr`. New code should import through that namespace:

```python
from mmctr.models import DNN
from mmctr.data import get_data_loader
from mmctr.utils import helper
from mmctr.utils.run_context import create_run_context
from mmctr.utils.tuning_protocol import evaluate_for_selection
```

The first package-migration step keeps the historical top-level `models`, `data`, and `utils`
packages in the distribution. They are compatibility implementation packages, not the preferred
public API. This bridge lets existing research scripts continue to run while model and data
modules move under `mmctr` in their dedicated refactor tasks.

The helper registry now imports model and data packages only when a factory is called. This makes
both of these orders valid:

```python
from models.ctr_models.dnn import DNN  # legacy compatibility
from mmctr.utils import helper
```

```python
from mmctr.utils import helper
from mmctr.models import DNN
```

Deep implementation imports such as `models.mm_ctr_models.MCCA` remain legacy during this bridge.
Do not add new external integrations against those paths. Public model classes and factories
should be obtained from `mmctr.models`.

The compatibility bridge will remain until the corresponding model/data migration tasks provide
canonical implementations and migration notes. Removal requires a separate deprecation decision;
`PKG-001` does not silently delete old imports.
