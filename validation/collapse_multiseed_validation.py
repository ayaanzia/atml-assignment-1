from pathlib import Path
import json
import sys
import pandas as pd
import torch
from IPython.display import Image, display

REPO_ROOT = Path.cwd().resolve()
if REPO_ROOT.name == 'validation':
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from validation.collapse_multiseed import (
    OUT, VALIDATION_SEEDS, check_protocol_only, protocol_record, run_validation
)

print('CUDA available:', torch.cuda.is_available())
print(json.dumps(protocol_record(), indent=2))

protocol_check = check_protocol_only()
pd.DataFrame({
    'source_train': protocol_check['source_train_sizes'],
    'source_validation': protocol_check['source_validation_sizes'],
})

combined_results, aggregate_summary = run_validation(force=False)

display(combined_results)
display(aggregate_summary)
display(Image(filename=str(OUT / 'multiseed_accuracy.png')))
