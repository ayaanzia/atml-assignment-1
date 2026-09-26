from pathlib import Path
import sys, yaml, torch, pandas as pd, matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == 'task4': ROOT = ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
TASK = ROOT / 'task4'
RESULTS = TASK / 'results'
CACHE = RESULTS / 'cache'
CHECKPOINTS = RESULTS / 'checkpoints'
DATA_ROOT = TASK / 'data' / 'cifar'
for directory in (CACHE, CHECKPOINTS): directory.mkdir(parents=True, exist_ok=True)
SEED = 6304
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print('Device:', DEVICE)

from task4.methods.vanilla import seed_everything
seed_everything(SEED)

def config(name):
    with (TASK / 'configs' / f'{name}.yaml').open() as handle:
        return yaml.safe_load(handle)

VANILLA_CFG, GCSC_CFG, PROSER_CFG = map(config, ('vanilla', 'gcsc', 'proser'))
RUN_TRAINING = True                                                               
FORCE_CACHE = False                                                              

from task4.data.cifar10 import build_cifar10_loaders
from task4.models import resnet18_cifar

known = build_cifar10_loaders(DATA_ROOT, batch_size=128, seed=SEED, randaugment=False,
                                    download=True, pin_memory=DEVICE.type == 'cuda')
assert len(known['train_indices']) == 45000 and len(known['val_indices']) == 5000
assert all(loader.num_workers == 0 for key, loader in known.items() if hasattr(loader, 'num_workers'))
resnet18_cifar(10)

from task4.methods.vanilla import train_vanilla, known_accuracy
from task4.extract_outputs import extract_or_load

vanilla_path = CHECKPOINTS / 'vanilla_seed6304.pt'
if RUN_TRAINING or not vanilla_path.exists():
    vanilla, vanilla_history = train_vanilla(known['train'], known['val'], VANILLA_CFG, vanilla_path, DEVICE)
else:
    vanilla = resnet18_cifar(10)
    vanilla.load_state_dict(torch.load(vanilla_path, map_location='cpu', weights_only=False)['model'])
    vanilla = vanilla.to(DEVICE, memory_format=torch.channels_last)
print(f'Vanilla CIFAR-10 test CSA: {known_accuracy(vanilla, known["test"], DEVICE):.4f}')

                                                                                              
vanilla_cache = {
    'train_aug': extract_or_load(vanilla, known['train'], DEVICE, CACHE/'vanilla_train_aug.pt', FORCE_CACHE),
    'train_unaug': extract_or_load(vanilla, known['train_eval'], DEVICE, CACHE/'vanilla_train_unaug.pt', FORCE_CACHE),
    'val': extract_or_load(vanilla, known['val'], DEVICE, CACHE/'vanilla_val.pt', FORCE_CACHE),
    'test': extract_or_load(vanilla, known['test'], DEVICE, CACHE/'vanilla_test.pt', FORCE_CACHE),
}

from task4.scores import (msp_unknownness, mls_unknownness, energy_unknownness,
                          fit_shared_diagonal_gaussian, mahalanobis_unknownness)
from task4.evaluation.thresholds import calibrate_threshold

MAH_MEANS, MAH_VARIANCE = fit_shared_diagonal_gaussian(
    vanilla_cache['train_unaug']['features'], vanilla_cache['train_unaug']['labels'], eps=1e-6)

SCORE_FUNCTIONS = {
    'MSP': lambda cache: msp_unknownness(cache['logits'][:, :10]),
    'MLS': lambda cache: mls_unknownness(cache['logits'][:, :10]),
    'Energy': lambda cache: energy_unknownness(cache['logits'][:, :10]),
    'Mahalanobis': lambda cache: mahalanobis_unknownness(cache['features'], MAH_MEANS, MAH_VARIANCE),
}
                                                            
VANILLA_THRESHOLDS = {name: calibrate_threshold(fn(vanilla_cache['val']))
                      for name, fn in SCORE_FUNCTIONS.items()}
VANILLA_THRESHOLDS

from task4.methods.gcsc import train_gcsc

gcsc_known = build_cifar10_loaders(DATA_ROOT, 128, SEED, randaugment=True,
                                         download=False, pin_memory=DEVICE.type == 'cuda')
gcsc_path = CHECKPOINTS / 'gcsc_seed6304.pt'
if RUN_TRAINING or not gcsc_path.exists():
    gcsc, gcsc_history = train_gcsc(gcsc_known['train'], gcsc_known['val'], GCSC_CFG, gcsc_path, DEVICE)
else:
    gcsc = resnet18_cifar(10)
    gcsc.load_state_dict(torch.load(gcsc_path, map_location='cpu', weights_only=False)['model'])
    gcsc = gcsc.to(DEVICE, memory_format=torch.channels_last)

gcsc_cache = {split: extract_or_load(gcsc, gcsc_known[loader], DEVICE, CACHE/f'gcsc_{split}.pt', FORCE_CACHE)
              for split, loader in {'train_unaug':'train_eval', 'val':'val', 'test':'test'}.items()}

from task4.methods.proser import PROSER, train_proser, placeholder_unknownness

proser_path = CHECKPOINTS / 'proser_seed6304.pt'
if RUN_TRAINING or not proser_path.exists():
    proser = PROSER.from_vanilla_checkpoint(vanilla_path, num_dummy=5)
    proser, proser_history = train_proser(proser, known['train'], known['val'], PROSER_CFG, proser_path, DEVICE)
else:
    proser = PROSER(10, 5)
    proser.load_state_dict(torch.load(proser_path, map_location='cpu', weights_only=False)['model'])
    proser = proser.to(DEVICE, memory_format=torch.channels_last)

proser_cache = {split: extract_or_load(proser, known[loader], DEVICE, CACHE/f'proser_{split}.pt', FORCE_CACHE)
                for split, loader in {'train_unaug':'train_eval', 'val':'val', 'test':'test'}.items()}
                                                                                                 
PROSER_MLS_THRESHOLD = calibrate_threshold(mls_unknownness(proser_cache['val']['logits'][:, :10]))
PROSER_PLACEHOLDER_THRESHOLD = calibrate_threshold(placeholder_unknownness(proser_cache['val']['logits']))

from task4.data.cifar100_unknowns import build_unknown_loaders, NEAR_CLASSES, FAR_CLASSES

unknown = build_unknown_loaders(DATA_ROOT, 128, download=True, pin_memory=DEVICE.type == 'cuda')
assert len(unknown['near'].dataset) == len(unknown['far'].dataset) == 800
models = {'vanilla': vanilla, 'gcsc': gcsc, 'proser': proser}
for model_name, model in models.items():
    for group in ('near', 'far'):
        extract_or_load(model, unknown[group], DEVICE, CACHE/f'{model_name}_{group}.pt', FORCE_CACHE)
print('Cached one batched forward pass per model/group; subsequent evaluation is cache-only.')

from task4.evaluate_osr import evaluate_all

vanilla_table, trained_table, failures = evaluate_all(
    CACHE, RESULTS, unknown['dataset'].classes)
display(vanilla_table.round(4))
display(trained_table.round(4))

display(failures)
fig, axes = plt.subplots(2, 3, figsize=(8, 5), constrained_layout=True)
for ax, (_, row) in zip(axes.flat, failures.iterrows()):
    ax.imshow(unknown['dataset'].data[int(row.source_index)])
    ax.set_xlabel(f"{row.unknown_class} → {row.predicted_cifar10_class}\nu={row.mls_unknownness:.3f}, τ={row.threshold:.3f}")
    ax.set_xticks([]); ax.set_yticks([])
print('Figure: Incorrectly accepted near and far unknowns under the validation-calibrated Vanilla MLS threshold')
plt.show()

required = [
    CHECKPOINTS/'vanilla_seed6304.pt', CHECKPOINTS/'gcsc_seed6304.pt', CHECKPOINTS/'proser_seed6304.pt',
    RESULTS/'vanilla_score_comparison.csv', RESULTS/'trained_model_comparison.csv',
    RESULTS/'score_distribution_or_roc_figure.png', RESULTS/'failure_cases.csv',
]
assert all(path.exists() for path in required), [str(path) for path in required if not path.exists()]
pd.DataFrame({'artifact': [str(path.relative_to(TASK)) for path in required],
              'exists': [path.exists() for path in required]})
