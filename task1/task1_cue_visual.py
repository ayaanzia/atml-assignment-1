import os, sys
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from utils.config import DEVICE, STL10_CLASSES, make_rng, seed_everything, SEED
from utils import data as dm
from utils import metrics as mx
from utils.adain import AdaINStyleTransfer

seed_everything(SEED)
print(f'Device: {DEVICE}')

                                                            
_, test_ds = dm.get_datasets(download=True)
eval_subset = dm.select_eval_subset(test_ds, total=500, seed_name='eval_subset')
eval_ds = Subset(test_ds, eval_subset['indices'])
eval_loader = dm.make_loader(eval_ds, batch_size=100, shuffle=False)

eval_imgs_by_class = {c: [] for c in range(len(STL10_CLASSES))}
for images, labels, image_ids in eval_loader:
    for image, label, image_id in zip(images, labels, image_ids):
        eval_imgs_by_class[int(label)].append((image, image_id))

print({STL10_CLASSES[c]: len(images) for c, images in eval_imgs_by_class.items()})

                                                                 
adain_model = AdaINStyleTransfer(
    vgg_weights_path='models/vgg_normalised.pth',
    decoder_weights_path='models/decoder.pth',
).to(DEVICE)
adain_model.eval()

cue_conflict_pairs = [
    ('cat', 'truck'),
    ('bird', 'ship'),
    ('dog', 'car'),
    ('horse', 'airplane'),
    ('monkey', 'deer'),
]
target_per_bucket = 22
class_to_idx = {name: i for i, name in enumerate(STL10_CLASSES)}
rng = make_rng('cue_conflict_sampling')
accepted_by_bucket = {}

def sample_pairs(content_pool, style_pool, n):
    content_indices = rng.integers(0, len(content_pool), size=n)
    style_indices = rng.integers(0, len(style_pool), size=n)
    return [(content_pool[i], style_pool[j]) for i, j in zip(content_indices, style_indices)]

for class_a, class_b in cue_conflict_pairs:
    idx_a, idx_b = class_to_idx[class_a], class_to_idx[class_b]
    directions = [
        ('A_shape_B_texture', idx_a, idx_b),
        ('B_shape_A_texture', idx_b, idx_a),
    ]
    for direction, content_class, texture_class in directions:
        bucket = f'{class_a}-{class_b}:{direction}'
        accepted_by_bucket[bucket] = []
        sampled = sample_pairs(
            eval_imgs_by_class[content_class],
            eval_imgs_by_class[texture_class],
            target_per_bucket,
        )
        for (content, content_id), (style, style_id) in sampled:
            with torch.inference_mode():
                stylized = adain_model.style_transfer(
                    content.unsqueeze(0).to(DEVICE),
                    style.unsqueeze(0).to(DEVICE),
                    alpha=1.0,
                ).squeeze(0).cpu()
            check = mx.is_valid_cue_conflict(
                content.permute(1, 2, 0).numpy(),
                stylized.permute(1, 2, 0).numpy(),
            )
            if check['accepted']:
                accepted_by_bucket[bucket].append({
                    'image': stylized,
                    'shape': STL10_CLASSES[content_class],
                    'texture': STL10_CLASSES[texture_class],
                    'content_id': content_id,
                    'style_id': style_id,
                    'ssim': check['ssim_to_content'],
                })

print({bucket: len(records) for bucket, records in accepted_by_bucket.items()})
print('Total accepted:', sum(map(len, accepted_by_bucket.values())))

def show_bucket(bucket, columns=6):
    records = accepted_by_bucket[bucket]
    if not records:
        print(f'No accepted cue conflicts for {bucket}')
        return

    rows = int(np.ceil(len(records) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.8 * columns, 3.0 * rows), squeeze=False)
    for ax, record in zip(axes.flat, records):
        ax.imshow(record['image'].permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(
            f"shape: {record['shape']} | texture: {record['texture']}\n"
            f"{record['content_id']} + {record['style_id']} | SSIM {record['ssim']:.3f}",
            fontsize=8,
        )
        ax.axis('off')
    for ax in axes.flat[len(records):]:
        ax.axis('off')
    fig.suptitle(f'{bucket} — {len(records)} accepted conflicts', fontsize=15)
    fig.tight_layout()
    plt.show()

show_bucket('cat-truck:A_shape_B_texture')

show_bucket('cat-truck:B_shape_A_texture')

show_bucket('bird-ship:A_shape_B_texture')

show_bucket('bird-ship:B_shape_A_texture')

show_bucket('dog-car:A_shape_B_texture')

show_bucket('dog-car:B_shape_A_texture')

show_bucket('horse-airplane:A_shape_B_texture')

show_bucket('horse-airplane:B_shape_A_texture')

show_bucket('monkey-deer:A_shape_B_texture')

show_bucket('monkey-deer:B_shape_A_texture')
