"""Create reproducible source-group splits; never split augmented siblings."""
import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def prepare(root, output, kind, seed=42):
    root = Path(root).resolve()
    records = []
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(root)
        stem = re.sub(r'_(?:aug_\d+|original)$', '', path.stem)
        with Image.open(path) as image:
            image.load()
            digest = hashlib.sha256(image.convert('L').tobytes() + str(image.size).encode()).hexdigest()
        records.append(dict(path=str(relative), label=unicodedata.normalize('NFC', path.parent.name),
                            group=str(relative.parent / stem), digest=digest,
                            original=not bool(re.search(r'_aug_\d+$', path.stem)),
                            official_test=kind == 'devanagari' and relative.parts[0].lower() == 'test'))
    if not records:
        raise ValueError(f'No readable images found under {root}')
    # Unite exact duplicate pixels as well as filename-defined augmentation families.
    parents = {r['group']: r['group'] for r in records}
    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    hashes = {}
    for row in records:
        if row['digest'] in hashes:
            parents[find(row['group'])] = find(hashes[row['digest']])
        hashes[row['digest']] = row['group']
    groups = defaultdict(list)
    for row in records:
        row['group'] = find(row['group'])
        groups[row['group']].append(row)
    by_class = defaultdict(list)
    excluded_duplicates = []
    overlap_groups = 0
    for key, rows in groups.items():
        labels = {row['label'] for row in rows}
        if len(labels) != 1:
            raise ValueError(f'Conflicting labels on identical images: {labels}; group={key}')
        if any(r['official_test'] for r in rows):
            if not all(r['official_test'] for r in rows):
                overlap_groups += 1
            for row in rows:
                # Preserve the official test samples, excluding the entire related
                # training family (including augmentations) from train AND val.
                # Never move training copies into test and inflate its sample count.
                if row['official_test']:
                    row['split'] = 'test'
                else:
                    row['split'] = 'excluded'
                    excluded_duplicates.append(dict(path=row['path'], group=key,
                                                    reason='source_group_overlaps_official_test'))
        else:
            by_class[rows[0]['label']].append(key)
    rng = random.Random(seed)
    has_official_test = any(r['official_test'] for r in records)
    for label, keys in sorted(by_class.items()):
        rng.shuffle(keys)
        if len(keys) < 3:
            raise ValueError(f'{label}: fewer than three independent sources')
        val_n = max(1, round(len(keys)*0.2))
        test_n = 0 if has_official_test else max(1, round(len(keys)*0.2))
        for index, key in enumerate(keys):
            split = 'val' if index < val_n else 'test' if index < val_n+test_n else 'train'
            for row in groups[key]:
                row['split'] = split
    # Evaluate only unaugmented images, not synthetic copies of held-out crops.
    included = [r for r in records if r['split'] == 'train' or
                (r['split'] in ('val', 'test') and r['original'])]
    labels = sorted({r['label'] for r in records})
    for split in ('train', 'val', 'test'):
        absent = set(labels) - {r['label'] for r in included if r['split'] == split}
        if absent:
            raise ValueError(f'{split} missing labels: {sorted(absent)}')
    report = dict(images=len(records), source_groups=len(groups), classes=len(labels),
                  leaf_folders=len({str(Path(r['path']).parent) for r in records}),
                  used_images=dict(Counter(r['split'] for r in included)),
                  official_overlap_groups=overlap_groups,
                  excluded_training_overlap_images=len(excluded_duplicates),
                  omitted_heldout_augmentations=sum(r['split'] in ('val', 'test') and
                                                   not r['original'] for r in records),
                  note='Source-crop disjoint, not writer/page disjoint: writer and page IDs were not provided.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(root=str(root), kind=kind, seed=seed, labels=labels,
                                     report=report, records=included,
                                     excluded_records=excluded_duplicates), ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--kind', choices=['modi', 'devanagari'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.root, args.out, args.kind, args.seed)
