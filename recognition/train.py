"""Devanagari pretraining, Modi transfer, and independent held-out evaluation."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomAffine, InterpolationMode
from model import ARCH, build_model, prepare_image, tensor_image


class Crops(Dataset):
    def __init__(self, manifest, split):
        self.root = Path(manifest['root'])
        self.rows = [r for r in manifest['records'] if r['split'] == split]
        self.labels = {label: i for i, label in enumerate(manifest['labels'])}
        self.augment = RandomAffine(7, translate=(.04, .04), scale=(.95, 1.05),
                                    interpolation=InterpolationMode.BILINEAR, fill=255) if split == 'train' else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as image:
            image = prepare_image(image)
        if self.augment:
            image = self.augment(image)
        return tensor_image(image), self.labels[row['label']]


def device_for(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


@torch.no_grad()
def evaluate(model, loader, device, labels):
    model.eval()
    cm = torch.zeros((len(labels), len(labels)), dtype=torch.int64)
    loss_sum = correct5 = count = 0
    for x, y in loader:
        logits = model(x.to(device))
        loss_sum += nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item()
        prediction = logits.argmax(1).cpu()
        cm += torch.bincount(y*len(labels)+prediction, minlength=len(labels)**2).reshape(cm.shape)
        correct5 += (logits.topk(min(5, len(labels)), 1).indices.cpu() == y[:, None]).any(1).sum().item()
        count += len(y)
    tp = cm.diag().float()
    precision = tp / cm.sum(0).clamp_min(1)
    recall = tp / cm.sum(1).clamp_min(1)
    f1 = 2*precision*recall/(precision+recall).clamp_min(1e-12)
    return dict(loss=loss_sum/max(1,count), accuracy=tp.sum().item()/max(1,count),
                top5=correct5/max(1,count), macro_f1=f1.mean().item(), samples=count,
                per_class={label: dict(support=int(cm[i].sum()), recall=float(recall[i]),
                                      precision=float(precision[i]), f1=float(f1[i])) for i, label in enumerate(labels)},
                confusion_matrix=cm.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source', help='Devanagari checkpoint produced by this script')
    parser.add_argument('--evaluate', help='Evaluate a trained checkpoint on test only; no training')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs >= 1 and batch-size >= 2 are required')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    manifest = json.loads(Path(args.manifest).read_text())
    digest = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    labels = manifest['labels']
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    device = device_for(args.device)
    model = build_model(len(labels)).to(device)
    def loader(split):
        dataset = Crops(manifest, split)
        # Avoid a singleton batch with BatchNorm during scratch pretraining.
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=split == 'train',
                          num_workers=args.workers, drop_last=split == 'train' and len(dataset) % args.batch_size == 1)
    if args.evaluate:
        ckpt = torch.load(args.evaluate, map_location='cpu', weights_only=True)
        if ckpt['labels'] != labels or ckpt['arch'] != ARCH or ckpt['manifest_sha256'] != digest:
            raise ValueError('Checkpoint labels, architecture or split manifest do not match')
        model.load_state_dict(ckpt['state_dict'])
        report = evaluate(model, loader('test'), device, labels)
        (output/'test_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print({k:v for k,v in report.items() if k not in ('per_class','confusion_matrix')})
        return
    if manifest['kind'] == 'modi' and not args.source:
        raise ValueError('Modi fine-tuning requires --source: first train the Devanagari model.')
    source_meta = None
    if args.source:
        source = torch.load(args.source, map_location='cpu', weights_only=True)
        if source['arch'] != ARCH or source['kind'] != 'devanagari':
            raise ValueError('Expected this implementation\'s Devanagari checkpoint')
        weights = {k:v for k,v in source['state_dict'].items() if not k.startswith('fc.')}
        incompatible = model.load_state_dict(weights, strict=False)
        if set(incompatible.missing_keys) != {'fc.weight','fc.bias'} or incompatible.unexpected_keys:
            raise ValueError('Incomplete backbone transfer')
        source_meta = dict(path=str(Path(args.source).resolve()), epoch=source['epoch'],
                           manifest_sha256=source['manifest_sha256'])
    backbone = [p for name,p in model.named_parameters() if not name.startswith('fc.')]
    optimizer = torch.optim.AdamW([{'params':backbone, 'lr':1e-4 if args.source else 1e-3},
                                  {'params':model.fc.parameters(), 'lr':1e-3}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    train_loader, val_loader = loader('train'), loader('val')
    best = -1.; stale = 0; history = []
    for epoch in range(args.epochs):
        model.train()
        frozen = bool(args.source) and epoch < args.warmup
        for p in backbone:
            p.requires_grad_(not frozen)
        # Freeze source running statistics on the tiny Modi dataset throughout.
        if args.source:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d): module.eval()
        train_loss = seen = 0
        for x,y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device), label_smoothing=.05)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            train_loss += loss.item()*len(y); seen += len(y)
        report = evaluate(model, val_loader, device, labels)
        row = dict(epoch=epoch+1, head_only=frozen, train_loss=train_loss/max(1,seen),
                   val_loss=report['loss'], val_accuracy=report['accuracy'], val_macro_f1=report['macro_f1'])
        print(json.dumps(row), flush=True); history.append(row)
        (output/'history.json').write_text(json.dumps(history, indent=2))
        if report['macro_f1'] > best:
            best = report['macro_f1']; stale = 0
            torch.save(dict(arch=ARCH, kind=manifest['kind'], labels=labels, epoch=epoch+1,
                            manifest_sha256=digest, source=source_meta,
                            preprocessing='gray64-white-background-pad58-normalize-half',
                            state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}), output/'best.pt')
            (output/'validation_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience and epoch >= args.warmup:
            break
    print(f'Best validation macro-F1: {best:.4f}. Test split has not been evaluated.')


if __name__ == '__main__':
    main()
