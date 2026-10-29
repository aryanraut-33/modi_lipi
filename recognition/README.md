# Devanagari → Modi character recognition

This implements handwritten offline character recognition (HCR/HOCR). If “hOCR”
means the HTML OCR output format, that format is not a trainable recognition model.
The target is a **single segmented Modi akshara**, including its vowel modifier;
the prediction is its Devanagari folder label, such as `ख`, `खा`, or `खि`.
It is not whole-page OCR or language translation.

## What is actually pretrained?

No Hindi/Devanagari-trained checkpoint was supplied or found in this workspace.
The code builds one by supervised training on Devanagari first, then transfers
its visual backbone into a new Modi classifier. No pretrained-model accuracy
or completed training is claimed. An arbitrary third-party CNN checkpoint cannot
be loaded into this architecture; it would need its matching model definition.

The proposed architecture is **ResNet-18 adapted to 64×64 grayscale**: one input
channel, a 3×3 stride-1 first convolution, no initial max pool, global average
pooling, and a 512-feature classification layer. These small-image changes
preserve stroke detail. It starts with random weights for Devanagari pretraining.
Torchvision's default pretrained weights are ImageNet weights, not Hindi OCR
weights, and this implementation does not mislabel them as Hindi pretraining.

This is a practical transfer baseline, not a reproduction of the original DHCD
paper or a claim that ResNet is optimal for Modi. A smaller CNN is a worthwhile
comparison when only a few independent samples exist per target class.

## Dataset findings and split policy

`Modi_Augmented_Dataset` contains 16,560 PNGs across 276 leaf folders, corresponding
to 2,760 original crops, each with five generated augmentations. Every leaf
folder contains ten originals. Sampled images are 64×64 grayscale.

Labels come from the **leaf** folder, never the top-level consonant folder.
The two `श - 1` and `श - 2` branches map to the same 12 Devanagari labels by
default, giving **264 classes** (22 base labels × 12 forms). This assumes the
branches are alternate shapes of the same akshara. If experts intend them as
different semantic classes, change the label policy before training.

The deterministic seed-42 split produced:

| Split | Used images | Policy |
|---|---:|---|
| Train | 9,936 | Training sources plus their stored augmentations |
| Validation | 552 | Original held-out crops only |
| Test | 552 | Original held-out crops only |

5,520 augmented held-out images are excluded. Source groups are derived by
stripping `_original` or `_aug_N` from filenames within each leaf folder.
Exact decoded-pixel duplicates are united before splitting. Conflicting labels
on identical pixels cause an error instead of silent training.

If a source group overlaps the official Devanagari test set, its training-side
members (including any augmented siblings) are excluded from both training and
validation. The official test images stay in test; no training copies are added
to test. No source files are deleted. The manifest records the exclusions under
`excluded_records`, with counts in `report`, so this cleaned training split is
auditable and should be disclosed when comparing with published benchmarks.

This protects against known augmentation-family leakage, **not writer/page
leakage**. Similar crops from the same manuscript may have different filenames.
Before reporting generalization, obtain writer/page IDs and an independent real
manuscript test set. Many classes have only two original test examples; per-class
scores will consequently be noisy. The closed-set model cannot recognize absent
characters or unknown conjuncts reliably.

## Run commands

Run from the repository root in an environment with the dependencies in
`recognition/requirements.txt`. CPU, CUDA and Apple MPS are supported. No GUI or
web service needs to be running.

### 1. Obtain and prepare the Devanagari source data

Download the image archive from the [UCI DHCD dataset page](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset)
and extract it locally. Select the directory containing `Train/<class>/*.png`
and `Test/<class>/*.png`, not a numeric CSV export. DHCD contains 92,000 images
in 46 classes. It is a Devanagari script dataset, originating in Nepali
handwriting research, and supports shared Devanagari character features rather
than Hindi language understanding. It does not cover the full Modi barakhadi.

```bash
python3 recognition/prepare.py \
  --root /absolute/path/to/DevanagariHandwrittenCharacterDataset \
  --kind devanagari --out recognition/manifests/devanagari.json

python3 recognition/train.py \
  --manifest recognition/manifests/devanagari.json \
  --out recognition/runs/devanagari --epochs 30 --batch-size 32
```

The supplied official `Test` folder is held out; 20% of remaining source groups
are used for validation. The best validation macro-F1 checkpoint is saved as
`recognition/runs/devanagari/best.pt`. For a source tree without an official Test
folder, the script uses its documented 60/20/20 grouped split instead.

### 2. Prepare the Modi data

```bash
python3 recognition/prepare.py \
  --root Modi_Augmented_Dataset --kind modi \
  --out recognition/manifests/modi.json
```

This manifest is already generated for the current local dataset. Regenerate it
if images change or the project moves (its dataset root is absolute). Keep the
manifest fixed while comparing experiments; it records source groups, labels,
split membership, pixel hashes, and seed.

### 3. Fine-tune the source recognizer on Modi

```bash
python3 recognition/train.py \
  --manifest recognition/manifests/modi.json \
  --source recognition/runs/devanagari/best.pt \
  --out recognition/runs/modi --epochs 40 --warmup 5 --batch-size 32
```

The transfer works as follows:

1. Load every compatible convolutional/residual feature weight from the
   Devanagari checkpoint, while discarding its 46-class output layer.
2. Initialize a new 264-class output layer. Output neurons are assigned using
   the stored sorted Unicode label list; no integer class mapping is guessed.
3. For five epochs, freeze the visual backbone and train only the new head.
   This lets the Modi labels adapt to the existing stroke features.
4. Unfreeze the backbone and fine-tune all weights. AdamW starts the backbone
   at `1e-4` and the head at `1e-3`, with cosine learning-rate decay. BatchNorm
   running statistics stay frozen throughout Modi transfer to reduce noisy
   updates from the small target dataset; its affine parameters can fine-tune.
5. Use cross-entropy with 0.05 label smoothing, weight decay, mild online affine
   augmentation, gradient clipping and early stopping on validation macro-F1.
   No flips are used, because flips change character identity.

Both domains use identical polarity normalization, aspect-preserving padding,
and grayscale normalization. The border median detects dark backgrounds and
inverts them. Full crops are preserved, including upper and lower modifiers.
Inspect unusual scans for border-based polarity errors before training.

The pretrained features are a starting point, not guaranteed positive transfer:
Modi shapes differ from Devanagari. Compare against a scratch-trained baseline
under the same splits in any research report. Nothing in this code promises a
particular target accuracy.

### 4. Evaluate once after model selection

```bash
python3 recognition/train.py \
  --manifest recognition/manifests/modi.json \
  --evaluate recognition/runs/modi/best.pt --out recognition/runs/modi
```

Outputs include top-1/top-5 accuracy, macro-F1, per-class precision/recall/F1 and
a confusion matrix (rows=true, columns=predicted) in `test_metrics.json`.
Training only looks at validation; test is an explicit separate command.
The checkpoint stores its manifest hash and requires the same split on evaluation.
`history.json` contains learning curves; `validation_metrics.json` describes
the selected checkpoint. Training is not resumable in this initial version;
checkpoints are intended for transfer and inference.

### 5. Predict a segmented glyph

```bash
python3 recognition/predict.py \
  --checkpoint recognition/runs/modi/best.pt \
  --image /absolute/path/to/a/modi-glyph.png
```

Import `Recognizer` from `recognition/predict.py` with the recognition directory
on the Python import path, construct it once, then call:

```python
from PIL import Image
predictions = recognizer.predict(Image.fromarray(glyph.full_glyph_img))
```

The existing `SegmentedGlyph` output therefore connects directly to this Phase 4
recognizer. Concatenate top predictions in line order for a first transcription;
retain top-k alternatives for review. Softmax scores are uncalibrated and should
not be presented as proven confidence. Word spacing, language correction and
whole-document transliteration require additional work; segmentation mistakes
will propagate into recognition. This change does not wire inference into the web UI.

## Validation performed

```bash
python3 recognition/smoke_test.py
```

This exercises training, replacement of a 2-class fixture head with a 3-class
head, frozen and unfrozen epochs, checkpoint serialization, evaluation and
inference. Fixtures use a few Modi crops with synthetic source-stage metadata
strictly to test code. They are NOT genuine Hindi pretraining and are deleted
after the test. Full Devanagari and Modi training requires running the commands
above; no trained production checkpoint is bundled.

## Research and implementation references

- Acharya, Pant and Gyawali, [Deep Learning Based Large Scale Handwritten
  Devanagari Character Recognition](https://ashokpant.github.io/publications/ashok_2015_deep.pdf):
  the source dataset and a character-CNN baseline using dropout and augmentation.
- [UCI DHCD](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset):
  dataset source and licensing/citation information.
- [Torchvision ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html):
  implementation used here; stock pretrained weights are ImageNet.

Research supports using character classifiers for this setting; the choice of
this adapted ResNet and its specific transfer schedule is an engineering baseline,
not a published Modi result.
