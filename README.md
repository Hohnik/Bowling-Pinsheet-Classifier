# Pinsheet Scanner

Scan German nine-pin score sheets with two ONNX models:

1. YOLO26n detects and orders the 15 × 8 pin fields.
2. A compact multilabel CNN classifies the nine pins in each crop.

OpenCV DNN runs both models. PyTorch and Ultralytics are training-only dependencies.

## Use

```bash
just scan data/sheets/007.png
just scan -v data/sheets/007.png
```

The normal command prints only the 120 throw scores. `-v` also saves each pipeline
stage under `runs/pipeline/<image>/`.

## Label

Each model has one labeling tool:

```bash
just label-sheets data/sheets/*.png
just label-crops data/sheets/*.png
```

`label-sheets` edits field boxes in `data/annotations/field_extractor/`.
`label-crops` edits pin states in
`data/annotations/pin_classifier/labels.csv`. Both tools save when Space is
pressed and skip saved work when restarted. Add `--review` to revisit existing
annotations.

### Why the annotation formats differ

The extractor is an object detector. YOLO requires one matching `.txt` file for
each source image, with a variable number of normalized boxes. The classifier
has one fixed target per crop: a nine-character pin vector. A single CSV avoids
creating 1,664 tiny label files. Each model still has its own annotation directory.

## Train

Each model has one training command:

```bash
just train-sheets  # Modal L4; writes models/field_extractor.onnx
just train-crops   # local MPS/CUDA/CPU; writes models/pin_classifier.onnx
```

The sheet trainer prepares its YOLO dataset internally. For local sheet training,
run `uv run --group train scripts/train_sheets.py` directly.

Both trainers share the same camera and print degradation pipeline: motion and
Gaussian blur, defocus, sensor noise, JPEG compression, contrast and gamma
changes, shadows, and downscaling. The classifier also uses crop-safe rotation,
translation, scale, shear, and perspective transforms. Mosaic stays
extractor-only because combining crops would invalidate their nine-pin labels.

## Library

```python
from pin import Scanner

scanner = Scanner()
result = scanner.scan_sheet("data/sheets/001.jpeg")
print([throw.score for throw in result.throws])
```

Reuse one `Scanner` so both ONNX models are loaded only once.

## Pipeline

1. Letterbox the sheet photograph to 1280 × 1280.
2. Extract pin fields with `models/field_extractor.onnx`.
3. Sort and crop the fields in sheet order.
4. Classify each crop with `models/pin_classifier.onnx`.
5. Convert cumulative clearing diagrams into throw scores.

## Structure

```text
data/sheets/                            source sheet photographs
data/crops/                             generated field crops
data/annotations/field_extractor/       YOLO boxes, one text file per sheet
data/annotations/pin_classifier/labels.csv  nine-bit labels, one row per crop
models/                                 the two runtime ONNX models
scripts/label_sheets.py  sheet box labeler
scripts/label_crops.py   crop pin labeler
scripts/train_sheets.py  sheet detector training
scripts/train_crops.py   crop classifier training
scripts/scan.py          complete two-model pipeline
src/pin/                 OpenCV runtime library
tests/                   regression tests
```

## Development

```bash
just check
just test
```
