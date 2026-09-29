# Use `just <recipe>` to execute a task
@_default:
    -just --list --unsorted

# Format and lint the project
[group('ci')]
check:
    @uv run ruff format
    @uv run ruff check --fix

# Run regression tests
[group('ci')]
test:
    uv run pytest -q

# Label pin-field boxes for the field extractor
[group('labeling')]
label-sheets *images:
    uv run scripts/label_sheets.py {{ images }}

# Label the nine pins for the pin classifier
[group('labeling')]
label-crops *images:
    uv run scripts/label_crops.py {{ images }}

# Train the field extractor on a Modal L4 and export ONNX
[group('train')]
train-sheets *args:
    uv run --group train modal run scripts/train_sheets.py {{ args }}

# Train the pin classifier and export ONNX
[group('train')]
train-crops *args:
    uv run --group train scripts/train_crops.py {{ args }}

# Run both models; add -v to save pipeline images
[group('inference')]
scan *args:
    @uv run scripts/scan.py {{ args }}
