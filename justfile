_default:
    @just --list --unsorted

# Lint and format with ruff
lint:
    uv run --extra dev ruff check . --fix && uv run --extra dev ruff format .

# K-fold cross-validate then retrain the CNN classifier
train-classifier *args:
    uv run --extra train pinsheet-scanner train {{ args }}

# Train the YOLO detector for pin diagram bounding boxes
train-detector *args:
    uv run pinsheet-scanner train-detector {{ args }}

# Hyperparameter tuning with Optuna
tune *args:
    uv run --extra train pinsheet-scanner tune {{ args }}

# Scan one or more score sheets and print results
scan *args:
    uv run pinsheet-scanner scan {{ args }}

# Draw every detected diagram box on a sheet image
boxes image output="example.png":
    uv run python scripts/show_detections.py {{ image }} --output {{ output }}

# Harvest high-confidence crops from sheet(s) into the training set
collect *args="sheets/*":
    uv run pinsheet-scanner collect {{ args }}

# Extract crops from one or more score sheet images
extract *args="sheets/*":
    uv run pinsheet-scanner extract {{ args }}

# Open the labeling UI to annotate ground-truth pin states
label *args:
    uv run pinsheet-scanner label {{ args }}

# Compare ground-truth labels against CNN predictions
accuracy *args:
    uv run pinsheet-scanner accuracy {{ args }}
