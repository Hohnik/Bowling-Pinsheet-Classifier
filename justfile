# Use `just <recipe>` to execute a task
@_default:
    -just --list --unsorted

alias lint := check
alias format := check

# Format and lint the project
check:
    @uv run ruff format
    @uv run ruff check --fix

# Run regression tests
test:
    uv run pytest -q

# Scan one or more sheets
scan *args:
    uv run scripts/scan.py {{ args }}

# Draw all detected boxes
boxes image output="detections.png":
    uv run scripts/show_detections.py {{ image }} --output {{ output }}
