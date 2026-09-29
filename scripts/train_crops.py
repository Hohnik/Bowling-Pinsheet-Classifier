"""Train and export the nine-pin crop classifier."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from pin.augmentation import crop_augmenter

VALIDATION_SOURCES = {
    "data/sheets/003.png",
    "data/sheets/007.png",
    "data/sheets/011.png",
}
IMAGE_SIZE = 128


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No crop labels in {path}")
    return rows


_CROP_AUGMENTER = crop_augmenter()


def augment(image: np.ndarray) -> np.ndarray:
    return _CROP_AUGMENTER(image=image)["image"]


def make_model() -> object:
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(1, 16, 5, stride=2, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 96, 3, stride=2, padding=1),
        nn.BatchNorm2d(96),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(96 * 4 * 4, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 9),
    )


def train(  # noqa: C901
    labels: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_name: str | None,
) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset

    random.seed(117)
    np.random.seed(117)
    torch.manual_seed(117)
    _CROP_AUGMENTER.set_random_seed(117)
    rows = load_rows(labels)
    train_rows = [row for row in rows if row["source"] not in VALIDATION_SOURCES]
    validation_rows = [row for row in rows if row["source"] in VALIDATION_SOURCES]

    class CropDataset(Dataset):
        def __init__(self, items: list[dict[str, str]], training: bool) -> None:
            self.items = items
            self.training = training

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            row = self.items[index]
            image = cv2.imread(row["crop"])
            if image is None:
                raise FileNotFoundError(f"Could not load crop: {row['crop']}")
            image = cv2.resize(
                image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
            )
            if self.training:
                image = augment(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            inputs = torch.from_numpy(gray).unsqueeze(0).float().div(255.0)
            targets = torch.tensor(
                [int(value) for value in row["pins"]], dtype=torch.float32
            )
            return inputs, targets

    if device_name is None:
        device_name = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    device = torch.device(device_name)
    train_loader = DataLoader(
        CropDataset(train_rows, True), batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        CropDataset(validation_rows, False), batch_size=batch_size
    )
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    best_exact = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(inputs)

        model.eval()
        correct_bits = 0
        correct_crops = 0
        bit_count = 0
        crop_count = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                predictions = model(inputs.to(device)).sigmoid().cpu() >= 0.6
                expected = targets >= 0.5
                correct_bits += int((predictions == expected).sum())
                correct_crops += int((predictions == expected).all(dim=1).sum())
                bit_count += expected.numel()
                crop_count += len(inputs)
        bit_accuracy = correct_bits / bit_count
        exact_accuracy = correct_crops / crop_count
        print(
            f"epoch={epoch:03d} loss={total_loss / len(train_rows):.4f} "
            f"bit_accuracy={bit_accuracy:.4f} exact_accuracy={exact_accuracy:.4f}"
        )
        if exact_accuracy >= best_exact:
            best_exact = exact_accuracy
            best_state = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model")
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE),
        output,
        input_names=["images"],
        output_names=["pins"],
        opset_version=17,
        dynamo=False,
    )
    print(f"Best validation exact accuracy: {best_exact:.4f}")
    print(f"Exported crop classifier to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/annotations/pin_classifier/labels.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/pin_classifier.onnx")
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", help="Training device: mps, cuda, or cpu")
    args = parser.parse_args()
    train(
        args.labels,
        args.output,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.device,
    )


if __name__ == "__main__":
    main()
