"""Prepare data, train YOLO26n on score sheets, and export ONNX."""

from __future__ import annotations

import argparse
import io
import shutil
import tarfile
from pathlib import Path

import modal

from pin.augmentation import serialized_photo_transforms

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VALIDATION_IMAGES = {"003_png", "007_png", "011_png"}
REMOTE_ROOT = Path("/tmp/pinsheet")


def image_key(path: Path) -> str:
    return f"{path.stem}_{path.suffix[1:].lower()}"


def prepare_dataset(source: Path, annotations: Path, output: Path) -> Path:
    images = sorted(
        path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No sheet images found in {source}")
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)

    validation_count = 0
    for image in images:
        key = image_key(image)
        label = annotations / f"{key}.txt"
        if not label.exists():
            raise FileNotFoundError(f"Missing annotation: {label}")
        split = "val" if key in VALIDATION_IMAGES else "train"
        validation_count += split == "val"
        destination = output / split / "images" / f"{key}{image.suffix.lower()}"
        destination.symlink_to(image.resolve())
        shutil.copy2(label, output / split / "labels" / f"{key}.txt")

    config = output / "dataset.yaml"
    config.write_text(
        f"path: {output.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n\n"
        "names:\n"
        "  0: pin_field\n"
    )
    print(
        f"Prepared {len(images) - validation_count} training and {validation_count} validation sheets"
    )
    return config


def detector_trainer() -> type:
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import YAML

    class PinFieldTrainer(DetectionTrainer):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.args.augmentations = serialized_photo_transforms()
            YAML.save(self.save_dir / "args.yaml", vars(self.args))

    return PinFieldTrainer


def train(
    source: Path,
    annotations: Path,
    dataset: Path,
    output: Path,
    epochs: int,
    image_size: int,
    batch: int,
    device: str | None,
    name: str,
) -> Path:
    from ultralytics import YOLO

    data = prepare_dataset(source, annotations, dataset)
    model = YOLO("yolo26n.pt")
    options: dict[str, object] = {
        "data": str(data.resolve()),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch,
        "val": False,
        "project": str(Path("runs").resolve()),
        "name": name,
        "exist_ok": True,
        "workers": 4,
        "seed": 117,
        "hsv_h": 0.015,
        "hsv_s": 0.25,
        "hsv_v": 0.35,
        "degrees": 12.0,
        "translate": 0.12,
        "scale": 0.35,
        "shear": 2.0,
        "perspective": 0.001,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.75,
        "close_mosaic": 10,
        "mixup": 0.0,
        "cutmix": 0.0,
    }
    if device is not None:
        options["device"] = device
    model.train(trainer=detector_trainer(), **options)
    if model.trainer is None:
        raise RuntimeError("Training did not produce a trainer")

    best = Path(model.trainer.best)
    trained = YOLO(best)
    metrics = trained.val(data=str(data.resolve()), imgsz=image_size, device=device)
    print(f"mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")
    exported = Path(
        trained.export(
            format="onnx", imgsz=image_size, opset=17, simplify=True, dynamic=False
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, output)
    print(f"Exported sheet detector to {output}")
    return Path(model.trainer.save_dir)


modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "albumentations==2.0.8",
        "onnx==1.23.1",
        "onnxslim==0.1.97",
        "ultralytics==8.4.166",
    )
    .add_local_dir("data/sheets", "/inputs/sheets")
    .add_local_dir("data/annotations/field_extractor", "/inputs/annotations")
    .add_local_dir("src/pin", "/root/pin")
)
app = modal.App("pinsheet-field-extractor-training", image=modal_image)


@app.function(gpu="L4", cpu=8, memory=32768, timeout=2 * 60 * 60)
def train_on_modal(epochs: int, image_size: int, batch: int, name: str) -> bytes:
    shutil.rmtree(REMOTE_ROOT, ignore_errors=True)
    REMOTE_ROOT.mkdir(parents=True)
    run = train(
        Path("/inputs/sheets"),
        Path("/inputs/annotations"),
        REMOTE_ROOT / "data/sheets",
        REMOTE_ROOT / "models/field_extractor.onnx",
        epochs,
        image_size,
        batch,
        "0",
        name,
    )
    (run / "weights/last.pt").unlink(missing_ok=True)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as output:
        output.add(
            REMOTE_ROOT / "models/field_extractor.onnx",
            "models/field_extractor.onnx",
        )
        output.add(run, f"runs/{name}")
    return archive.getvalue()


def unpack(archive: bytes) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        source.extractall(Path.cwd(), filter="data")


@app.local_entrypoint()
def modal_main(
    epochs: int = 300,
    image_size: int = 1280,
    batch: int = 4,
    name: str = "field_extractor",
) -> None:
    unpack(train_on_modal.remote(epochs, image_size, batch, name))
    print(f"Downloaded models/field_extractor.onnx and runs/{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/sheets"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/field_extractor"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/sheet_training"))
    parser.add_argument(
        "--output", type=Path, default=Path("models/field_extractor.onnx")
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", help="Training device: mps, 0, or cpu")
    parser.add_argument("--name", default="field_extractor")
    args = parser.parse_args()
    train(
        args.source,
        args.annotations,
        args.dataset,
        args.output,
        args.epochs,
        args.image_size,
        args.batch,
        args.device,
        args.name,
    )


if __name__ == "__main__":
    main()
