#!/usr/bin/env python3
"""Train a compact imitation-learning policy for Duckiematrix Duckiebots.

The model predicts two continuous motor commands from a preprocessed camera
image: left_action and right_action.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only affects terminal nicety
    tqdm = None


DEFAULT_RUN_DIR = Path(
    "~/duckietown-data/imitation_learning/expert_data/run_001_20260707_094333"
).expanduser()
TARGET_COLUMNS = ("left_action", "right_action")
IMAGE_COLUMN = "image"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    run_dir: str
    image_dir: str
    split_mode: str
    train_runs: list[str]
    val_runs: list[str]
    output_dir: str
    model: str
    pretrained: bool
    train_backbone: bool
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    val_fraction: float
    num_workers: int
    seed: int
    device: str
    limit: int | None
    skip_missing: bool


class DuckiebotActionDataset(Dataset):
    def __init__(
        self,
        run_dirs: list[Path],
        image_dir_arg: Path | None,
        transform: Callable[[Image.Image], torch.Tensor],
        limit: int | None = None,
        skip_missing: bool = False,
        split_name: str = "dataset",
    ) -> None:
        self.run_dirs = run_dirs
        self.image_dir_arg = image_dir_arg
        self.transform = transform
        self.split_name = split_name
        self.samples = self._load_samples(limit=limit, skip_missing=skip_missing)

        if not self.samples:
            raise ValueError(f"No samples found for {split_name}")

    def _load_samples(
        self,
        limit: int | None,
        skip_missing: bool,
    ) -> list[tuple[Path, torch.Tensor]]:
        samples: list[tuple[Path, torch.Tensor]] = []
        missing: list[Path] = []

        for run_dir in self.run_dirs:
            actions_file = run_dir / "actions.csv"

            if not actions_file.exists():
                raise FileNotFoundError(f"actions.csv not found: {actions_file}")

            image_dir = resolve_image_dir(run_dir, self.image_dir_arg)

            with actions_file.open(newline="") as file:
                reader = csv.DictReader(file)
                self._validate_columns(reader.fieldnames)

                for row in reader:
                    image_name = Path(row[IMAGE_COLUMN]).name
                    image_path = image_dir / image_name

                    if not image_path.exists():
                        missing.append(image_path)
                        if skip_missing:
                            continue

                    target = torch.tensor(
                        [float(row[column]) for column in TARGET_COLUMNS],
                        dtype=torch.float32,
                    )
                    samples.append((image_path, target))

                    if limit is not None and len(samples) >= limit:
                        break

            if limit is not None and len(samples) >= limit:
                break

        if missing and not skip_missing:
            example = missing[0]
            raise FileNotFoundError(
                f"{len(missing)} image files referenced by actions.csv are missing. "
                f"First missing file: {example}"
            )

        return samples

    @staticmethod
    def _validate_columns(fieldnames: list[str] | None) -> None:
        if fieldnames is None:
            raise ValueError("actions.csv has no header")

        missing_columns = [
            column
            for column in (IMAGE_COLUMN, *TARGET_COLUMNS)
            if column not in fieldnames
        ]

        if missing_columns:
            raise ValueError(
                "actions.csv is missing required columns: "
                + ", ".join(missing_columns)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, target = self.samples[index]

        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        return image_tensor, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch imitation-learning policy for Duckiematrix."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=(
            "Single run directory with actions.csv, or split root containing "
            "train/ and val/ directories with run subdirectories."
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory with processed images. Defaults to images_processed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/imitation_learning"),
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Optional checkpoint folder prefix. Output folder becomes "
            "<experiment-name>_<timestamp>."
        ),
    )
    parser.add_argument(
        "--model",
        choices=("mobilenet_v3_small", "resnet18"),
        default="mobilenet_v3_small",
    )
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="Fine-tune the full CNN instead of only training the regression head.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for quick smoke tests.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip rows whose preprocessed image file is missing.",
    )
    parser.set_defaults(pretrained=True)
    return parser.parse_args()


def resolve_image_dir(run_dir: Path, image_dir_arg: Path | None) -> Path:
    if image_dir_arg is not None:
        image_dir = image_dir_arg.expanduser()
        if not image_dir.is_absolute():
            image_dir = run_dir / image_dir

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        return image_dir

    image_dir = run_dir / "images_processed"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return image_dir


def find_run_dirs(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    run_dirs = sorted(
        path
        for path in split_dir.iterdir()
        if path.is_dir() and (path / "actions.csv").exists()
    )

    if not run_dirs:
        raise FileNotFoundError(f"No run directories with actions.csv found in {split_dir}")

    return run_dirs


def resolve_dataset_splits(run_dir: Path) -> tuple[str, list[Path], list[Path]]:
    train_dir = run_dir / "train"
    val_dir = run_dir / "val"

    if train_dir.is_dir() or val_dir.is_dir():
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise FileNotFoundError(
                f"Split root must contain both train/ and val/: {run_dir}"
            )

        return "run_wise", find_run_dirs(train_dir), find_run_dirs(val_dir)

    if (run_dir / "actions.csv").exists():
        return "random_frame", [run_dir], []

    raise FileNotFoundError(
        f"Expected either actions.csv or train/ and val/ below: {run_dir}"
    )


def experiment_folder_name(experiment_name: str | None, model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if experiment_name is None:
        return f"{timestamp}_{model_name}"

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_name):
        raise ValueError(
            "--experiment-name must start with an alphanumeric character and "
            "contain only letters, digits, underscores, dots, or hyphens"
        )

    return f"{experiment_name}_{timestamp}"


def make_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_mobilenet_v3_small(pretrained: bool) -> nn.Module:
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    except AttributeError:
        model = models.mobilenet_v3_small(pretrained=pretrained)
    except TypeError:
        model = models.mobilenet_v3_small(pretrained=pretrained)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(TARGET_COLUMNS))
    return model


def build_resnet18(pretrained: bool) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=pretrained)
    except TypeError:
        model = models.resnet18(pretrained=pretrained)

    model.fc = nn.Linear(model.fc.in_features, len(TARGET_COLUMNS))
    return model


def build_model(model_name: str, pretrained: bool, train_backbone: bool) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        model = build_mobilenet_v3_small(pretrained=pretrained)
        head_modules = [model.classifier]
    elif model_name == "resnet18":
        model = build_resnet18(pretrained=pretrained)
        head_modules = [model.fc]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not train_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

        for module in head_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    return model


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")

    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(
    dataset: Dataset,
    val_fraction: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

    val_size = max(1, int(round(len(dataset) * val_fraction)))
    train_size = len(dataset) - val_size

    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def iterate(loader: DataLoader, description: str):
    if tqdm is None:
        return loader

    return tqdm(loader, desc=description, leave=False, position=1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for images, targets in iterate(loader, f"train {epoch}"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.mean(torch.abs(predictions.detach() - targets)).item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_mae / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for images, targets in iterate(loader, f"val {epoch}"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        predictions = model(images)
        loss = criterion(predictions, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.mean(torch.abs(predictions - targets)).item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_mae / total_samples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: TrainConfig,
    history: list[dict[str, float | int]],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "target_columns": TARGET_COLUMNS,
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
            "config": asdict(config),
            "history": history,
        },
        path,
    )


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "epoch",
                "train_loss",
                "train_mae",
                "val_loss",
                "val_mae",
            ),
        )
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    split_mode, train_run_dirs, val_run_dirs = resolve_dataset_splits(run_dir)
    image_dir_description = str(args.image_dir) if args.image_dir is not None else "images_processed"
    output_root = args.output_dir.expanduser()
    experiment_dir = output_root / experiment_folder_name(
        experiment_name=args.experiment_name,
        model_name=args.model,
    )
    experiment_dir.mkdir(parents=True, exist_ok=False)

    set_seed(args.seed)
    device = resolve_device(args.device)

    config = TrainConfig(
        run_dir=str(run_dir),
        image_dir=image_dir_description,
        split_mode=split_mode,
        train_runs=[str(path) for path in train_run_dirs],
        val_runs=[str(path) for path in val_run_dirs],
        output_dir=str(experiment_dir),
        model=args.model,
        pretrained=args.pretrained,
        train_backbone=args.train_backbone,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        device=str(device),
        limit=args.limit,
        skip_missing=args.skip_missing,
    )

    with (experiment_dir / "config.json").open("w") as file:
        json.dump(asdict(config), file, indent=2)
        file.write("\n")

    transform = make_transforms(args.image_size)

    if split_mode == "run_wise":
        train_dataset = DuckiebotActionDataset(
            run_dirs=train_run_dirs,
            image_dir_arg=args.image_dir,
            transform=transform,
            limit=args.limit,
            skip_missing=args.skip_missing,
            split_name="train",
        )
        val_dataset = DuckiebotActionDataset(
            run_dirs=val_run_dirs,
            image_dir_arg=args.image_dir,
            transform=transform,
            limit=args.limit,
            skip_missing=args.skip_missing,
            split_name="val",
        )
        train_loader = make_loader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            shuffle=True,
        )
        val_loader = make_loader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            shuffle=False,
        )
    else:
        dataset = DuckiebotActionDataset(
            run_dirs=train_run_dirs,
            image_dir_arg=args.image_dir,
            transform=transform,
            limit=args.limit,
            skip_missing=args.skip_missing,
            split_name="dataset",
        )
        train_loader, val_loader = make_loaders(
            dataset=dataset,
            val_fraction=args.val_fraction,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            device=device,
        )

    model = build_model(
        model_name=args.model,
        pretrained=args.pretrained,
        train_backbone=args.train_backbone,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())

    print(f"Run dir:          {run_dir}")
    print(f"Split mode:       {split_mode}")
    print(f"Image dir:        {image_dir_description}")
    print(f"Experiment dir:   {experiment_dir}")
    print(f"Train runs:       {len(train_run_dirs)}")
    print(f"Val runs:         {len(val_run_dirs) if val_run_dirs else 'random split'}")
    print(f"Samples:          {len(train_loader.dataset) + len(val_loader.dataset)}")
    print(f"Train/val:        {len(train_loader.dataset)}/{len(val_loader.dataset)}")
    print(f"Device:           {device}")
    print(f"Model:            {args.model} pretrained={args.pretrained}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    epoch_iterator = range(1, args.epochs + 1)

    if tqdm is not None:
        epoch_iterator = tqdm(
            epoch_iterator,
            desc="epochs",
            total=args.epochs,
            position=0,
        )

    for epoch in epoch_iterator:
        train_loss, train_mae = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        val_loss, val_mae = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
        }
        history.append(row)
        write_history(experiment_dir / "history.csv", history)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=experiment_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config,
                history=history,
            )

        save_checkpoint(
            path=experiment_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            config=config,
            history=history,
        )

        metrics = (
            f"train_loss={train_loss:.6f} train_mae={train_mae:.6f} "
            f"val_loss={val_loss:.6f} val_mae={val_mae:.6f} "
            f"best_val_loss={best_val_loss:.6f}"
        )

        if tqdm is not None:
            epoch_iterator.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_mae": f"{val_mae:.4f}",
                    "best": f"{best_val_loss:.4f}",
                }
            )
            tqdm.write(f"Epoch {epoch:03d}/{args.epochs:03d} {metrics}")
        else:
            print(f"Epoch {epoch:03d}/{args.epochs:03d} {metrics}")

    print(f"Done. Best checkpoint: {experiment_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
