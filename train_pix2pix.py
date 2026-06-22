from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from data import Pix2PixMapsDataset
from models import PatchDiscriminator, UNetGenerator

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining_args = config_parser.parse_known_args()

    config_defaults: dict[str, Any] = {}
    if config_args.config:
        if yaml is None:
            raise ImportError("PyYAML is required when using --config. Install it with `pip install pyyaml`.")
        with open(config_args.config, "r", encoding="utf-8") as config_file:
            config_defaults = yaml.safe_load(config_file) or {}

    parser = argparse.ArgumentParser(description="Train a Pix2Pix-style UNet on the Maps dataset.")
    parser.add_argument("--config", type=str, default=config_args.config)
    parser.add_argument("--data-root", type=str, required="data_root" not in config_defaults)
    parser.add_argument("--output-dir", type=str, default="runs/pix2pix_maps")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--load-size", type=int, default=286)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source-interpolation", type=str, default="bicubic", choices=["nearest", "bilinear", "bicubic", "lanczos"])
    parser.add_argument("--target-interpolation", type=str, default="nearest", choices=["nearest", "bilinear", "bicubic", "lanczos"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--generator-lr", type=float, default=2e-4)
    parser.add_argument("--discriminator-lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--weighted-l1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--road-mask-threshold", type=float, default=0.9)
    parser.add_argument("--road-loss-weight", type=float, default=5.0)
    parser.add_argument("--lambda-edge", type=float, default=0.1)
    parser.add_argument("--generator-norm", type=str, default="instance", choices=["batch", "instance"])
    parser.add_argument("--discriminator-norm", type=str, default="instance", choices=["batch", "instance"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--d-update-interval", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--no-early-stopping", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-ddp", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(**config_defaults)
    return parser.parse_args(remaining_args)


def seed_everything(seed: int, rank: int = 0) -> None:
    adjusted_seed = seed + rank
    random.seed(adjusted_seed)
    np.random.seed(adjusted_seed)
    torch.manual_seed(adjusted_seed)
    torch.cuda.manual_seed_all(adjusted_seed)


def resolve_validation_split(data_root: Path) -> str | None:
    for split_name in ("val", "test"):
        if (data_root / split_name).exists():
            return split_name
    return None


def init_weights(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias.data)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.zeros_(module.bias.data)


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.bind(("127.0.0.1", 0))
        connection.listen(1)
        return int(connection.getsockname()[1])


def setup_distributed(rank: int, world_size: int, port: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    return image_tensor.mul(0.5).add(0.5).clamp(0.0, 1.0)


def weighted_reconstruction_loss(
    generated: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    road_weight: float,
) -> torch.Tensor:
    target_brightness = denormalize_image(target).mean(dim=1, keepdim=True)
    road_mask = (target_brightness >= threshold).to(target.dtype)
    pixel_weights = 1.0 + (road_weight - 1.0) * road_mask
    return (generated - target).abs().mul(pixel_weights).mean()


class SobelEdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.l1_loss = nn.L1Loss()

    def _to_grayscale(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            return image
        return 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gray_gen = self._to_grayscale(generated)
        gray_target = self._to_grayscale(target)

        gray_gen = F.pad(gray_gen, (1, 1, 1, 1), mode="replicate")
        gray_target = F.pad(gray_target, (1, 1, 1, 1), mode="replicate")

        edge_gen_x = F.conv2d(gray_gen, self.sobel_x)
        edge_gen_y = F.conv2d(gray_gen, self.sobel_y)
        edge_gen = torch.sqrt(edge_gen_x.pow(2) + edge_gen_y.pow(2) + 1e-6)

        edge_target_x = F.conv2d(gray_target, self.sobel_x)
        edge_target_y = F.conv2d(gray_target, self.sobel_y)
        edge_target = torch.sqrt(edge_target_x.pow(2) + edge_target_y.pow(2) + 1e-6)

        return self.l1_loss(edge_gen, edge_target)


def autocast_context(device: torch.device, enabled: bool):
    return autocast(device_type=device.type, enabled=enabled)


def save_preview(
    output_path: Path,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    generated_image: torch.Tensor,
) -> None:
    source_array = denormalize_image(source_image).permute(1, 2, 0).cpu().numpy()
    target_array = denormalize_image(target_image).permute(1, 2, 0).cpu().numpy()
    generated_array = denormalize_image(generated_image).permute(1, 2, 0).cpu().numpy()

    preview_array = np.concatenate([source_array, generated_array, target_array], axis=1)
    preview_image = Image.fromarray((preview_array * 255.0).astype(np.uint8))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_image.save(output_path)


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: Adam,
    discriminator_optimizer: Adam,
    generator_scheduler: LambdaLR,
    discriminator_scheduler: LambdaLR,
    scaler: GradScaler,
    best_val_l1: float,
    epochs_without_improvement: int,
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": unwrap_model(generator).state_dict(),
            "discriminator": unwrap_model(discriminator).state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "generator_scheduler": generator_scheduler.state_dict(),
            "discriminator_scheduler": discriminator_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_l1": best_val_l1,
            "epochs_without_improvement": epochs_without_improvement,
            "args": vars(args),
        },
        checkpoint_path,
    )


def maybe_load_checkpoint(
    resume_path: str | None,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: Adam,
    discriminator_optimizer: Adam,
    generator_scheduler: LambdaLR,
    discriminator_scheduler: LambdaLR,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float, int]:
    if not resume_path:
        return 0, math.inf, 0

    checkpoint = torch.load(resume_path, map_location=device)
    unwrap_model(generator).load_state_dict(checkpoint["generator"])
    unwrap_model(discriminator).load_state_dict(checkpoint["discriminator"])
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
    generator_scheduler.load_state_dict(checkpoint["generator_scheduler"])
    discriminator_scheduler.load_state_dict(checkpoint["discriminator_scheduler"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_l1 = float(checkpoint.get("best_val_l1", math.inf))
    epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
    return start_epoch, best_val_l1, epochs_without_improvement


def create_scheduler(optimizer: Adam, total_epochs: int, warmup_epochs: int | None = None) -> LambdaLR:
    def decay_lambda(epoch: int) -> float:
        warm = warmup_epochs if warmup_epochs is not None else total_epochs // 2
        if epoch < warm:
            return 1.0
        decay_progress = (epoch - warm) / max(1, total_epochs - warm)
        return max(0.0, 1.0 - decay_progress)

    return LambdaLR(optimizer, lr_lambda=decay_lambda)


def calculate_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((prediction - target) ** 2).item()
    if mse <= 0.0:
        return 99.0
    return 20.0 * math.log10(2.0) - 10.0 * math.log10(mse)


def build_dataloaders(
    args: argparse.Namespace,
    world_size: int,
    rank: int,
) -> tuple[DataLoader, DataLoader | None, DistributedSampler | None]:
    data_root = Path(args.data_root)
    train_dataset = Pix2PixMapsDataset(
        root_dir=data_root,
        split="train",
        image_size=args.image_size,
        load_size=args.load_size,
        augment=True,
        source_side=args.source_side,
        source_interpolation=args.source_interpolation,
        target_interpolation=args.target_interpolation,
    )
    validation_split = resolve_validation_split(data_root)
    val_dataset = None
    if validation_split is not None and (world_size == 1 or is_main_process(rank)):
        val_dataset = Pix2PixMapsDataset(
            root_dir=data_root,
            split=validation_split,
            image_size=args.image_size,
            load_size=args.load_size,
            augment=False,
            source_side=args.source_side,
            source_interpolation=args.source_interpolation,
            target_interpolation=args.target_interpolation,
        )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, args.batch_size),
            shuffle=False,
            sampler=None,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
            drop_last=False,
        )

    return train_loader, val_loader, train_sampler


def validate(
    generator: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
    epoch: int,
    output_dir: Path,
) -> dict[str, float]:
    generator.eval()
    l1_losses: list[float] = []
    edge_losses: list[float] = []
    psnr_values: list[float] = []
    saved_preview = False
    l1_loss = nn.L1Loss().to(device)
    edge_loss = SobelEdgeLoss().to(device)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
            source_images = batch["source"].to(device, non_blocking=True)
            target_images = batch["target"].to(device, non_blocking=True)

            with autocast_context(device, amp_enabled):
                generated_images = generator(source_images)
                batch_l1 = l1_loss(generated_images, target_images)
                batch_edge = edge_loss(generated_images, target_images)

            l1_losses.append(float(batch_l1.item()))
            edge_losses.append(float(batch_edge.item()))
            psnr_values.append(calculate_psnr(generated_images.float(), target_images.float()))

            if not saved_preview and is_main_process(rank):
                save_preview(
                    output_dir / "samples" / f"epoch_{epoch:04d}.png",
                    source_images[0],
                    target_images[0],
                    generated_images[0],
                )
                saved_preview = True

    generator.train()
    return {
        "val_l1": float(np.mean(l1_losses)) if l1_losses else math.inf,
        "val_edge": float(np.mean(edge_losses)) if edge_losses else math.inf,
        "val_psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
    }


def train_worker(rank: int, world_size: int, port: int, args: argparse.Namespace) -> None:
    distributed = world_size > 1
    if distributed:
        setup_distributed(rank, world_size, port)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    seed_everything(args.seed, rank)

    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "train_args.json", "w", encoding="utf-8") as args_file:
            json.dump(vars(args), args_file, indent=2)

    train_loader, val_loader, train_sampler = build_dataloaders(args, world_size, rank)

    generator = UNetGenerator(norm_type=args.generator_norm).to(device)
    discriminator = PatchDiscriminator(norm_type=args.discriminator_norm).to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    if distributed:
        generator = DDP(generator, device_ids=[rank])
        discriminator = DDP(discriminator, device_ids=[rank])
    elif torch.cuda.device_count() > 1 and device.type == "cuda":
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    generator_optimizer = Adam(generator.parameters(), lr=args.generator_lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=args.discriminator_lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    generator_scheduler = create_scheduler(generator_optimizer, args.epochs, args.warmup_epochs)
    discriminator_scheduler = create_scheduler(discriminator_optimizer, args.epochs, args.warmup_epochs)

    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    reconstruction_loss = nn.L1Loss().to(device)
    edge_loss = SobelEdgeLoss().to(device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    start_epoch, best_val_l1, epochs_without_improvement = maybe_load_checkpoint(
        args.resume,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        generator_scheduler,
        discriminator_scheduler,
        scaler,
        device,
    )

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if SummaryWriter is not None and is_main_process(rank) else None

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        generator.train()
        discriminator.train()

        if is_main_process(rank):
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        else:
            progress_bar = train_loader

        running_metrics = {
            "generator_loss": 0.0,
            "discriminator_loss": 0.0,
            "gan_loss": 0.0,
            "l1_loss": 0.0,
            "edge_loss": 0.0,
        }
        step_count = 0

        generator_optimizer.zero_grad(set_to_none=True)
        discriminator_optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress_bar, start=1):
            source_images = batch["source"].to(device, non_blocking=True)
            target_images = batch["target"].to(device, non_blocking=True)

            if not torch.isfinite(source_images).all() or not torch.isfinite(target_images).all():
                continue

            update_d = step % args.d_update_interval == 0

            with autocast_context(device, amp_enabled):
                generated_images = generator(source_images)
                if update_d:
                    discriminator_real = discriminator(source_images, target_images)
                    discriminator_fake = discriminator(source_images, generated_images.detach())
                    fake_labels = torch.zeros_like(discriminator_fake)
                    valid_labels = torch.ones_like(discriminator_real)
                    discriminator_loss = 0.5 * (
                        adversarial_loss(discriminator_real, valid_labels) +
                        adversarial_loss(discriminator_fake, fake_labels)
                    )
                    discriminator_loss = discriminator_loss / args.accumulation_steps

            if update_d:
                if not torch.isfinite(discriminator_loss):
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    generator_optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(discriminator_loss).backward()

            with autocast_context(device, amp_enabled):
                generated_images = generator(source_images)
                discriminator_fake_for_generator = discriminator(source_images, generated_images)
                valid_labels = torch.ones_like(discriminator_fake_for_generator)
                gan_loss = adversarial_loss(discriminator_fake_for_generator, valid_labels)
                reconstruction_term = reconstruction_loss(generated_images, target_images)
                if args.weighted_l1:
                    reconstruction_term = weighted_reconstruction_loss(
                        generated_images,
                        target_images,
                        threshold=args.road_mask_threshold,
                        road_weight=args.road_loss_weight,
                    )
                l1_loss = reconstruction_term * args.lambda_l1
                edge = edge_loss(generated_images, target_images) * args.lambda_edge
                generator_loss = (gan_loss + l1_loss + edge) / args.accumulation_steps

            if not torch.isfinite(generator_loss):
                discriminator_optimizer.zero_grad(set_to_none=True)
                generator_optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(generator_loss).backward()

            should_step = step % args.accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                if args.gradient_clip_norm > 0:
                    scaler.unscale_(discriminator_optimizer)
                    scaler.unscale_(generator_optimizer)
                    nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=args.gradient_clip_norm)
                    nn.utils.clip_grad_norm_(generator.parameters(), max_norm=args.gradient_clip_norm)

                scaler.step(discriminator_optimizer)
                scaler.step(generator_optimizer)
                scaler.update()
                discriminator_optimizer.zero_grad(set_to_none=True)
                generator_optimizer.zero_grad(set_to_none=True)

            if update_d:
                running_metrics["discriminator_loss"] += float(discriminator_loss.item() * args.accumulation_steps)
            running_metrics["generator_loss"] += float(generator_loss.item() * args.accumulation_steps)
            running_metrics["gan_loss"] += float(gan_loss.item())
            running_metrics["l1_loss"] += float(l1_loss.item())
            running_metrics["edge_loss"] += float(edge.item())
            step_count += 1

            if is_main_process(rank):
                averaged_metrics = {name: value / max(1, step_count) for name, value in running_metrics.items()}
                progress_bar.set_postfix({
                    "d_loss": f"{averaged_metrics['discriminator_loss']:.4f}",
                    "g_loss": f"{averaged_metrics['generator_loss']:.4f}",
                    "l1": f"{averaged_metrics['l1_loss']:.4f}",
                    "edge": f"{averaged_metrics['edge_loss']:.4f}",
                })

        generator_scheduler.step()
        discriminator_scheduler.step()

        train_metrics = {name: value / max(1, step_count) for name, value in running_metrics.items()}
        metrics = dict(train_metrics)

        validation_available = val_loader is not None and (epoch + 1) % args.sample_every == 0
        if validation_available:
            if is_main_process(rank):
                print(f"Starting validation for epoch {epoch + 1}...")
            validation_model = unwrap_model(generator) if distributed else generator
            metrics.update(validate(validation_model, val_loader, device, amp_enabled, rank, epoch + 1, output_dir))
        else:
            metrics.update({"val_l1": math.inf, "val_edge": math.inf, "val_psnr": 0.0})

        if writer is not None:
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(metric_name, metric_value, epoch + 1)
            writer.add_scalar("lr/generator", generator_optimizer.param_groups[0]["lr"], epoch + 1)
            writer.add_scalar("lr/discriminator", discriminator_optimizer.param_groups[0]["lr"], epoch + 1)

        should_stop = False
        if is_main_process(rank):
            if validation_available and metrics["val_l1"] < best_val_l1:
                best_val_l1 = metrics["val_l1"]
                epochs_without_improvement = 0
                save_checkpoint(
                    output_dir / "checkpoints" / "best.pt",
                    epoch,
                    generator,
                    discriminator,
                    generator_optimizer,
                    discriminator_optimizer,
                    generator_scheduler,
                    discriminator_scheduler,
                    scaler,
                    best_val_l1,
                    epochs_without_improvement,
                    args,
                )
            elif validation_available:
                epochs_without_improvement += 1

            latest_checkpoint = output_dir / "checkpoints" / "latest.pt"
            save_checkpoint(
                latest_checkpoint,
                epoch,
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                generator_scheduler,
                discriminator_scheduler,
                scaler,
                best_val_l1,
                epochs_without_improvement,
                args,
            )

            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt",
                    epoch,
                    generator,
                    discriminator,
                    generator_optimizer,
                    discriminator_optimizer,
                    generator_scheduler,
                    discriminator_scheduler,
                    scaler,
                    best_val_l1,
                    epochs_without_improvement,
                    args,
                )

            summary = " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
            print(f"Epoch {epoch + 1}: {summary}")

            if val_loader is not None and not args.no_early_stopping and args.patience > 0 and epochs_without_improvement >= args.patience:
                should_stop = True

        if distributed:
            sync_tensor = torch.tensor(
                [best_val_l1, float(epochs_without_improvement), float(should_stop)],
                device=device,
            )
            dist.broadcast(sync_tensor, src=0)
            best_val_l1 = float(sync_tensor[0].item())
            epochs_without_improvement = int(sync_tensor[1].item())
            should_stop = bool(sync_tensor[2].item())

        if should_stop:
            if is_main_process(rank):
                print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    if writer is not None:
        writer.close()

    cleanup_distributed()


def main() -> None:
    args = parse_args()
    gpu_count = torch.cuda.device_count()

    if args.use_ddp and gpu_count > 1:
        port = get_free_port()
        torch.multiprocessing.spawn(
            train_worker,
            args=(gpu_count, port, args),
            nprocs=gpu_count,
            join=True,
        )
    else:
        train_worker(rank=0, world_size=1, port=0, args=args)


if __name__ == "__main__":
    main()
