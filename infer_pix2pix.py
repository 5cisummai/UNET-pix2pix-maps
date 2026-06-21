from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models import UNetGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pix2Pix UNet inference on satellite images.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="runs/inference")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--paired-input", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--source-side", type=str, default="left", choices=["left", "right"])
    return parser.parse_args()


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    return image_tensor.mul(0.5).add(0.5).clamp(0.0, 1.0)


def load_source_image(image_path: Path, image_size: int, paired_input: bool, source_side: str) -> tuple[torch.Tensor, Image.Image]:
    original_image = Image.open(image_path).convert("RGB")

    if paired_input:
        width, height = original_image.size
        midpoint = width // 2
        left_image = original_image.crop((0, 0, midpoint, height))
        right_image = original_image.crop((midpoint, 0, width, height))
        original_image = left_image if source_side == "left" else right_image

    resized_image = original_image.resize((image_size, image_size), Image.BICUBIC)
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).mul(2.0).sub(1.0)
    return image_tensor.unsqueeze(0), original_image


def save_output(source_image: Image.Image, generated_tensor: torch.Tensor, output_path: Path) -> None:
    generated_array = denormalize_image(generated_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    generated_image = Image.fromarray((generated_array * 255.0).astype(np.uint8))
    resized_source = source_image.resize(generated_image.size, Image.BICUBIC)

    comparison = Image.new("RGB", (generated_image.width * 2, generated_image.height))
    comparison.paste(resized_source, (0, 0))
    comparison.paste(generated_image, (generated_image.width, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> UNetGenerator:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = UNetGenerator().to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator


def resolve_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in input_path.iterdir() if path.suffix.lower() in extensions)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = load_checkpoint(args.checkpoint, device)
    input_images = resolve_input_images(Path(args.input_path))
    if not input_images:
        raise FileNotFoundError(f"No input images found at {args.input_path}")

    for image_path in tqdm(input_images, desc="Inference"):
        source_tensor, source_image = load_source_image(
            image_path,
            image_size=args.image_size,
            paired_input=args.paired_input,
            source_side=args.source_side,
        )
        source_tensor = source_tensor.to(device)

        with torch.no_grad():
            generated_tensor = generator(source_tensor)

        output_path = Path(args.output_dir) / f"{image_path.stem}_generated.png"
        save_output(source_image, generated_tensor, output_path)


if __name__ == "__main__":
    main()
