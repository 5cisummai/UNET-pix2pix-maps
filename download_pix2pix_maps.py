from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


DATASET_URL = "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz"
EXPECTED_SPLITS = ("train", "val")


class DownloadProgressBar(tqdm):
    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify the Pix2Pix Maps dataset.")
    parser.add_argument("--output-dir", type=str, default="datasets")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading maps.tar.gz") as progress_bar:
        urllib.request.urlretrieve(DATASET_URL, filename=destination_path, reporthook=progress_bar.update_to)


def safe_extract(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = target_dir / member.name
            if not member_path.resolve().is_relative_to(target_dir.resolve()):
                raise RuntimeError(f"Unsafe path found in archive: {member.name}")
        archive.extractall(path=target_dir)


def verify_dataset_root(dataset_root: Path) -> None:
    missing_splits = [split_name for split_name in EXPECTED_SPLITS if not (dataset_root / split_name).exists()]
    if missing_splits:
        raise FileNotFoundError(f"Missing expected splits in {dataset_root}: {', '.join(missing_splits)}")

    for split_name in EXPECTED_SPLITS:
        image_count = sum(1 for path in (dataset_root / split_name).iterdir() if path.is_file())
        if image_count == 0:
            raise FileNotFoundError(f"No images found in split: {dataset_root / split_name}")


def install_dataset(output_dir: Path, force: bool) -> Path:
    dataset_root = output_dir / "maps"
    if dataset_root.exists() and not force:
        verify_dataset_root(dataset_root)
        return dataset_root

    if dataset_root.exists() and force:
        shutil.rmtree(dataset_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pix2pix_maps_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        archive_path = temp_dir / "maps.tar.gz"
        extract_root = temp_dir / "extract"

        download_archive(archive_path)
        archive_hash = sha256_file(archive_path)
        print(f"Downloaded archive SHA256: {archive_hash}")

        safe_extract(archive_path, extract_root)
        extracted_dataset_root = extract_root / "maps"
        verify_dataset_root(extracted_dataset_root)

        shutil.move(str(extracted_dataset_root), str(dataset_root))

    return dataset_root


def main() -> None:
    args = parse_args()
    dataset_root = install_dataset(Path(args.output_dir), force=args.force)
    print(f"Dataset installed at: {dataset_root}")
    print("Expected training command:")
    print(f"python3 train_pix2pix.py --config configs/pix2pix_maps.yaml --data-root {dataset_root}")


if __name__ == "__main__":
    main()
