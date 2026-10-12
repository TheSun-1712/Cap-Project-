"""Data download, verification, extraction, and resolution preprocessor for Real-IAD.

This script:
1. Downloads `realiad_1024/toothbrush.zip` and `realiad_jsons.zip` from Hugging Face Hub.
2. Verifies the SHA256 checksum (expected: e8540fd8...3e6c30c8 for toothbrush.zip).
3. Unpacks archives into the exact directory layout required by Anomalib:
     datasets/Real-IAD/realiad_1024/toothbrush/
     datasets/Real-IAD/realiad_jsons/realiad_jsons/toothbrush.json
4. Optionally downscales 1024px images to 512px and 256px to save download bandwidth
   and enable rapid baseline ablation without downloading additional gigabytes.
"""

import argparse
import hashlib
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional
from PIL import Image

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

EXPECTED_TOOTHBRUSH_SHA256 = (
    "e8540fd8f86be2e4e83764836696dbf9cb819bc57dd6942adbc6b4b43e6c30c8"
)
DATASET_REPO = "Real-IAD/Real-IAD"


def compute_sha256(filepath: Path) -> str:
    """Compute sha256 checksum of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_real_iad(
    dest_dir: Path,
    category: str = "toothbrush",
    token: Optional[str] = None,
) -> None:
    """Download the gated Real-IAD dataset files from Hugging Face."""
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is not installed. Run `pip install huggingface_hub`.")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    realiad_1024_dir = dest_dir / "realiad_1024"
    realiad_jsons_dir = dest_dir / "realiad_jsons"
    realiad_1024_dir.mkdir(parents=True, exist_ok=True)
    realiad_jsons_dir.mkdir(parents=True, exist_ok=True)

    zip_1024_name = f"realiad_1024/{category}.zip"
    jsons_zip_name = "realiad_jsons.zip"

    print(f"[*] Downloading {zip_1024_name} from Hugging Face ({DATASET_REPO})...")
    try:
        downloaded_1024 = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=zip_1024_name,
            repo_type="dataset",
            local_dir=str(dest_dir),
            token=token,
        )
        print(f"[+] Downloaded 1024 zip to: {downloaded_1024}")
    except Exception as e:
        print(f"[!] Error downloading {zip_1024_name}: {e}")
        print("    Ensure you have requested access at: https://huggingface.co/datasets/Real-IAD/Real-IAD")
        print("    And provide your Hugging Face token with `--token <HF_TOKEN>` or via `hf auth login`.")
        return

    print(f"[*] Downloading {jsons_zip_name} from Hugging Face ({DATASET_REPO})...")
    try:
        downloaded_jsons = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=jsons_zip_name,
            repo_type="dataset",
            local_dir=str(dest_dir),
            token=token,
        )
        print(f"[+] Downloaded jsons zip to: {downloaded_jsons}")
    except Exception as e:
        print(f"[!] Error downloading {jsons_zip_name}: {e}")
        return

    # Checksum verification
    target_1024_path = Path(downloaded_1024)
    print(f"[*] Verifying SHA-256 for {target_1024_path.name}...")
    computed_hash = compute_sha256(target_1024_path)
    print(f"    Computed SHA256: {computed_hash}")
    if category == "toothbrush":
        if computed_hash.lower() == EXPECTED_TOOTHBRUSH_SHA256.lower():
            print("[+] Checksum matches expected hash!")
        else:
            print(f"[!] Warning: Hash differs from known hash ({EXPECTED_TOOTHBRUSH_SHA256[:16]}...)")


def extract_real_iad(dest_dir: Path, category: str = "toothbrush") -> None:
    """Extract downloaded zip files into the Anomalib directory structure."""
    dest_dir = Path(dest_dir)
    zip_1024 = dest_dir / "realiad_1024" / f"{category}.zip"
    if not zip_1024.exists():
        # Check root
        zip_1024_alt = dest_dir / f"{category}.zip"
        if zip_1024_alt.exists():
            zip_1024 = zip_1024_alt

    zip_jsons = dest_dir / "realiad_jsons.zip"
    if not zip_jsons.exists():
        zip_jsons_alt = dest_dir / "realiad_jsons" / "realiad_jsons.zip"
        if zip_jsons_alt.exists():
            zip_jsons = zip_jsons_alt

    if zip_1024.exists():
        target_dir = dest_dir / "realiad_1024"
        print(f"[*] Unpacking {zip_1024} into {target_dir}...")
        with zipfile.ZipFile(zip_1024, "r") as zf:
            zf.extractall(target_dir)
        print(f"[+] Extracted 1024 images.")
    else:
        print(f"[!] Could not find {zip_1024} to extract.")

    if zip_jsons.exists():
        target_dir = dest_dir / "realiad_jsons"
        print(f"[*] Unpacking {zip_jsons} into {target_dir}...")
        with zipfile.ZipFile(zip_jsons, "r") as zf:
            zf.extractall(target_dir)
        print(f"[+] Extracted metadata JSONs.")
    else:
        print(f"[!] Could not find {zip_jsons} to extract.")


def generate_downscaled_resolutions(
    dest_dir: Path,
    category: str = "toothbrush",
    target_resolutions=(512, 256),
) -> None:
    """Downsample 1024px images to 512px and 256px to save bandwidth and enable multi-res studies."""
    dest_dir = Path(dest_dir)
    src_cat_dir = dest_dir / "realiad_1024" / category
    if not src_cat_dir.exists():
        # Check if extracted directly
        alt_dir = dest_dir / "realiad_1024"
        if (alt_dir / "OK").exists() or (alt_dir / "NG").exists():
            src_cat_dir = alt_dir
        else:
            print(f"[!] Source directory {src_cat_dir} does not exist. Run extract first.")
            return

    for res in target_resolutions:
        dst_cat_dir = dest_dir / f"realiad_{res}" / category
        print(f"[*] Generating {res}x{res} dataset in {dst_cat_dir}...")
        dst_cat_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        for root_path, _, files in os.walk(src_cat_dir):
            rel_dir = Path(root_path).relative_to(src_cat_dir)
            target_sub_dir = dst_cat_dir / rel_dir
            target_sub_dir.mkdir(parents=True, exist_ok=True)

            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    src_file = Path(root_path) / f
                    dst_file = target_sub_dir / f
                    if not dst_file.exists():
                        with Image.open(src_file) as img:
                            # Use NEAREST for mask ground truths to preserve binary values, BILINEAR for images
                            resample_filter = Image.NEAREST if "mask" in str(src_file).lower() else Image.BILINEAR
                            resized = img.resize((res, res), resample=resample_filter)
                            resized.save(dst_file)
                    img_count += 1

        print(f"[+] Created {img_count} downscaled images at resolution {res}x{res}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and unpack Real-IAD toothbrush dataset.")
    parser.add_argument("--dest-dir", type=str, default="./datasets/Real-IAD", help="Dataset root directory")
    parser.add_argument("--category", type=str, default="toothbrush", help="Category name")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token")
    parser.add_argument("--download", action="store_true", help="Download archives from Hugging Face Hub")
    parser.add_argument("--extract", action="store_true", help="Extract archives into Anomalib structure")
    parser.add_argument("--generate-res", action="store_true", help="Downsample 1024 images to 512 and 256")
    parser.add_argument("--all", action="store_true", help="Run download, extract, and downscale")

    args = parser.parse_args()
    dest = Path(args.dest_dir)

    if args.all or args.download:
        download_real_iad(dest, category=args.category, token=args.token)

    if args.all or args.extract:
        extract_real_iad(dest, category=args.category)

    if args.all or args.generate_res:
        generate_downscaled_resolutions(dest, category=args.category)

    if not any([args.download, args.extract, args.generate_res, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
