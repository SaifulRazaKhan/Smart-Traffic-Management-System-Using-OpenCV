"""
Download script for AIM-IISc UVH-26 subset from kalyan1729/trafficmanagementdataset
Downloads ~2.5 to 3.0 GB of real Bengaluru urban traffic CCTV imagery and annotations.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "kalyan1729/trafficmanagementdataset"
REPO_TYPE = "dataset"
TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trafficmanagementdataset"))
TARGET_BYTES = int(2.5 * 1024 * 1024 * 1024)  # 2.5 GB in bytes


def format_bytes(b):
    if b >= 1024 * 1024 * 1024:
        return f"{b / (1024 * 1024 * 1024):.2f} GB"
    elif b >= 1024 * 1024:
        return f"{b / (1024 * 1024):.2f} MB"
    elif b >= 1024:
        return f"{b / 1024:.2f} KB"
    return f"{b} B"


def main():
    print("=" * 70)
    print(" AIM-IISc UVH-26 / Bengaluru Traffic Dataset Downloader")
    print(f" Source:      {REPO_ID}")
    print(f" Target Dir:  {TARGET_DIR}")
    print(f" Target Size: ~2.5 GB (2 to 4 GB range)")
    print("=" * 70)

    os.makedirs(TARGET_DIR, exist_ok=True)
    api = HfApi()

    # Step 1: Download the Ground-Truth Annotation File (~63 MB)
    annot_file = "UVH-26-Train/UVH-26-MV-Train.json"
    print(f"\n[Step 1] Downloading consensus ground-truth annotations: {annot_file}...")
    try:
        annot_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=annot_file,
            repo_type=REPO_TYPE,
            local_dir=TARGET_DIR
        )
        annot_size = os.path.getsize(annot_path)
        print(f" -> Successfully downloaded annotations: {format_bytes(annot_size)}")
    except Exception as e:
        print(f" -> Note: Annotation download skipped/failed ({e}), proceeding with image frames.")
        annot_size = 0

    # Step 2: List files in UVH-26-Train/data/000
    print("\n[Step 2] Fetching image file list from Hugging Face repository...")
    image_items = list(api.list_repo_tree(
        repo_id=REPO_ID,
        path_in_repo="UVH-26-Train/data/000",
        repo_type=REPO_TYPE
    ))
    
    # Filter valid image files
    image_files = [f for f in image_items if getattr(f, 'size', None) and f.path.lower().endswith(('.png', '.jpg'))]
    print(f" -> Found {len(image_files)} image frames in UVH-26-Train/data/000.")

    # Step 3: Select images up to target size (~2.5 GB)
    selected_files = []
    cumulative_bytes = annot_size
    for f in image_files:
        if cumulative_bytes >= TARGET_BYTES:
            break
        selected_files.append((f.path, f.size))
        cumulative_bytes += f.size

    print(f"\n[Step 3] Selected {len(selected_files)} images totaling {format_bytes(cumulative_bytes)}.")

    # Step 4: Multi-threaded parallel download
    print(f"\n[Step 4] Starting download with 8 parallel worker threads...")
    t0 = time.time()
    downloaded_bytes = annot_size
    completed_count = 0
    total_count = len(selected_files)

    def download_one(item):
        path, size = item
        dest = os.path.join(TARGET_DIR, path.replace("/", os.sep))
        # Skip if already fully downloaded
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            return size, True
        hf_hub_download(
            repo_id=REPO_ID,
            filename=path,
            repo_type=REPO_TYPE,
            local_dir=TARGET_DIR
        )
        return size, False

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(download_one, item): item for item in selected_files}
        for future in as_completed(future_map):
            completed_count += 1
            try:
                size, was_cached = future.result()
                downloaded_bytes += size
            except Exception as exc:
                item = future_map[future]
                print(f"Warning: Failed to download {item[0]}: {exc}")

            if completed_count % 50 == 0 or completed_count == total_count:
                pct = (completed_count / total_count) * 100
                elapsed = time.time() - t0
                speed = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                print(f" -> Progress: {completed_count}/{total_count} ({pct:.1f}%) | "
                      f"Downloaded: {format_bytes(downloaded_bytes)} | Speed: {speed:.2f} MB/s")

    elapsed_total = time.time() - t0
    print("\n" + "=" * 70)
    print(f" DOWNLOAD COMPLETE!")
    print(f" Total files:     {completed_count + (1 if annot_size > 0 else 0)}")
    print(f" Total volume:    {format_bytes(downloaded_bytes)}")
    print(f" Time elapsed:    {elapsed_total:.1f} seconds")
    print(f" Stored in:       {TARGET_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
