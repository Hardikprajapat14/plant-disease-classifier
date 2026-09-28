"""
download_dataset.py
--------------------
Downloads the PlantVillage dataset from Kaggle and unzips it into ./dataset,
so you don't have to manually download it through the browser.

Prerequisites:
    1. pip install kaggle   (already in requirements.txt)
    2. Get a Kaggle API token:
         - Log in to kaggle.com
         - Go to Account -> Settings -> API -> "Create New Token"
         - This downloads a file called kaggle.json
    3. Place kaggle.json here:
         - Mac/Linux: ~/.kaggle/kaggle.json   (chmod 600 ~/.kaggle/kaggle.json)
         - Windows:   C:\\Users\\<you>\\.kaggle\\kaggle.json

Usage:
    python download_dataset.py
    python download_dataset.py --output_dir ./dataset
"""

import argparse
import os
import zipfile

DATASET_SLUG = "emmarex/plantdisease"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="dataset")
    args = parser.parse_args()

    # Import here (not at module load) so a missing kaggle.json gives a clean
    # error message instead of an import-time crash.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as exc:
        raise SystemExit(
            "Kaggle credentials not found.\n\n"
            "1. Log in to kaggle.com -> Account -> Settings -> API -> "
            "'Create New Token' (downloads kaggle.json)\n"
            "2. Move it to ~/.kaggle/kaggle.json (Mac/Linux) or "
            "C:\\Users\\<you>\\.kaggle\\kaggle.json (Windows)\n"
            "3. On Mac/Linux run: chmod 600 ~/.kaggle/kaggle.json\n"
            "4. Re-run this script.\n\n"
            f"Original error: {exc}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset '{DATASET_SLUG}' to '{args.output_dir}'...")
    api.dataset_download_files(DATASET_SLUG, path=args.output_dir, quiet=False)

    zip_path = None
    for fname in os.listdir(args.output_dir):
        if fname.endswith(".zip"):
            zip_path = os.path.join(args.output_dir, fname)
            break

    if zip_path is None:
        raise SystemExit("Download finished but no .zip file was found. Check the output above.")

    print(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(args.output_dir)

    os.remove(zip_path)

    print(f"\nDone. Dataset ready at: {args.output_dir}")
    print("Check that it contains one subfolder per class before training, e.g.:")
    print(f"  {args.output_dir}/Tomato___healthy/")
    print(f"  {args.output_dir}/Potato___Early_blight/")
    print("\nIf the extracted files are nested inside an extra subfolder, "
          "point --data_dir in train_model.py at that inner folder instead.")


if __name__ == "__main__":
    main()
