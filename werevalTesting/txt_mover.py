from pathlib import Path
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("speaker", type=str)
args = parser.parse_args()
TARGET_DIR = Path(r"C:\Users\reaso\PycharmProjects\TestOriginalUrythmic\RnV-main\RnV-main\testmhub\text")
SOURCE_DIR = Path(rf"C:\Users\reaso\PycharmProjects\TestOriginalUrythmic\RnV-main\RnV-main\textek\{args.speaker}")  # <-- choose which folder to inject

# --- Step 1: Empty target folder ---
if TARGET_DIR.exists():
    for item in TARGET_DIR.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
else:
    TARGET_DIR.mkdir(parents=True)

# --- Step 2: Copy files from source folder ---
for item in SOURCE_DIR.iterdir():
    if item.is_file():
        shutil.copy2(item, TARGET_DIR / item.name)

