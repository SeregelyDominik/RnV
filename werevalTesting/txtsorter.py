from pathlib import Path
import shutil

SRC_DIR = Path(r"D:\testingstuff\bme\HungarianDysartriaDatabase2\HungarianDysartriaDatabase\text")
DST_DIR = Path("textek")

DST_DIR.mkdir(parents=True, exist_ok=True)

for txt_file in SRC_DIR.glob("*.txt"):
    parts = txt_file.stem.split("_")

    if len(parts) < 2:
        print(f"Skipping (unexpected name): {txt_file.name}")
        continue

    folder_name = parts[1]  # "010"

    target_dir = DST_DIR / folder_name
    target_dir.mkdir(exist_ok=True)

    shutil.move(txt_file, target_dir / txt_file.name)

print("Done.")
