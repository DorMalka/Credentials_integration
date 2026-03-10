from pathlib import Path
from PIL import Image

# Source and destination
SRC_DIR = Path("/Users/dormalka/Downloads/LivDet 2015/Testing/Hi_Scan/Fake/Gelatine/")
DST_DIR = Path("/Users/dormalka/Desktop/Dor/Paper/sourceafis-demo/livdet_preproc_png/Testing/Hi_Scan/Fake/Gelatine/")

DST_DIR.mkdir(parents=True, exist_ok=True)

bmp_files = sorted(SRC_DIR.glob("*.bmp"))

print(f"[i] Found {len(bmp_files)} bmp files")

for bmp_path in bmp_files:
    png_name = bmp_path.stem + ".png"
    dst_path = DST_DIR / png_name

    with Image.open(bmp_path) as img:
        img.save(dst_path, "PNG")

    print(f"[✓] {bmp_path.name} → {png_name}")

print("[✓] Conversion complete")