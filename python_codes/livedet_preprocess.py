from pathlib import Path
from PIL import Image

# Source and destination
SRC_DIR = Path("/Users/dormalka/Downloads/LivDet 2015/Testing/Digital_Persona/Live/")
DST_DIR = Path("/Users/dormalka/Desktop/Dor/Paper/sourceafis-demo/livdet_preproc_png/Testing/Digital_Persona/Live/")

DST_DIR.mkdir(parents=True, exist_ok=True)

png_files = sorted(SRC_DIR.glob("*.png"))

print(f"[i] Found {len(png_files)} png files")

for png_path in png_files:
    png_name = png_path.stem + ".png"
    dst_path = DST_DIR / png_name

    with Image.open(png_path) as img:
        img.save(dst_path, "PNG")

    print(f"[✓] {png_path.name} → {png_name}")

print("[✓] Conversion complete")