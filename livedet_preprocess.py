from pathlib import Path
from PIL import Image

SRC = Path("/Users/dormalka/Downloads/LivDet 2015")
DST = Path("/Users/dormalka/Desktop/Dor/Paper/livdet_preproc_png")
DST.mkdir(parents=True, exist_ok=True)

MAX_SIDE = 800         
OUT_EXT = ".png"
TARGET_DIRS = [
    SRC / "Training" / "Hi_Scan" / "Live",
    SRC / "Testing"  / "Hi_Scan" / "Live",
    SRC / "Testing"  / "Hi_Scan" / "Fake" / "Latex",
]

def convert_one(bmp: Path):
    rel = bmp.relative_to(SRC)
    out = (DST / rel).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(bmp).convert("L")
    w, h = img.size
    m = max(w, h)
    if m > MAX_SIDE:
        s = MAX_SIDE / float(m)
        img = img.resize((int(round(w * s)), int(round(h * s))), Image.BILINEAR)

    img.save(out, format="PNG", optimize=True)

count = 0
for folder in TARGET_DIRS:
    for bmp in folder.glob("002_0_*.bmp"):
        convert_one(bmp)
        count += 1

print(f"Done. Converted {count} images into: {DST}")