#!/usr/bin/env python3
# OpenAFIS ISO Template Extractor (SecuGen SDK based)
# Works with BMP / PNG / JPG / TIFF
# Requires: numpy, pillow, cffi
# Requires: sgfplib (SecuGen FDx Pro SDK) in same directory

import argparse
import sys
import numpy as np
from cffi import FFI
from PIL import Image


# =========================
# FFI definitions from sgfplib.h
# =========================

ffi = FFI()
ffi.cdef("""
typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned long DWORD;
typedef void* HSGFPM;

enum SGFDxDeviceName {
   SG_DEV_FDU05 = 0x06
};

enum SGFDxTemplateFormat {
   TEMPLATE_FORMAT_ISO19794 = 0x0300
};

enum SGImpressionType {
   SG_IMPTYPE_LP = 0x00
};

enum SGFingerPosition {
   SG_FINGPOS_UK = 0x00
};

typedef struct tagSGFingerInfo {
    WORD FingerNumber;
    WORD ViewNumber;
    WORD ImpressionType;
    WORD ImageQuality;
} SGFingerInfo;

DWORD SGFPM_Create(HSGFPM* phFpm);
DWORD SGFPM_Terminate(HSGFPM hFpm);
DWORD SGFPM_Init(HSGFPM hFpm, DWORD devName);
DWORD SGFPM_SetTemplateFormat(HSGFPM hFpm, WORD format);
DWORD SGFPM_GetMaxTemplateSize(HSGFPM hFpm, DWORD* size);
DWORD SGFPM_CreateTemplate(HSGFPM hFpm, SGFingerInfo* fpInfo, BYTE* rawImage, BYTE* minTemplate);
DWORD SGFPM_GetTemplateSize(HSGFPM hFpm, BYTE* minTemplate, DWORD* size);
""")


# =========================
# Argument parsing
# =========================

parser = argparse.ArgumentParser(
    description="Extract ISO 19794-2 template from fingerprint image (bmp/png/jpg/tif)"
)
parser.add_argument("input_file", type=str, help="Input image file")
parser.add_argument("output_file", type=str, help="Output ISO file")
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file


# =========================
# Load & preprocess image
# =========================

try:
    img = Image.open(input_file).convert("L")  # force grayscale
except Exception as e:
    print(f"❌ Failed to open image: {e}")
    sys.exit(1)

# Resize to match SecuGen SG_DEV_FDU05 sensor (300x400)
img = img.resize((300, 400), Image.LANCZOS)

raw = np.array(img, dtype=np.uint8)

if raw.shape != (400, 300):
    print("❌ Unexpected image size after resize:", raw.shape)
    sys.exit(1)

print(f"[i] Loaded image: {input_file}")
print(f"[i] Image shape: {raw.shape}")


# =========================
# Load SecuGen library
# =========================

try:
    lib = ffi.dlopen("sgfplib")
except OSError:
    print("❌ Could not load sgfplib.")
    print("Make sure sgfplib (SecuGen SDK) is in this directory.")
    sys.exit(1)


# =========================
# Initialize FPM
# =========================

fpm = ffi.new("HSGFPM*")

r = lib.SGFPM_Create(fpm)
if r != 0:
    raise RuntimeError(f"SGFPM_Create failed: {r}")

r = lib.SGFPM_Init(fpm[0], lib.SG_DEV_FDU05)
if r != 0:
    raise RuntimeError(f"SGFPM_Init failed: {r}")

r = lib.SGFPM_SetTemplateFormat(fpm[0], lib.TEMPLATE_FORMAT_ISO19794)
if r != 0:
    raise RuntimeError(f"SGFPM_SetTemplateFormat failed: {r}")


# =========================
# Allocate template buffer
# =========================

max_size = ffi.new("DWORD*")
r = lib.SGFPM_GetMaxTemplateSize(fpm[0], max_size)
if r != 0:
    raise RuntimeError(f"SGFPM_GetMaxTemplateSize failed: {r}")

template_buffer = ffi.new("BYTE[]", max_size[0])


# =========================
# Create template
# =========================

fp_info = ffi.new("SGFingerInfo*", {
    "FingerNumber": lib.SG_FINGPOS_UK,
    "ViewNumber": 0,
    "ImpressionType": lib.SG_IMPTYPE_LP,
    "ImageQuality": 0
})

raw_ptr = ffi.cast("BYTE*", raw.ctypes.data)

r = lib.SGFPM_CreateTemplate(fpm[0], fp_info, raw_ptr, template_buffer)
if r != 0:
    raise RuntimeError(f"SGFPM_CreateTemplate failed: {r}")


# =========================
# Get actual template size
# =========================

size = ffi.new("DWORD*")
r = lib.SGFPM_GetTemplateSize(fpm[0], template_buffer, size)
if r != 0:
    raise RuntimeError(f"SGFPM_GetTemplateSize failed: {r}")

template_data = ffi.buffer(template_buffer)[0:size[0]]

lib.SGFPM_Terminate(fpm[0])


# =========================
# Write ISO file
# =========================

with open(output_file, "wb") as f:
    f.write(template_data)

print(f"✅ ISO template written: {output_file}")
print(f"[i] Template size: {size[0]} bytes")