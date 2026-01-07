#DO NOT RUN THIS SCRIPT UNLESS YOU HAVE THE ORIGINAL DATA FILE.
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from PIL import Image
import os, io
from collections import defaultdict

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
MRI_DIR   = os.path.dirname(UTILS_DIR)
MODEL_DIR = os.path.join(MRI_DIR, "Models")
DATA_DIR  = os.path.join(MODEL_DIR, "data")

PARQUET_PATH = DATA_DIR                   
TEST_IMG_DIR = os.path.join(UTILS_DIR, "Test_Data")

TRAIN_OUT = os.path.join(DATA_DIR, "train.parquet")
TEST_OUT  = os.path.join(DATA_DIR, "test.parquet")

os.makedirs(TEST_IMG_DIR, exist_ok = True)

table = pq.read_table(PARQUET_PATH)
rows = table.to_pylist()

def decode_image(img):
    if isinstance(img, dict):
        if "bytes" in img:
            return Image.open(io.BytesIO(img["bytes"]))
        else:
            arr = np.array(img["data"], dtype = np.uint8).reshape(img["shape"])
            return Image.fromarray(arr)
    elif isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img))
    else:
        return Image.fromarray(img)

picked = {}
remaining = []

for row in rows:
    label = int(row["label"])
    if label not in picked:
        picked[label] = row
    else:
        remaining.append(row)

for label, row in picked.items():
    img = decode_image(row["image"]).convert("RGB")
    img.save(os.path.join(TEST_IMG_DIR, f"class_{label}.png"))

remaining = np.array(remaining, dtype = object)

rng = np.random.default_rng(seed = 67)
rng.shuffle(remaining)

split = int(0.8 * len(remaining))
train_rows = remaining[:split]
test_rows  = remaining[split:]

pq.write_table(pa.Table.from_pylist(list(train_rows)), TRAIN_OUT)
pq.write_table(pa.Table.from_pylist(list(test_rows)), TEST_OUT)

print(f"Extracted {len(picked)} class samples → {TEST_IMG_DIR}")
print(f"Train samples saved: {len(train_rows)} → train.parquet")
print(f"Test samples saved:  {len(test_rows)} → test.parquet")