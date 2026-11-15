"""Repackage the synthetic/ folder and the AeBAD test set into small, upload-
ready folders. Resizes images to 384px (the notebook trains at 224, so this is
ample) and saves as JPEG. Does NOT touch your originals or regenerate anything.

Run from your project folder:
    python shrink_for_upload.py

Produces:
    synthetic_small/        images/*.jpg + labels.csv   (masks are dropped: training doesn't use them)
    aebad_test_small/        the real test images only   (ground_truth dropped: not used)
Then zip THOSE and upload them instead.
"""
import csv
from pathlib import Path
import cv2

MAX_SIDE = 384
QUALITY = 90
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def shrink(src, dst):
    img = cv2.imread(str(src))
    if img is None:
        return False
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        sc = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    return True


def do_synthetic(src_dir, out_dir):
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    labels = list(csv.DictReader(open(src_dir / "labels.csv")))
    new_rows, n = [], 0
    for r in labels:
        src = src_dir / "images" / r["filename"]
        if not src.exists():
            continue
        new_name = Path(r["filename"]).with_suffix(".jpg").name
        if shrink(src, out_dir / "images" / new_name):
            r["filename"] = new_name
            new_rows.append(r)
            n += 1
    with open(out_dir / "labels.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=labels[0].keys())
        wr.writeheader()
        wr.writerows(new_rows)
    print(f"synthetic: wrote {n} images -> {out_dir}")


def do_test(src_dir, out_dir):
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    n = 0
    for p in src_dir.rglob("*"):
        if p.suffix.lower() not in EXTS:
            continue
        rel = p.relative_to(src_dir).with_suffix(".jpg")
        if shrink(p, out_dir / rel):
            n += 1
    print(f"test: wrote {n} images -> {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", default="synthetic")
    p.add_argument("--test", default="AeBAD/AeBAD_S/test")
    p.add_argument("--out-synthetic", default="synthetic_small")
    p.add_argument("--out-test", default="aebad_test_small")
    a = p.parse_args()
    do_synthetic(a.synthetic, a.out_synthetic)
    do_test(a.test, a.out_test)
    print("\nDone. Now zip and upload the *_small folders:")
    print("    zip -r synthetic.zip synthetic_small")
    print("    zip -r aebad_test.zip aebad_test_small")