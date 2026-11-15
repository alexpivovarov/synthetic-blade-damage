"""
synth_damage.py - add synthetic damage to UNDAMAGED aero-engine blade photos.

Damage types match the AeBAD dataset:
    breakdown  small dark holes through the blade
    ablation   dark, sooty burn patches
    fracture   a chunk missing from the blade edge (outline changes)
    groove     small notches on the edge + thin scratch lines

Usage:
    python synth_damage.py --input AeBAD_S/train/good --output synthetic --per-image 4
    python synth_damage.py ... --types breakdown,groove      # only some types

Output:
    synthetic/images/      damaged (label 1) and clean (label 0) images
    synthetic/masks/       white = where damage was added (black for clean images)
    synthetic/labels.csv   filename, label, defect_type, source_image
    synthetic/preview.jpg  original | one example of each damage type
    synthetic/blade_check.jpg  green outline = where the script thinks the blade is
    synthetic/skipped.txt  images skipped because the blade wasn't found

Install:  python -m pip install opencv-python numpy "rembg[cpu]"
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

TYPES = ["breakdown", "ablation", "fracture", "groove"]


# ---------------------------------------------------------------------------
# 1. Find the blade: everything that doesn't look like the background
# ---------------------------------------------------------------------------
def _smooth_background(lab, iters=4, keep=0.6):
    """Fit a smooth surface (2nd-order polynomial per colour channel) to the
    image, refitting each time on the pixels that fit best. Backgrounds in
    these photos are smooth gradients; the blade is what doesn't fit.
    Returns the signed residual per channel (h, w, 3)."""
    h, w = lab.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    x, y = (xx / w - 0.5).ravel(), (yy / h - 0.5).ravel()
    A = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], 1)
    V = lab.reshape(-1, 3)
    use = np.ones(len(V), bool)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[use], V[use], rcond=None)
        R = V - A @ coef
        res = np.linalg.norm(R, axis=1)
        use = res <= np.quantile(res, keep)     # keep best-fitting pixels
    return R.reshape(h, w, 3), use.reshape(h, w)


def _edge_sharpness(grey, comp):
    """Average gradient strength along a region's outline. Real objects have
    crisp outlines; shadows and lighting blobs have soft ones."""
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ring = cv2.dilate(comp, np.ones((3, 3), np.uint8)) - cv2.erode(comp, np.ones((3, 3), np.uint8))
    hgt, wid = comp.shape
    ring[:1], ring[-1:], ring[:, :1], ring[:, -1:] = 0, 0, 0, 0   # ignore photo border
    return float(mag[ring > 0].mean()) if (ring > 0).any() else 0.0


_SESSION = None


def blade_mask_ai(img):
    """Find the blade with a pretrained background-removal network (U2-Net,
    via the rembg library). Trained on thousands of photos to separate 'the
    main object' from its background - far more robust to shadows, gradients
    and blades touching the frame than the hand-made detector below."""
    global _SESSION
    from rembg import new_session, remove
    if _SESSION is None:
        _SESSION = new_session("u2netp")          # small model, ~4.6 MB download
    m = np.array(remove(img, session=_SESSION, only_mask=True))
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])   # the blade = largest object
    m = np.where(lbl == biggest, 255, 0).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cs, -1, 255, -1)          # fill holes inside the blade
    frac = (m > 0).mean()
    if frac < 0.01 or frac > 0.9:
        return None
    return m


def blade_mask(img, method="ai"):
    if method == "ai":
        return blade_mask_ai(img)
    return blade_mask_classic(img)


def blade_mask_classic(img):
    """Estimate which pixels are blade. Returns None if unsure.

    1. Model the background as a smooth colour gradient.
    2. Candidate regions = pixels clearly BRIGHTER than it, clearly DARKER
       than it, or clearly a different COLOUR. Kept separate, so a white blade
       and its dark shadow can't merge into one blob.
    3. Pick the candidate with the crispest outline (blade, not shadow).
    """
    h, w = img.shape[:2]
    s = max(1, max(h, w) // 160)                  # work on a ~160px copy
    small = cv2.resize(img, (w // s, h // s), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    R, inl = _smooth_background(lab, keep=0.5)
    L, chroma = R[..., 0], np.hypot(R[..., 1], R[..., 2])
    # spread of the pixels that really are background (the best-fitting half);
    # using ALL pixels overestimates it when the blade is large
    noise = np.median(np.abs(L[inl])) * 1.4826 + 1e-3
    noise = max(noise, 2.0)

    cands = []
    k = np.ones((3, 3), np.uint8)
    min_area = 0.01 * L.size
    cthr = max(8.0, 4 * np.median(chroma[inl]))
    for raw in (L > 4 * noise, L < -4 * noise, chroma > cthr):
        m = raw.astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            comp = np.where(lbl == i, 255, 0).astype(np.uint8)
            comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k, iterations=2)
            # thickness = radius of the largest circle that fits inside.
            # Blades are wide; shadow edges and reflections are thin slivers.
            thick = cv2.distanceTransform(comp, cv2.DIST_L2, 5).max()
            score = _edge_sharpness(grey, comp) * np.sqrt(thick)
            cands.append((score, thick, comp))
    if not cands:
        return None
    score, thick, m = max(cands, key=lambda c: c[0])
    if thick < 0.03 * min(m.shape):               # winner is a thin sliver -> unsure
        return None

    # fill holes inside the blade (e.g. dark grooves on a white blade)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cs, -1, 255, -1)

    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    m = np.where(cv2.GaussianBlur(m, (0, 0), s) > 127, 255, 0).astype(np.uint8)
    frac = (m > 0).mean()
    if frac < 0.01 or frac > 0.9:                 # implausible -> don't guess
        return None
    return m


def blade_edge(mask):
    """Pixels on the blade outline (not on the photo's border)."""
    edge = mask - cv2.erode(mask, np.ones((5, 5), np.uint8))
    b = 3
    edge[:b], edge[-b:], edge[:, :b], edge[:, -b:] = 0, 0, 0, 0
    return edge


def _poly_terms(x, y):
    return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y,
                     x ** 3, y ** 3, x * x * y, x * y * y], 1)


def background_estimate(img, mask):
    """Smooth model of the whole background (a 3rd-order colour surface fitted
    to the pixels that are NOT blade). Captures vignettes and gradients."""
    h, w = img.shape[:2]
    away = cv2.dilate(mask, np.ones((9, 9), np.uint8)) == 0
    ys, xs = np.nonzero(away)
    if len(xs) < 50:
        return np.full(img.shape, img.reshape(-1, 3).mean(0), np.float32)
    step = max(1, len(xs) // 20000)               # subsample for speed
    ys, xs = ys[::step], xs[::step]
    A = _poly_terms(xs / w - 0.5, ys / h - 0.5)
    coef, *_ = np.linalg.lstsq(A, img[ys, xs].astype(np.float32), rcond=None)
    yy, xx = np.mgrid[0:h, 0:w]
    full = _poly_terms(xx.ravel() / w - 0.5, yy.ravel() / h - 0.5) @ coef
    return np.clip(full.reshape(h, w, 3), 0, 255).astype(np.float32)


def local_background(img, mask, region, rng):
    """What the background behind `region` should look like.

    For a region at the blade's edge (fracture): inpaint it, i.e. flow in
    colours from the real background pixels right next to it, so the seam
    matches. (Measured on real AeBAD photos, this matched the seam ~6x better
    than estimating the background from the whole image.)
    For a region inside the blade (holes): use the whole-image model.
    Either way, add grain with the same strength as the photo's own noise."""
    h, w = img.shape[:2]
    imgf = img.astype(np.float32)
    bgpix = cv2.dilate(mask, np.ones((7, 7), np.uint8)) == 0
    near = (cv2.dilate(region, np.ones((25, 25), np.uint8)) > 0) & bgpix

    if region.any() and near.sum() >= 20:
        unknown = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        ys, xs = np.nonzero(region)
        pad = 40
        y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
        bg = imgf.copy()
        bg[y0:y1, x0:x1] = cv2.inpaint(img[y0:y1, x0:x1], unknown[y0:y1, x0:x1],
                                       7, cv2.INPAINT_TELEA)
        grain_src = near
        grain_where = (unknown > 0).astype(np.float32)   # only the filled-in pixels
    else:
        bg = background_estimate(img, mask)
        grain_src = bgpix
        grain_where = np.ones((h, w), np.float32)

    # grain: same strength as the fine noise in the real background
    hf = imgf - cv2.GaussianBlur(imgf, (0, 0), 1.5)
    sigma = float(hf[grain_src].std()) if grain_src.any() else 2.0
    grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.6)
    grain *= sigma / max(1e-3, float(grain.std()))
    return np.clip(bg + (grain * grain_where)[..., None], 0, 255).astype(np.float32)


def pick(mask, rng):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    i = rng.integers(len(xs))
    return int(xs[i]), int(ys[i])


def jagged_polygon(cx, cy, r, rng, n=14, rough=0.35):
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = r * rng.uniform(1 - rough, 1 + rough, n)
    return np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)


# ---------------------------------------------------------------------------
# 2. One function per damage type. Each returns (new_image, damage_mask)
# ---------------------------------------------------------------------------
def add_breakdown(img, mask, rng):
    """Small holes punched through the blade: you see the background through
    them, with a dark broken rim around each one."""
    h, w = img.shape[:2]
    s = min(h, w)
    out = img.astype(np.float32)
    # holes are inside the blade: use the whole-image background model + grain
    bg = local_background(img, mask, np.zeros_like(mask), rng)
    dmg = np.zeros((h, w), np.uint8)
    inner = cv2.erode(mask, np.ones((25, 25), np.uint8))
    for _ in range(rng.integers(1, 5)):
        p = pick(inner, rng)
        if p is None:
            break
        r = rng.uniform(0.012, 0.035) * s
        hole = np.zeros((h, w), np.uint8)
        cv2.fillPoly(hole, [jagged_polygon(p[0], p[1], r, rng, rough=0.25)], 255)
        soft = cv2.GaussianBlur(hole.astype(np.float32) / 255, (0, 0), max(0.8, r * 0.1))[..., None]
        # inside the hole: background (seen through it), slightly shadowed
        out = out * (1 - soft) + bg * rng.uniform(0.55, 0.85) * soft
        # dark torn rim around the hole
        rim = cv2.dilate(hole, np.ones((5, 5), np.uint8)).astype(np.float32) / 255
        rim = np.clip(cv2.GaussianBlur(rim, (0, 0), 1.2) - soft[..., 0], 0, 1)[..., None]
        out *= (1 - rng.uniform(0.4, 0.7) * rim)
        dmg |= hole
    return out, dmg


def add_ablation(img, mask, rng):
    """Dark, uneven burn / soot patch."""
    h, w = img.shape[:2]
    s = min(h, w)
    p = pick(mask, rng)
    if p is None:
        return img.astype(np.float32), np.zeros((h, w), np.uint8)
    r = rng.uniform(0.07, 0.18) * s
    # irregular blob = smooth noise shaping a soft disc
    noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), r * 0.2)
    noise = cv2.normalize(noise, None, 0, 1, cv2.NORM_MINMAX)
    yy, xx = np.mgrid[0:h, 0:w]
    disc = np.clip(1 - np.hypot(xx - p[0], yy - p[1]) / r, 0, 1)
    blob = np.clip(disc * (0.6 + 1.2 * noise) - 0.15, 0, 1) * (mask > 0)
    blob = cv2.GaussianBlur(blob.astype(np.float32), (0, 0), max(1.5, r * 0.06))
    blob *= (mask > 0)                               # never spill off the blade
    strength = rng.uniform(0.6, 0.9)
    soot = np.array([rng.uniform(15, 35), rng.uniform(20, 40), rng.uniform(30, 55)], np.float32)  # BGR brownish black
    # grainy texture inside the burn
    grain = 1 + rng.normal(0, 0.15, (h, w)).astype(np.float32)
    a = np.clip(strength * blob * grain, 0, 1)[..., None]
    out = img.astype(np.float32) * (1 - a) + soot * a
    return out, (blob > 0.15).astype(np.uint8) * 255


def add_fracture(img, mask, rng):
    """Break off a chunk at the blade edge: replace it with background."""
    h, w = img.shape[:2]
    s = min(h, w)
    edge = blade_edge(mask)
    r = rng.uniform(0.07, 0.16) * s
    # only break where the whole chunk fits inside the photo - a "break"
    # where the blade runs out of frame just looks like a missing bit of photo
    m = int(r * 1.5) + 2
    room = np.zeros_like(edge)
    room[m:h - m, m:w - m] = 255
    p = pick(edge & room, rng)
    if p is None:
        return img.astype(np.float32), np.zeros((h, w), np.uint8)
    chunk = np.zeros((h, w), np.uint8)
    cv2.fillPoly(chunk, [jagged_polygon(p[0], p[1], r, rng, n=10, rough=0.45)], 255)
    # remove blade pixels only - but reach a few pixels past the detected
    # outline, otherwise a thin bright sliver of the real edge survives
    poly = chunk.copy()
    grown = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    chunk &= grown
    # anything the cut separates from the main blade falls away with the chunk
    # (otherwise bright fragments are left floating in the background)
    rest = grown & ~chunk
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(rest)
    if n > 2:
        main = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        chunk |= np.where((lbl > 0) & (lbl != main), 255, 0).astype(np.uint8)
    # The blade's soft shadow and edge blur reach further out than its outline.
    # Inside the break, clear that band too - otherwise a "ghost line" of the
    # old edge stays visible across the gap.
    R = max(9, int(0.045 * s))
    wide = cv2.dilate(mask, np.ones((2 * R + 1, 2 * R + 1), np.uint8))
    halo = poly & wide & ~grown
    fill = chunk | halo
    bg = local_background(img, wide, fill, rng)
    # crisp against the remaining blade (a sharp broken edge), but feathered
    # softly where the cleared band meets untouched background
    crisp = cv2.GaussianBlur(chunk.astype(np.float32) / 255, (0, 0), 1.0)
    feather = cv2.GaussianBlur(fill.astype(np.float32) / 255, (0, 0), 3.5)
    alpha = np.maximum(crisp, feather)
    keep = (grown & ~chunk) > 0                       # blade that remains
    alpha[keep] = crisp[keep]
    soft = alpha[..., None]
    out = img.astype(np.float32) * (1 - soft) + bg * soft
    # darker broken edge along the new outline
    new_edge = cv2.dilate(chunk, np.ones((3, 3), np.uint8)) & (mask & ~chunk)
    e = cv2.GaussianBlur(new_edge.astype(np.float32) / 255, (0, 0), 1.0)[..., None]
    out *= (1 - 0.25 * e)
    return out, chunk


def add_groove(img, mask, rng):
    """Small notches cut into the edge, plus thin scratch lines."""
    h, w = img.shape[:2]
    s = min(h, w)
    dmg = np.zeros((h, w), np.float32)
    edge = blade_edge(mask)
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.mean(), ys.mean()) if len(xs) else (w / 2, h / 2)

    for _ in range(rng.integers(2, 5)):             # notches: short, inward
        p = pick(edge, rng)
        if p is None:
            break
        ang = np.arctan2(cy - p[1], cx - p[0]) + rng.normal(0, 0.4)
        L = rng.uniform(0.03, 0.08) * s
        q = (int(p[0] + L * np.cos(ang)), int(p[1] + L * np.sin(ang)))
        cv2.line(dmg, p, q, 1.0, max(2, int(rng.uniform(3, 6) * s / 512)), cv2.LINE_AA)

    for _ in range(rng.integers(0, 3)):             # scratches: long, faint
        p = pick(cv2.erode(mask, np.ones((15, 15), np.uint8)), rng)
        if p is None:
            break
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(0.05, 0.15) * s
        q = (int(p[0] + L * np.cos(ang)), int(p[1] + L * np.sin(ang)))
        cv2.line(dmg, p, q, rng.uniform(0.4, 0.8), max(1, int(s / 512)), cv2.LINE_AA)

    dmg *= (mask > 0)
    soft = cv2.GaussianBlur(dmg, (0, 0), 0.8)
    out = img.astype(np.float32) * (1 - rng.uniform(0.5, 0.8) * soft[..., None])
    return out, (dmg > 0.2).astype(np.uint8) * 255


ADDERS = {"breakdown": add_breakdown, "ablation": add_ablation,
          "fracture": add_fracture, "groove": add_groove}


# ---------------------------------------------------------------------------
# 3. Random photo variation, applied to damaged AND clean images
# ---------------------------------------------------------------------------
def photometric_jitter(img, rng):
    """So the model can't learn "edited-looking = damaged"."""
    img = img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    img = img + rng.normal(0, rng.uniform(2, 6), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


MIN_CONTRAST = 30   # average brightness change (0-255) the damage must cause


def make_damaged(img, mask, dtype, rng):
    h, w = img.shape[:2]
    min_pixels = 0.001 * h * w                      # reject barely-visible damage
    for _ in range(20):
        out, dmg = ADDERS[dtype](img, mask, rng)
        m = dmg > 0
        if m.sum() < min_pixels:
            continue
        # contrast check: the damaged pixels must actually look different
        # (e.g. a dark burn on dark metal is invisible -> try again)
        change = np.abs(out - img.astype(np.float32)).mean(axis=2)[m].mean()
        if change >= MIN_CONTRAST:
            return photometric_jitter(out, rng), dmg
    return None, None          # couldn't draw visible damage: caller skips it


# ---------------------------------------------------------------------------
# 4. Run over a folder
# ---------------------------------------------------------------------------
def make_preview(examples, path, tile=220):
    """examples: list of [original, breakdown, ablation, fracture, groove]."""
    def lab(im, text):
        im = cv2.resize(im, (tile, tile))
        cv2.putText(im, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return im
    names = ["original"] + TYPES
    rows = []
    for ex in examples[:5]:
        tiles = [lab(im.copy(), n) for im, n in zip(ex, names) if im is not None]
        rows.append(np.hstack(tiles))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="folder of UNDAMAGED blade images")
    p.add_argument("--output", default="synthetic")
    p.add_argument("--per-image", type=int, default=4,
                   help="damaged versions per clean image (types cycle in turn)")
    p.add_argument("--clean-per-image", type=int, default=1,
                   help="jittered clean copies (label 0) per image")
    p.add_argument("--types", default=",".join(TYPES))
    p.add_argument("--size", type=int, default=512,
                   help="resize so the longest side is at most this many pixels")
    p.add_argument("--detector", choices=["ai", "classic"], default="ai",
                   help="ai = pretrained background removal (needs rembg); "
                        "classic = hand-made detector, no extra install")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    for t in types:
        if t not in ADDERS:
            raise SystemExit(f"Unknown type {t}; choose from {TYPES}")
    if args.detector == "ai":
        try:
            import rembg  # noqa: F401
        except ImportError:
            raise SystemExit('The AI blade detector needs rembg:\n'
                             '    python -m pip install "rembg[cpu]"\n'
                             'or run with --detector classic')
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in Path(args.input).rglob("*")
                   if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not files:
        raise SystemExit(f"No images found in {args.input}")

    rows, examples, skipped, checks = [], [], [], []
    no_damage = 0
    show = set(rng.choice(len(files), size=min(5, len(files)), replace=False).tolist())
    check = set(rng.choice(len(files), size=min(12, len(files)), replace=False).tolist())
    for idx, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        # AeBAD photos are ~3000px: shrink first (much faster; the model
        # only sees small images anyway). Real test images get the same resize.
        h0, w0 = img.shape[:2]
        if max(h0, w0) > args.size:
            sc = args.size / max(h0, w0)
            img = cv2.resize(img, (round(w0 * sc), round(h0 * sc)), interpolation=cv2.INTER_AREA)
        mask = blade_mask(img, args.detector)
        if mask is None:                            # couldn't find the blade:
            skipped.append(str(f))                  # skip rather than guess
            continue
        # include sub-folders in the name so e.g. background/0001.png and
        # view/0001.png don't overwrite each other
        rel = f.relative_to(args.input)
        tag = "_".join(rel.with_suffix("").parts)
        for k in range(args.per_image):
            dtype = types[(idx + k) % len(types)]
            fake, dmg = make_damaged(img, mask, dtype, rng)
            if fake is None:                        # never save a "damaged"
                no_damage += 1                      # label with no damage
                continue
            name = f"{tag}_{dtype}{k}.png"
            cv2.imwrite(str(out / "images" / name), fake)
            cv2.imwrite(str(out / "masks" / name), dmg)
            rows.append([name, 1, dtype, str(rel)])
        for k in range(args.clean_per_image):
            name = f"{tag}_clean{k}.png"
            cv2.imwrite(str(out / "images" / name), photometric_jitter(img, rng))
            cv2.imwrite(str(out / "masks" / name), np.zeros(img.shape[:2], np.uint8))
            rows.append([name, 0, "good", str(rel)])
        if idx % 20 == 0:
            print(f"  {idx + 1}/{len(files)} images processed")
        if idx in check:                            # outline of detected blade
            ov = img.copy()
            cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cs, -1, (0, 255, 0), 3)
            checks.append(cv2.resize(ov, (200, 200)))
        if idx in show:                             # one of each type for the preview
            ex = [img]
            for t in TYPES:
                d = make_damaged(img, mask, t, rng)[0] if t in types else None
                if d is None:                       # dropped: say so in the preview
                    d = np.full_like(img, 255)
                    cv2.putText(d, "dropped (not visible)", (10, d.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, d.shape[1] / 600, (0, 0, 255), 2)
                ex.append(d)
            examples.append(ex)

    with open(out / "labels.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["filename", "label", "defect_type", "source_image"])
        wr.writerows(rows)
    if examples:
        make_preview(examples, out / "preview.jpg")
    if checks:                                      # grid of blade outlines
        while len(checks) % 4:
            checks.append(np.full_like(checks[0], 255))
        grid = np.vstack([np.hstack(checks[i:i + 4]) for i in range(0, len(checks), 4)])
        cv2.imwrite(str(out / "blade_check.jpg"), grid)
    if skipped:
        (out / "skipped.txt").write_text("\n".join(skipped))
        print(f"Skipped {len(skipped)} images where the blade wasn't found (see skipped.txt)")
    if no_damage:
        print(f"Dropped {no_damage} damaged versions where the damage couldn't be drawn")
    n_dmg = sum(r[1] for r in rows)
    print(f"Done: {n_dmg} damaged + {len(rows) - n_dmg} clean images -> {out}/")
    print(f"Check {out}/preview.jpg and {out}/blade_check.jpg (green = detected blade)")


if __name__ == "__main__":
    main()