# Synthetic damage for aero-engine blade inspection

Can cheaply generated synthetic damage stand in for real defect photos when training a damage detector? This project tests that on the AeBAD-S aero-engine blade dataset.

## Method

- **Generator (`synthetic_damage.py`).** A procedural, classical image-processing pipeline. It locates the blade with a pretrained U²-Net segmentation model (via `rembg`), then draws the four AeBAD-S damage types onto real defect-free blade photos:
  - **Breakdown:** holes punched through the blade, with a dark torn rim
  - **Ablation:** dark, sooty burn patches (coating loss)
  - **Fracture:** a chunk missing from the blade edge, filled by local inpainting
  - **Groove:** notches cut into the edge, plus thin scratch lines

  Defect masks are random polygons and random-walk shapes, blended in with feathered edges, darkening and grain matched to each photo's own noise. A contrast check rejects damage that doesn't visibly change the pixels.
- **Dataset produced:** 1,787 synthetic damaged images from 457 undamaged AeBAD-S training images (4 variants each). 64 source images were skipped where the blade couldn't be located reliably, and 41 variants were dropped as not visible enough. 457 lightly jittered clean copies serve as the negative class.
- **Detector:** an ImageNet-pretrained ResNet-18, fine-tuned as a binary "damaged vs good" classifier. **The test set is always 100% real AeBAD images.**

## Results

Recall and precision on real damage, averaged over 3 random seeds. N is the number of real damaged images in training.

| N real | Synthetic + real: recall | Synthetic + real: precision | Real only: recall | Real only: precision |
|---:|---:|---:|---:|---:|
| 0 | 0.42 | 0.90 | – | – |
| 10 | 0.46 | 0.93 | 0.67 | 0.79 |
| 25 | 0.55 | 0.91 | 0.82 | 0.77 |
| 50 | 0.62 | 0.88 | 0.92 | 0.74 |
| 100 | 0.77 | 0.84 | 0.97 | 0.74 |

- In all 12 seed × N comparisons, adding the synthetic set **lowered recall** and **raised precision** compared with training on the same real images alone. The synthetic data makes the model more cautious rather than better.
- Per damage type (preliminary, single seed), simple geometric damage transferred well (breakdown recall ≈ 0.95) and subtle edge and texture damage did not (fracture ≈ 0.24, groove ≈ 0.17).


## Lessons from building the generator

Synthesis artefacts are a real risk: a detector can score well by spotting the generator's fingerprints instead of damage. Two examples found and fixed:

- Early fracture fills left a faint "ghost line" along the blade's old edge, with a fill colour that didn't match the background. Fixed with local inpainting, noise-matched grain and a feathered seam.
- Dark damage on dark blades was labelled "damaged" while barely changing any pixels. Fixed with a contrast check.

## Next steps

- Compare models with a threshold-independent metric (e.g. average precision) to tell a genuine loss from a shifted decision threshold.
- Re-run the per-type analysis across seeds.
- Test what closes the realism gap: better blending, re-weighting real images, or generative models.

## Data

AeBAD is **not** included in this repository. Download it from the original authors' release, and see their licence before use.
