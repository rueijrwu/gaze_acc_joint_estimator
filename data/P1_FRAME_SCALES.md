# P1 frame scales

`p1_frame_scales.pkl` contains 88,978 per-frame P1 similarity scales from the [global mean-reference experiment](../experiments/archive/p1_mean_reference_similarity_scale/README.md). One reference is the equal-weight mean of the 20 centered P1 fixation patterns (four captures × five gazes).

```python
import pickle

with open("data/p1_frame_scales.pkl", "rb") as stream:
    scales = pickle.load(stream)

g = scales["p1_g"]
```

Each array is aligned in saved source-frame order. Match later-stage frames by the pair `capture`, `source_row`; `source_frame` and `timestamp_ms` are also preserved. Additional labels are `fixation` (0..4) and `target_theta_deg` (−10°, −5°, 0°, +5°, +10°). The export includes calibration captures 1–4 and their fixation frames.

`p1_g` is a positive, unclipped, dimensionless scale mapping reference size to measured frame size. Divide centered measured coordinates by `p1_g` to remove scale. The original similarity fit also estimated rotation and translation; those are retained in the [experiment arrays](../experiments/archive/p1_mean_reference_similarity_scale/results/arrays.npz).

`reference_vertices_px` contains the single reference triangle. `metadata` records the reference convention, frame key, schema version, and source hashes.

The original exporter and fit are retained in the linked archive. The pickle remains the reusable scale dataset; its metadata retains the original source paths and hashes.
