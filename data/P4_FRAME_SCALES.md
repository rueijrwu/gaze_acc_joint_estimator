# P4 frame scales

`p4_frame_scales.pkl` contains 88,978 per-frame P4 similarity scales from the [P4 global mean-reference experiment](../experiments/archive/p4_mean_reference_similarity_scale/README.md). One fixed reference is the equal-weight mean of 20 centered P4 fixation patterns (four captures × five gazes).

```python
import pickle

with open("data/p4_frame_scales.pkl", "rb") as stream:
    scales = pickle.load(stream)

g = scales["p4_g"]
```

Match later-stage frames by `capture`, `source_row`. Arrays also preserve `source_frame`, `timestamp_ms`, `fixation` (0..4), and `target_theta_deg` (−10°, −5°, 0°, +5°, +10°). The export includes calibration captures 1–4 and their fixation frames. Frame identity and ordering match the [P1 scale export](P1_FRAME_SCALES.md), with alignment checked for every frame.

`p4_g` is a positive, unclipped, dimensionless scale mapping the fixed P4 reference to each measured P4 frame. Divide centered coordinates by this scale to remove scale. Rotation and translation from the similarity fit remain in the [experiment arrays](../experiments/archive/p4_mean_reference_similarity_scale/results/arrays.npz).

`reference_vertices_px` contains the global P4 reference. `metadata` records the reference convention, schema version, frame key, and source hashes. P4 scales use raw P4 triangles, before any P1 scale correction.

The original exporter and fit are retained in the linked archive. The pickle remains the reusable scale dataset; its metadata retains the original source paths and hashes.
