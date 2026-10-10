# External distortion-tracking evidence inventory

## Source identity and retrieval

The exact repository name `rueijrwu/distortion_tracker` was not available: the read-only GitHub lookup returned “Repository not found” / “Could not resolve to a Repository.” The accessible similarly named repository is [`rueijrwu/distortion_tracking`](https://github.com/rueijrwu/distortion_tracking).

Its pinned source revision is [`1d2a0874c79f3f174b36a197c3fbbffd8a32fe60`](https://github.com/rueijrwu/distortion_tracking/tree/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60) (2026-10-09, “docs: update distortion model summaries to include P4 Z magnification study details”). Read-only lookup found current `main` at [`7b3f28e7a7817aa701c84d1ecba13dbdd229d96d`](https://github.com/rueijrwu/distortion_tracking/tree/7b3f28e7a7817aa701c84d1ecba13dbdd229d96d) (2026-10-09, “Remove obsolete sequence files and scripts; update p1_finite.seq for consistency in parameters and structure”). Between these revisions, the P1/P4 data, `Theory.md`, `Summary.md`, and `Script/raytracing.py` are unchanged; the cleanup removes old sequence files and `Script/distortion_grid.py`, and renames/updates other lens files. Relevant blobs are identical at both revisions, including `Script/raytracing.py` (`c03b0c6521bad8bc1a49bd676fa42367e4403903`). No newer centroid-position data were added in this revision.

The checkout is a temporary read-only inventory at `/tmp/distortion_tracking_inventory`; it is detached at the pinned revision. No external-repository files were modified. The repository’s `AGENTS.md` requires a Windows-specific Python executable for its scripts; that executable is unavailable in this Linux environment, and no repository scripts or CODE V sessions were run.

## What the committed grids measure

The [theory](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Theory.md#L15-L30) defines the P1/P4 coordinates as relative to the center ray at the same state. The P1 grid collector [traces the center ray and subtracts it from each grid ray](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Script/distortion_grid.py#L113-L143); the P4 collector calls that same function. Thus the saved 3×3 pattern coordinates describe center-ray-relative shape. Their nine-point means are pattern centroids, not the absolute image positions of the P1/P4 chief rays or fitted optical origins.

The committed P1 CSV header is:

`eye_rotation_deg,z_mm,field_x_relative,field_y_relative,paraxial_x_mm,paraxial_y_mm,real_x_mm,real_y_mm,radial_distortion_pct,tangential_distortion_pct`

It contains 184,059 rows; the `z_mm=0` slice has 3,609 rows (401 rotations × 9 fields). Metadata records P1 model SHA-256 `21406BCF60699186469B18292DE848698C235B1DC2D16032ABC2171688F01515` and absolute `Cornea_ENT_D` THI Z from −5 to +5 mm.

The committed P4 structured pickle contains 184,059 rows with fields `eye_rotation_deg`, `accommodation_d`, `field_x_relative`, `field_y_relative`, `paraxial_x_mm`, `paraxial_y_mm`, `real_x_mm`, `real_y_mm`, `radial_distortion_pct`, and `tangential_distortion_pct`. It spans 0–5 D in 0.1 D steps and −20° to +20° in 0.1° steps. Metadata records P4 model SHA-256 `5e0715c70016fbcec8a49956dd2ec824a50f5c4c52ca472b0225a2523db614ac`. The P1 and P4 coordinates are millimeters from separate optical setups and model files; they must not be treated as a shared absolute coordinate frame. `Summary.md` also warns that the separate P4 Z study uses a different lens revision.

## Separate uncentered position export named by source

`Script/raytracing.py` contains a distinct path that traces uncentered chief-ray positions: [sweep dimensions and arrays](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Script/raytracing.py#L113-L129), [P1/P4 output samples](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Script/raytracing.py#L191-L217), and [saved NPZ fields](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Script/raytracing.py#L223-L235). It names the output `p14_sigma_atch_perfect_src_z.npz`, uses 101 rotations from −20° to +20°, 51 accommodations from 0 to 5 D, and 21 image-surface Z positions from +1 to −1 mm. It stores `data_p11`, `data_p12`, `data_p10`, `data_p41`, `data_p42`, and `data_p40`; the P4 records explicitly request the CODE V `y` output coordinate.

Neither `p14_sigma_atch_perfect_src_z.npz` nor its source lens `p14_sigma_atch_perfect_src.len` is present in the pinned or current Git tree or under `/home/aplab`. The source mentions the export but does not commit the generated file. Consequently, the committed centered grids cannot establish the requested absolute P4-versus-P1 center shift or verify a 50-arcmin bound.

If that NPZ is later supplied, preserve its signed source axes and units and calculate, at each matched `(theta,Z)`, `c4 = mean(data_p41, data_p42, data_p40)`, `c1 = mean(data_p11, data_p12, data_p10)`, and `H = c4 - c1`. Compare `H(A=4)-H(A=0)` at fixed theta/Z with the matched A=0 gaze response/sensitivity. A local ratio `ΔH / (dH/dtheta)` is a local sensitivity diagnostic, not an exact inverse angle for large changes. Do not convert these source coordinates to ACC pixels without a documented calibration, and do not treat a centroid as an optical origin.

## Read-only evidence files

- [Source summary](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Summary.md)
- [Model definitions and coordinate convention](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/Theory.md)
- [P4 Z magnification report](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/data/p4_z_magnification/p4_z_magnification.md)
- [P4 Z metadata](https://github.com/rueijrwu/distortion_tracking/blob/1d2a0874c79f3f174b36a197c3fbbffd8a32fe60/data/p4_z_magnification/metadata.json)
