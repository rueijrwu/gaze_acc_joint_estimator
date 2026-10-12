# Shared P4 template-size comparison

## Navigation

- [Canonical scientific record](../RECORD.md) and [current status](../../CURRENT_STATUS.md)
- [Artifact index: plots, tables and audit links](report/REPORT.md)
- [Independent audit narrative](audits/AUDIT.md) and [machine audit](audits/free_size_output_audit.json)
- [Frozen run plan](provenance/plans/PLAN_AT_RUN.md), [import manifests](provenance/imports/), and [archived run history](../archive/reverse_transform/cleanup_2026_10_12/pruned_family/provenance/RUN_LOGS.md)

The family adds one shared numerical size coordinate for the centered P4 template while retaining the fixed centroid, physical P1-derived `g`, capture radii and corrected Stage 01–04 calibration. The paired fixed-size controls are preserved as inputs. The reported `λ` is a template-geometry parameter, not physical magnification.

## Artifact layout

- `scripts/`: family path helpers, model, production runner, held validation, conditional size profile, reporting and audit utilities.
- `inputs/production/` and `inputs/training/`: immutable Stage 01–04 bundles for production and temporal training.
- `inputs/fixed_controls/{production,training,held}/`: immutable matched fixed-size controls.
- `results/{production,training}/`: free-size Stage 05 fits.
- `validation/temporal/`: held omitted-vertex outputs for both pairs.
- `validation/template_size_profile/`: primary conditional profile.
- `report/`: generated Markdown index, figures and derived CSV summaries.
- `audits/` and `provenance/`: canonical audit evidence, frozen plan, imports, one hash-verified fit-time source snapshot, and relocation metadata.

## Reproduction

The saved production commands and exact settings are in the [archived run log](../archive/reverse_transform/cleanup_2026_10_12/pruned_family/provenance/RUN_LOGS.md). The Stage 05 runner is `python scripts/run.py --backend cupy --case all --starts 2 --maxiter 150 --state-maxiter 400`; temporal scoring, profile generation and report generation use `scripts/validation.py`, `scripts/profile.py`, and `scripts/report_free_size.py`. This is a replay from the copied Stage 01–04 bundles, not a fresh end-to-end Stage 01/02 calibration; historical calibration outputs and duplicate source snapshots are preserved in the archive.

For an optional replay in a copied family, first rebase that copy’s `provenance/relocation_manifest.json` `path_aliases` to the copied family’s own input paths. Preserve the copied Stage 01–04 inputs and all fixed controls; clear only copied free-fit outputs under `results/{production,training}/`, copied held outputs under `validation/temporal/`, and copied profile outputs under `validation/template_size_profile/` before starting. Do not edit frozen source snapshots, summaries or protocol metadata to make a copy appear portable. The original source/import manifests keep their historical path strings; the [relocation map](provenance/relocation_manifest.json) documents their current destinations. Raw detections and shared implementation libraries remain at repository `data/` and `distortion_model/`.

`provenance/plans/PLAN_AT_RUN.md` is a byte-preserved run-time snapshot. Its historical repository-root link strings are intentionally unchanged; use the relocation map to resolve moved paths.
