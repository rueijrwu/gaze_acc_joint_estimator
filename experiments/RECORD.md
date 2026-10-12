# Experiment record

Updated 2026-10-12. The active experiment is the controlled shared P4 template-size comparison. [CURRENT_STATUS.md](../CURRENT_STATUS.md) is the concise user-facing summary; the active family contains the immutable fit artifacts, implementation snapshots and audit trail.

## Active result: free shared template size

The active family is [p1_scale_barrel_first_free_size](p1_scale_barrel_first_free_size/). It adds one shared numerical size coordinate `λ` to the centered P4 template while keeping the centroid, P1-derived physical `g`, Stage 01–04 calibration, capture radii and matched priors fixed. `λ` describes template geometry; it is not physical magnification.

| Pair | P4 forward / inverse all-row weighted RMS, fixed → free (px) | Held weighted RMS, fixed → free (px) | Free λ | A lower contacts, fixed → free |
|---|---:|---:|---:|---:|
| Primary, 0.50 D / 2× | 0.818739 / 0.984475 → 0.739213 / 0.924861 | 1.640546 → 1.501476 | 0.984661 | 3 → 28 |
| Robustness, 0.25 D / 1× | 1.059744 / 1.262006 → 0.844091 / 1.047138 | 1.870972 → 1.579695 | 0.987431 | 5 → 12 |

The temporal split has 66,268 training rows, 4,571 gap rows, 18,139 test rows and 54,417 omitted-vertex scores per pair. All 12 held-state certificates passed with zero invalid rows. Every capture-average held score improves, but 5/20 primary and 4/20 robustness cells regress. Largest reported changes are primary C1 +10° (+0.240 px) and robustness C1 +5° (+0.496 px). The complete cell comparisons and figures are in the [artifact index](p1_scale_barrel_first_free_size/report/REPORT.md).

The free primary fit retains a gaze-associated A pattern: near +10° minus near-zero fitted mean A is +0.07395 D in C1, −0.27675 D in C2, −0.38484 D in C3, and −0.27059 D in C4. Thus C2–C4 are lower at +10°, while C1 is not. Fixation order, nominal gaze, and capture/demand are confounded. This is a fitted-state pattern, not a causal gaze effect or a separately measured residual bias.

Across the jointly changed 0.25 D / 1× and 0.50 D / 2× settings, the fixed/free RMS change across the 20 fixation-mean A values is 0.097422 to 0.042241 D; after removing each capture offset it is 0.076329 to 0.041862 D. Because both anchor width and optical multiplier change, this is not an isolated width effect. The selected size minimizes the anchored objective, not the data-only component: the λ=0.979738 profile point has lower data loss but higher optical penalty than λ=0.984661. The local data-only rank is 17/18 and the slope/accommodation scale-null remains. The fit is a conditional geometric result; it does not establish physical accommodation or magnification. Independent output/equation checks passed; see the [audit narrative](p1_scale_barrel_first_free_size/audits/AUDIT.md) and [machine audit](p1_scale_barrel_first_free_size/audits/free_size_output_audit.json).

## Historical findings

| Earlier branch | Finding retained for context | Archive |
|---|---|---|
| Corrected barrel-first, fixed centered-energy template | The 0.25 D / 1× Stage 05 model had 1.870972 px weighted held RMS; data-only rank 16/17 and gaze-associated A remained. It is the paired fixed-size reference, not the current active model. | [Historical family](archive/reverse_transform/historical_families/p1_scale_barrel_first_p4/) |
| Stage 05 anchor / optical-width sensitivity | Six variants traded conditional held geometry, gaze/A patterns and lower-bound contacts. Capture-scope variants were not promoted; sensitivities do not identify a physiological anchor width. | [Historical family](archive/reverse_transform/historical_families/p1_scale_barrel_first_sensitivity/) |
| P1-scale/capturewise P4 and earlier learned-reference/order variants | These pre-correction branches used different reference/priors and do not isolate the barrel-order change. A learned-reference temporal result improved aggregate score but had 5/20 capture/fixation and 24/60 vertex-cell regressions. | [Historical family](archive/reverse_transform/historical_families/p4_gaze_accommodation_decoupling/); [superseded comparison](archive/reverse_transform/superseded_p1_scale_capturewise_p4/) |
| Shared-keystone S0 chain and early reverse-transform fits | Earlier conditional models retained gaze/fixation-associated A and substantial scale/state ambiguity. In S0, fitted mean A was lower at nominal +10° than zero in all captures; capture, demand and fixation order were confounded. | [Archived shared-keystone results](archive/reverse_transform/shared_keystone/); [earlier raw-keystone archive](archive/raw_keystone/); [pre-cleanup record](archive/reverse_transform/cleanup_2026_10_12/documents/EXPERIMENTS_RECORD.md) |
| Descriptive P4 geometry and scale diagnostics | Useful measurements of fixation-dependent triangle geometry and P1 scale response, but no distortion-free reference or independent accommodation truth. | [Superseded baseline archive](archive/reverse_transform/superseded_baseline/) |

## Provenance and navigation

The active-family [README](p1_scale_barrel_first_free_size/README.md) is the reproduction/navigation guide. The fixed run plan and its hashes, imported artifact manifests, source snapshots, arrays, summaries, protocols and audit evidence are retained under that family. Its [cleanup relocation map](p1_scale_barrel_first_free_size/provenance/relocation_manifest.json) documents moved family paths and immutable embedded provenance paths; [PRE_CLEANUP_BACKUPS.json](archive/reverse_transform/cleanup_2026_10_12/PRE_CLEANUP_BACKUPS.json) records the archived pre-cleanup documents and exact hashes. Raw reviewed data and shared implementation libraries remain in their original locations.
