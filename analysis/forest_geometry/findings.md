# E41S: fixed Random Forest × geometry comparison

The bounded forest and compact geometry addition do not improve classification.
Retain the current submission; neither experiment candidate is promoted (D63S).
This tests the specific fixed settings and eight descriptors, not every possible
forest or morphology representation. No SME annotation was used.

The organiser assembled artificial visual groups from crops of around 15 electrode
source images. Parent IDs are unavailable, so these crop holdouts can share source
images across folds. They are development diagnostics, not new-source accuracy,
IID significance or manufacturing outcome validation. The three revealed drop
labels are failure diagnostics, not an external test.

## Fixed comparison

All 31 known crops (7/7/17 labels) are retained. The four factorial candidates use
the same complete phase-quality gate, content ordering, outer crop holdouts and
training-fold median imputation. L1 scaling/C selection is training-only. Forest
settings are 500 trees, depth 3, leaf minimum 4, sqrt feature sampling, balanced
classes, bootstrap and seed 0. No search or winner selection follows these results.

| Candidate | Inputs | Correct | Balanced accuracy | Recall B1 / B2 / B3 | Balanced log loss |
|---|---:|---:|---:|---|---:|
| Ungated L1, preserved procedure reference | 29 | 21/31 | 0.607843 | 5/7 · 2/7 · 14/17 | 0.889949 |
| Gated base L1 | 29 | 19/31 | 0.568627 | 4/7 · 3/7 · 12/17 | 0.986486 |
| Gated base forest | 29 | 18/31 | 0.464986 | 2/7 · 2/7 · 14/17 | 1.015598 |
| Gated geometry L1 | 37 | 18/31 | 0.521008 | 5/7 · 1/7 · 12/17 | 1.090002 |
| **Gated geometry forest, declared primary** | **37** | **17/31** | **0.417367** | **2/7 · 1/7 · 14/17** | **1.032144** |
| Quality-only forest, acquisition control | 2 | 8/31 | 0.380952 | 3/7 · 5/7 · 0/17 | 1.097458 |
| Availability-only forest, acquisition control | 37 bits | 7/31 | 0.333333 | 0/7 · 7/7 · 0/17 | 1.087983 |

Changing gated L1 to the forest loses 0.103641 balanced accuracy; adding the eight
measurements to that forest loses another 0.047619. Geometry changes exactly one
base-forest bet, turning a correct B2 crop into B1. Both forests preserve more B3
recall at a cost to minority groups. Adding geometry to L1 fixes two crops but
breaks three; B2 recall falls from 3/7 to 1/7. Weak quality/availability controls
do not rule out remaining acquisition shortcuts in numerical appearance inputs.

## Measurements and sensitivity

The eight reuse exact shape spread, solidity lower tail, median/upper-tail void
width, centroid-spacing variability, 512px bright-window dispersion and 256px
directional phase-association definitions. Raw replay reproduces all 32 values
on four fixed examples. Usable counts are 29/29/29/27/27/29/29/25 of 31, or 224/248
cells. Phase-quality and computational coverage gates precede imputation; unknown
quality fails closed. Counts are coverage, not independent observations. Widths
are pixels, bright fragments are unreviewed, and graph proximity is not contact.

Only the new eight inputs are perturbed; old 29 remain fixed. The original fits
change 0/31 geometry-forest bets under either threshold shift (−5/+5), and 1/31
under the bright-object floor change to 100px². Largest score moves are
0.039777/0.042574/0.040143 respectively. Geometry L1 changes 2/31, 1/31 and 2/31
bets, largest moves 0.086366/0.298867/0.092951. These are partial-input mask
sensitivity checks, not whole-pipeline radiometric invariance or mask accuracy.
Nominal-unavailable cells remain unavailable. Raster dispersion/correlation and
the fixed void floor are invariant to the bright floor by definition.

Primary seeds 1 and 2 change no bets and both score 0.417367 balanced accuracy;
seed 0 remains primary. Stability of poor predictions is not correctness confidence.
Replacing one observed input family by its training median changes 6/31 bets for
existing pore inputs, 3/31 for particle texture, 2/31 for existing bright inputs,
and 1/31 each for raw BSE anchors and new spacing. Other new families change none.
This is fixed-model sensitivity: correlated hybrid inputs can be implausible.
It is not refit ablation, causal importance or independent feature validity.

## Revealed-drop failures and explanations

Every main candidate retains the same B1/B2 swaps and correct B3 bet: 1/3 on the
revealed development drop. The low-contrast true-B2 crop has only 2/8 usable extra
measurements (widths), giving 12/37 observed numerical inputs. The geometry-forest
B1 score is 0.520628; its true-B2 score is 0.259383. Several leading path terms
are imputed Inlens inputs, explicitly marked unobserved. They are fitted assumptions,
not measured sample texture. The other mistake has B2 score 0.428062 versus true
B1 0.389146. Model probabilities are uncalibrated balanced-target scores, not the
probability the assignment is correct.

Forest terms average tree-root probabilities plus per-split probability changes;
their class difference exactly reconstructs the declared chosen-versus-B3 margin
(or chosen-versus-runner-up when the bet is B3). This is noncausal, path-order and
correlation dependent, not SHAP. L1 terms reconstruct linear decision margins;
their units differ. Four actual geometry panels show measurement definitions and
per-example quality exclusions in the [HTML report](report.html).

## Verification and next step

Independent review replays all 217 held-out fits within 9.7e-17, checks actual
train-fold imputation/scaling and forest settings, and reconstructs all explanation
margins within 2.0e-15. Known fits and their receipt were saved before raw drop
extraction/truth loading. All 102 raw TIFFs and 241 protected original/prior
experiment artifacts remain unchanged. Rejected numerical values, row/ID ordering,
class mapping, saved fits, report links/images and registry prefix are delivery
checks. Full suite: 232 tests pass, 15 existing warnings, 199.94s.

An early run was interrupted after two L1 comparators when a newly added test
assertion compared DataFrames incorrectly; that partial trial is preserved. Only
the assertion was corrected; model/settings stayed fixed. The complete trial
uses a fresh directory. A bare pytest command collected archived source snapshots;
the documented full-suite command `pytest tests -q` passes.

Existing eight inventory entries acquire this evidence, with no new duplicate KPI,
expert confirmation or production role. Frozen v2 and QC remain unchanged. Next
bounded classification experiment is the separately proposed supervised head on
frozen DINO features (M3), subject to actual cache/pooling provenance and the same
dependence and nuisance checks. Genuine parent IDs remain necessary for source
holdouts. Browser policy blocks local-file navigation; scientific images and HTML
links are verified, while native browser layout remains unverified.

[Protocol](protocol.md) · [Geometry extraction](protocol_geometry.md) ·
[Results](output/evaluation/known_summary.csv) ·
[Receipt](output/evaluation/receipt.json) ·
[Delivery checks](output/evaluation/delivery_validation.json)
