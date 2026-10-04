# E39S — no-annotation classification audit

**Neither M1 nor M2 supports replacing frozen v2.** The fixed primary model change
reduces crop-held-out balanced accuracy; the additional appearance features do not
recover that loss. The two error types on the first drop remain informative, but
correcting one revealed example is development evidence, not a new validation win.

## What the new organiser information changes

The real electrode data were cropped from around 15 source images and organised
into artificial visual batches. This supersedes literal supplier-lot provenance;
Batch 3 remains the challenge reference. Around 15 source images does not identify
15 independent specimens. Parent-image IDs remain unavailable, so related crops may
cross validation folds. Every score below is a **crop-held-out development diagnostic**,
not source-held-out generalisation, independent significance or manufacturing truth.
E40S found no recoverable overlaps but cannot detect non-overlapping related crops.

## Fixed candidates, shared folds

All 31 known crops remain included (7/7/17). The protocol was fixed before scores,
after exploratory work and first-drop feedback. Complete E37S mask-dependency gates,
fold-local median imputation/scaling and training-only three-value C selection are
used for the new candidates. Logistic class balancing and LDA's equal priors define
a balanced target; scores are uncalibrated model outputs, not empirical posteriors.

| Candidate | Features | Correct crops | Balanced accuracy | Recall B1 / B2 / B3 | Balanced log loss |
|---|---:|---:|---:|---|---:|
| Ungated L1, frozen-procedure reference | 29 | 21/31 | 0.608 | 5/7 · 2/7 · 14/17 | 0.890 |
| Gated L1, matched comparator | 29 | 19/31 | 0.569 | 4/7 · 3/7 · 12/17 | 0.986 |
| **Gated L2 multinomial, M1 primary** | 29 | 17/31 | **0.445** | 3/7 · 1/7 · 13/17 | 0.962 |
| Gated shrinkage LDA, comparator | 29 | 17/31 | 0.445 | 3/7 · 1/7 · 13/17 | 1.900 |
| Appearance-only L2, comparator | 24 | 11/31 | 0.300 | 3/7 · 0/7 · 8/17 | 1.261 |
| **Gated + appearance L2, M2 incremental primary** | 53 | 17/31 | **0.473** | 3/7 · 2/7 · 12/17 | 1.092 |

L2 changes eight bets versus gated L1: three newly correct and five newly wrong.
Adding appearance to L2 changes eight: three newly correct and three newly wrong;
the balanced-accuracy increment is 0.028, with worse log loss and five Batch 3 crops
called Batch 2. This is not evidence to promote the feature family. Matched L1
out-of-fold scores reproduce E37S to numerical precision. No new IID p-values,
confidence intervals, feature search, winner selection or calibration are reported.

## Mask-independent image panel and nuisance checks

Twenty-four fixed descriptors pool nine 512 px measurement tiles in each of BSE,
ETD and Inlens: relative Gaussian band energies at 2/8/32 px, fine/coarse spatial
IQR and sigma-8 gradient axial moments/coherence. All 24 are available on all 34
supplied crops. Exact measured union coverage is **14.56–19.33%**; tiles/channels
are coverage rather than extra labelled observations. Image gradients do not
identify plate axes, and Gaussian bands are overlapping, not orthogonal power bins.

At fixed original trim/coordinates, nonclipping positive gain/offset changes cancel
numerically (maximum feature change 2.98e-16); appearance-only outer bets change on
**0/31** crops. Gamma 0.7 and 1.4 change **5/31 and 8/31** bets, respectively, with
maximum class-score movements 0.256 and 0.303. Step-8 quantisation changes **0/31**
bets, maximum score movement 0.0207. Descriptor median absolute movements under
the two gamma changes are 0.284/0.321 known-feature SD. These are paired descriptor
family checks, not augmentation, end-to-end old-KPI robustness or acquisition-free
material evidence. No chemical/phase, expert-mask or battery-performance validation.

The strongest absolute Spearman correlation of any new descriptor with existing
BSE correlation length is 0.492, and with FFT slope 0.385. This descriptive check
does not prove novelty, independence or material specificity. Six actual-image
panels show raw crops, fine Gaussian-band responses and sigma-8 gradient magnitude.

## Revealed failure diagnostics

All six fitted candidates use only the original 31 known labels. Only after saving
the known-data stage were the three already-revealed drop labels read. Ungated L1,
gated L1, gated L2 and LDA keep the original B1/B2 swaps (1/3 correct). Appearance-only
corrects `fn0mhxef` but loses the baseline crop (1/3). The augmented candidate corrects
`fn0mhxef` and retains `xrv9xvzb` (2/3); **low-contrast true-B2 `3e122cbj` remains
wrongly B1 under every candidate**. These are development failure diagnostics;
their outcomes neither select a model nor establish external accuracy.

Exact linear decision-function contributions, observed/imputed status and comparison
classes are saved for every outer prediction and drop diagnostic. They explain model
evidence, not physical causation. Imputed phase terms are unobserved inputs.

## Delivery / next step

The report, model/source receipts, all scores, confusions, contributions and paired
nuisance checks are saved here. Frozen v2 predictions/models and QC remain unchanged;
no 34-site submission fit or automatic promotion. The inventory adds 24 descriptors
as experimental image appearance, with human review pending and accuracy benefit
unsupported. Next bounded classification experiment: **M3, frozen DINOv2 supervised
crop representation**, with the same parent-dependence and nuisance caveats. Parent
mapping remains the priority for measuring generalisation to new source images.

Targeted model/extraction tests passed. Full-suite, artifact and register checks are
recorded in `output/evaluation/validation.json`. Scientific figures and response maps
are inspected separately; native browser layout remains unverified because browser
policy blocks local-file navigation. No real expert annotations were created.
