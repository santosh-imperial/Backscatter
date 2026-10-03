# Fixed bright-object graph audit (E26G / task E)

Known-batch development. Proximity edges are geometric constructions, not electrical or physical contacts. Expert masks and specimen independence remain unconfirmed. No primary KPI, classifier or verdict input changes.

Three nearest distinct centroid neighbours, undirected union; 8-connected bright masks after existing smoothing/opening; component floor 50 px², clipped objects excluded. Sensitivity repeats all sites at thresholds −5/0/+5 and area floors 50/100 px². These settings were fixed before new outputs. At least 10 nodes/10 edges is a pilot coverage gate, not additional statistical n.

Bright-usable excludes observed low contrast; quality-matched additionally excludes grey-pore preparation flags. Ordinary-reference sensitivity additionally removes known long-void reference sites, without declaring a clean baseline. Grey-pore is not required for a bright-only measurand. Missing flags abstain; unresolved pore-mode diagnostics remain disclosed.

Intervals are 4000-resample percentile site-bootstrap intervals, conditional on provisional site independence. Threshold/object-floor sensitivity and deletion ranges are separate from sampling uncertainty. No p-values, classifier AUC or manufacturing defect accuracy were estimated.

## Bright-usable nominal comparisons

| descriptor | incoming − reference | n incoming / reference | median difference | site-bootstrap 95% interval |
|---|---|---|---|---|
| `graph_edge_length_median_px` | Batch_1 − Batch_3 | 5/17 | -4.5385 | [-37.108, 29.44] |
| `graph_edge_length_median_px` | Batch_2 − Batch_3 | 7/17 | 16.525 | [-53.831, 65.485] |
| `graph_edge_length_median_px` | Batch_2 − Batch_1 | 7/5 | 21.063 | [-53.619, 67.032] |
| `graph_edge_length_iqr_ratio` | Batch_1 − Batch_3 | 5/17 | -0.027692 | [-0.18225, 0.21389] |
| `graph_edge_length_iqr_ratio` | Batch_2 − Batch_3 | 7/17 | -0.14915 | [-0.29566, 0.16694] |
| `graph_edge_length_iqr_ratio` | Batch_2 − Batch_1 | 7/5 | -0.12146 | [-0.27784, 0.13805] |
| `graph_edge_axial_strength` | Batch_1 − Batch_3 | 5/17 | -0.0093737 | [-0.070089, 0.066384] |
| `graph_edge_axial_strength` | Batch_2 − Batch_3 | 7/17 | 0.027639 | [-0.064087, 0.097382] |
| `graph_edge_axial_strength` | Batch_2 − Batch_1 | 7/5 | 0.037013 | [-0.084355, 0.11653] |
| `graph_log_area_assortativity` | Batch_1 − Batch_3 | 5/17 | -0.063479 | [-0.1142, 0.014204] |
| `graph_log_area_assortativity` | Batch_2 − Batch_3 | 7/17 | 0.021676 | [-0.070929, 0.079785] |
| `graph_log_area_assortativity` | Batch_2 − Batch_1 | 7/5 | 0.085155 | [-0.030154, 0.13103] |

## Sensitivity and baseline controls

All pairwise views and six threshold/floor variants are saved in comparisons.csv. Acquisition and node/loading-only ridge controls use alpha=1 with imputation and scaling refitted inside each held-out-site fold. Within-batch correlations accompany pooled results. Positive predictability suggests redundancy/confounding; poor prediction does not establish novelty or acquisition invariance.

| descriptor | acquisition LOO R² | node/loading LOO R² | strongest within-batch acquisition correlation |
|---|---|---|---|
| `graph_edge_length_median_px` | -0.147 | 0.496 | Batch_2/H: rho=-0.955, n=7 |
| `graph_edge_length_iqr_ratio` | -0.406 | 0.088 | Batch_1/etd_boundary_sharpness: rho=1.000, n=5 |
| `graph_edge_axial_strength` | -0.362 | -0.635 | Batch_1/bright_sep: rho=0.718, n=5 |
| `graph_log_area_assortativity` | -0.152 | -0.289 | Batch_1/bse_std: rho=-0.800, n=5 |

## Retain/defer decision

Retain the fixed graph as an exploratory image-review tool. Defer all four descriptors as material/QC drivers until independent component review, sensitivity and specimen/process validation. No GNN, learned defect classifier or battery-mechanism advantage is claimed. Finite-frame neighbour bias remains despite removing clipped objects; object counts and edges do not increase n.

The four overlay sites were prespecified, with crops fixed at frame centre rather than chosen for separation. Full-frame graphs are cropped only for display; values come from full-site tables. Both clipping and sub-floor exclusions are recorded in coverage fields. Node/edge source tables include exact coordinates and component labels.

Pre-presentation checks: C01–C06/C09/C13/C14/C16/C17/C21/C22/C28–C30. Syntax and graph tests validate conventions, not expert segmentation accuracy or unseen-batch generalisation.
