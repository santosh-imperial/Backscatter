# Measured qualification · task J / E28J

Retain the response maps for exploratory BSE appearance review. Defer all
three summaries as material/QC drivers. No evidence here establishes useful
information beyond existing geometry or acquisition variables.

The all-known Batch1−Batch3 coarse-energy-share difference is
-0.01691, site-bootstrap95% [-0.03304, -0.00590]
at n7/17 sites. In the quality-matched view it is
-0.01384 [-0.03049, +0.00730], n5/13.
All nine nominal quality-matched intervals include zero. Filtering changes
both the population and sample size; this does not prove the difference was
an acquisition artifact or that the batches are equivalent. These are
correlated exploratory contrasts, with no multiplicity-adjusted test.

Nominal versus full-resolution ranks have Spearman rho
0.987–0.994; median
absolute descriptor movement is 0.038–0.047
of the pooled between-site IQR. Gamma changes move some sites more; complete
paired values and maxima are in `sensitivity_summary.csv`. The non-clipping
affine control agrees to numerical precision, as expected by construction.
This is not evidence that real contrast loss or preparation is corrected.

All twelve acquisition and FFT/geometry/tensor site-LOO ridge R² results are
negative. That means these prespecified linear controls predict poorly at
this n; it does not establish new information or acquisition invariance.
Strong within-batch correlations remain in the correlation table, with
small group counts and different sampled versus full-frame measurands.

Valid convolution interiors cover only 2.53–3.63%
of each trimmed frame. Four windows are coverage, not independent samples;
spatial representativeness and specimen independence remain unvalidated.
Wavevectors are normal to image stripes, not graphite plate-instance axes.
Expert phase validity, battery mechanisms, manufacturing outcomes and
unseen generalisation remain open. Primary/classifier/verdict inputs are
unchanged.

Eight mathematical convention/invariance tests pass. They do not validate
material identity or independent segmentation/defect accuracy.
Review: C01–C07/C09/C11/C13/C14/C16/C17/C21–C23/C28–C30.
