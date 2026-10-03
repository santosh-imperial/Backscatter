# Artifact examples

Collector candidate: Batch_2/img_epqdaau9 shows a continuous bright bottom layer in BSE and the matched ETD view. Morphology and BSE brightness are consistent with a current collector; copper identity requires acquisition/sample information or compositional analysis. This region should be excluded from electrode-only KPIs.

Histogram evidence below is calculated from full-resolution original TIFF intensities, after excluding a four-pixel colored export border. Counts use 256 bins, one per 8-bit gray level. Missing levels refer to the inclusive 1st–99th percentile interval.

- Batch_3/img_ufdvpb81_BSE: 57 empty levels within 3–117 (out of 115 levels). Missing examples: [4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 30, 31, 33].
- Batch_2/img_i9jiqjwl_BSE: 50 empty levels within 0–127 (out of 128 levels). Missing examples: [1, 4, 6, 9, 11, 14, 17, 19, 22, 24, 27, 29, 32, 34, 37, 40, 42, 45, 47, 50].
- Batch_3/img_71vgq3fw_BSE: 0 empty levels within 23–118 (out of 96 levels). Missing examples: [].

Regular skipped intensity levels support quantized intensity remapping, consistent with contrast stretching of a finite-bit-depth image. They cannot establish when or where the remapping occurred; acquisition electronics, export software and post-acquisition processing need to be distinguished using raw data and provenance. A continuous histogram does not prove absence of processing.

The previous report did not explicitly identify either collector inclusion or histogram combing. Its intensity, entropy, threshold and texture metrics remain exploratory and may be affected by these artifacts.
