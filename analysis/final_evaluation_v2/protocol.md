# Final evaluation protocol · E44S / D70S

Date: 2026-10-04. Input: `Hackathon-Polaron-eval`.

Use the frozen `categoriser-v2-D52` classifier. It uses 29 morphology and appearance features. Its primary family ID is `material`. This ID does not establish the material origin of texture. Frame height and explicit session statistics are excluded from the primary family.

Verify the model, source and dependency hashes before inference. Require one matching BSE, ETD and Inlens image for each sample. Reject exact copies of known images as evaluation data. Preserve the original TIFF files.

Assign each sample to the class with the highest model score. Apply that assignment to all three detector files. Keep full precision in the CSV files. State that the scores are uncalibrated. Report the runner-up, score margin, quality flags and feature contributions.

Run the scorer in a fresh Python process. Use a new feature cache. Save first predictions before the report is built. Verify the explanations against the saved classifier. Preserve every existing prediction and experiment file.

Keep baseline morphology ranks separate from batch assignments. Do not pool this folder into one QC batch. The six samples can have different source batches. No true labels are available at the start of this run. Save future labels in a separate directory.

Do not fit models, change features, adjust scores or select comparators on this folder. The confidence experiment remains closed. No external submission is part of this task. Record the predictions under E44S after the files are sealed.

Commands from the repository root:

```bash
MPLCONFIGDIR=/private/tmp/polaron-v2-mpl /opt/anaconda3/bin/python3 -m analysis.submission_v2.score_folder --model-version categoriser-v2-D52 --input /Users/santoshkumarsaravanan/Documents/Polaron/Hackathon-Polaron-eval --out analysis/final_evaluation_v2/scored --cache-dir _scratch/final_evaluation_v2_features
MPLCONFIGDIR=/private/tmp/polaron-v2-mpl /opt/anaconda3/bin/python3 -m analysis.submission_test.compose_submission --model-version categoriser-v2-D52 --primary-family material --run analysis/final_evaluation_v2/scored --out analysis/final_evaluation_v2/submission
MPLCONFIGDIR=/private/tmp/polaron-v2-mpl /opt/anaconda3/bin/python3 -m analysis.submission_v2.build_report --run analysis/final_evaluation_v2/scored --out analysis/final_evaluation_v2/submission
```

The commands use the preserved runner and renderer. Add a short STE handoff beside the original generated report. Do not edit the sealed score files to change display text.
