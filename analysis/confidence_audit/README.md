# Confidence audit and one fixed L2 geometry probe

[Report](report.html) · [Findings](findings.md) · [Protocol](protocol.md).

E42S/D64S/D65S follows the user's clarification that correctness, confidence and
explanations all matter. Ten fixed candidates are deduplicated; one added L2+8
probe uses known31 only. No calibration or competition utility is inferred.

As-run sequence (already complete; outputs refuse overwrite):

```sh
/opt/anaconda3/bin/python3 -m analysis.confidence_audit.probe
/opt/anaconda3/bin/python3 -m analysis.confidence_audit.audit
/opt/anaconda3/bin/python3 -m analysis.confidence_audit.build_report
/opt/anaconda3/bin/python3 -m analysis.confidence_audit.verify
```

The report uses a metadata-corrected derivative `output/explanations.csv`; its
separate correction receipt preserves the original family field and unchanged
contributions. Existing models/raws/E39/E41/submissions are immutable. A new study
needs a fresh declared folder/protocol. The current submission and QC are unchanged.
