"""Append E39S appearance descriptors without promoting existing metric roles."""
import json
from pathlib import Path

from .appearance import CHANNELS, SUFFIXES

ROOT=Path(__file__).resolve().parents[2]
REGISTER=ROOT/"analysis/morphology/metric_register.json"
DEFINITIONS={
    "dog_fine_fraction_median":("Fine Gaussian-band energy fraction", "Median across nine tiles of mean((G2-G8)^2) divided by the sum of fine/mid/coarse band energies."),
    "dog_mid_fraction_median":("Mid Gaussian-band energy fraction", "Median across nine tiles of mean((G8-G32)^2) divided by the sum of fine/mid/coarse band energies."),
    "dog_coarse_fraction_median":("Coarse Gaussian-band energy fraction", "Median across nine tiles of mean((G32-mean(G32))^2) divided by the sum of fine/mid/coarse band energies, centring within the measurement tile."),
    "dog_fine_fraction_iqr":("Spatial variation of fine-band fraction", "75th minus 25th percentile of the nine tile fine Gaussian-band energy fractions."),
    "dog_coarse_fraction_iqr":("Spatial variation of coarse-band fraction", "75th minus 25th percentile of the nine tile coarse Gaussian-band energy fractions."),
    "gradient_axis_cos2_median":("Image-gradient axial cosine moment", "Median across nine tiles of (mean(gx^2)-mean(gy^2))/(mean(gx^2)+mean(gy^2)), with sigma8 Gaussian derivatives."),
    "gradient_axis_sin2_median":("Image-gradient axial sine moment", "Median across nine tiles of 2*mean(gx*gy)/(mean(gx^2)+mean(gy^2)), with sigma8 Gaussian derivatives."),
    "gradient_axis_coherence_median":("Image-gradient axial coherence", "Median across nine tiles of sqrt(cos2^2+sin2^2), with sigma8 Gaussian-derivative energy-weighted moments."),
}


def main():
    data=json.loads(REGISTER.read_text())
    data["updated"]="2026-10-04"
    data["dataset_construction"]={"date":"2026-10-04","source":"Organiser via Santosh","decision":"D59S",
        "statement":"Real electrode images cropped from around 15 source images and organised into artificial batches sharing visual features.",
        "interpretation":"The batches are curated visual groups, not verified supplier lots. Batch 3 remains the challenge reference. Parent-image IDs and independent specimen counts are unknown; related crops may cross site folds. Around 15 source images is not 15 confirmed independent specimens."}
    baseline=data["supplier_baseline_confirmation"]
    baseline.setdefault("history",[])
    if not any(h.get("decision")=="D59S" for h in baseline["history"]):
        baseline["history"].append({"date":"2026-10-04","decision":"D59S","previous_statement":baseline["statement"],
            "reason":"New organiser clarification supersedes literal supplier-lot provenance, preserving the reference rule."})
    baseline["statement"]="Batch 3 is the promised challenge reference; the organiser confirms the three folders are artificial visual batches constructed from cropped real electrode images, not verified supplier lots."
    for metric in data["metrics"]:
        note="D59S: sites are crops from around 15 source images assigned to artificial visual groups. Unknown shared-parent dependence qualifies historical/new CV, permutation and resampling; crop count is not verified independent n."
        if note not in metric["limitations"]: metric["limitations"].append(note)
    existing={entry["id"] for entry in data["metrics"]}
    for channel in CHANNELS:
        for suffix in SUFFIXES:
            key=f"appearance_{channel.lower()}_{suffix}"
            if key in existing: continue
            label,definition=DEFINITIONS[suffix]
            data["metrics"].append({"id":key,"label":f"{channel} {label.lower()}","family":"Mask-independent image appearance",
                "keys":[key],"channels":[channel],"units":"dimensionless",
                "definition":definition+" Fixed 3x3 grid of 512px measurement tiles; 96px Gaussian context, truncate3; contextual centring/standardisation; original BSE band trim shared across channels. Any invalid tile makes channel outputs unavailable.",
                "interpretation":"Acquisition-sensitive image signature without phase masks; image axes and pixel-scale texture, not confirmed plate/coating axes or chemistry.",
                "implementation_status":"computed","evidence_status":"acquisition_sensitive_development_audit","qc_role":"experimental_classification_only",
                "limitations":["Nine windows provide coverage, not independent samples; measured union covers 14.6-19.3% of supplied fields.",
                    "Affine intensity invariance holds only for fixed trim and positive gain/offset without clipping/rounding; gamma changes these descriptors.",
                    "Gaussian bands overlap and are not orthogonal power bins. Gradient direction differs from elongated-object axes.",
                    "No phase chemistry, human mask/instance accuracy or battery-performance validation. No QC/submission promotion; E39S accuracy benefit unsupported.",
                    "Crops share around 15 unknown source images and artificial visual labels; crop-LOO is not source-independent validation."],
                "evidence_paths":["analysis/classification_m1_m2/output/appearance_features.csv","analysis/classification_m1_m2/output/appearance_definition.json",
                    "analysis/classification_m1_m2/report.html","analysis/classification_m1_m2/output/evaluation/known_summary.csv"],
                "experiments":["E39S"],"visual_kind":"m2_appearance","review_status":"unreviewed","review_notes":"No human SME available; classification success cannot validate physical interpretation.",
                "history":[{"date":"2026-10-04","status":"computed","reason":"Fixed before E39S scores; 34 crop vectors, paired nuisance tests and actual response maps. No benefit demonstrated; separate experimental appearance role."}]})
    if not any(entry["id"]=="classification_m1_m2_E39S" for entry in data["methods"]):
        data["methods"].append({"id":"classification_m1_m2_E39S","label":"M1/M2 no-annotation classifier/appearance audit",
            "implementation_status":"computed","evidence_status":"accuracy_gain_not_established","qc_role":"experimental_only",
            "description":"Six fixed nested crop-LOO candidates with complete phase dependency gating; 24 image-signature descriptors, response maps and all31-crop intensity challenges. Unknown parent leakage; current submission/QC unchanged.",
            "experiments":["E39S"],"evidence_paths":["analysis/classification_m1_m2/report.html"]})
    if "analysis/classification_m1_m2/output/appearance_features.csv" not in data["sources"]:
        data["sources"].append("analysis/classification_m1_m2/output/appearance_features.csv")
    REGISTER.write_text(json.dumps(data,indent=2)+"\n")


if __name__=="__main__": main()
