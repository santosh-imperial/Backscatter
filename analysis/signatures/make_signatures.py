"""L4 — batch signatures against the Batch 3 baseline (docs/batch_signatures.md).

Run from the repository root:  /opt/anaconda3/bin/python3 analysis/signatures/make_signatures.py
Writes analysis/signatures/{batch_signatures.csv, family_counts.csv, fig_signature_strips.png, fig_signature_crops.png}.
Sites are n (7/7/17); low-contrast sites are excluded from bright-phase descriptors (stats.usable_values), grey-pore sites kept.
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from polaron_qc import PRIMARY_KPIS, MATERIAL_KPIS, KPI_TRUST, BATCH_COLORS, stats as ST, features as FE
OUT = os.path.join(ROOT, "analysis", "signatures")
ACQ = {"bse_p1", "bse_std", "bse_empty_bin_frac", "H", "bright_sep", "etd_boundary_sharpness", "etd_curtain_frac", "etd_curtain_anisotropy",
       "bse_p50", "etd_p50", "inlens_p50", "band_rows", "etd_empty_bin_frac", "inlens_empty_bin_frac", "sigma_l", "sigma_r", "gray_levels"}

def latest(b):
    fs = sorted(glob.glob(f"analysis_cache/features/{b}_*sites.parquet"), key=os.path.getmtime); return pd.read_parquet(fs[-1])

def family(c):
    if c in ACQ: return "acquisition"
    if c.startswith(("etd_crack_density", "crack_g_", "crack_p_", "curtain_", "inlens_", "ridge_")) or c == "etd_grad_energy": return "texture (ETD/Inlens; acquisition-sensitive)"
    if KPI_TRUST.get(c, "") == "exploratory" or c.startswith(("bright_void", "local_bright", "long_void")): return "secondary (battery geometry, exploratory)"
    if c.startswith(("pore_", "crack_", "bright_", "profile_")) or c in ("corr_len_px", "fft_slope", "graphite_frac"): return "morphology (BSE geometry)"
    return None

def values(df, c):
    if c in MATERIAL_KPIS or c.startswith(("pore_", "crack_", "bright_")):
        return ST.usable_values(df, c).dropna()
    return pd.Series(df[c].to_numpy(float), index=df.site).dropna()

def main():
    S = pd.concat([latest(b) for b in ("Batch_1", "Batch_2", "Batch_3")], ignore_index=True); S["site"] = S.site.astype(str)
    acq = pd.read_csv("analysis_cache/acquisition_sites.csv"); acq["site"] = acq.site.astype(str)
    S = S.merge(acq[["batch", "site"] + [c for c in acq.columns if c not in S.columns]], on=["batch", "site"], how="left")
    ref = S[S.batch == "Batch_3"]; rows = []
    for c in S.columns:
        fam = family(c)
        if fam is None or not pd.api.types.is_numeric_dtype(S[c]): continue
        r = values(ref, c)
        if len(r) < 10 or r.nunique() < 4: continue
        rec = dict(feature=c, family=fam, trust=KPI_TRUST.get(c, ""), primary=c in PRIMARY_KPIS, ref_median=float(np.median(r)), ref_mad=float(ST.mad(r)), ref_max=float(r.max()), n_ref=len(r))
        for b, tag in (("Batch_1", "b1"), ("Batch_2", "b2")):
            v = values(S[S.batch == b], c)
            rec[f"{tag}_n"] = len(v)
            if len(v) < 4 or rec["ref_mad"] <= 0:
                rec.update({f"{tag}_shift_mad": np.nan, f"{tag}_p": np.nan, f"{tag}_median": float(np.median(v)) if len(v) else np.nan}); continue
            pt = ST.permutation_test(r.to_numpy(), v.to_numpy(), statistic="hl_shift", n_mc=5000, seed=0)
            rec.update({f"{tag}_shift_mad": float((np.median(v) - rec["ref_median"]) / rec["ref_mad"]), f"{tag}_p": float(pt["p"]), f"{tag}_median": float(np.median(v)),
                        f"{tag}_above_ref_median": int((v > rec["ref_median"]).sum()), f"{tag}_above_ref_max": int((v > rec["ref_max"]).sum()),
                        f"{tag}_below_ref_min": int((v < r.min()).sum())})
        rows.append(rec)
    T = pd.DataFrame(rows); T["min_p"] = T[["b1_p", "b2_p"]].min(axis=1); T["monotone"] = np.sign(T.b1_shift_mad) == np.sign(T.b2_shift_mad)
    T = T.sort_values("min_p"); T.to_csv(os.path.join(OUT, "batch_signatures.csv"), index=False)
    fc = T.groupby("family").agg(scanned=("feature", "size"), p_lt_05=("min_p", lambda x: int((x < 0.05).sum())), p_lt_01=("min_p", lambda x: int((x < 0.01).sum())))
    fc.loc["all"] = [len(T), int((T.min_p < 0.05).sum()), int((T.min_p < 0.01).sum())]; fc.to_csv(os.path.join(OUT, "family_counts.csv")); print(fc)
    # figure 1: strip plots
    panels = [("inlens_particle_texture", "Inlens particle-interior texture (SD, levels)"), ("etd_crack_density_graphite", "ETD ridge density in graphite (T=0.4)"),
              ("ridge_p97", "ETD ridge strength p97"), ("bright_d50", "additive D50 (px)  — primary"), ("profile_bright_8", "additive fraction, depth bin 8/10"),
              ("pore_count_per_Mpx", "pore count per Mpx"), ("bright_void_distance_d50_px", "additive→void distance D50 (px) — secondary"), ("etd_boundary_sharpness", "ETD boundary sharpness — acquisition flag")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2)); rng = np.random.default_rng(0)
    for ax, (k, title) in zip(axes.ravel(), panels):
        row = T[T.feature == k].iloc[0]
        for x, b in enumerate(("Batch_3", "Batch_1", "Batch_2")):
            d = S[S.batch == b]; v = d[k].astype(float).to_numpy(); gp = d.grey_pore.astype(bool).to_numpy(); lc = d.bright_low_contrast.astype(bool).to_numpy(); c = BATCH_COLORS[b]
            for val, g, l in zip(v, gp, lc):
                if np.isfinite(val):
                    ax.scatter(x + rng.uniform(-.14, .14), val, s=24, color=BATCH_COLORS["Batch_3 (grey-pore group)"] if (g and b == "Batch_3") else c, alpha=.9, zorder=3, edgecolor="#e34948" if l else "none", linewidth=1.2)
            if np.isfinite(v).any(): ax.hlines(np.nanmedian(v), x - .25, x + .25, color=c, lw=2.2, zorder=4)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["B3 baseline\n(n=17)", "B1\n(n=7)", "B2\n(n=7)"], fontsize=8); ax.set_title(title, fontsize=9); ax.tick_params(axis="y", labelsize=8)
        ax.text(0.02, 0.97, f"B1 {row.b1_shift_mad:+.1f} MAD, p {row.b1_p:.3g}\nB2 {row.b2_shift_mad:+.1f} MAD, p {row.b2_p:.3g}", transform=ax.transAxes, va="top", fontsize=7.5, color="#444")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    h = [plt.Line2D([], [], marker="o", ls="", color=BATCH_COLORS["Batch_3"], label="Batch 3 baseline site"), plt.Line2D([], [], marker="o", ls="", color=BATCH_COLORS["Batch_3 (grey-pore group)"], label="Batch 3 grey-pore site (fallback threshold)"),
         plt.Line2D([], [], marker="o", ls="", color=BATCH_COLORS["Batch_1"], label="Batch 1 site"), plt.Line2D([], [], marker="o", ls="", color=BATCH_COLORS["Batch_2"], label="Batch 2 site"),
         plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", color="#e34948", label="low-contrast site (ring)"), plt.Line2D([], [], color="#777", lw=2.2, label="median")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Batch signatures against the Batch 3 baseline — site-level values (shift in baseline MADs; site-level permutation p, Hodges–Lehmann)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96)); fig.savefig(os.path.join(OUT, "fig_signature_strips.png"), dpi=130); plt.close(fig)
    # figure 2: matched crops
    sites = [("Batch_1", "f1vzngrs"), ("Batch_2", "3806gxp0"), ("Batch_3", "9luzk4jm")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.8))
    for j, (b, s) in enumerate(sites):
        r = S[(S.batch == b) & (S.site == s)].iloc[0]
        for i, det in enumerate(("ETD", "Inlens")):
            img = FE.load_image(os.path.join("Dataset", b), s, det); H, W = img.shape[:2]; cy, cx = H // 2, W // 2; w = 640
            ax = axes[i, j]; ax.imshow(img[cy - w // 2:cy + w // 2, cx - w // 2:cx + w // 2], cmap="gray", vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
            val = r.etd_crack_density_graphite if det == "ETD" else r.inlens_particle_texture
            ax.set_title(f"{b} · {s} · {det}\n{'ETD ridge density (graphite)' if det == 'ETD' else 'Inlens particle texture'} = {val:.3g}", fontsize=9)
    fig.suptitle("One 640 px window per site (image centre), display 0–255 fixed; illustrative only — the site values are whole-strip measurements.\n"
                 "Texture differences are acquisition-sensitive: an expert should judge whether what differs is graphite ridging / particle-interior structure or imaging (charging, session).", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(OUT, "fig_signature_crops.png"), dpi=110); plt.close(fig)
    T2 = T.set_index("feature")
    for k in ("inlens_particle_texture", "etd_crack_density_graphite", "pore_count_per_Mpx", "bright_void_distance_d50_px", "profile_bright_8"):
        r = T2.loc[k]; print(f"{k}: B1 above ref median {r.b1_above_ref_median}/{r.b1_n}, above max {r.b1_above_ref_max}, below min {r.b1_below_ref_min} | B2 above median {r.b2_above_ref_median}/{r.b2_n}, above max {r.b2_above_ref_max}, below min {r.b2_below_ref_min}")

if __name__ == "__main__":
    main()
