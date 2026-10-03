# Battery application review of the SEM morphology programme

Date: 2026-10-03. Author: an AI specialist reviewer, not a credentialed battery engineer. Status: research recommendations; no new production metric, verdict rule or performance model adopted by this memo.

**The highest-value extension is to measure the local environment of the Si/SiOx-like bright particles, while reviewing long voids and graphite plate alignment.** Those observations can explain what changed in manufacturing and identify questions for electrochemical validation. They cannot establish capacity, lifetime or fast-charge capability from the current images.

Santosh confirmed directly in this conversation on 2026-10-03 that the material is a **fresh graphite–Si/SiOx electrode**. Exact Si versus SiOx composition, binder formulation, supplier recipe, the chemical identity of every bright pixel, collector orientation and site/specimen independence remain unspecified. Visible voids therefore belong to an as-manufactured/prepared specimen; calling them cycling-induced cracks would be incorrect. Chemistry-specific mechanisms below describe possible behaviour during formation or later operation, not damage already demonstrated in these fresh samples.

## Evidence and image provenance

I read `CLAUDE.md`, `docs/problem_and_findings.md`, `docs/decision_log.md` Part B, `analysis/morphology/metric_register.json` and relevant parts of `polaron_qc/physics.py`. I visually inspected the existing atlas contact sheet, local-width and collector-band panels, and raw benchmark crops from `Batch_3/hzumfsms`, `Batch_3/71vgq3fw` and `Batch_1/4ih2ggld`.

These inspected crops show large interparticle dark spaces, elongated dark features, bright particulate sections and differing boundary contrast. The collector panel shows a continuous bright bottom band and nearby dark separation; this panel alone cannot determine collector adhesion or whether the band includes a stitching artefact. The raw crops are selected illustrations, not a random or statistically independent validation sample. Current masks remain predictions awaiting expert review.

The existing inventory already measures phase fractions, bright-section size/shape/spacing, pore alignment, void width and through-image-height profiles. `bright_to_void_distance` is proposed; `etd_plate_orientation` and `binder_network_frac` are deferred. Local-width evidence has geometry tests but expert review remains pending. This memo proposes interpretations and candidate measurements; it does not change those statuses.

## Primary-source mechanism anchors

The source IDs are also recorded with retrieval scope in `analysis/battery/sources.json`. Findings below are qualitative; published performance numbers and material-specific cutoffs are deliberately not transferred to this product.

| ID | Study and verified evidence | Applicable limit |
|---|---|---|
| S1 | [Sheng et al., *Effect of Calendering on Electrode Wettability in Lithium-Ion Batteries* (2014)](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2014.00056/full): graphite wetting-balance experiments, SEM and porosimetry. Light calendering improved wetting, while stronger compression worsened it. The study discusses cavities connected by narrow passages. | One graphite recipe and electrolyte; its optimum and pore measurements are not our specifications. Wetting and electrochemical transport are related but distinct tests. |
| S2 | [Pietsch et al., *Quantifying microstructural dynamics and electrochemical activity of graphite and silicon-graphite lithium ion battery anodes* (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5052642/): operando 3-D tomography and volume correlation, with numerical diffusion calculations. Plate alignment produced transport anisotropy; Si-containing electrodes exhibited locally pronounced deformation. | The dynamic and 3-D evidence exceeds what an uncycled 2-D section supplies. |
| S3 | [*Unravelling electro-chemo-mechanical processes in graphite/silicon composites* (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12623239/): operando/electrochemical graphite/Si and graphite/SiOx experiments. Sparse contact and enclosure impeded utilization; contact could outweigh size in reaction order. Pores and the carbon-binder domain affected deformation. | Material-specific contact ranges are not our acceptance thresholds. |
| S4 | [*Chemical staining for fundamental studies and optimization of binders in Li-ion battery negative electrodes* (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12913984/): binder-selective mapping, microscopy and electrical/ionic measurements. Slurry mixing changed carbon-binder agglomeration; drying changed binder distribution and adhesion. The work demonstrates why common aqueous binders are difficult to identify with ordinary SEM alone. | A binder-looking texture is not a validated binder map. Staining is a specialist laboratory method, not an assumed capability of this dataset. |
| S5 | [Nikpour et al., *Li-ion Electrode Microstructure Evolution during Drying and Calendering* (2022)](https://www.mdpi.com/2313-0105/8/9/107): original graphite/PVDF and NMC experiments combining cross-section SEM/EDX with transport measurements. Drying-induced component redistribution altered coating and collector-contact resistance. Calendering effects depended on drying history. | PVDF-based experimental recipes; do not assume the same migration strength or optimum for this unknown binder. Direct page fetch failed; original publisher-indexed experimental/results sections were retrieved. |
| S6 | [Ebner et al., *Tortuosity Anisotropy in Lithium-Ion Battery Electrodes* (2014)](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aenm.201301278): the original publisher abstract reports experimental tomography linking particle shape and fabrication-induced alignment to anisotropy, with a geometrical modelling approximation. | Abstract verified, full paper not retrieved. Use only the stated shape/alignment mechanism, not a calibrated prediction from our masks. |
| S7 | [Liu et al., *Size-dependent fracture of silicon nanoparticles during lithiation* (2012)](https://pubmed.ncbi.nlm.nih.gov/22217200/): original research abstract describing in situ TEM and size-dependent first-lithiation fracture of crystalline Si nanoparticles. | Abstract verified. Particle state, size scale, surface and reaction mechanism differ from commercial SiOx/composite microparticles; no universal critical size is adopted. |
| S8 | [JEOL, cross-section-polisher operating principles](https://www.jeol.com/products/science/cp.php): differential ion-milling rates in mixed hard/soft materials generate beam-direction streaks. | Official instrument explanation, not an independent electrode-performance experiment. |
| S9 | [JEOL, cooled cross-section-polisher examples](https://www.jeol.com/products/scientific/cp/IB-19520CCP.php): preparation temperature altered interface voids and bonding-agent deformation in demonstration specimens. | Examples are non-electrode materials; they establish a preparation alternative to test, not that our voids are artefacts. |
| S10 | [Kötzer, Vedel Jensen and Baddeley, *Geometric identities in stereological particle analysis* (2005 research report)](https://data.math.au.dk/publications/thiele/2005/imf-thiele-2005-10.pdf): the authors' university-hosted overview explicitly represents the volume of an arbitrary solid as an integral of its horizontal section areas. | A mathematical/sampling source, not a battery experiment. Spatial sampling is distinct from assumptions required to recover individual 3-D particle sizes. |
| S11 | [*Comparison of three-dimensional analysis and stereological techniques for quantifying lithium-ion battery electrode microstructures*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4999027/): original graphite/LCO/LMO tomography study comparing slicewise and volumetric measurements. Full-slice means reproduced volume fractions across axes; individual sections varied. Connected 3-D pore networks could appear disconnected in 2-D. | Supports representative sampling and the 2-D/3-D distinction; it does not validate our segmentation. |

## Proposed inspection programme

Checks B01–B06 are stable cross-reference IDs for the live metric inventory and visual atlas. The binder review is supporting context rather than a seventh core check. Every new measurement name below is a **candidate ID, status proposed/unvalidated, excluded from verdicts**. Existing metric IDs retain their inventory status. Summarise observations per site; individual particles, width samples and windows do not become independent batch samples. Choose spatial scales from measurement resolution and reference material structure before testing folder separation.

Programme priority matches the live inventory: **B01/B02 high; B03/B04/B05/B06 medium**. Scientific or measurement value and practical feasibility are stated separately below; a feature can have high scientific value while remaining in the medium implementation tier because its validation is incomplete.

### B01. Local bright-particle neighbourhood: solid adjacency, void access and accommodation

**Observable.** Look for bright particles with predominantly void-adjacent outlines, bright particles almost enclosed by residual solid, and locally dense bright-rich regions with little nearby visible void. Show the source crop with perimeter segments and the surrounding phase mask, rather than reducing everything to one mean distance.

**Metric mapping.** Existing: `bright_to_void_distance` (proposed), `bright_frac`, `bright_d50`, `bright_d90`, `bright_solidity`, `void_local_width_d50_px`. New candidates: `bright_void_adjacent_perimeter_frac`, `bright_residual_solid_adjacent_perimeter_frac`, and `bright_local_pore_frac` in fixed annuli around reviewed particles. Keep a per-site distribution and coverage/abstention count. Residual-solid adjacency includes unresolved graphite, binder and conductive carbon; label it accordingly.

**Process hypotheses.** Mixing/dispersal differences, altered particle morphology, local packing during coating or calendering, and supplier surface/coating changes. These are alternatives for investigation, not attributions.

**Battery question.** Could the changed neighbourhood balance electron supply, electrolyte access and room for subsequent expansion differently? S3 motivates checking both sparse adjacency and enclosure; neither extreme establishes electrical isolation or blocked electrolyte in this section.

**Counter-explanations and validation.** Off-centre cuts, out-of-plane contacts, particle rims, unresolved thin binder, resin-filled space and imperfect thresholds can change perimeter classes. Begin with the independent annotation benchmark; use BSE/ETD agreement as corroboration, not truth. For application validation obtain chemical maps, serial sections or tomography, and matched formation/impedance outcomes.

**Priority/feasibility.** Highest priority; feasible from reviewed masks without fitting a new classifier. Select illustrative particles across size, contrast and acquisition groups, not only dramatic examples.

### B04. Local bright-phase homogeneity rather than centroid spacing alone

**Observable.** Identify connected bright assemblies, patches with high bright fraction, bright-free patches and whether bright-rich patches are also void-poor. Distinguish one irregular particle from several touching ones before calling an agglomerate.

**Metric mapping.** Existing: `bright_nn_mean_px`, `bright_nn_csr_ratio`, `bright_width_d90_d10`, `bright_aspect_aw`, `bright_solidity`, `profile_bright`, `profile_bright_abs_slope`. New candidates: `local_bright_frac_dispersion` and `local_bright_pore_association` at fixed spatial scales. The E18 component-floor audit already makes spacing sensitive to small rim fragments; preserve that sensitivity rather than selecting the floor with the best batch difference.

**Process hypotheses.** Incomplete dispersal, slurry segregation, altered powder distribution, coating nonuniformity or drying rearrangement. Supplier formulation could change while site-average loading remains similar.

**Battery question.** The spatial arrangement highlighted by S3 suggests checking for local reaction/mechanical heterogeneity during formation. Uniformity is a comparison target, not automatically a better design: intentional gradients can be useful.

**Counter-explanations and validation.** Section plane, touching-mask mergers, inconsistent resolution and rim contamination can mimic clustering. Current uniform-point spacing simulations do not reproduce finite particle exclusion or nonuniform phase support. Review component instances; compare the same object definitions/scales and phase fraction, then validate any association with independently measured process and cell outcomes.

**Priority/feasibility.** Medium programme priority with high scientific value; local phase fractions are inexpensive once masks are reviewed. A more realistic particle-pattern null is a later method study, not a prerequisite for descriptive overlays.

### B02. Long-void location and confirmed collector interface

**Observable.** Separate internal long voids from gaps immediately adjacent to a verified collector. Record their depth, orientation, width, lateral span and relation to particle boundaries. Review `hzumfsms`, `0grcilhi` and `ufdvpb81` as reference anomalies, together with ordinary and preparation-flagged sites.

**Metric mapping.** Existing: `crack_frac`, `pore_max_d`, `longest_void_px`, `crack_orientation_deg_median`, `delamination_index`, `columns_interrupted_frac`, `crack_local_width_d50_px`. New candidates: `long_void_depth_distribution`, `collector_adjacent_gap_fraction` and `collector_adjacent_gap_width_px`. Only compute interface metrics where the collector is independently located. Keep a separate raw-interface analysis because the general material extraction trims edge bands.

**Process hypotheses.** Drying shrinkage, poor adhesion/cohesion, excessive or uneven calendering, cutting/handling damage, and ion-polishing damage. Fresh material rules out assigning these features to prior electrochemical cycling.

**Battery question.** If a real interface separation persists in 3-D, it may affect adhesion and collector contact. S5 supports treating drying and contact resistance together; S9 motivates checking preparation temperature before asserting a manufacturing defect.

**Counter-explanations and validation.** Natural interplate pores, stitching, mounting/sectioning, thermal milling damage and out-of-plane bypasses. Compare channels and adjacent preparation repeats; obtain peel/adhesion and coating-to-collector contact measurements. The existing column-intersection fraction describes geometry, not the fraction of electronically disconnected electrode.

**Priority/feasibility.** Highest priority for manual review; quantitative interface analysis is conditional on visible, confirmed boundaries and adequate raw margins.

### B03. Graphite plate alignment relative to the collector

**Observable.** Annotate clearly recognizable graphite plate sections and compare their major-axis angles relative to a verified collector normal, including changes with depth. Compare plate orientation and void orientation as separate quantities.

**Metric mapping.** Existing: `pore_horizontal_alignment`, `pore_alignment_strength`, `pore_elong`, `crack_orientation_deg_median`. `bright_horizontal_alignment` describes bright particles, not graphite. `etd_plate_orientation` is deferred. Candidate: `reviewed_graphite_plate_alignment` with per-site annotation coverage and axial-angle distribution. An orientation measure from connected residual solid would need instance validation first.

**Process hypotheses.** Changes in graphite flake shape, shear/coating history and calendering pressure; also section-plane differences.

**Battery question.** S2/S6 motivate an anisotropy check because plate alignment affects the direction of electrolyte pathways. Any implication for through-thickness transport requires knowing which image direction corresponds to the collector normal.

**Counter-explanations and validation.** ETD beam curtains, section tilt and grazing cuts can generate directional features. Compare reviewed plate outlines across channels; curtain/ridge angles remain acquisition diagnostics. Validate with orthogonal sections, tomography or suitable directional transport measurements. The absence of a spanning pore path in 2-D is not evidence of a blocked 3-D network.

**Priority/feasibility.** Medium programme priority with high scientific value; a small manual plate review is feasible, reliable automated instances are more demanding.

### B05. Narrow versus broad interparticle voids and depth-dependent packing

**Observable.** Distinguish broad cavities from narrow, elongated separations and inspect void-width distributions together with phase fraction and depth. Look for a compact surface-adjacent region, stacked narrow gaps or repeated void-poor zones, not only a mean pore diameter.

**Metric mapping.** Existing: `void_local_width_d50_px`, `void_local_width_d90_px`, `crack_local_width_d50_px`, `pore_frac`, `pore_width_d90_d50`, `profile_pore`, `profile_pore_abs_slope`. Candidate: `local_width_depth_profile`, retaining edge-exclusion and threshold-sensitivity information. A claimed bottleneck metric must wait for resolved, reviewed geometry.

**Process hypotheses.** Packing/calendering, particle-shape distribution, local coatings/segregation and preparation differences.

**Battery question.** S1 supports inspecting width and arrangement because wetting depends on the network rather than pore fraction alone. Use this as a reason for a wetting experiment, not a wetting-rate prediction. Greater observed void area can coexist with lower solid packing and fewer useful contacts.

**Counter-explanations and validation.** Grey-pore/fallback thresholds, digital boundaries, unpruned skeleton junctions, clipped cavities and unresolved porosity. E24 widths omit edge-connected voids; preserve that exclusion in the report. Complete expert width checks, compare segmentation variants and obtain volumetric porosity/wetting or 3-D measurements. A medial-axis width is not a 3-D pore throat.

**Priority/feasibility.** Medium programme priority; completing measurement checks has high immediate value because extraction exists. Battery interpretation remains conditional. Depth-profile extension is feasible once orientation and coverage are verified.

### Supporting context: binder/carbon-domain gradients and residual-solid texture

**Observable.** Review fine material between plates, bridges, surface-rich fine domains and their relation to pores. Ask whether the same structure is present in BSE and ETD before labelling it binder or carbon black.

**Metric mapping.** Existing: `graphite_frac` is residual solid, not chemically pure graphite; `profile_pore`, `profile_bright`, `etd_crack_density_graphite` and curtain/sharpness diagnostics. `binder_network_frac` remains deferred. Candidate: `reviewed_fine_domain_area_frac`, explicitly morphology-only until chemistry is mapped.

**Process hypotheses.** Mixing order, binder/carbon dispersion and drying migration. S4 and S5 motivate the process questions but also show the importance of binder-specific measurement.

**Battery question.** Could unresolved fine-domain redistribution change adhesion, local ionic access or electronic contacts? The current three-class masks cannot answer which binder is present or whether its network is electrically functional.

**Counter-explanations and validation.** Graphite rims, polishing debris, relief, charging and resin are plausible alternatives. Seek recipe information first, then a chemistry-appropriate mapping method; EDS may help for some elements/binders but does not automatically resolve common carbon-rich components. Binder-selective staining is an external research option with material/chemistry limitations, not an action proposed for this hackathon.

**Priority/feasibility.** High organiser-question priority; low readiness for a numeric QC metric from current images.

### B06. Expert review of fresh bright-particle interiors and possible fracture appearances

**Observable.** Review genuine internal voids, fracture-like lines, porous substructure, composite boundaries and coarse sections. Preserve the distinction between an intrinsically porous Si-containing particle, an aggregate of smaller particles and a preparation ridge.

**Metric mapping.** Existing: `bright_d90`, `bright_max_d`, `bright_solidity`, `etd_crack_density_particles`; Inlens texture metrics remain excluded because of acquisition confounding. Candidate: `reviewed_bright_internal_void_fraction` only after matching BSE/ETD boundaries and expert labels.

**Process hypotheses.** Supplier powder synthesis/milling, intrinsic composite structure, processing damage or ion-milling relief. No cycling-induced fracture can be concluded in fresh specimens.

**Battery question.** S7 supports a material-specific relation between particle scale and subsequent Si fracture susceptibility. It does not justify applying a crystalline nanoparticle cutoff to SiOx microparticles, or calling coarser sections a higher total binder load.

**Counter-explanations and validation.** Charging, curtain direction, off-centre cuts, finite resolution and unresolved coatings. Obtain supplier powder morphology/chemistry and high-quality images. Validate mechanical susceptibility by matched formation/cycling imaging and electrochemical tests, not texture alone.

**Priority/feasibility.** Medium; useful expert image review now, automated interior interpretation after acquisition normalisation and independent labels.

## Physics wording that should be corrected before external presentation

This is a read-only audit of the current physics layer; implementation changes are assigned separately. Qualifying a claim as “direction only” does not make an unsupported direction reliable.

| Current location / wording | Safer replacement or action |
|---|---|
| `physics.py` `CONSEQUENCE_TEXT['pore_frac']`: directly links more visible pore area to more favourable transport and lower coating density. | “Higher segmented 2-D void-area fraction. This changes the observed packing; implications for electrolyte access, contacts and volumetric loading require representative porosity and transport measurements.” Density also needs representative volume, composition and mass/thickness measurements. S1 demonstrates a non-monotonic processing/wetting relation. |
| `CONSEQUENCE_TEXT['pore_d50']`: larger equivalent diameters directly imply better transport. | “Coarser segmented void sections. Equivalent diameter does not establish connected passages, wetting or ionic conductivity.” Refer separately to local width and validated network measurements. |
| `CONSEQUENCE_TEXT['bright_d50']` and `additive_size_reading`: bigger sections imply longer lithiation times; numeric squared diameter ratio. | “Coarser observed Si/SiOx-like bright sections. Size is one possible kinetic/mechanical factor; contact, chemistry, porosity and electrolyte access can dominate.” If retained, the squared-length result belongs in a clearly labelled hypothetical equal-diffusivity model panel, not a battery-rate forecast. S3 explicitly reports contact-dependent reaction order. |
| `additive_size_reading` caveat: 2-D area-weighted quantiles underestimate 3-D sizes equally in both batches. | “2-D section-size distributions have sampling, shape and orientation bias; equal bias across batches is unverified.” Even a uniform physical size distribution can produce differently weighted section distributions. Compare observed sections; stereological inversion requires validated model assumptions. |
| `additive_mechanics_reading`: size alone raises total “expansion load on the binder.” | Separate phase amount from individual-section size. At fixed true phase volume and lithiation strain, resizing redistributes local expansion; it does not establish greater total expansion. Binder stress additionally depends on confinement, interfaces, binder properties and state of charge. Report morphology and a conditional local-mechanics hypothesis, not a monotone load direction. |
| `1 - etd_crack_density_particles` called an intact-share proxy. | Rename to “fraction of analysed interior pixels without detected qualifying dark ridges,” or omit. It is neither intact-particle share nor a validated fracture measurement. `etd_crack_density_particles` needs “ridge coverage, expert interpretation pending” wording. |
| `physics.py` high weighting: bright fraction described as capacity share; pore elongation as anisotropy. | Bright area fraction is a phase-amount descriptor pending recipe/chemical/sampling validation. Capacity share needs phase utilization/chemistry. Void aspect ratio describes 2-D shape, while alignment separately describes directional preference; neither establishes transport anisotropy quantitatively. |

Keep the existing explicit warning that `columns_interrupted_frac`, `delamination_index` and section-path outputs do not establish electronic discontinuity or bound 3-D tortuosity. Ensure “along the coating” is conditional on the image-to-collector orientation, rather than silently equating it to image horizontal.

### Additional stereology wording check

Do not say that graphite plate alignment itself violates the area-fraction/volume-fraction relation. The volume identity in S10 integrates sections with a fixed orientation; isotropic particle orientation is not necessary for that identity. This is a mathematical inference from the verified section-volume representation, rather than an empirical battery result.

Recommended plan wording: **“Area fractions can estimate phase volume fractions under representative, unbiased spatial sampling; plate alignment alone does not invalidate that relation. Section selection, segmentation and unresolved phases are not validated here, so report observed 2-D fractions.”** For irregular reference domains, proper area/volume weighting and the ratio-estimation design also matter. A single convenient fixed-location plane through a layered/heterogeneous coating can be unrepresentative even though the identity is valid. Orientation and shape remain substantial issues for particle-size inversion, surface/length estimators and transport anisotropy. None of this promotes our current area fractions to calibrated 3-D volume or recipe mass fractions.

S11 supplies direct battery evidence for this distinction: its complete slicewise averages matched volumetric pore fractions across axes, whereas individual-section values differed. This agreement follows complete voxel coverage; it does not mean our few selectively acquired sections are representative. Its disconnected 2-D sections also caution against interpreting missing section paths as failed volumetric pore connectivity.

## What is not identifiable from these images

- Measured capacity, energy density, cycle life, fast-charge limit, heat generation or a quantitative plating probability.
- Lithium plating or electrochemically formed SEI; these are fresh electrodes and the SEM geometry does not identify those species.
- 3-D electronic/ionic percolation, pore-throat distribution or effective conductivity from a single section.
- Actual binder identity/network, Si versus SiOx composition or every particle's coating/nanoporosity from greyscale appearance alone.
- Whether a dark space is wholly manufacturing-induced rather than preparation/handling-related without comparative preparation evidence.
- Absolute physical widths until scale calibration is confirmed; all proposed lengths stay in pixels.

## Recommended next implementation and validation order

1. Finish the independent segmentation benchmark and review a small, deliberately varied set of particle neighbourhoods and long voids. Store uncertain or unmeasurable regions explicitly.
2. Add descriptive bright-boundary adjacency and local phase-fraction overlays; extend the register with proposed/computed/expert-reviewed stages. Validate perimeter labels before comparing batches.
3. Add long-void depth/context and a separate confirmed-collector review. Preserve raw bands for this purpose while keeping them out of general bright-phase measurements.
4. Review graphite plate instances and collector orientation before attempting automated orientation extraction. Reuse existing void-width/profile outputs with their measurement uncertainty.
5. Seek matched adhesion/contact, wetting and formation/impedance outcomes for the most plausible changes. Add performance-related interpretations only after those tests, while keeping the five frozen primary KPIs separate.

These are prioritised experiments, not a new list of defect thresholds. Preserve all pairwise comparisons, usable site counts, acquisition subgroups and localized-reference anomalies. Do not tune morphology definitions to the known folder labels.

## Questions for the organisers and supplier

1. Which Si-containing product is used: elemental Si, SiOx, carbon-coated/composite, porous or dense? Are elemental maps or supplier powder examples available?
2. What are the nominal graphite/Si-containing phase fractions, graphite shape/size specifications, binder/carbon formulation and intended morphology differences?
3. What were the coating, drying, calendering and target loading/thickness/porosity conditions? Are these images before or after calendering?
4. Which image edge is the collector and which is the separator-facing surface; was orientation held constant? Can pixel size be verified?
5. Were sections prepared with identical resin/mounting, ion-milling temperature/beam/rocking conditions and image-export settings? What explains the grey-pore and low-contrast groups?
6. Which sites share the same specimen/roll position, and how were locations selected? Are replicate specimens available?
7. What actual manufacturing, adhesion, wetting, formation, impedance or cycling outcomes define acceptable/defective variation? Are the long reference voids intentional accepted morphology or preparation/production defects?

## Review checks applied

Applied C03 (acquisition confounds), C04 (implemented versus proposed capability), C05 (observed morphology versus material naming), C06/C14 (sites and unresolved specimen independence), C07 (absence of detected drift is not acceptance), C08 (localized anomalies), C13 (2-D geometry and no performance numbers), C16 (known-batch development provenance), C17 (uncertainty terminology), and C28 (predictions are not independent labels). Literature mechanisms, inspected-image observations and proposed measurements are separated above. No experiments, new performance figures, production-code edits, cache edits or commits were made in this specialist review.
