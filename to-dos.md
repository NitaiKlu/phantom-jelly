# Paper to-dos (through segmentation; radiomics/ML not included)

Checklist for **everything before** “Radiomics and machine learning architecture.” Do not start the paper edits until the **Blocked** items have answers (or an explicit decision to omit/soft-claim).

Legend: **[I can draft]** = I can write from the current manuscript + review. **[Blocked — you supply]** = I do not have the experimental fact. **[Decision]** = you choose among options, then I implement.

---

## 0. Information I cannot complete (hand these over)

Copy this section out, fill it, and return it. Until then I can restructure and deflate rhetoric, but I cannot make the methods reproducible.

### 0.1 Numbers and dataset composition
- [ ] **[Blocked]** Canonical validated-segment count (use one number everywhere; review assumed **2,302**).
- [ ] **[Blocked]** Attrition: CT-detected → NM-detected → 1:1 matched → validated. Exclusion reasons and \(n\) for each (overlap, merge, no match, bubbles, too small, …).
- [ ] **[Blocked]** Insert **volume** range (ml) overall and for the held-out session (min–max, and mean if you have it).
- [ ] **[Blocked]** Per-session table: date; session nickname; \(n\) containers; \(n\) SPECT/CT scans; \(n\) validated segments; activity range; volume range. Confirm whether held-out is “session 0” or a **date**.
- [ ] **[Blocked]** Scans per container; whether all 21 jars appear in every session.
- [ ] **[Blocked]** Independent-sample note: is activity density constant within a scan for most acquisitions? If you know “\(N\) scans / \(M\) unique activity labels,” give it. If not, say “unknown — omit numeric independence claim.”
- [ ] **[Decision]** Units: report **kBq/ml** with µCi/ml in parentheses (recommended), or µCi/ml only.

### 0.2 Phantom recipe and activity assay
- [ ] **[Blocked]** Gelatin: grade/bloom, manufacturer/product.
- [ ] **[Blocked]** How irregular shapes are **actually formed** (set–break–embed, pipette droplets, inject into set background, shake while gelling, …). Step order: background first vs inserts first.
- [ ] **[Blocked]** Container: type, volume, wall material, fill height, approximate dimensions; how it sat on the pallet.
- [ ] **[Blocked]** Contrast agent: product name, iodine concentration (mg I/mL), and confirm 10 mL/L.
- [ ] **[Blocked]** Pigment: juice species/brand, **or** “food dye, product X,” **or** decision to call it only “dark pigment” if brand is unknown.
- [ ] **[Blocked]** \(^{99\mathrm{m}}\)Tc chemical form (e.g. sodium pertechnetate).
- [ ] **[Blocked]** Activity **assay**: instrument (dose calibrator model), geometry, residual-syringe correction (yes/no), clocks used, decay correction to which time (start of SPECT / mid-frame).
- [ ] **[Blocked]** How activity density (µCi/ml) was computed: assayed activity ÷ **prepared** gel volume, or assayed activity ÷ **CT** volume, or another rule.
- [ ] **[Blocked]** Background assayed for contamination? Aliquots of insert mix counted for uniformity?
- [ ] **[Blocked]** Timing: mix → gelation temperature/time → first scan; session duration; any observation of iodine/Tc diffusion.
- [ ] **[Blocked]** Typical inserts per jar; intended size range.
- [ ] **[Blocked]** Measured HU (mean ± SD) of background vs inserts, if it exists. If not, say so — we will not invent “tissue-equivalent” numbers.
- [ ] **[Blocked]** Air bubbles: ignored / excluded / hole-filled in CT masks?
- [ ] **[Blocked / Decision]** Attenuation mismatch: do you have data on AC bias from iodine in inserts? If none: we will **discuss as a limitation**, not claim matched attenuation.

### 0.3 SPECT/CT protocol
- [ ] **[Blocked]** SPECT: collimator; photopeak energy window; scatter window and whether scatter correction was on; projections (number, angular step); orbit (circular vs contour); radius or auto-contour; matrix; voxel size; zoom; time/view (reconcile with “3 s per degree”).
- [ ] **[Blocked]** Reconstruction: software/product (Flash3D, xSPECT, …); AC yes (already stated); scatter; collimator-detector response / resolution recovery; where the 8 mm Gaussian was applied; **justification of 48 iterations × 1 subset** (clinical preset name is enough).
- [ ] **[Blocked]** SPECT calibration to Bq/ml or counts-only.
- [ ] **[Blocked]** CT: kVp; mAs or automatic exposure; collimation; slice thickness; kernel; voxel size; same CT used for AC and for 60 HU mask (yes/no).
- [ ] **[Blocked]** QC / CT–SPECT registration: vendor default rigid only, or extra registration? Any residual known?

### 0.4 Segmentation and matching
- [ ] **[Blocked]** Why **60 HU** (empirical from histogram / visual / other). Any sensitivity test (40 vs 80 HU)?
- [ ] **[Blocked]** CT post-processing: 3D connected components, minimum volume, hole fill, split of touching objects (yes/no + parameters).
- [ ] **[Blocked]** SPECT resampling onto CT grid (or reverse) and interpolation.
- [ ] **[Blocked]** 1:1 matching rule: overlap/IoU, centroid distance, Hungarian, distance cutoff.
- [ ] **[Blocked]** Why NM **8% global** then **40% of local peak**; why recursive split thresholds 60% volume and 70% secondary peak.
- [ ] **[Blocked]** Exclusion fraction for merged/overlapping inserts (even an approximate %).

### 0.5 Figures
- [ ] **[Blocked]** Figure 1 left: what are the **orange rings** (scanner geometry vs rendering artifact)? Delete if clutter.
- [ ] **[Blocked]** Figure 1 right: jar volume or a ruler measurement for the caption.
- [ ] **[Blocked]** Figure 2: display windows (CT HU; SPECT counts or Bq/ml) if you want them in the caption.

### 0.6 Scope decisions (no lab work, but you must choose)
- [ ] **[Decision]** Primary clinical hook: **SPECT quantification / dosimetry**, with PET SUV as analogue — not PET-first.
- [ ] **[Decision]** Abstract NEMA sentence: **soften** (no “outperforming” until quantified) vs keep if you will add RC metrics in the later section.
- [ ] **[Decision]** Two abstract variants (I will draft both, you pick): **A** conservative methods-paper; **B** results-forward (still no unearned NEMA win).
- [ ] **[Decision]** \(^{177}\)Lu / PET: “protocol transferable in spirit, demonstrated only with \(^{99\mathrm{m}}\)Tc” vs drop Lu/PET from abstract entirely.

---

## 1. Global consistency (do first, after 0.1 numbers land)

- [ ] **[I can draft]** Replace every 3,000 / “over 2,000” / “approximately 2,000” / “more than two thousand” with the canonical validated \(n\), plus exclusion \(n\) once.
- [ ] **[I can draft]** Define **SPECT** at first use; stop using **NM** in prose (keep NM only if a figure label already says NM, and expand in the caption).
- [ ] **[I can draft]** Split ground truth in every relevant sentence: **CT = morphological reference (volume/shape)**; **activity density = assayed preparation** (wording depends on 0.2).
- [ ] **[I can draft]** Replace “absolute ground truth” with “high-resolution morphological reference.”
- [ ] **[I can draft]** Replace “tissue-equivalent” with a claim we can support (water-equivalent gelatin ± measured HU from 0.2).
- [ ] **[I can draft]** Replace “isotope-agnostic” / “same protocol extends to \(^{177}\)Lu” with the Decision in 0.6.
- [ ] **[I can draft]** SI units per 0.1 decision.
- [ ] **[I can draft]** Rename held-out set from “session 0” to the date/name you supply.
- [ ] **[I can draft]** Kill filler: paradigm shift, clinical imperative, life-saving treatments, big data, data factory (or one use of “data factory” max), “models collapse,” “massive,” “vast,” “extensive sessions.”

---

## 2. Title, authors, front matter

- [ ] **[I can draft]** Keep title unless you want SPECT-quantification wording aligned with the new intro hook.
- [ ] **[Blocked]** Author 2 affiliation: institution missing (`roeexs@gmail.com` only). Supply or we leave as is.
- [ ] **[Decision]** Add a short running title / keywords later if the target venue needs them (not required for the current arXiv article class).

---

## 3. Abstract rewrite

- [ ] **[I can draft]** Draft **Variant A** (methods-paper): gelatin library, 6 sessions / 21 containers / 75 scans / \(n\) segments, activity range, cold background, CT morphology vs SPECT observation, held-out session, \(^{99\mathrm{m}}\)Tc as model tracer. Results as secondary.
- [ ] **[I can draft]** Draft **Variant B** (results-forward): same facts + held-out \(R^2\) numbers; NEMA mentioned only as “compared with a sphere RC curve,” no “outperforming” unless 0.6 says you will quantify later.
- [ ] **[I can draft]** Either variant: drop unexplained “NM”; drop isotope-agnostic; state two ground truths; state held-out is a new session/preparation.
- [ ] **[Decision]** You pick A or B (or a merge). Then one pass to match the final numbers from §0.1.

---

## 4. Introduction — restructure (compress + contributions)

- [ ] **[I can draft]** Cut PVE textbook to **one paragraph**: finite resolution → underestimation in small / high-gradient lesions; spill-out vs spill-in named once; tissue fraction named once.
- [ ] **[I can draft]** Move the longer two-mechanism PVC history (deconvolution, anatomy-guided, noise sensitivity) **out of the intro** — either delete or a short appendix “PVE and PVC background.”
- [ ] **[I can draft]** Reframe motivation as **quantitative SPECT** (dosimetry / activity concentration), with PET SUV thresholds as the well-known clinical analogue — per 0.6.
- [ ] **[I can draft]** Fix citations: Jentzen 2011 for **SPECT recovery**, not PET cut-offs; keep Soret, Erlandsson, Hoffman, Kessler, Gallivanone, Zeintl where they belong.
- [ ] **[I can draft]** Shape-vs-volume paragraph stays (circles vs ellipses → stochastic lesions). Soften “unpredictable RCs” to “RC is not a function of volume alone.”
- [ ] **[I can draft]** Related work, 3–5 sentences: spherical NEMA RC; irregular/3D-printed/anthropomorphic phantoms; gelatin/agar if we cite generic practice. **Do not** cite Salmon, Tran-Gia, Southekal, Matsubara, De Nijs as support for gelatin+radiomics unless we actually discuss them (drop from intro closing).
- [ ] **[I can draft]** Delete “over 3,000” / “precisely calibrated” until assay text exists; then “calibrated” only if 0.2 supports it.
- [ ] **[I can draft]** Delete “recovery models collapse” and “big data environment.”
- [ ] **[I can draft]** **Contributions paragraph** at end of intro (your `paper/additions.md` item 1), e.g.:
  1. stochastic contrast-labeled gelatin inserts with known morphology (CT) and known activity (assay);
  2. SPECT/CT library of \(n\) 1:1 pairs, held-out session;
  3. scope sentence: proof of concept on cold-background \(^{99\mathrm{m}}\)Tc SPECT, not a patient PVC tool, not a \(^{177}\)Lu/PET demonstration.
- [ ] **[I can draft]** Explicit **not-claims**: no warm background; overlapping lesions excluded; not a deployed clinical correction.

---

## 5. Methods — structural change

- [ ] **[I can draft]** **Delete** subsection “General algorithm and experimental rationale” (repeats intro + later methods; “four-stage” vs three bullets; “two-phase” wording).
- [ ] **[I can draft]** Optional replacement: **one figure** — flowchart Prepare → Scan / shake / decay → Segment & 1:1 match → (later) model. Only if you want a visual; otherwise skip.
- [ ] **[I can draft]** New Methods order:
  1. Phantom preparation
  2. Activity assay and decay *(new)*
  3. SPECT/CT acquisition and reconstruction
  4. Segmentation, matching, exclusions
  5. Dataset composition

---

## 6. Phantom preparation (expand in main text; do not appendix the recipe)

- [ ] **[I can draft once 0.2 arrives]** Base matrix: water, temperature, gelatin g/L, bloom/product, stir/gelation.
- [ ] **[I can draft once 0.2 arrives]** Insert mix: Tc form, contrast product + mg I/mL + 10 mL/L, pigment.
- [ ] **[I can draft once 0.2 arrives]** **Shape-formation protocol** (the missing method): how lumps are made and embedded.
- [ ] **[I can draft once 0.2 arrives]** Containers and filling; placement on scanner.
- [ ] **[I can draft]** Fix internal contradiction: **do not** claim matched attenuation after adding iodine. Rewrite: matched gelatin base; iodine added for CT contrast; expected ΔHU / AC implication as limitation if no measurement.
- [ ] **[I can draft]** Air bubbles: one sentence from 0.2.
- [ ] **[I can draft]** Shake between scans = **pose resampling of the same physical inserts**, not independent new lesions.
- [ ] **[I can draft]** Decay between scans = **count-level / SNR diversity**, same morphology.
- [ ] **[I can draft]** Appendix only: bath model, lot numbers, exact stir times — if you provide them. Otherwise omit.

---

## 7. New subsection: Activity assay and decay

- [ ] **[I can draft once 0.2 arrives]** Formula for activity density; decay correction; time reference.
- [ ] **[I can draft]** State clearly this — not CT — is the activity ground truth.
- [ ] **[I can draft]** If assay details never arrive: we **cannot** keep “known/calibrated activity density” as a strong claim; we must downgrade to “nominal activity from preparation, assay not reported” (worse for reviewers). Flag this if 0.2 is incomplete.

---

## 8. SPECT/CT acquisition and reconstruction (expand; appendix only the DICOM dump)

- [ ] **[I can draft once 0.3 arrives]** SPECT paragraph: camera (keep Siemens Symbia Pro.specta, Shaare Zedek), collimator, windows, scatter, orbit, matrix, voxel size, time/view.
- [ ] **[I can draft once 0.3 arrives]** Reconstruction paragraph: product name, OSEM 48×1 explained, AC, scatter, RR/CDR, 8 mm FWHM Gaussian and whether that is the site clinical kernel.
- [ ] **[I can draft once 0.3 arrives]** CT paragraph: kVp, exposure, slice thickness, kernel, voxel size; same CT for AC and morphological mask.
- [ ] **[I can draft]** One sentence: hybrid vendor registration only, unless 0.3 says otherwise.
- [ ] **[I can draft]** Main text = the list above. Appendix = full recon preset / DICOM fields **if** you paste them.

---

## 9. Segmentation, matching, exclusions

- [ ] **[I can draft once 0.4 arrives]** CT: 60 HU + justification; connected components / min volume / holes / splits.
- [ ] **[I can draft]** Call CT volume “volume at 60 HU,” not absolute truth. Add sensitivity sentence if you have it; if not, one limitation sentence.
- [ ] **[I can draft once 0.4 arrives]** NM: 8% then 40%; recursive split as **merged-blob splitter**, not “to preserve irregular morphology.”
- [ ] **[I can draft]** Note: NM thresholds are applied to the **PVE-blurred** image; features will live in that mask.
- [ ] **[I can draft once 0.4 arrives]** Matching algorithm and 1:1 rule.
- [ ] **[I can draft once 0.4 arrives]** Exclusions + \(n\) or %; **scope:** adjacent/overlapping lesions not in the dataset (spill-in from neighbors not studied).
- [ ] **[I can draft]** Cold-background limitation sentence (spill-out dominated; spill-in from warm tissue not tested).

---

## 10. Dataset composition (replace repeating bullets)

- [ ] **[I can draft]** Keep Table 1; **delete** bullets that only repeat it.
- [ ] **[I can draft once 0.1 arrives]** Add Table 2 (or appendix table): per-session composition.
- [ ] **[I can draft]** Table 1 fields: sessions, containers, scans, validated \(n\), excluded \(n\), activity range, **volume range**, held-out \(n\) + activity + volume ranges.
- [ ] **[I can draft]** Rewrite held-out paragraph: new preparation/session; range is a **shift inside** the global range, not “unique”; it is a **stress test**, not a guarantee against training-distribution bias.
- [ ] **[I can draft]** One sentence on ROI count vs independent activity labels (from 0.1, or qualitative if unknown).

---

## 11. Figures (this part of the paper only)

- [ ] **[I can draft]** Figure 1 caption: volume rendering; red = contrast inserts; blue = background; explain or remove orange rings (0.5); photo = physical jars + scale/volume; do not repeat the recipe.
- [ ] **[I can draft]** Figure 2: label panels SPECT vs CT, image vs mask on the figure if not already; add windows/scale from 0.5; caption = PVE blur vs CT reference, not a methods dump.
- [ ] **[Decision]** Optional small **attrition flowchart** if exclusions are large; else a table row is enough.
- [ ] **[Decision]** Optional pipeline flowchart if we dropped §5 rationale without a figure.

**I cannot redraw the PNGs** unless you provide new renders (rings removed, scale bar, panel letters). I can only change captions/LaTeX layout.

---

## 12. Appendix (only what we agreed to park)

- [ ] **[I can draft]** Appendix A (optional): compressed PVE/PVC background moved out of the intro.
- [ ] **[I can draft if materials arrive]** Appendix B: reagent lots, bath model, DICOM/recon preset list.
- [ ] **[I can draft if 0.1 is a wide table]** Per-scan file list stays out of main text; per-session table stays in main unless huge.

Do **not** appendix: recipe, assay, SPECT/CT protocol, segmentation rule, exclusion rule.

---

## 13. What this pass does **not** include (later)

- [ ] Radiomics and machine learning architecture (that section will change separately).
- [ ] Results, NEMA RC head-to-head metrics, conclusion.
- [ ] Limitations section as a whole (we **will** plant limitation *sentences* in Methods: cold background, iodine/AC, 60 HU, independence, overlaps excluded, \(^{99\mathrm{m}}\)Tc-only).
- [ ] Data-availability / ethics / funding.
- [ ] Implementing `paper/additions.md` item 3 (“explain more about the radiomics method”) — that belongs to the next section.

---

## 14. Order of work when we start editing

1. You return **§0** (even partial: mark unknown vs not applicable).
2. I lock **numbers, units, SPECT wording, ground-truth split**.
3. Abstract variants → you pick.
4. Intro rewrite + contributions.
5. Methods restructure (delete rationale; new subsection order).
6. Fill phantom, assay, imaging, segmentation from §0.
7. Tables + held-out wording.
8. Figure captions.
9. Appendix only if content exists.
10. One consistency pass (segment counts, session name, µCi vs kBq, no NM in prose).

---

**Stop condition for “I cannot write this yet”:** if §0.2 assay and §0.3 collimator/windows/recon/CT geometry are still blank, we can still do §1 rhetoric, §4 intro, §5 delete-rationale, and abstract **scaffolds** with placeholders (`XX`, `TODO`). We should not submit or compile a “final” methods text around empty protocol fields.

When you are ready, send §0 in any format (bullet answers, email dump, lab notes). Partial answers are fine; I will list what remains unknown in the manuscript as explicit limitations rather than inventing it.
