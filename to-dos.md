# Paper to-dos (through segmentation; radiomics/ML not included)

**Status (2026-08-30).** Methods through segmentation, abstract, intro, discussion, and consistency pass are in the manuscript. Radiomics/ML + Results are **not** rewritten yet. NEMA RC comparison is blocked on sphere measurements (or fitted \(a,b\)).

Legend: **[I can draft]** = I can write from the current manuscript + review. **[Blocked — you supply]** = I do not have the experimental fact. **[Decision]** = you choose among options, then I implement.

---

## 0. Information I cannot complete (hand these over)

Answers received. Applied where the paper needed them; ignored items were omitted rather than invented.

### 0.1 Numbers and dataset composition
- [x] **[Blocked]** Canonical validated-segment count (use one number everywhere; review assumed **2,302**).
  Applied: **2,302** everywhere.
- [x] **[Blocked]** Attrition: CT-detected → NM-detected → 1:1 matched → validated. Exclusion reasons and \(n\) for each (overlap, merge, no match, bubbles, too small, …).
  Applied qualitatively (1:1 only; many-to-one excluded). Exact exclusion \(n\) not supplied — omitted.
- [x] **[Blocked]** Insert **volume** range (ml) overall and for the held-out session (min–max, and mean if you have it).
  **Ignored** — not in the paper.
- [x] **[Blocked]** Per-session table: date; session nickname; \(n\) containers; \(n\) SPECT/CT scans; \(n\) validated segments; activity range; volume range. Confirm whether held-out is “session 0” or a **date**.
  Applied as Table 2 (sessions 1–5 + held-out). Activity converted to kBq/ml from the µCi/ml ranges below.

  | date | Number of segmentation | Activity_Density_uCiml_NM_40 | scans | jars |
  | --- | --- | --- | --- | --- |
  | 1 | 386 | 5.5-7.3 | 11 | 3 |
  | 2 | 327 | 3.4-4.7 | 13 | 3 |
  | 3 | 190 | 3.04-3.96 | 10 | 3 |
  | 4 | 583 | 0.6-0.9 | 15 | 4 |
  | 5 | 482 | 1.7-2.5, 8.4-11.8 | 13 | 4 |
  | held-out (was 0) | 334 | 3.6-5.2, 7.1-10.2 | 13 | 4 |

- [x] **[Blocked]** Scans per container; whether all 21 jars appear in every session.
  Covered by Table 2.
- [x] **[Blocked]** Independent-sample note: is activity density constant within a scan for most acquisitions? If you know “\(N\) scans / \(M\) unique activity labels,” give it. If not, say “unknown — omit numeric independence claim.”
  **Omitted** (unknown).
- [x] **[Decision]** Units: report **kBq/ml** with µCi/ml in parentheses (recommended), or µCi/ml only.
  Applied: **kBq/ml** only. Global range from the table: **22.2–436.6 kBq/ml** (not the old 0.12–80.07 µCi/ml).

### 0.2 Phantom recipe and activity assay
- [x] **[Blocked]** Gelatin: grade/bloom, manufacturer/product. **Ignored.**
- [x] **[Blocked]** How irregular shapes are **actually formed**. **Ignored.**
- [x] **[Blocked]** Container: type, volume, wall material, fill height. **Ignored.**
- [x] **[Blocked]** Contrast agent: product name, iodine concentration. **Ignored** (kept 10 mL/L).
- [x] **[Blocked]** Pigment: juice species/brand. **Ignored** (kept “dark-pigmented juice”).
- [x] **[Blocked]** \(^{99\mathrm{m}}\)Tc chemical form. **Ignored.**
- [x] **[Blocked]** Activity **assay** protocol. **Ignored.**
- [x] **[Blocked]** How activity density was computed. **Ignored.**
- [x] **[Blocked]** Background assayed for contamination? **Ignored.**
- [x] **[Blocked]** Timing: mix → gelation → first scan. **Ignored.**
- [x] **[Blocked]** Typical inserts per jar; intended size range. **Ignored.**
- [x] **[Blocked]** Measured HU of background vs inserts. **Ignored** (no numbers).
- [x] **[Blocked]** Air bubbles. **Ignored.**
- [x] **[Blocked / Decision]** Attenuation mismatch: discuss as a limitation.
  Applied: Methods sentence + **Discussion** section.

### 0.3 SPECT/CT protocol
- [ ] **[Blocked]** SPECT: collimator; energy window; scatter; orbit; matrix; voxel size; time/view.
  **Later** — “Ignore for now, I will hand it later.” Existing 360° / 3 s per degree kept.
- [x] **[Blocked]** Reconstruction: software/product, scatter, RR, 48×1 justification.
  **Ignored.** Existing OSEM 48×1, AC, 8 mm Gaussian kept.
- [ ] **[Blocked]** SPECT calibration to Bq/ml or counts-only.
  **Later** — “I'll hand it later.”
- [x] **[Blocked]** CT: kVp, mAs, slice thickness, kernel, voxel size.
  Extra CT geometry **ignored**. Applied: **same CT used for AC and morphological mask.**
- [x] **[Blocked]** QC / CT–SPECT registration. **Ignored.**

### 0.4 Segmentation and matching
- [x] **[Blocked]** Why **60 HU**. Applied: most promising cut in a small-scale experiment.
- [x] **[Blocked]** CT post-processing (connected components, min volume, holes). **Ignored.**
- [x] **[Blocked]** SPECT resampling onto CT. **Ignored.**
- [x] **[Blocked]** 1:1 matching rule. Applied: exact one-CT to one-SPECT; many-to-one excluded.
- [x] **[Blocked]** Why NM **8%** then **40%**; recursive 60%/70%. Thresholds kept; recursive rule recast as **merged-blob splitter**. Extra justification **ignored.**
- [x] **[Blocked]** Exclusion fraction. **Ignored** (no %).

### 0.5 Figures
- [x] **[Blocked]** Figure 1 orange rings. **Ignored** (PNG not redrawn). Captions rewritten.
- [x] **[Blocked]** Figure 1 jar volume / ruler. **Ignored.**
- [x] **[Blocked]** Figure 2 display windows. **Ignored.**

### 0.6 Scope decisions
- [x] **[Decision]** Primary clinical hook: SPECT quantification / dosimetry, PET SUV as analogue. Applied.
- [x] **[Decision]** Abstract NEMA sentence: keep “outperforming” until RC metrics exist.
  Kept for now; **still unearned** until NEMA points or \(a,b\) arrive.
- [x] **[Decision]** Abstract variant **A only**. Applied.
- [x] **[Decision]** \(^{177}\)Lu / PET: transferable in spirit, demonstrated only with \(^{99\mathrm{m}}\)Tc. Applied.

---

## 1. Global consistency (do first, after 0.1 numbers land)

- [x] **[I can draft]** Replace every 3,000 / “over 2,000” / “approximately 2,000” with **2,302**.
- [x] **[I can draft]** Define **SPECT** at first use; stop using **NM** in prose.
- [x] **[I can draft]** Split ground truth: CT = morphological reference; activity density = preparation target.
- [x] **[I can draft]** Replace “absolute ground truth” with “high-resolution morphological reference.”
- [x] **[I can draft]** Replace “tissue-equivalent” (title + body → water-equivalent gelatin / stochastic gelatin).
- [x] **[I can draft]** Replace “isotope-agnostic” with transferable-in-spirit / \(^{99\mathrm{m}}\)Tc-only demonstration.
- [x] **[I can draft]** SI units: kBq/ml.
- [x] **[I can draft]** Rename held-out set from “session 0” to **held-out session**.
- [x] **[I can draft]** Kill filler (paradigm shift, data factory, models collapse, etc.).

---

## 2. Title, authors, front matter

- [x] **[I can draft]** Title aligned: dropped “Tissue-Equivalent.”
- [x] **[Blocked]** Author 2 affiliation: **left as email only.**
- [x] **[Decision]** Running title / keywords: not required for arXiv.

---

## 3. Abstract rewrite

- [x] **[I can draft]** Draft **Variant A** (methods-paper) and put it in `paper/body.tex`.
- [x] **[I can draft]** Variant B not drafted (A only).
- [x] **[I can draft]** Drop unexplained “NM”; drop isotope-agnostic; two ground truths; held-out is a new session.
- [x] **[Decision]** A only.

---

## 4. Introduction — restructure (compress + contributions)

- [x] **[I can draft]** Cut PVE textbook to one paragraph.
- [x] **[I can draft]** PVC history compressed in-place (no appendix).
- [x] **[I can draft]** Reframe motivation as quantitative SPECT; PET SUV as analogue.
- [x] **[I can draft]** Fix citations (Jentzen for SPECT RC, not PET cut-offs).
- [x] **[I can draft]** Soften “unpredictable RCs” to “RC is not a function of volume alone.”
- [x] **[I can draft]** Related work without Salmon / Tran-Gia / Southekal / Matsubara / De Nijs as gelatin+ML support.
- [x] **[I can draft]** Delete “over 3,000” / “precisely calibrated.”
- [x] **[I can draft]** Delete “recovery models collapse” and “big data environment.”
- [x] **[I can draft]** Contributions paragraph at end of intro (`paper/additions.md` item 1).
- [x] **[I can draft]** Explicit not-claims: cold background; overlaps excluded; not a patient PVC tool.

---

## 5. Methods — structural change

- [x] **[I can draft]** Delete “General algorithm and experimental rationale.”
- [x] **[I can draft]** Pipeline flowchart (Figure 1).
- [x] **[I can draft]** New Methods order: phantom → SPECT/CT → segmentation → dataset.
  Assay subsection **not** added (0.2 ignored).

---

## 6. Phantom preparation (expand in main text; do not appendix the recipe)

- [x] **[I can draft once 0.2 arrives]** Base matrix: kept 65°C, 19 g/L, ~1 h stir. Bloom/product **ignored.**
- [x] Insert mix: kept juice + 10 mL/L contrast + \(^{99\mathrm{m}}\)Tc. Product details **ignored.**
- [x] Shape-formation protocol. **Ignored.**
- [x] Containers and filling. **Ignored.**
- [x] **[I can draft]** Do not claim matched attenuation after iodine. Applied + Discussion.
- [x] Air bubbles. **Ignored.**
- [x] **[I can draft]** Shake = pose resampling of the same physical inserts.
- [x] **[I can draft]** Decay = count-level / SNR diversity, same morphology.
- [x] Appendix reagent lots. **Omitted** (not provided).

---

## 7. New subsection: Activity assay and decay

- [x] Assay formula / decay clock. **Ignored** — no new subsection.
- [x] **[I can draft]** Activity density is a preparation target, not CT. Stated in abstract/methods/discussion.
- [x] **[I can draft]** Downgraded “calibrated”; Discussion notes dose-calibrator protocol not reported.

---

## 8. SPECT/CT acquisition and reconstruction (expand; appendix only the DICOM dump)

- [ ] SPECT collimator/windows/etc. **Waiting** on later hand-off.
- [x] Reconstruction extras. **Ignored**; kept existing OSEM 48×1 / 8 mm.
- [x] CT kVp/slice. **Ignored**; same-CT-for-AC-and-mask sentence added.
- [x] Vendor registration sentence. **Omitted** (ignored).
- [x] DICOM appendix. **Omitted.**

---

## 9. Segmentation, matching, exclusions

- [x] CT 60 HU + small-scale justification. Post-processing details ignored.
- [x] Call CT volume “volume at 60 HU.” Applied; limitation in Discussion.
- [x] SPECT 8% then 40%; recursive split as merged-blob splitter.
- [x] SPECT thresholds applied to the PVE-blurred image.
- [x] 1:1 matching; clustered inserts out of scope.
- [x] Exclusion \(n\)/% **ignored**; qualitative exclusion rule applied.
- [x] Cold-background limitation (Methods + Discussion).

---

## 10. Dataset composition (replace repeating bullets)

- [x] Keep Table 1; delete repeating bullets.
- [x] Add Table 2: per-session composition.
- [x] Table 1 fields applied except **volume range** and **excluded \(n\)** (ignored).
- [x] Held-out paragraph: new session; range inside global range; stress test.
- [x] ROI vs independent activity labels. **Omitted** (unknown).

---

## 11. Figures (this part of the paper only)

- [x] Figure captions rewritten (CT volume rendering; SPECT vs CT segmentation). Orange rings / scale **ignored** (no PNG redraw).
- [x] No attrition flowchart (decision: no need).
- [x] Pipeline flowchart added.

---

## 12. Appendix (only what we agreed to park)

- [x] Appendix A (PVE background). **Omitted** — intro compressed instead.
- [x] Appendix B (lots / DICOM). **Omitted** — no materials.
- [x] Per-session table kept in main text.

---

## 13. What this pass does **not** include (later)

- [ ] Radiomics and machine learning architecture — **planned, not rewritten.** Still describes old LR+RF / elastic net.
- [ ] Results, NEMA RC head-to-head metrics, conclusion — **planned, not rewritten.**
- [x] Limitations / Discussion section. **Done** (cold background, iodine/AC, 60 HU, 1:1 exclusions, pose resampling, \(^{99\mathrm{m}}\)Tc-only, assay not reported).
- [ ] Data-availability / ethics / funding (`paper/additions.md` item 2: code and dataset URLs — need links).
- [ ] Implementing more radiomics-method explanation in the paper (belongs with the ML rewrite).

---

## 14. Order of work when we start editing

1. [x] You return **§0**.
2. [x] Lock numbers, units, SPECT wording, ground-truth split.
3. [x] Abstract variant A.
4. [x] Intro rewrite + contributions.
5. [x] Methods restructure (delete rationale; new subsection order; flowchart).
6. [x] Fill phantom / imaging / segmentation from §0 (ignored items omitted).
7. [x] Tables + held-out wording.
8. [x] Figure captions.
9. [x] Appendix skipped (no content).
10. [x] Consistency pass (2,302; held-out session; kBq/ml; no NM in prose).

---

## Remaining (next pass): radiomics / ML / Results / RC

Decisions already made for that pass:
- Cannot re-run the model zoo. Use `reports/results.md` as-is (reduced set: 1,766 + 315).
- Activity RMSE in **kBq/ml**.
- RC uses **SPECT volume** (`Volume_[ml]_NM_40`).
- Best **reported** models (session-CV to select): **Ratio+LightGBM** for activity; **physics \(a V_{\mathrm{NM}}+b\)** for volume. Do not keep LR+RF / elastic net as the named winners.
- Figures planned: held-out identity plots (activity, volume); RC recovery-vs-volume + error-by-volume (once NEMA exists).

**Blocked for NEMA RC.** Same scanner is not enough. Repo has no sphere table and no fitted \(a,b\). Need **one** of:
- six NEMA hot-sphere \(A_{\mathrm{SPECT}}\) values (same 40% SPECT definition as gelatin) at the stated \(4.1\,\mu\mathrm{Ci/ml}\) fill, **or**
- already-fitted MIRD \(a,b\).

Until that arrives, do not claim “outperforming a classical NEMA RC curve.”
