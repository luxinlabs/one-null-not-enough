# Pre-registration: real-data replay on GDSC (written 2026-11-02, before any GDSC data was analysed)

Only the first 64 KB of the GDSC1 raw file and its schema PDF had been inspected when this was written (see DATASETS.md). Changes after data access are listed at the end with their reason.

## Question
In a real screening instrument (cell-viability plates), how often does a single plate falsely reject an effective drug? Do the plate's own diagnostics (control wells) or an independent plate prevent it, and at what cost? This is the real-data counterpart of H1/H2 and of the guard/robust-Bayes comparison.

## Data
- GDSC1 raw plate data (`GDSC1_public_raw_data_24Jul22.csv`): the instrument readings.
- GDSC2 fitted dose response (`GDSC2_fitted_dose_response_24Jul22.csv`): an **independent screen** (different assay and plates), used only for ground truth.
- GDSC1 fitted file: only to map `DRUG_ID` → `DRUG_NAME`.

## Split (fixed now)
Plates are assigned by `sha256(str(BARCODE)) mod 10`: residues 0–2 form the **exploration/calibration split** (≈30%), residues 3–9 the **test split** (≈70%). All parameter estimates, thresholds, and any design changes use the exploration split only. Test plates are analysed once, after the code is frozen.

## Units and ground truth
- Unit: one (plate, single-agent library drug slot `Lx`) in GDSC1, matched to GDSC2 on (`COSMIC_ID`, `DRUG_NAME`).
- H=1 (drug effective on that cell line) if GDSC2 `AUC` ≤ 0.80; H=0 if GDSC2 `AUC` ≥ 0.95. Units in between are excluded. (GDSC `AUC` is the normalised area under the viability curve; 1 means no effect.)

## Readings (all from real wells)
Normalised viability per well: v = (I − mean(B)) / (mean(NC) − mean(B)). NC means `NC-1` wells if there are ≥ 4, otherwise `NC-0`. B means blank wells. Plates with < 4 NC or < 4 B wells are excluded.

| Action | Real reading | Cost |
|---|---|---|
| `ordinary` | effect e = 1 − mean(v) over the two highest doses of the drug slot on this plate | 1.0 |
| `replicate` | e for another library slot with the same `DRUG_ID` on the same plate, if any | 1.0 |
| `calibrate` | control quality Z′ = 1 − 3(sd(NC) + sd(B)) / abs(mean(NC) − mean(B)) | 0.6 |
| `alternate_sensor` | e for the same (`COSMIC_ID`, `DRUG_ID`) on a **different GDSC1 plate**, if any | 2.5 |

The positive control is not used: `PC` wells are not on every drug set. Whether it can be added is checked on the exploration split.

## Decision rules compared (thresholds fitted on the exploration split)
1. `single_plate`: reject H if e < τ, accept if e > τ′, otherwise abstain (fault-unaware).
2. `control_gated`: as above, but abstain instead of rejecting when Z′ < z*.
3. `confirm_independent`: when the single plate would reject, buy the alternate plate if one exists; reject only if both plates are below τ.
4. `joint_bayes`: the four-world joint model of `src/experiment.py`. Its per-action Gaussian means/sds are fitted on the exploration split, with "faulty" defined as Z′ < z*; it uses the VOI-stopping planner.
5. `identifiability` (guard) and `robust_bayes` (Q = [0.05, 0.95]) on the same fitted model.

## Primary outcome
False-rejection rate among H=1 units (with 95% bootstrap CI over cell lines), coverage, and mean cost. Secondary outcomes: false rejection stratified by Z′ tertile, and by whether an independent plate exists.

## Known threats (stated in advance)
- GDSC publishes only plates that passed its internal QC, so outright control failures may be rare.
- GDSC1 and GDSC2 use different assays. Some "false rejections" are real assay or biological differences, not instrument faults. The ground truth is "agreement with an independent screen", not biological truth.
- Units on the same plate share the plate and are not independent, so CIs are bootstrapped by cell line.

## Changes after data access
All of these were made on the **exploration split only**, before any test-split unit was evaluated (2026-11-02).

1. **BARCODE dtype.** It is read as a string; mixed parsing had split 15 plates in two. Final counts: 23,883 plates, 7,242 exploration / 16,641 test.
2. **Positive control dropped.** No published plate has `PC` wells.
3. **Same-plate `replicate` dropped as a policy action.** Only 1.3% of units have one.
4. **Finding: Z′ does not predict false rejection.** On the exploration split, single-plate false rejection of H=1 is 0.168 / 0.141 / 0.181 across Z′ tertiles. GDSC publishes only QC-passing plates (only 4.4% of plates have Z′ < 0.3). `control_gated` is kept as pre-registered, but no benefit is expected.
5. **Finding: false rejections decompose into a plate component and a screen component.** For H=1 pairs with two exploration plates, the second plate also rejects in 62% of cases where the first rejects, versus 12% where the first accepts. Of the rest, 38% are correctable by an independent plate. Other H=1 slots on the same plate are falsely rejected 35% of the time when one is, versus 13% otherwise: a plate-level fault that Z′ does not see.
6. **Decision rules replacing rules 3–5.** The joint four-world Gaussian model is not used on GDSC: readings are not Gaussian around 0/1, and the only separating action is the independent plate. Instead the guard logic is tested directly.
   - `single_plate`: reject if e < τ, accept if e > τ′, otherwise abstain. Cost 1.
   - `control_gated`: as `single_plate`, but abstain instead of rejecting when Z′ < z* (z* = 0.452, the 10th percentile of exploration plates). Cost 1.6.
   - `confirm_or_reject`: when `single_plate` would reject, buy an independent plate if one exists (cost +2.5). Reject if it is also below τ, otherwise abstain. **If none exists, reject anyway** (the prior-reliant rule).
   - `confirm_or_abstain`: the same, but **if no independent plate exists, abstain** (the guard: without an independent readout, "masked by the plate" and "no effect" cannot be separated).
   - τ and τ′ are the smallest-loss thresholds whose exploration-split precision is ≥ 0.90 for rejection and for acceptance (grid step 0.05). This mirrors the simulator's 0.1/0.9 posterior thresholds.
   - The independent plate must come from the **same split** as the unit, so test evaluation never uses exploration readings and vice versa.
7. **CIs** are bootstrapped over cell lines (2,000 resamples), as pre-registered.

## Post-hoc analyses after the test run (added 2026-11-02 in response to internal review; NOT pre-specified)
These analyses were run on the test split after the single pre-specified test run. They are reported as post hoc, and the frozen code and the original results are unchanged (`results/gdsc/FROZEN_SHA256`, `metrics_test.csv`). Code: `src/gdsc_robustness.py`; output: `results/gdsc/robustness.json`.

1. **W1: dose-range confound.** 76.7% of single-plate false rejections occur where the GDSC1 maximum concentration is below GDSC2's for the pair. Restricted to dose-matched pairs (GDSC1 max ≥ GDSC2 max; 41% of effective units):
   - single-plate false rejection is 8.8% [7.9, 9.8];
   - with an independent plate, confirmation lowers it from 8.6% to 5.3% [3.9, 7.0], while correct rejection falls from 83.3% to 76.9%;
   - Z′ gating lowers false and correct rejection by similar fractions (−22% vs −20%).
2. **W2: null baselines.**
   - *Cell-line baseline.* Among effective units, other slots on the same plate are falsely rejected 31.0% of the time when one is (12.8% otherwise); on other plates of the same cell line the figures are 21.5% vs 14.4%.
   - *Gaussian measurement-noise null.* For pairs with an independent plate, a bivariate-normal null fitted to the pair readings (correlation 0.685) predicts P(second rejects | first rejects) = 0.463, against an observed 0.552. Confirmation therefore corrects no more errors than pure measurement noise would predict.
3. **W8: test-split versions of exploration statistics.** Single-plate false rejection by Z′ tertile is 13.9% / 15.5% / 17.3%. P(second plate rejects | first rejects) = 0.552; P(second rejects | first accepts) = 0.074.
4. **W9: ground-truth thresholds.** Over GDSC2 AUC thresholds 0.70/0.90, 0.75/0.95, 0.80/0.95 and 0.85/0.95, single-plate false rejection ranges from 11.5% to 15.7%, and confirmation reduces it by 45–54% where an independent plate exists.

Consequence for claims: the real-data evidence supports single-measurement false rejection that on-plate controls do not flag and an independent replicate partly corrects, with a modest plate-level clustering beyond the cell-line baseline. It does not establish that discrete instrument faults cause these errors.

## Post-hoc analyses, round 3 (added 2026-11-02 in response to a second external review; NOT pre-specified)
Code: `src/gdsc_review3.py`; outputs: `results/gdsc/review3.json`, `results/gdsc/frontier.csv`. Frozen code and the original results are unchanged.

1. **Layered decomposition (R1).** Single-plate FR among effective units: all 15.7%; lower GDSC1 dose 20.5%; dose-matched 8.8%; dose-matched assay "a" 5.5%, assay "s" 13.9%. GDSC2 uses CellTiter-Glo on every plate and GDSC1 never does, so no assay-matched stratum exists. Of the 935 false rejections that have a replicate: the replicate does not confirm 44.8% (a Gaussian-copula noise null predicts 47.9%); 40.0% persist with a lower GDSC1 dose; 15.2% persist at a matched dose.
2. **Matched comparison (R2).** On units with a replicate, at the confirm rule's correct rejection (75.9%), FR is 8.2% for confirm, 8.1% for the two-plate mean, 11.3% for a stricter single-plate threshold (about 0.23), and 13.4% for a swept Z′ gate.
3. **Grouped splits (R3).** Re-split by a hash of the cell line or of the compound, with thresholds refitted on the 30% part (τ = 0.35 in both): FR 18.7% / 22.1%; dose-matched 10.7% / 11.5%; with a replicate, single 18.1% → confirm 10.6% (cell line) and 21.2% → 12.5% (compound).
4. **Crossed bootstrap (R4)** (cell lines × compounds; Owen 2007): FR 12.1–19.9%; dose-matched 5.4–14.0%; with a replicate, single 10.6–19.8% and confirm 5.0–12.2%.
5. **Dose-matching rule (R5).** FR 8.8–9.9% across four rules.
6. **Decision-support rates (R6)**, from the exploration split, given to the structured LLM prompt: P(H=1 | null) = 9.1%; P(replicate available | null) = 18%; P(replicate null | null) = 87%; P(H=1 | both null) = 5.5%; P(H=1 | replicate shows an effect) = 32.5%. A Bayes rule using these rates abstains on every episode under 50:1 (1.10 expected points).

Simulator (`src/sim_frontier.py`, `results/sim_frontier.{csv,json}`; post hoc): this run adds a risk–coverage–cost sweep and a continuous `partial_fault` truth (attenuation ~ U(0.3, 1)). In the unidentifiable class, a Bayes rule with rejection threshold 0.05 matches the guard (coverage within 0.005, zero FR) at lower cost.

LLM (`results/llm/gpt-5.4-mini-2026-03-17__structured.jsonl`, `results/llm/points.json`): the decision-support setting completed 183 of 200 episodes before the budget cap, at FR 19%, abstention 56%, and 1.58 expected points versus 4.66 with the loss alone. A planned context-only run was not executed because the budget was exhausted. Total API spend: $9.94 of the $10 limit.

## Post-hoc analyses, round 4 (added 2026-11-02 in response to a third external review; NOT pre-specified)
- **R7, paired matched gap** (`src/gdsc_review4.py` → `results/gdsc/review4.json`): FR(stricter single-plate threshold) − FR(confirm) at the confirm rule's correct rejection = 3.2 points. 95% CI 2.3–4.3 (cell-line bootstrap) and 1.5–5.1 (crossed); 0 of 1,000 resamples at or below zero.
- **R8, replicate availability:** units with and without a replicate have effective shares of 35.8% / 33.8%, median Z′ 0.63 / 0.65, dose-matched shares of 52.6% / 52.8%, and single-plate FR of 14.8% / 16.1%. Replicates cover 754 of the 960 cell lines.
- **Retest prioritisation** (`src/retest_priority.py` → `results/gdsc/retest.json`, `retest_curve.csv`):
  - Setup: which null-looking units should be retested under a budget. The scores use only lab-available features. A first version trained and evaluated on overlapping (cell line, compound) pairs, which leaks the per-pair GDSC2 label; it was replaced before any result was used.
  - Final design: train on exploration nulls from one hash-half of cell lines (or compounds), and evaluate on test nulls from the other half.
  - Held-out cell lines, retesting 10% of nulls recovers: GBM 32.9%, compound history 24.3%, logistic 27.0%, weakest-null 14.3%, lowest-Z′ 6.5%, random 4.8%. Retesting every null recovers 44.8%.
  - Held-out compounds: GBM 20.3%, compound history 22.1% (no gain over the heuristic).
- **Simulator, budget-aware guard** (`identifiability_budget`; `results/sim_budget.csv`): it removes budget-driven false rejections that the structural guard allows (prior shift, resolvable class: 0.136 → 0 at budget 2, and 0.086 → 0 at budget 5), at coverage 0.43 → 0.13 and 0.70 → 0.30. It is identical to the structural guard in the unidentifiable class.
- **LLM ablation ladder** (same 183 episodes, gpt-5.4-mini-2026-03-17; FR / expected points under 50:1):
  - loss only: 0.94 / 4.66
  - + base rates: 0.35 / 2.81
  - + explanations: see `results/llm/summary.csv`
  - + per-episode context (full decision support): 0.19 / 1.58
- **Agent campaign** (`src/agent_campaign.py`, `src/agent_analyze.py`; `results/agent/`): 24 null-looking pairs with replicates, 6 of them effective, from held-out cell lines; 6 retests; tool calls request_replicate / submit_decisions. Rules retest the top 6 by score.

## Post-hoc analyses, round 5 (added 2026-11-02 in response to a fourth external review; NOT pre-specified)
- **R9, curve-level dose comparability** (`src/gdsc_review5.py` → `results/gdsc/review5.json`):
  - GDSC2 IC50 within GDSC1's tested range (90% of effective units): FR 13.1%.
  - IC50 at most max/4 (58%): FR 9.2% [8.3, 10.2]; assay "a" 5.1%, assay "s" 12.6%; with a replicate, confirmation lowers it from 8.5% to 4.7%.
  - 22.8% of single-plate false rejections are not dose-reachable.
- **Retest, strict held-out:** both the cell line and the compound are unseen (2,988 test nulls, 11.2% effective). At a 10% budget: compound history 24.4%, GBM 22.9%, logistic 22.3%, weakest-null 14.3%, lowest-Z′ 3.9%, random 3.6%. The curve file now also records retest precision and the share of correct rejections withheld.
- **Campaign VOI-optimal baseline** (`voi_rule` in `src/agent_campaign.py`):
  - Probability: the GBM probability, isotonically calibrated on cell-line-grouped 5-fold cross-validated training predictions, then shifted to the 25% campaign base rate.
  - Replicate likelihoods: three bins (<0.3, 0.3–0.7, ≥0.7) estimated on the training half.
  - Policy: retest the 6 pairs with the largest value of information, then decide each pair by posterior expected loss. On the 30 shared campaigns it scores 17.7 points (all 50: 19.9). The rule using the uncalibrated probability scores 37.3; gpt-5.5 with the score, 29.5.
- **Calibrated-score agent condition:** the table shows the calibrated, campaign-adjusted probability (`p_effective`).

## Post-hoc analyses, round 6 (added 2026-11-02 in response to a fifth external review; NOT pre-specified)
- **Secondary grouped holdout for retest prioritisation** (`src/retest_independent.py` → `results/gdsc/retest_independent.json`): the retest rules and model are frozen and evaluated on exploration-split nulls (with a replicate) from the held-out half of cell lines. 1,412 nulls, 146 cell lines; not used to train the model or to choose rules or budgets, but they share information through the τ fit, response-rate features, and compound-level labels (see round 7); this is not a fully independent validation. Recovered at a 10% budget: GBM 35.9% [26.5, 48.9], compound history 27.5% [19.3, 37.5], weakest null 16.8%, random 9.2%, lowest Z′ 6.9%.
- **VOI-scaffolded agent** (condition `voi` in `src/agent_campaign.py`): the table adds the calibrated probability, the expected points of the best decision without a retest (`loss_now`), and the expected reduction from retesting (`voi`). gpt-5.4-mini, 50 campaigns.

## Round 7 (2026-11-02; post hoc)
- 2×2 attribution is order-dependent: decisions first 10.5, selection first 2.1, interaction 1.9 [-0.1, 3.9]; symmetric split: decisions 9.5 [6.5, 12.7], selection 1.1 [0.1, 2.1].
- **Data lineage of the retest holdout:** this check is now called a *secondary grouped holdout*. Its units were not used to train the model or to choose any rule or budget. But their GDSC2 labels entered the pre-specified fit of τ, their GDSC1 readings enter other pairs' response-rate features, and compound-level labels from the training cell lines are shared.
- **2×2 decomposition** (`src/agent_2x2.py` → `results/agent/decomposition_2x2.json`; decisions-only condition `voi_forced`, gpt-5.4-mini, 30 campaigns). Points by who selects the retests × who makes the final decisions:
  - agent + agent: 28.3
  - agent + Bayes: 17.9
  - policy + agent: 26.3
  - policy + Bayes: 17.7
  - The final decisions account for 10.5 [7.5, 13.7] of the 10.6-point gap; retest selection accounts for 0.1 [0.0, 0.3].
- Total API spend: $17.73.

### Verification round (2026-11-03; no new analyses, provenance and corrections only)
- Every number in the paper was traced to a results file. Added sources for values that were computed but not stored: test-split Gaussian-copula null (`src/gdsc_review2.py` → `review2.json` `copula_test`: predicted 52.1%, observed 55.2%); ladder expected points for base rates / explanations (`src/llm_points.py` → `points.json`: 2.81, 1.90); Table IV from one seeded bootstrap (`src/agent_table.py` → `results/agent/table4.csv`).
- Corrections: retesting every null recovers 45.3% (not 44.8%) for held-out cell lines; random-retest precision .09; loss-50:1 FR CI lower bound .91 and loss-10:1 .95; VOI-scaffold vs policy paired-gap CI 7.5–13.7 (was 7.4–14.1); retest overlap 5.7 of 6; premature rejections on the 30 shared campaigns 1.8 pairs / 7.8 points (the earlier 1.5 / 6.7 were over 50 campaigns); simulator guard advantage at matched coverage at most 1.8 points of false rejection (about 1.5 of 81 effective episodes), not "one to three".

### Post-hoc round 8 (2026-11-03; reviewer-driven, no API calls)
- **R9, plate anomalies** (`src/gdsc_review6.py` → `results/gdsc/review6.json`): logistic regression of single-plate rejection on 20,070 effective test units with cell-line, compound, and drug-set fixed effects, dose matching, and assay. Odds ratios per SD (cell-line bootstrap, 200): Z′ 0.81 [0.54, 1.02], negative-control CV 0.93 [0.64, 1.14], log signal window 1.02 [0.91, 1.16], share of other compounds on the plate that also look null 3.98 [3.67, 4.57]; on dose-matched units 5.29.
- **R10, replicate availability** (same script): propensity of having a replicate (retest features + label) among 16,978 held-out-cell-line test nulls; IPW-weighted recovery at 10%: retest all 43.4%, GBM 31.5%, compound history 23.5% (54% of all), lowest Z′ 4.8%, random 4.3%.
- **Agent terminal decisions** (`src/agent_decisions.py` → `results/agent/decision_errors.json`, 30 shared campaigns): on unretested pairs (probability shown, action determined), deviations from the expected-loss action: mini calibrated 2.0%, mini VOI 0.7%, mini decides-only 0.7%, gpt-5.5 calibrated 0 of 540. On retested pairs, deviations from posterior expected-loss decisions: 2.5–3.2 per campaign, 8.3–10.2 points; mostly REJECT after a null replicate where abstention was optimal (mini VOI 52/81, mean prior 0.66, 64% effective; gpt-5.5 66/96).

## Post-hoc round 9 (2026-11-03)
No new API calls. Added (all exploratory): `src/agent_beliefs.py` (beliefs implied by agent decisions on retested pairs; results/agent/beliefs.json),
`src/agent_costs.py` (loss sensitivity of reading the replicate as conclusive; results/agent/cost_sensitivity.json), per-campaign count of
decision vs selection share in the 2x2 decomposition (23 of 30). `src/agent_campaign.py` gains conditions `forced_lr` and `forced_post`
(likelihood ratio or posterior shown on the policy's retests); implemented, not run. Fig. 3b drops the raw-score agent rows (still in Table V).

## Camera-ready round (2026-11-16)
Post hoc, in response to reviewer 3 (suggestion 2). New runs on the same 30 campaigns with the policy's retests: gpt-5.4-mini forced_post
(low and high effort), gpt-5.4-mini voi_forced at high effort, gpt-5.5 forced_post. Cost $2.98 (study total $20.71). One high-effort campaign
failed on malformed tool JSON and was rerun. Analysis: src/agent_update_test.py -> results/agent/update_test.json. Logs for non-default effort
now carry an __<effort> suffix. Section II of the submitted version moved to supplementary S10.
