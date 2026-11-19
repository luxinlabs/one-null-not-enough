# Experimental protocol, analysis plan, and results

All numbers below are from a **synthetic simulator**; none is laboratory evidence. Unless stated, results use the 200 test seeds (100–299), 95% bootstrap CIs (2000 resamples), and `n_mc=256`. Reproduce with:

```bash
python src/run_grid.py          # 110 tasks, 363,600 episodes, ~1 min on 8 cores; resumable
python src/analyze.py           # results/analysis/metrics.csv, contrasts.csv
python src/figures.py           # results/figures/F1-F4 (.png, .pdf)
```

Provenance: `results/grid/meta.json` (code hashes, versions, wall time). Each row carries its full configuration.

## Design

### E0 — sanity checks (implemented)
Four latent worlds: (effect absent/present) × (instrument healthy/faulty), with effect ~ Bernoulli(0.5) and fault ~ Bernoulli(`fault_rate`) drawn independently per seed. Ordinary readings follow effect × instrument health. World (no effect, healthy) and (effect, faulty) both yield mean zero. Calibration reads fault; positive control reads instrument health; alternate sensor reads scientific effect. Gaussian noise has equal variance across actions, so equal means imply identical distributions (Figure F1). Unit tests (`tests/`, 18) check equivalence, common random numbers, every data-generating process, the rejection guard, robust Bayes, and that every policy and scenario runs.

### E1 — restricted-action identifiability
Action classes: `resolvable` (all five actions), `unidentifiable` (ordinary/replicate only), and the E4 restrictions `no_calibrate`, `no_positive_control`, `no_alternate`. Budget is swept directly (2, 5, 10); the pilot-only `budget_limited` scenario is not used in the grid. Equivalence is determined analytically: a pair of posterior-supported worlds that disagree on H is unseparable when every action in the class gives both the same mean. Note that `no_alternate` is itself unidentifiable for (H=0, faulty) vs (H=1, faulty): with a faulty instrument only the alternate sensor reads H.

### Policies
| Policy | Model | Rule |
|---|---|---|
| `scientific_only` | fault-unaware, H only | ordinary readings |
| `replicate` | joint (H, fault) | repeat ordinary readings |
| `calibrate_first` | joint | calibrate until fault resolved, then read H |
| `joint_eig`, `joint_risk` | joint | myopic entropy / 0-1 risk reduction per cost; full budget |
| `joint_risk_voi` | joint | `joint_risk`, stop when best expected risk reduction ≤ `voi_tol` (1e-3) |
| `identifiability_stop` | joint | risk reduction over actions that separate a consequential pair; stop when none |
| `identifiability` | joint | `identifiability_stop` + **rejection guard**: never reject H while an unseparable consequential pair (both worlds > `support_eps`=1e-3) remains under the action class |
| `identifiability_eig` | joint | `identifiability` with the entropy objective (E4) |
| `oracle_fault` | joint, told the true fault state | `joint_risk_voi` (privileged upper bound, E4) |
| `robust_bayes` | joint, credal set of fault priors Q | plans and stops like `joint_risk_voi` under the nominal prior; rejects only if P(H) < 0.1, and accepts only if P(H) > 0.9, for **every** fault prior in Q. Primary Q = [0.05, 0.95], fixed before its first run. P(H) is monotone in the fault prior, so the endpoints suffice. |

Decision rule for all: accept if P(H)>0.9, reject if P(H)<0.1, else abstain. "Remove abstention" (E4) is evaluated post hoc as the forced decision P(H)>0.5.

### Data-generating processes (E3)
The planning model always assumes `specified`. Misspecifications (parameters in `src/experiment.py`):
- `drift`: main-sensor offset +0.1 per measurement step.
- `intermittent`: a faulty instrument misbehaves on each measurement with probability 0.5.
- `correlated_noise`: AR(1) noise, ρ=0.8.
- `imperfect_control`: calibration reads 0.5 instead of 1 when faulty; the positive control falsely passes a faulty instrument 30% of the time.
- `unknown_fault`: a fault type outside the model; main-sensor gain 0.4, invisible to calibration.
- Prior shift: truth fault rate 0.5, planner prior 0.1 (overconfident that the instrument is healthy).

### Grid
| Block | Seeds | Conditions | Episodes |
|---|---|---|---|
| `e2` | test | σ{0.25, 0.5, 1} × budget{2, 5, 10} × fault rate{0.1, 0.3, 0.5}, matched prior, 2 action classes, 9 policies | 97,200 |
| `e3` | test | 6 truths × prior{matched, shifted} × σ{0.5, 1} × budget{5, 10}, 2 action classes, 9 policies | 172,800 |
| `e4` | test | 6 truths × 5 action classes × 8 policies, σ=0.5, budget 5 | 48,000 |
| `nmc` | test | n_mc{64, 128, 256, 512} × 3 planning-stream repeats, 6 joint policies | 28,800 |
| `sens` | validation | voi_tol × support_eps ∈ {1e-4, 1e-3, 1e-2}² | 7,200 |
| `credal` | test | Q ∈ {[0.01, 0.99], [0.2, 0.8]} × truth fault rate {0.1, 0.5} × prior {matched, 0.1}; robust Bayes, guard, `joint_risk_voi` | 9,600 |

## Results

### H1 — fault-unaware models falsely reject masked effects: **supported**
Main cell (σ=0.5, budget 5, fault rate 0.5): `scientific_only` false-rejection rate among true-H worlds is 0.56 [0.45, 0.66], and 1.00 when the instrument is faulty. For every joint-model policy it is ≤ 0.01. The masked-effect false rejection of the fault-unaware model stays at 0.34–1.00 under every E3 truth (F3a).

### H2 — joint models with controls beat naive retry at matched or lower cost: **supported when faults are common, mixed when rare**
At fault rate 0.5, resolvable: `joint_risk_voi` minus `replicate` is +0.415 [0.340, 0.495] correct decisions at −0.88 [−0.98, −0.79] cost. At fault rate 0.1 it is −0.065 [−0.125, −0.005] correct (replicate decides more often on the strength of a prior that says faults are rare) but −0.040 [−0.070, −0.015] false rejection. Across the 27 E2 cells, `joint_risk_voi` has significantly more correct decisions in 15.

### H3 — identifiability awareness beyond joint risk minimization: **no gain under a correct, fault-common prior; large false-rejection reduction at a large coverage cost when faults are rare or the prior is wrong**
- **Matched prior, fault rate 0.5:** `identifiability` and `joint_risk_voi` differ in at most 1 of 200 seeds in the main cell. The joint posterior already abstains when the action class cannot separate the hypotheses, so the guard never fires: its effect is exactly 0 in every matched-prior E3 cell.
- **Rare faults (0.1), matched prior, unidentifiable:** `joint_risk_voi` rejects on the strength of the prior. False rejection is 0.11 [0.05, 0.19], and **1.00** for masked effects. `identifiability` has 0.00 false rejection but coverage 0.33 [0.27, 0.40] vs 0.89 [0.84, 0.93]. Under 0-1 loss with a correct prior, the Bayes decision is the better one.
- **Prior shift (planner 0.1, truth 0.5), unidentifiable:** `joint_risk_voi` false rejection is 0.49 [0.39, 0.60] (0.91 for masked effects). `identifiability` has 0.00, at coverage 0.18 vs 0.90 (F3b). This is the regime where the guard is worth having: the rejections it blocks are prior-driven, and the prior is wrong.
- **The guard never helps in the resolvable class.** There, every consequential pair is separable in principle, so the guard never fires. Prior-shift false rejection remains 0.09 vs 0.10 when the budget runs out first.
- **Guard ablation** (`identifiability` vs `identifiability_stop`): the decisions differ only in the unidentifiable and `no_alternate` classes. There, false rejection falls by up to 0.22 and correct decisions by up to 0.60.
- **Cost:** `identifiability` spent significantly more than `joint_risk_voi` in 139 of 150 E2/E3 cells, and significantly less in none. The sensitivity block shows that the ordering depends on the tolerances: with `support_eps`=1e-2, `identifiability` spends less (2.41 vs 2.70 in the unidentifiable class). Do not make cost claims.

### E3 — misspecification other than the prior barely affects the joint model
Matched prior, σ=0.5, budget 5, resolvable: masked-effect false rejection for `joint_risk_voi` is ≤ 0.07 under every truth. `calibrate_first` is fragile to an imperfect control (0.32) and to an unknown fault (0.27), because it trusts a single diagnostic. With a matched prior, `identifiability` and `joint_risk_voi` differ only through the stopping rule. At σ=1 and budget 10, `identifiability` keeps measuring after VOI stopping, gaining at most +0.04 correct decisions at higher cost.

### E4 — ablations (pooled over the six truths unless stated)
- **Oracle fault state:** coverage 0.84 vs 0.63 (resolvable), 0.51 vs 0.20 (unidentifiable). Knowing the fault state is worth about 20–30 points of coverage. This is the headroom any fault-diagnosis method could close.
- **Entropy instead of risk objective:** worse. False rejection is 0.058 vs 0.025 and AURC 0.048 vs 0.029 (resolvable).
- **Remove abstention (forced decision):** false rejection rises to 0.18 (resolvable) and 0.47 (unidentifiable) for every joint policy. Abstention is essential.
- **Restricted controls:** removing calibration or the positive control costs little, because the other one substitutes. Removing the alternate sensor makes (H=0, faulty) vs (H=1, faulty) unidentifiable. There the guard cuts false rejection from 0.035 to 0.000 at coverage 0.27 vs 0.40.

### Robustness of the analysis
- **Monte Carlo precision:** `joint_risk_voi` coverage varies by ≤ 0.04 across n_mc ∈ {64…512} and planning-stream repeats. Mean cost falls slightly with n_mc (4.20 → 4.05, resolvable). n_mc=256 is adequate.
- **Thresholds (validation seeds):** coverage of every policy is insensitive to `voi_tol` and `support_eps` over two orders of magnitude; cost is not (see H3, Cost).
- **σ=0.25** saturates the resolvable class: one alternate-sensor reading settles H.

### Deviation from the preregistered plan (disclosed)
The first grid run (archived in `results/grid_v1`, `results/analysis_v1`) used an `identifiability` policy that only stopped measuring. It omitted step 3 of `PROPOSAL.md`, "abstain from hypothesis rejection". We found this after inspecting the test-seed results: in unidentifiable cells with a low fault prior, the stopping-only policy falsely rejected *more* often than `joint_risk_voi`. We added the guard as specified in the proposal and kept the stopping-only version as `identifiability_stop`. Both are reported. No threshold was tuned on test seeds.

### Robust (credal) Bayes baseline: matches the guard where the action class is unidentifiable; much more conservative elsewhere
σ=0.5, budget 5; coverage / false rejection of true H:

| Condition | Action class | `joint_risk_voi` | `identifiability` (guard) | `robust_bayes` Q=[0.05,0.95] | `robust_bayes` Q=[0.2,0.8] |
|---|---|---|---|---|---|
| fault 0.5, matched prior | resolvable | 0.60 / 0.01 | 0.60 / 0.01 | 0.46 / 0.01 | 0.55 / 0.01 |
| | unidentifiable | 0.18 / 0.00 | 0.18 / 0.00 | 0.16 / 0.00 | 0.18 / 0.00 |
| fault 0.1, matched prior | resolvable | 0.82 / 0.00 | 0.83 / 0.00 | 0.48 / 0.00 | 0.58 / 0.00 |
| | unidentifiable | 0.89 / 0.11 | 0.33 / 0.00 | 0.24 / 0.00 | 0.32 / 0.00 |
| fault 0.5, planner prior 0.1 | resolvable | 0.69 / 0.10 | 0.70 / 0.09 | 0.36 / 0.00 | 0.46 / 0.00 |
| | unidentifiable | 0.90 / 0.49 | 0.18 / 0.00 | 0.11 / 0.00 | 0.18 / 0.00 |

Q=[0.05,0.95] values come from the E2/E3 blocks; Q=[0.2,0.8] values from the `credal` block (same seeds).
- **Unidentifiable class:** robust Bayes and the guard both remove all false rejection; their false-rejection rates are identical in all 81 E2/E3 cells. With the narrow set Q=[0.2,0.8], robust Bayes also matches the guard's coverage (0.18 vs 0.18; 0.32 vs 0.33). Here the guard is essentially a credal rule.
- **Identifiable classes:** the guard never fires and costs no coverage. Robust Bayes withholds decisions whenever the prior could still move P(H) across a threshold: −5 to −25 points of coverage with Q=[0.2,0.8], −14 to −35 with Q=[0.05,0.95]. In return it also removes the prior-driven false rejections that the guard misses when the budget runs out under a shifted prior (0.09 → 0.00).
- Wider Q is uniformly more conservative (coverage 0.07–0.42 at Q=[0.01,0.99]); false rejection is ≤ 0.01 for every Q.
- `robust_bayes` spends exactly what `joint_risk_voi` spends (same planner), so it is cheaper than the guard. The guard's extra measurements buy coverage, not safety.

## Interpretation and limits
- The identifiability guard is a **prior-free rejection certificate**. When the action class cannot distinguish "effect masked by a fault" from "no effect", it refuses to reject, whatever the fault prior says. This makes it equivalent to the joint Bayes decision when the prior already gives faults substantial mass. It is more conservative, and under 0-1 loss worse, when the prior is right and faults are rare. It is much safer when the prior understates faults.
- Against a **robust/credal Bayes rule**, the guard is not new protection: in the unidentifiable class a narrow credal set gives the same false rejection and coverage. What the guard adds is **targeting**. It abstains only where the action class makes the hypotheses indistinguishable, so it keeps full coverage where evidence can settle H. Robust Bayes abstains wherever the prior still matters, which protects against more (including budget-limited cases under a wrong prior) at a large coverage cost.
- A defensible method claim is therefore narrow: an action-relative equivalence check gives credal-Bayes safety against fault-masked falsification without paying credal-Bayes coverage costs in identifiable settings. Reviewers will ask whether a credal rule restricted to unidentifiable directions is the same thing. It probably is, and that should be stated as a characterisation rather than claimed as novelty.
- The finite four-world model makes equivalence checkable exactly. Continuous or high-dimensional models need an approximate certificate, which is where LLM-IDEA-style tooling competes.
- Synthetic success is not wet-lab validation.

## Figures (`results/figures/`)
- F1: mean reading per world and action; the boxed pair is observationally equivalent under ordinary/replicate.
- F2: false rejection vs budget with rare faults; in the resolvable class `identifiability` coincides with `joint_risk_voi`, and in the unidentifiable class with `robust_bayes` (0.00).
- F3: (a) masked-effect false rejection by data-generating fault type; (b) coverage vs false rejection in the unidentifiable class under matched, rare-fault, and shifted priors; diamonds are `robust_bayes`.
- F4: E4 coverage and false rejection by action class, with `robust_bayes` and the oracle bound.

## Real-data replay: GDSC1 plates (pre-registered in docs/PREREG_GDSC.md)
Real cell-viability plates (GDSC1 raw data, 23,883 plates). Ground truth H comes from an **independent screen** (GDSC2 AUC ≤ 0.80 effective, ≥ 0.95 ineffective). Thresholds were fitted on the 30% exploration split (τ=0.3, τ′=0.7, z*=0.452). Code was frozen (`results/gdsc/FROZEN_SHA256`) before the single test run. Test split: 58,343 labelled (plate, drug) units over 960 cell lines and 122 drugs. 95% CIs are bootstrapped over cell lines.

| Rule | False rejection of H=1 | Coverage | Correct rejection of H=0 | Mean cost |
|---|---|---|---|---|
| `single_plate` | 0.157 [0.147, 0.167] | 0.786 | 0.832 | 1.00 |
| `control_gated` (Z′) | 0.126 [0.117, 0.135] | 0.653 | 0.646 | 1.60 |
| `confirm_or_reject` | 0.136 [0.127, 0.145] | 0.766 | 0.812 | 1.44 |
| `confirm_or_abstain` (guard) | 0.026 [0.022, 0.030] | 0.342 | 0.224 | 1.44 |

Stratified by whether an independent plate exists (30% of units have one):
- **With an independent plate:** confirmation cuts false rejection from 0.148 to 0.082 [0.069, 0.096] (−45%). Correct rejection only falls from 0.826 to 0.759 (−8%), so the independent plate is a **specific** diagnostic.
- **Without one:** the guard abstains on every would-be rejection (false rejection 0.000, but correct rejection also 0.000). It is the unidentifiable case of the simulator, and the guard's cost is the same: coverage collapses (0.180).
- **Plate controls are not specific.** Z′ gating removes ~20% of false *and* ~22% of correct rejections; on the exploration split, false rejection did not vary across Z′ tertiles (0.17/0.14/0.18).
- **About half of single-plate false rejections are not plate faults.** Even after independent confirmation 0.082 remain. These are disagreements between the GDSC1 and GDSC2 screens (concentrated in a few drugs, e.g. Wee1 inhibitor, crizotinib), which no plate-level action can resolve.

Interpretation: real single-plate screening shows the failure mode (H1: ~16% false falsification against an independent screen). The diagnostic that separates "masked by the plate" from "no effect" is an independent readout, not on-plate controls. The guard's trade-off on real data matches the simulator: it is safe exactly where the separating action is unavailable, and expensive there.

## LLM agents on real GDSC episodes (Part 2)
**Design (`src/llm_agent.py`, `src/llm_analyze.py`).**
- Episodes: 400 GDSC test units where a single plate looks null (effect < τ). They are stratified 100 each by ground truth (GDSC2) × independent-plate availability, and reweighted to the population of would-be rejections.
- Compounds and cell lines are anonymised.
- The agent sees plate-1 viability at the top two doses and the plate's Z′. It can request the independent plate (cost 2.5), then ACCEPT, REJECT, or ABSTAIN. Output is JSON; there were no invalid replies.
- Models: `gpt-5.5` and `gpt-5.4-mini`, reasoning effort "low", one sample per episode.
- Prompts: `base`, and `fault_aware`, which adds one sentence saying that plate failures can mask effects and are not always visible in controls.
- Total API spend: $5.63, against a $9.50 cap.

**Results.** All metrics are conditional on would-be rejections. "False rejection" is the share of effective compounds that the agent still rejects; 95% bootstrap CIs.

| Agent | False rejection | Correct rejection | Abstain | Requests independent plate | Experimental cost |
|---|---|---|---|---|---|
| single plate (rule) | 1.00 | 1.00 | 0.00 | 0% | 1.00 |
| Z′-gated (rule) | 0.75 [0.68, 0.81] | 0.79 | 0.21 | 0% | 1.60 |
| confirm if possible (rule) | 0.84 [0.79, 0.88] | 0.96 | 0.05 | 100% | 1.73 |
| guard (rule) | 0.13 [0.10, 0.18] | 0.26 | 0.76 | 100% | 1.73 |
| gpt-5.4-mini | 0.95 [0.92, 0.98] | 0.99 | 0.01 | 18% | 1.14 |
| gpt-5.4-mini + fault hint | 0.89 [0.84, 0.93] | 0.89 | 0.10 | 33% | 1.23 |
| gpt-5.5 | 0.78 [0.71, 0.84] | 0.90 | 0.11 | 22% | 1.16 |
| gpt-5.5 + fault hint | 0.72 [0.66, 0.79] | 0.80 | 0.20 | 35% | 1.24 |

**Findings.**
- **LLM agents commit fault-masked falsification.** gpt-5.4-mini rejects 95% of effective compounds that one plate makes look null, close to the single-plate rule. gpt-5.5 rejects 78%.
- **Their diagnostic requests follow the wrong signal.** Both models request the independent plate in 98–100% of episodes with low Z′ (< 0.45), except gpt-5.4-mini base at 63%. With normal Z′ they request it in only 6–15%. On real data, Z′ does not predict false rejection (see the GDSC section). Agents rely on the conventional QC heuristic rather than the separating measurement: even when an independent plate exists, they request it in only 20–35% of episodes; the rule requests it every time.
- **A fault-awareness hint helps little and non-specifically.** It raises requests (≈ 20% → 35%) and abstention, but correct rejections fall about as much as false rejections. In F5 both hinted agents sit near the non-specific diagonal, like Z′ gating.
- gpt-5.5 is more selective than Z′ gating (−22 pts false vs −10 pts correct rejection). It is still less specific than always confirming with an independent plate (−16 vs −4).
- **Limits:** one prompt family, one sample per episode, low reasoning effort, two models from one provider, and anonymised compounds (the agent cannot use prior knowledge of the compound).

Figure F5 (`results/figures/F5_llm_agents`): false vs correct rejection for rules and agents.

## Revision after strict internal review (2026-11-02)
### GDSC: post-hoc robustness (`src/gdsc_robustness.py`, `results/gdsc/robustness.json`; not pre-specified)
- **Dose-range confound.** 76.7% of single-plate false rejections occur where GDSC1's maximum concentration is below GDSC2's.
  - On dose-matched pairs (41% of effective units), single-plate false rejection is 8.8% [7.9, 9.8], against the headline 15.7%.
  - Where an independent plate exists, confirmation lowers it from 8.6% to 5.3% (−38%).
  - Z′ gating cuts false and correct rejection alike (−22% / −20%).
- **Measurement-noise null.** A bivariate Gaussian fitted to the two plate readings (r = 0.685) predicts P(second rejects | first rejects) = 0.463; the observed value is 0.552. Confirmation corrects no more than independent noise would. The corrected errors therefore cannot be attributed to discrete faults.
- **Cell-line baseline.** Same-plate co-rejection is 31.0% vs 12.8%; same cell line on other plates is 21.5% vs 14.4%. A modest plate-level excess remains.
- **Test-split Z′ tertiles.** False rejection is 13.9 / 15.5 / 17.3%, rising with Z′.
- **Ground-truth thresholds.** Single-plate false rejection is 11.5–15.7%; confirmation reduces it by 45–54%.
- **Revised claim.** Real data show single-measurement false rejection that controls do not flag and replicates partly correct. They do not show that instrument faults cause it.

### LLM prompt ablations (total API spend $8.63 of the $9.50 cap)
- **Removing the conventional Z′ threshold from the prompt (no-Z′-hint).**
  - Requests still track Z′: gpt-5.5 requests in 88% of low-Z′ vs 10% of normal-Z′ episodes; gpt-5.4-mini in 48% vs 3%.
  - False rejection is unchanged: gpt-5.5 83% → 84% on the same 200 episodes; gpt-5.4-mini 0.96 → 0.97.
- **Explicit loss** (false rejection = 10× abstention). gpt-5.4-mini's false rejection is 0.98.
- **Repeat sampling.** Over three samples, gpt-5.4-mini's false rejection ranges 0.95–0.97 (base) and 0.96–0.97 (no-Z′-hint).
- **Invalid replies.** 10 of 4,200 episodes, counted as abstentions.

## Second strict review (2026-11-02): additional analyses (`src/gdsc_review2.py`, `results/gdsc/review2.json`; post hoc)
- **Exploration-split replication.** Dose-matched single-plate false rejection is 9.7% (test: 8.8%), and 75.3% of false rejections have the lower GDSC1 dose (test: 76.7%). Confirmation on dose-matched pairs lowers false rejection from 9.1% to 5.1%. The copula null predicts 47.6% co-rejection; 53.9% is observed.
- **Replicate plates are technical replicates.** 98.9% share the cell stock (CELL_ID) and 8.9% the seeding date. Co-rejection is 70.3% for same-date replicates (n=74) and 53.9% for different-date replicates (n=861).
- **Assay.** Single-plate false rejection is 12.0% for GDSC1 assay "a" and 19.8% for assay "s".
- **Copula null on the test split.** It predicts 52.1%; 55.2% is observed (the Gaussian null predicts 46.3%).
- **Decision-theoretic check of the LLM loss prompts.**
  - P(effective | one plate looks null) = 0.090.
  - Under 10:1, rejection is near-optimal: single plate 0.999, best rule 0.975, gpt-5.4-mini 1.011.
  - Under 50:1, abstention is optimal (1.10; confirm-or-abstain 1.53; single plate 4.60). gpt-5.4-mini (snapshot 2026-03-17, confirmed in all 471 replies) still rejected 95% of effective compounds and scored 4.49.
- **API spend** is $9.04 in total.
- **Licence.** GDSC grants non-transferable use for internal research and education, so per-unit GDSC-derived data are excluded from the supplement.

## Review round 3 (post hoc)
- `PYTHONPATH=src python src/gdsc_review3.py` runs the layered decomposition, the matched frontier, the grouped splits, the crossed bootstrap, the dose-rule sensitivity, and the decision-support rates (≈12 s). Outputs: `results/gdsc/review3.json` and `frontier.csv`.
- `PYTHONPATH=src python src/gdsc_figure.py` draws Figure F6.
- `PYTHONPATH=src python src/sim_frontier.py` runs the simulator risk–coverage–cost sweep, including `partial_fault` (32,000 episodes, ≈6 s).
- `PYTHONPATH=src python src/llm_agent.py --model gpt-5.4-mini-2026-03-17 --prompt structured --n 200` runs the decision-support agent; it stopped at the $9.90 cap after 183 episodes. `src/llm_points.py` computes expected points on the shared episodes.
- See `docs/PREREG_GDSC.md` (round 3) for the numbers.

## Review round 4 (post hoc)
- `PYTHONPATH=src python src/gdsc_review4.py`: the paired matched-gap CI and the replicate-availability check (`results/gdsc/review4.json`).
- `PYTHONPATH=src python src/retest_priority.py`: retest prioritisation with disjoint cell lines or compounds (`results/gdsc/retest.json`, `retest_curve.csv`). It also writes per-unit `null_features_*.csv.gz`, which is not redistributed.
- `PYTHONPATH=src python src/agent_campaign.py --model <m> --condition plain|score --n 50`: multi-step tool-using retest campaigns. `--rules-only` writes the rule baselines (`results/agent/rules.csv`); `src/agent_analyze.py` summarises; `src/retest_figure.py` draws F7.
- `PYTHONPATH=src python src/llm_agent.py --prompt baserate|baserate_explain --episodes-from results/llm/gpt-5.4-mini-2026-03-17__structured.jsonl`: the LLM ablation ladder.
- `src/sim_frontier.py` now also writes `results/sim_budget.csv` (structural vs budget-aware guard).
- Total API spend: $15.22.
