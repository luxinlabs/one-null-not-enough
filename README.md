# When One Null Is Not Enough

**Auditing Single-Measurement Falsification in Autonomous Labs**

Luxin Zhang · AI4AutoSci 2026 Workshop, 2026 IEEE International Conference on Big Data (IEEE BigData 2026), Phoenix, AZ, USA, December 14–17, 2026

This repository contains the code, analysis plan, aggregate results, and supplementary material for the paper. Everything here supports the numbers reported in the paper and lets them be regenerated from the public GDSC data.

---

## Background

Self-driving laboratories and language-model agents increasingly plan experiments, run them on automated hardware or in software, and decide which hypotheses survive. Each decision depends on a reading the system produced itself. A reading that shows no effect is usually taken as evidence against the hypothesis. The same reading can also occur when the hypothesis is true and the measurement failed to show it, for example because of an instrument fault, a plate problem, a dose range that is too low, or noise.

We call this **single-measurement falsification**: rejecting a hypothesis on one null reading when another available measurement could have overturned it. Autonomous experimentation makes this error more costly than it is in a conventional laboratory, for three reasons:

1. **Budgets favour single measurements.** Primary screens are often run without replicates, and only apparent hits are followed up.
2. **The error is silent.** A false positive is caught when the hit is confirmed. A false negative leaves the queue and produces no later signal.
3. **The decision is increasingly delegated** to agents that lack the experience that tells a scientist when a result looks wrong.

Real data give no ground truth for a null, and the mechanism behind any single error cannot be observed. We therefore audit the problem empirically. We replay real robotic screening plates as if an autonomous laboratory were deciding from them, and score each decision against an independent screen of the same compound–cell-line pairs.

## Research questions and main findings

The study replays **23,883 cell-viability plates** from the GDSC1 drug screen against the independent **GDSC2** screen. The analysis plan was frozen on an exploration split before a single run on the test split, which has 58,343 labelled units.

| Question | Main finding |
|---|---|
| **RQ1.** How often does one plate disagree with an independent reference? | Where both screens tested comparable doses, a single-plate rule rejected about **9%** of the units the reference classifies as effective (15.7% overall, mostly from lower GDSC1 doses). |
| **RQ2.** Which diagnostic separates a false null from a true one? | Plate control quality (Z′) did **not** flag these rejections. A replicate plate lowered them by **3.2 points** (95% CI 1.5–5.1) more than a stricter threshold, at the same correct-rejection rate. |
| **RQ3.** When only some nulls can be retested, which ones should be? | Retesting the 10% of nulls ranked highest by the compound's activity on other cell lines recovered about **half** of the benefit of retesting every null. Ranking by control quality was no better than random. |
| **RQ4.** What do LLM agents do with the same evidence? | Agents chose nearly the same retests as a model-optimal value-of-information policy but scored about **10 points worse**, because they treated a confirming null replicate as conclusive. When told the replicate's likelihood ratio, they matched the policy. |

An illustrative simulator isolates the case that a replicate cannot resolve: a fault that every allowed measurement shares. In that case an identifiability guard serves as an auditing device rather than a more accurate rule.

**Scope.** Real-data labels mean agreement with GDSC2, not biological truth. A discordant rejection is therefore not a confirmed false negative. The agent results describe the evaluated models and prompts (OpenAI gpt-5.5 and gpt-5.4-mini), not an inherent limit of language models. See the paper's Discussion and Limitations for details.

## Repository structure

| Path | Contents |
|---|---|
| `supplementary_tables.pdf` | Supplementary material S1–S10: simulator tables, extra ablations, data partitions, scoring, resampling, data provenance, agent beliefs and the direct likelihood-ratio test, and the formal setting with the explanatory bound |
| `docs/PREREG_GDSC.md` | Analysis plan, frozen before the single test run; post hoc amendments are marked |
| `docs/EXPERIMENTS.md` | Experiment log |
| `src/` | All analysis, experiment, agent, and figure code |
| `results/gdsc/` | GDSC replay results (RQ1–RQ3) and the frozen code hashes (`FROZEN_SHA256`) |
| `results/llm/` | Single-step agent episodes: raw replies (`*.jsonl`), summaries, and points |
| `results/agent/` | Multi-step retest campaigns: raw replies (`*.jsonl`), tables, decomposition, and update tests |
| `results/analysis/`, `results/sim_*` | Simulator results |
| `results/figures/` | All figures |
| `data/SHA256SUMS` | Checksums of the GDSC release 8.4 files used |
| `tests/` | Simulator tests |

### Where each result comes from

| Paper item | Result files |
|---|---|
| Table II, Findings 1, 4 | `results/gdsc/metrics_test.csv`, `robustness.json` |
| Table III, Finding 2 | `results/gdsc/review2.json`–`review5.json`, `robustness.json` |
| Findings 3, 5, 6 | `results/gdsc/review2.json`, `review3.json`, `review4.json`, `review6.json`, `robustness.json` |
| Table IV, Fig. 7a, Finding 7 | `results/gdsc/retest.json`, `retest_curve.csv`, `retest_independent.json`, `review6.json` |
| Table V, Findings 8–9 | `results/llm/summary.csv`, `points.json` |
| Tables VI–VII, Fig. 7b, Findings 10–11 | `results/agent/table4.csv`, `decomposition_2x2.json`, `decision_errors.json`, `beliefs.json`, `update_test.json` |
| Finding 12, Table S1 | `results/analysis/`, `results/sim_frontier.json`, `results/sim_budget.csv` |

## Data

The study uses **GDSC release 8.4** (files dated 24 July 2022) from the Wellcome Sanger Institute. In line with the GDSC terms of use, per-unit GDSC data are **not redistributed** here; they can be regenerated with the scripts below.

1. Download the following files from the [GDSC release directory](https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/) into `data/raw/gdsc/`. The directory is a moving alias, so check that the files match the release used here.
   - `GDSC1_public_raw_data_24Jul22.csv`
   - `GDSC1_fitted_dose_response_24Jul22.csv`
   - `GDSC2_fitted_dose_response_24Jul22.csv`
   - `screened_compounds_rel_8.4.csv`
   - `GDSC_Raw_Data_Description.pdf`
2. Verify the checksums:
   ```bash
   cd data/raw/gdsc && shasum -a 256 -c ../../SHA256SUMS
   ```

## Reproducing the results

Python 3.13 was used; the code also runs on recent Python 3 versions. Install the dependencies and run all scripts from the repository root:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

**GDSC replay (RQ1–RQ3).**

```bash
python src/gdsc_prepare.py                 # per-plate controls and per-unit readings
python src/gdsc_replay.py --split explore  # fit thresholds on the exploration split
python src/gdsc_replay.py --split test     # single pre-specified test run
python src/gdsc_robustness.py
python src/gdsc_review2.py                 # ... through gdsc_review6.py
python src/retest_priority.py
python src/retest_independent.py
```

**LLM agents (RQ4).**

- Single-step episodes: `src/llm_agent.py`, then `src/llm_analyze.py` and `src/llm_points.py`.
- Retest campaigns: `src/agent_campaign.py`, then `src/agent_table.py`, `src/agent_2x2.py`, `src/agent_decisions.py`, `src/agent_beliefs.py`, and `src/agent_costs.py`.
- Direct likelihood-ratio and updated-probability test: `src/agent_update_test.py`.

New agent runs require an `OPENAI_API_KEY` environment variable and incur API costs; the full study cost US$21.72. The saved replies in `results/llm/` and `results/agent/` let every analysis be re-run without an API key. Served model snapshots are recorded in the logs and listed in supplement S8.

**Simulator.**

```bash
python src/run_grid.py
python src/analyze.py
python src/sim_frontier.py
python src/figures.py
```

Random seeds are fixed: NumPy seed 0 for bootstrap and campaign sampling, `random_state=0` for gradient boosting, and simulator test seeds 100–299 (supplement S8).

## Citation

If you use this code or these results, please cite:

```bibtex
@inproceedings{zhang2026onenull,
  author    = {Luxin Zhang},
  title     = {When One Null Is Not Enough: Auditing Single-Measurement Falsification in Autonomous Labs},
  booktitle = {Proceedings of the 2026 IEEE International Conference on Big Data (BigData)},
  note      = {AI4AutoSci 2026 Workshop},
  year      = {2026}
}
```

Please also cite the GDSC resource (Yang et al., *Nucleic Acids Research*, 2013; Iorio et al., *Cell*, 2016) when using the data.

## Acknowledgment of AI assistance

In line with the IEEE policy disclosed in the paper, Claude (Anthropic) was used through Claude Code, under the author's direction, to implement the analysis, experiment, and figure code to the author's experimental design. The OpenAI models gpt-5.4-mini and gpt-5.5 are subjects of the experiments, not authoring tools. The author designed the study and is responsible for all content.

## Contact

Luxin Zhang · luxin.zhang92@outlook.com

Questions and issues are welcome through [GitHub Issues](https://github.com/luxinlabs/one-null-not-enough/issues).
