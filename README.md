# When One Null Is Not Enough

Supplementary material for *When One Null Is Not Enough: Auditing Single-Measurement Falsification in Autonomous Labs* (Luxin Zhang, AI4AutoSci 2026 workshop at IEEE BigData 2026).

The paper replays 23,883 GDSC1 cell-viability plates against the independent GDSC2 screen and asks four questions:
1. how often a single plate leads to a discordant rejection;
2. which diagnostics reduce it;
3. which nulls to retest first;
4. how LLM agents decide from the same evidence.

## Contents

| Path | What it is |
|---|---|
| `supplementary_tables.pdf` | Supplement S1–S10: simulator tables, extra ablations, data partitions, scoring, resampling, data provenance, agent beliefs and the direct likelihood-ratio test, and the formal setting with the explanatory bound |
| `docs/PREREG_GDSC.md` | Analysis plan, frozen before the single test run, with post hoc amendments marked |
| `docs/EXPERIMENTS.md` | Experiment log |
| `src/` | All analysis, experiment, agent, and figure code |
| `results/` | Aggregate results behind every table and figure, agent replies (`results/llm/*.jsonl`, `results/agent/*.jsonl`), and figures |
| `data/SHA256SUMS` | Checksums of the GDSC release 8.4 files used |
| `tests/` | Simulator tests |

## Reproducing

Install the dependencies with `pip install -r requirements.txt`. Then run scripts from the repository root with `PYTHONPATH=src`.

- **GDSC replay (RQ1–RQ3).** Per-unit GDSC data are not redistributed, in line with the GDSC terms of use. Download GDSC release 8.4 (raw data and fitted dose–response files) from the Sanger release site, check them against `data/SHA256SUMS`, then run:
  - `src/gdsc_prepare.py`
  - `src/gdsc_replay.py --split explore`
  - `src/gdsc_replay.py --split test`
  - `src/gdsc_robustness.py`
  - `src/gdsc_review2.py` to `src/gdsc_review6.py`
  - `src/retest_priority.py`
  - `src/retest_independent.py`
- **LLM agents (RQ4).**
  - Single-step episodes: `src/llm_agent.py`, `src/llm_analyze.py`, `src/llm_points.py`.
  - Campaigns: `src/agent_campaign.py`, `src/agent_table.py`, `src/agent_2x2.py`, `src/agent_decisions.py`, `src/agent_beliefs.py`, `src/agent_costs.py`.
  - Direct likelihood-ratio and updated-probability test: `src/agent_update_test.py`.
  - Running the agents needs an `OPENAI_API_KEY` in the environment. The saved replies let you re-run all analyses without one.
- **Simulator.** `src/run_grid.py`, `src/analyze.py`, `src/sim_frontier.py`, `src/figures.py`.

Random seeds and the model snapshots served by the API are listed in supplement S8.

## Notes

The real-data labels mean agreement with GDSC2, not biological truth. A discordant rejection is therefore not a confirmed false negative.

## Contact

Luxin Zhang, luxin.zhang92@outlook.com
