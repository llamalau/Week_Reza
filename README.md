![ABM animation](figs/abm.gif)

## BMI 500 — HW 2: Agent-Based Modeling of Pandemic Spread

- **Name/Contact**: Your Name — your.email@example.com
- **Question answered**: HW 2 — Agent-Based Modeling of Pandemic Spread (A and B)
- **Repository structure (enumerated by parts)**:
  - `HW2.py` — Part A (base model) and Part B (social distancing) implementation
  - `HW2_ABM.ipynb` — Reproducible notebook generating figures and tables
  - `figs/` — Saved plots (created when running the notebook)
  - `requirements.txt` — Minimal Python dependencies

## Objective
Simulate pandemic spread with an ABM and study social distancing effects on infection and recovery.

### Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python HW2.py  # quick demo run with plots
```
Or open `HW2_ABM.ipynb` and run all cells to regenerate figures into `figs/`.

## A. Base model
- **i. Environment/initialization**
  - 75×75 bounded grid; 100 agents.
  - States: S=95, I=5, R=0; random positions.
- **ii. Behaviors**
  - Movement: random up/down/left/right; may stay if `move_prob<1`.
  - Transmission: if S shares a cell with any I, S→I with prob `p`.
  - Recovery: I→R with prob `q` per step.
- **iii. Simulation**
  - 200 steps; record counts S/I/R each step; plot trajectories.
  - See `figs/base_single.png`.
- **iv. Sensitivity**
  - Sweep: `p∈{0.05,0.1}`, `q∈{0.02,0.05}`; multi-seed averages.
  - `p↑` → higher/faster peak; `q↑` → shorter outbreak/tail.
  - Averaging smooths stochastic runs; common trend emerges.
  - Outputs: `figs/sensitivity_table.txt`, `figs/avg_random_inits.png`.

## B. Social distancing and interventions
- **i. Measures and parameters**
  - Reduced movement: adjust `move_prob` (↓ → fewer contacts).
  - Avoid infected: adjust `avoidance` (↑ → move away when near I).
  - 200-step runs; average across seeds; plot S/I/R.
  - Peak-control (healthcare capacity): primarily `move_prob` (policy); also `avoidance`. Lowering disease `p` helps but is not a policy knob.
- **ii. Comparison (no distancing vs distancing)**
  - Peak infected: lower with distancing.
  - Time to peak: delayed with distancing.
  - Spread/duration: slower spread; often longer to peak but lower burden.
  - Public health: flatten peak to avoid system overwhelm.
- **iii. Additional sensitivity**
  - Vary `move_prob` (e.g., 1.0→0.35): stricter distancing → lower/delayed peaks, smaller final R.
  - Vary `avoidance`: higher avoidance further reduces peak.

### Figures
- `figs/base_single.png` — Single-run S/I/R
- `figs/avg_random_inits.png` — Averages across seeds
- `figs/sensitivity_table.txt` — p, q, peak, t_peak, final_R (averaged)
- `figs/distancing_compare.png` — No distancing vs distancing (averages)
- `figs/abm.gif` — Retro pixel animation of agent dynamics

## Key insights
- `move_prob↓` and `avoidance↑` reduce/delay peaks; `p↑` amplifies peaks; `q↑` shortens tails.
- Sparse settings reduce contacts; density/interaction design matters for dynamics.

## Relevance to model-based ML
- Interpretable generative rules; parameters enable calibration/sensitivity akin to ML hyperparameters; seed-averaging reduces variance.

## Suggestions for improvements
- More contact (Moore neighborhood), heterogeneous agents, hotspots, adaptive policies, vaccination; parameter inference (e.g., ABC).

## How each part maps to code
- Part A — Base model: `SimCfg`, `run_sim`, `plot_curves`, `sweep` in `HW2.py`.
- Part B — Social distancing: `move_prob` and `avoidance` fields in `SimCfg`; examples in `demo()` and notebook.

Run `HW2_ABM.ipynb` to save figures in `figs/`.

### AI tool disclosure
This repository was authored with assistance from an AI coding assistant for code edits and documentation organization.

### References
- Python docs and scientific stack (`numpy`, `matplotlib`).
- Standard agent-based epidemic modeling literature (cellular automata, random walks, SIR dynamics).


