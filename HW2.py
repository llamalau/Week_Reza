from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


# States
S, I, R = 0, 1, 2


@dataclass(frozen=True)
class SimCfg:
    grid: int = 75
    agents: int = 100
    steps: int = 200
    p: float = 0.05  # infection prob per step when sharing a cell
    q: float = 0.02  # recovery prob per infected per step
    move_prob: float = 1.0  # social distancing: probability an agent moves
    avoidance: float = 0.0  # social distancing: probability of avoidance behavior if S near I
    seed: int | None = None


def run_sim(cfg: SimCfg) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    # positions (y, x) in [0, grid-1]
    pos = rng.integers(0, cfg.grid, size=(cfg.agents, 2), endpoint=False, dtype=np.int64)
    state = np.full(cfg.agents, S, dtype=np.int8)
    # init 5 infected (rest susceptible)
    state[rng.choice(cfg.agents, size=5, replace=False)] = I
    sus_hist, inf_hist, rec_hist = np.empty(cfg.steps + 1, int), np.empty(cfg.steps + 1, int), np.empty(cfg.steps + 1, int)
    sus_hist[0], inf_hist[0], rec_hist[0] = (state == S).sum(), (state == I).sum(), (state == R).sum()

    # movement choices: up, down, left, right
    moves = np.array([[1, 0], [-1, 0], [0, -1], [0, 1]], dtype=np.int64)

    for t in range(1, cfg.steps + 1):
        # Decide who moves (social distancing)
        will_move = rng.random(cfg.agents) < cfg.move_prob
        # Propose random directions for movers
        dir_idx = rng.integers(0, 4, size=cfg.agents)
        step_vec = moves[dir_idx]

        # Avoidance: S agents try to move away from cells with I if possible
        if cfg.avoidance > 0:
            # linear cell index for quick occupancy lookups
            lin = pos[:, 0] * cfg.grid + pos[:, 1]
            inf_cells = np.zeros(cfg.grid * cfg.grid, dtype=np.int32)
            np.add.at(inf_cells, lin[state == I], 1)
            inf_here = inf_cells[lin] > 0
            need_avoid = (state == S) & will_move & inf_here & (rng.random(cfg.agents) < cfg.avoidance)
            if need_avoid.any():
                candidates = (pos[need_avoid, None, :] + moves[None, :, :]).clip(0, cfg.grid - 1)
                cand_lin = candidates[:, :, 0] * cfg.grid + candidates[:, :, 1]
                # score directions by infected presence (prefer fewer infected)
                cand_score = inf_cells[cand_lin]
                # break ties randomly via tiny noise, then row-wise argmin
                best = np.argmin(cand_score + 1e-6 * rng.random(cand_score.shape), axis=1)
                step_vec[need_avoid] = moves[best]

        # Apply movement
        pos[will_move] = (pos[will_move] + step_vec[will_move]).clip(0, cfg.grid - 1)

        # Transmission: S sharing a cell with any I gets infected with prob p
        lin = pos[:, 0] * cfg.grid + pos[:, 1]
        inf_count = np.bincount(lin[state == I], minlength=cfg.grid * cfg.grid)
        inf_present = inf_count > 0
        at_risk = (state == S) & inf_present[lin]
        to_infect = at_risk & (rng.random(cfg.agents) < cfg.p)
        state[to_infect] = I

        # Recovery
        rec_now = (state == I) & (rng.random(cfg.agents) < cfg.q)
        state[rec_now] = R

        sus_hist[t], inf_hist[t], rec_hist[t] = (state == S).sum(), (state == I).sum(), (state == R).sum()

    return {"S": sus_hist, "I": inf_hist, "R": rec_hist}


def plot_curves(runs: list[dict[str, np.ndarray]], title: str, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    steps = np.arange(runs[0]["S"].size)
    for lab, color in (("S", "tab:blue"), ("I", "tab:red"), ("R", "tab:green")):
        mean = np.mean([r[lab] for r in runs], axis=0)
        ax.plot(steps, mean, label=lab, color=color)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("agents")
    ax.legend()
    return ax


def run_sim_trace(cfg: SimCfg) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    pos = rng.integers(0, cfg.grid, size=(cfg.agents, 2), endpoint=False, dtype=np.int64)
    state = np.full(cfg.agents, S, dtype=np.int8)
    state[rng.choice(cfg.agents, size=5, replace=False)] = I
    moves = np.array([[1, 0], [-1, 0], [0, -1], [0, 1]], dtype=np.int64)
    pos_hist = np.empty((cfg.steps + 1, cfg.agents, 2), dtype=np.int64)
    state_hist = np.empty((cfg.steps + 1, cfg.agents), dtype=np.int8)
    pos_hist[0] = pos
    state_hist[0] = state

    for t in range(1, cfg.steps + 1):
        will_move = rng.random(cfg.agents) < cfg.move_prob
        dir_idx = rng.integers(0, 4, size=cfg.agents)
        step_vec = moves[dir_idx]
        if cfg.avoidance > 0:
            lin = pos[:, 0] * cfg.grid + pos[:, 1]
            inf_cells = np.zeros(cfg.grid * cfg.grid, dtype=np.int32)
            np.add.at(inf_cells, lin[state == I], 1)
            inf_here = inf_cells[lin] > 0
            need_avoid = (state == S) & will_move & inf_here & (rng.random(cfg.agents) < cfg.avoidance)
            if need_avoid.any():
                candidates = (pos[need_avoid, None, :] + moves[None, :, :]).clip(0, cfg.grid - 1)
                cand_lin = candidates[:, :, 0] * cfg.grid + candidates[:, :, 1]
                cand_score = inf_cells[cand_lin]
                best = np.argmin(cand_score + 1e-6 * rng.random(cand_score.shape), axis=1)
                step_vec[need_avoid] = moves[best]
        pos[will_move] = (pos[will_move] + step_vec[will_move]).clip(0, cfg.grid - 1)

        lin = pos[:, 0] * cfg.grid + pos[:, 1]
        inf_count = np.bincount(lin[state == I], minlength=cfg.grid * cfg.grid)
        inf_present = inf_count > 0
        at_risk = (state == S) & inf_present[lin]
        to_infect = at_risk & (rng.random(cfg.agents) < cfg.p)
        state[to_infect] = I

        rec_now = (state == I) & (rng.random(cfg.agents) < cfg.q)
        state[rec_now] = R

        pos_hist[t] = pos
        state_hist[t] = state

    return {"pos": pos_hist, "state": state_hist}


def render_gif(cfg: SimCfg, out_path: str = "figs/abm.gif", fps: int = 10) -> None:
    tr = run_sim_trace(cfg)
    pos, st = tr["pos"], tr["state"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, cfg.grid - 0.5)
    ax.set_ylim(-0.5, cfg.grid - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    def colors(arr: np.ndarray) -> list[str]:
        return np.where(arr == S, "#1f77b4", np.where(arr == I, "#d62728", "#2ca02c")).tolist()
    scat = ax.scatter(pos[0, :, 1], pos[0, :, 0], c=colors(st[0]), s=14)
    ax.set_title("step 0")
    def update(frame: int):
        scat.set_offsets(np.c_[pos[frame, :, 1], pos[frame, :, 0]])
        scat.set_color(colors(st[frame]))
        ax.set_title(f"step {frame}")
        return scat,
    ani = animation.FuncAnimation(fig, update, frames=pos.shape[0], interval=100, blit=True)
    ani.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def sweep(cfg: SimCfg, p_vals: list[float], q_vals: list[float], runs: int = 5, seeds: list[int] | None = None) -> list[tuple[float, float, int, int, int]]:
    res: list[tuple[float, float, int, int, int]] = []
    seeds = seeds or list(range(runs))
    for p in p_vals:
        for q in q_vals:
            inf_mat = []
            for s in seeds:
                out = run_sim(SimCfg(grid=cfg.grid, agents=cfg.agents, steps=cfg.steps, p=p, q=q, move_prob=cfg.move_prob, avoidance=cfg.avoidance, seed=s))
                inf_mat.append(out["I"])
            mean_I = np.mean(inf_mat, axis=0)
            peak = int(mean_I.max())
            t_peak = int(mean_I.argmax())
            final_R = int(np.mean([run_sim(SimCfg(grid=cfg.grid, agents=cfg.agents, steps=cfg.steps, p=p, q=q, move_prob=cfg.move_prob, avoidance=cfg.avoidance, seed=s))["R"][-1] for s in seeds]))
            res.append((p, q, peak, t_peak, final_R))
    return res


def demo() -> None:
    base = SimCfg()
    # Single run
    out = run_sim(base)
    plt.figure(figsize=(10, 6))
    plot_curves([out], "Single run (base)")

    # Sensitivity over p, q
    p_vals, q_vals = [0.05, 0.1], [0.02, 0.05]
    results = sweep(base, p_vals, q_vals, runs=8)
    print("Sensitivity (mean over runs): p, q, peak_I, t_peak, final_R")
    for row in results:
        print(row)

    # Average over multiple random initializations
    runs = [run_sim(SimCfg(seed=s)) for s in range(10)]
    plt.figure(figsize=(10, 6))
    plot_curves(runs, "Average over random initial conditions")

    # Social distancing: reduced movement and avoidance
    sd_cfg = SimCfg(move_prob=0.35, avoidance=0.8)
    sd_runs = [run_sim(SimCfg(p=0.1, q=0.02, move_prob=sd_cfg.move_prob, avoidance=sd_cfg.avoidance, seed=s)) for s in range(10)]
    ns_runs = [run_sim(SimCfg(p=0.1, q=0.02, move_prob=1.0, avoidance=0.0, seed=s)) for s in range(10)]
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    plot_curves(ns_runs, "No distancing (avg)", ax1)
    ax2 = plt.subplot(1, 2, 2)
    plot_curves(sd_runs, "With distancing (avg)", ax2)

    # Concise answers
    print("\nAnswers:")
    print("- Reduced movement: change 'move_prob'.")
    print("- Avoiding infected cells: change 'avoidance'.")
    print("- To reduce peak infected: primarily lower 'move_prob' and/or raise 'avoidance' (also lower 'p').")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()


