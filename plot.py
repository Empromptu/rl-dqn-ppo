"""
Curvas de aprendizaje a partir de runs/<nombre>__s<seed>/returns.csv.

Cada curva = media entre seeds (línea) ± desvío (sombra). Para promediar seeds
que registran episodios en pasos distintos, se interpola el retorno (suavizado
con media móvil) sobre una grilla común de global_step.

Uso:
    uv run python plot.py            -> genera figuras en plots/
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

KFMT = FuncFormatter(lambda v, _: f"{v/1000:.0f}k")

RUNS = "runs"
OUT = "plots"


def load_curve(name: str, window: int = 20, n_points: int = 200):
    files = sorted(glob.glob(f"{RUNS}/{name}__s*/returns.csv"))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    max_step = min(df["global_step"].max() for df in dfs)
    grid = np.linspace(0, max_step, n_points)
    curves = []
    for df in dfs:
        smooth = df["episodic_return"].rolling(window, min_periods=1).mean()
        curves.append(np.interp(grid, df["global_step"], smooth))
    curves = np.array(curves)
    return grid, curves.mean(0), curves.std(0), len(files)


def plot(ax, name, label, color, **kw):
    c = load_curve(name, **kw)
    if c is None:
        print(f"(sin runs para {name})")
        return
    x, m, s, n = c
    ax.plot(x, m, color=color, label=f"{label} ({n} seeds)")
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.2)


def finish(ax, title, ref=None, ref_label=None):
    if ref is not None:
        ax.axhline(ref, ls="--", color="gray", lw=1, label=ref_label)
    ax.set_title(title)
    ax.set_xlabel("pasos de entorno")
    ax.xaxis.set_major_formatter(KFMT)
    ax.set_ylabel("retorno por episodio")
    ax.grid(alpha=0.3)
    ax.legend()


def main():
    os.makedirs(OUT, exist_ok=True)

    figs = [
        ("dqn_acrobot", "DQN — Acrobot-v1", "DQN", "#1f77b4", -80, "referencia CleanRL/SB3 (≈ −80)"),
        ("ppo_acrobot", "PPO — Acrobot-v1", "PPO", "#d62728", -80, "referencia CleanRL/SB3 (≈ −80)"),
        ("ppo_pendulum", "PPO — Pendulum-v1 (acción continua)", "PPO", "#d62728", -230, "referencia sb3 (−230)"),
        ("dqn_highway", "DQN — highway-fast-v0", "DQN", "#1f77b4", None, None),
    ]
    for name, title, label, color, ref, ref_label in figs:
        fig, ax = plt.subplots(figsize=(7, 4))
        plot(ax, name, label, color, window=5 if "highway" in name else 20)
        finish(ax, title, ref, ref_label)
        fig.tight_layout()
        fig.savefig(f"{OUT}/{name}.png", dpi=120)
        plt.close(fig)

    # Comparación DQN vs PPO sobre el mismo entorno
    fig, ax = plt.subplots(figsize=(7, 4))
    plot(ax, "dqn_acrobot", "DQN", "#1f77b4")
    plot(ax, "ppo_acrobot", "PPO", "#d62728")
    finish(ax, "Acrobot-v1: DQN vs PPO", -80, "referencia (≈ −80)")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(f"{OUT}/acrobot_dqn_vs_ppo.png", dpi=120)
    plt.close(fig)

    # Diagnósticos de PPO: entropía / sigma, KL y clipfrac
    for name, ent_col in [("ppo_acrobot", "entropy"), ("ppo_pendulum", "sigma")]:
        files = sorted(glob.glob(f"{RUNS}/{name}__s*/losses.csv"))
        if not files:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        for f in files:
            df = pd.read_csv(f)
            seed = f.split("__")[1].split("/")[0]
            for ax, col in zip(axes, [ent_col, "approx_kl", "clipfrac"]):
                ax.plot(df["global_step"], df[col], label=seed, alpha=0.8)
                ax.set_title(col)
                ax.set_xlabel("pasos de entorno")
                ax.xaxis.set_major_formatter(KFMT)
                ax.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle(f"{name}: diagnósticos de PPO")
        fig.tight_layout()
        fig.savefig(f"{OUT}/{name}_diagnosticos.png", dpi=120)
        plt.close(fig)

    # Diagnóstico de DQN: Q medio y TD loss
    for name in ["dqn_acrobot", "dqn_highway"]:
        files = sorted(glob.glob(f"{RUNS}/{name}__s*/losses.csv"))
        if not files:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for f in files:
            df = pd.read_csv(f)
            seed = f.split("__")[1].split("/")[0]
            for ax, col in zip(axes, ["q_mean", "td_loss"]):
                ax.plot(df["global_step"], df[col], label=seed, alpha=0.8)
                ax.set_title(col)
                ax.set_xlabel("pasos de entorno")
                ax.xaxis.set_major_formatter(KFMT)
                ax.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle(f"{name}: diagnósticos de DQN")
        fig.tight_layout()
        fig.savefig(f"{OUT}/{name}_diagnosticos.png", dpi=120)
        plt.close(fig)

    print(f"Figuras en {OUT}/")


if __name__ == "__main__":
    main()
