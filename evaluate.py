"""
Evalúa un modelo entrenado sin exploración:
    DQN -> acción = argmax_a Q(s, a)          (epsilon = 0)
    PPO -> discreto: acción más probable; continuo: la media mu(s) de la gaussiana

Uso:
    uv run python evaluate.py runs/dqn_acrobot__s1 --episodes 20
    uv run python evaluate.py runs/ppo_pendulum__s1 --render          # ventana en vivo
    uv run python evaluate.py runs/dqn_highway__s1 --video            # graba mp4 en <run>/videos
    uv run python evaluate.py runs/dqn_highway__s1 --random           # baseline: política aleatoria
"""
import os
from dataclasses import dataclass

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
import tyro

import dqn_acrobot
import ppo_acrobot
import ppo_pendulum


@dataclass
class Args:
    run: tyro.conf.Positional[str]  # carpeta del run, ej: runs/dqn_acrobot__s1
    episodes: int = 20
    seed: int = 1000         # seeds distintas a las de entrenamiento
    render: bool = False
    video: bool = False
    random: bool = False     # ignorar el modelo y actuar al azar (baseline)


def main(args: Args):
    ckpt = torch.load(f"{args.run}/model.pt", weights_only=False)
    cfg = ckpt["args"]
    algo = os.path.basename(os.path.normpath(args.run)).split("_")[0]  # "dqn" o "ppo"
    env_id = cfg["env_id"]

    render_mode = "human" if args.render else ("rgb_array" if args.video else None)
    env = gym.make(env_id, render_mode=render_mode)
    if args.video:
        env = gym.wrappers.RecordVideo(env, f"{args.run}/videos", episode_trigger=lambda e: e < 3)
    if "highway" in env_id:
        env = gym.wrappers.FlattenObservation(env)
    continuous = isinstance(env.action_space, gym.spaces.Box)
    obs_dim = env.observation_space.shape[0]

    if algo == "dqn":
        model = dqn_acrobot.QNetwork(obs_dim, env.action_space.n, cfg["hidden"])
    elif continuous:
        model = ppo_pendulum.Agent(obs_dim, env.action_space.shape[0])
    else:
        model = ppo_acrobot.Agent(obs_dim, env.action_space.n)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def policy(obs):
        if args.random:
            return env.action_space.sample()
        if "obs_mean" in ckpt:  # PPO continuo: misma normalización que en entrenamiento
            obs = np.clip((obs - ckpt["obs_mean"]) / np.sqrt(ckpt["obs_var"] + 1e-8), -10, 10)
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if algo == "dqn":
                return int(model(x).argmax(1).item())
            if continuous:
                return np.clip(model.actor_mean(x)[0].numpy(), env.action_space.low, env.action_space.high)
            return int(model.actor(x).argmax(1).item())

    returns, lengths, crashes = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, R, L, info = False, 0.0, 0, {}
        while not done:
            obs, r, terminated, truncated, info = env.step(policy(obs))
            R += float(r)
            L += 1
            done = terminated or truncated
        returns.append(R)
        lengths.append(L)
        crashes.append(bool(info.get("crashed", False)))
        print(f"episodio {ep:>2}: retorno={R:8.2f}  largo={L}")
    env.close()

    tag = "aleatoria" if args.random else ("greedy" if algo == "dqn" else "determinística")
    print(f"\n{args.run} [{tag}] retorno medio = {np.mean(returns):.2f} ± {np.std(returns):.2f}  "
          f"(largo medio {np.mean(lengths):.1f})")
    if "highway" in env_id:
        print(f"tasa de choque = {np.mean(crashes):.0%}")


if __name__ == "__main__":
    main(tyro.cli(Args))
