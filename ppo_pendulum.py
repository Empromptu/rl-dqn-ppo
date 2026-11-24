"""
PPO con acción CONTINUA sobre Pendulum-v1 (grupo impar, "un pasito más").

Es el mismo algoritmo que ppo_acrobot.py (rollout, GAE, ratio, clip, crítico).
Lo único que cambia de fondo es la POLÍTICA:
    Acrobot:  pi(a|s) = Categorical(logits(s))      -> 3 acciones discretas
    Pendulum: pi(a|s) = Normal(mu(s), sigma)        -> torque continuo en [-2, 2]
DQN no puede hacer esto directamente: necesita max_a Q(s,a) y con infinitas
acciones no hay un máximo que se pueda calcular recorriendo las acciones.

Los cambios respecto de ppo_acrobot.py están marcados con  # >>> CONTINUO

Uso:
    uv run python ppo_pendulum.py --seed 1
"""
import csv
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.distributions import Normal


# Hiperparámetros: mezcla de rl-baselines3-zoo (Pendulum-v1) y cleanrl/ppo_continuous_action.py
@dataclass
class Args:
    env_id: str = "Pendulum-v1"
    seed: int = 1
    total_timesteps: int = 300_000
    learning_rate: float = 1e-3
    anneal_lr: bool = True
    num_envs: int = 4
    num_steps: int = 1024           # batch = 4 * 1024 = 4096
    gamma: float = 0.9              # horizonte corto: la recompensa es densa y lo importante es el corto plazo
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    bootstrap_truncation: bool = True   # en el corte por tiempo, sumar gamma * V(s_final) (ver rollout)
    norm_reward: bool = True
    norm_obs: bool = False              # ver README.md: normalizar obs acá ROMPE el aprendizaje
    run_dir: str = ""               # por defecto: runs/ppo_pendulum__s<seed>


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(env_id: str, gamma: float, norm_reward: bool = True, norm_obs: bool = True):
    def thunk():
        env = gym.make(env_id)
        # RecordEpisodeStatistics va PRIMERO para registrar la recompensa ORIGINAL (sin normalizar)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # >>> CONTINUO: la gaussiana puede samplear cualquier valor; el entorno solo acepta [-2, 2]
        env = gym.wrappers.ClipAction(env)
        # Normalizar observaciones (media 0, desvío 1, con estadísticas que se van actualizando).
        # APAGADO por defecto: en Pendulum las obs ya están en escala razonable (cos, sin en [-1,1],
        # velocidad en [-8,8]) y como las estadísticas siguen cambiando mientras la política cambia
        # (al empezar a girar, la varianza de la velocidad explota), la red ve entradas que significan
        # otra cosa con el tiempo -> el aprendizaje se degrada y se traba girando. Ver README.md.
        if norm_obs:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10), env.observation_space)
        # Escalar la recompensa según el desvío del retorno descontado -> el crítico ve valores de escala ~1
        if norm_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
        return env
    return thunk


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # >>> CONTINUO: el actor devuelve la MEDIA mu(s) de la gaussiana (una por dimensión de acción)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        # >>> CONTINUO: log(sigma) es un parámetro aprendible que NO depende del estado.
        # Arranca en 0 -> sigma = 1 (mucha exploración); el entrenamiento lo va achicando.
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        std = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        # >>> CONTINUO: log pi(a|s) = suma de log-densidades de cada dimensión (son independientes)
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x)


def main(args: Args):
    run_dir = args.run_dir or f"runs/ppo_pendulum__s{args.seed}"
    os.makedirs(run_dir, exist_ok=True)
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.gamma, args.norm_reward, args.norm_obs) for _ in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]   # >>> CONTINUO: Box(1,) en vez de Discrete(3)

    agent = Agent(obs_dim, act_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim))
    actions = torch.zeros((args.num_steps, args.num_envs, act_dim))  # >>> CONTINUO: floats
    logprobs = torch.zeros((args.num_steps, args.num_envs))
    rewards = torch.zeros((args.num_steps, args.num_envs))
    dones = torch.zeros((args.num_steps, args.num_envs))
    values = torch.zeros((args.num_steps, args.num_envs))

    returns_f = open(f"{run_dir}/returns.csv", "w", newline="")
    returns_w = csv.writer(returns_f)
    returns_w.writerow(["global_step", "episodic_return", "episodic_length"])
    losses_f = open(f"{run_dir}/losses.csv", "w", newline="")
    losses_w = csv.writer(losses_f)
    losses_w.writerow(["global_step", "pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "lr", "sigma"])

    global_step = 0
    start_time = time.time()
    next_obs = torch.as_tensor(envs.reset(seed=args.seed)[0], dtype=torch.float32)
    next_done = torch.zeros(args.num_envs)

    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # [ROLLOUT]
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            # >>> CONTINUO: guardamos la acción SIN clipear (la que salió de la gaussiana),
            # para que log pi(a|s) en el ratio corresponda a esa misma acción.
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.numpy())
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32)
            # Pendulum NUNCA termina: todos los episodios se truncan a los 200 pasos.
            # Si tratamos el corte como terminal, el crítico aprende que "el paso 200 vale 0",
            # pero ese estado no tiene nada especial (la observación ni siquiera incluye el tiempo).
            # Corrección: en los entornos truncados (y no terminados) sumamos a la recompensa
            # el valor estimado del estado donde se cortó: r <- r + gamma * V(s_final).
            # Así el futuro que existía se cuenta igual, y el done de abajo solo corta el GAE
            # entre episodios.
            if args.bootstrap_truncation and "final_obs" in infos:
                trunc_only = truncated & ~terminated
                if trunc_only.any():
                    final_obs = torch.as_tensor(np.stack(infos["final_obs"][trunc_only]), dtype=torch.float32)
                    with torch.no_grad():
                        rewards[step, trunc_only] += args.gamma * agent.get_value(final_obs).flatten()
            next_done = torch.as_tensor(np.logical_or(terminated, truncated), dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32)

            if "final_info" in infos and "episode" in infos["final_info"]:
                fi = infos["final_info"]
                for i in np.where(fi["_episode"])[0]:
                    returns_w.writerow([global_step, float(fi["episode"]["r"][i]), int(fi["episode"]["l"][i])])
            returns_f.flush()

        # [GAE]
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape(-1, obs_dim)
        b_actions = actions.reshape(-1, act_dim)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start:start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])

                # [RATIO]
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # [CLIP]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # [CRITICO]
                v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        sigma = agent.actor_logstd.exp().mean().item()
        losses_w.writerow([global_step, pg_loss.item(), v_loss.item(), entropy_loss.item(),
                           approx_kl.item(), float(np.mean(clipfracs)), optimizer.param_groups[0]["lr"], sigma])
        losses_f.flush()
        if iteration % 10 == 0:
            print(f"step={global_step:>7}  sigma={sigma:.3f}  kl={approx_kl.item():.4f}  "
                  f"clipfrac={np.mean(clipfracs):.3f}  sps={int(global_step / (time.time() - start_time))}")

    # Guardamos también las estadísticas de NormalizeObservation: sin ellas el modelo
    # recibiría observaciones en otra escala al evaluarlo.
    ckpt = {"model": agent.state_dict(), "args": vars(args)}
    if args.norm_obs:
        obs_rms = envs.envs[0].get_wrapper_attr("obs_rms")
        ckpt.update(obs_mean=obs_rms.mean, obs_var=obs_rms.var)
    torch.save(ckpt, f"{run_dir}/model.pt")
    returns_f.close()
    losses_f.close()
    envs.close()
    print(f"Listo en {time.time() - start_time:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
