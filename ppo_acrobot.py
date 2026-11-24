"""
PPO mínimo sobre Acrobot-v1 (grupo impar).

Script single-file, estilo CleanRL. Componentes que pide el TP, marcadas en el código:
    [ROLLOUT]  juntar N pasos con la política actual en varios entornos en paralelo
    [GAE]      calcular la ventaja A_t con Generalized Advantage Estimation
    [RATIO]    pi_nueva(a|s) / pi_vieja(a|s)
    [CLIP]     objetivo recortado que impide que la política cambie demasiado
    [CRITICO]  red de valor V(s), entrenada por regresión contra los returns

Uso:
    uv run python ppo_acrobot.py --seed 1
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
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Hiperparámetros (valores por defecto de cleanrl/ppo.py)
# ---------------------------------------------------------------------------
@dataclass
class Args:
    env_id: str = "Acrobot-v1"
    seed: int = 1
    total_timesteps: int = 500_000
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True          # bajar el lr linealmente a 0 durante el entrenamiento
    num_envs: int = 4               # entornos en paralelo
    num_steps: int = 128            # pasos por entorno en cada rollout -> batch = 4 * 128 = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4          # cuántas veces recorremos el mismo rollout
    norm_adv: bool = True
    clip_coef: float = 0.2          # epsilon del clip
    ent_coef: float = 0.01          # peso del bonus de entropía (exploración)
    vf_coef: float = 0.5            # peso de la pérdida del crítico
    max_grad_norm: float = 0.5
    run_dir: str = ""               # por defecto: runs/ppo_acrobot__s<seed>


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Inicialización ortogonal (detalle de implementación estándar de PPO)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# Actor-Crítico: dos redes separadas
#   actor:   s -> logits sobre las 3 acciones -> distribución Categorical
#   critico: s -> V(s), un escalar
# ---------------------------------------------------------------------------
class Agent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            # std chico -> al principio todas las acciones son casi equiprobables
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        dist = Categorical(logits=self.actor(x))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


def main(args: Args):
    run_dir = args.run_dir or f"runs/ppo_acrobot__s{args.seed}"
    os.makedirs(run_dir, exist_ok=True)
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    # Entornos vectorizados. SAME_STEP: cuando un episodio termina, step() ya devuelve
    # la observación inicial del episodio nuevo (semántica clásica de gym/CleanRL).
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(args.env_id)) for _ in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    agent = Agent(obs_dim, n_actions)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Buffers del rollout: forma (num_steps, num_envs, ...)
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim))
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
    logprobs = torch.zeros((args.num_steps, args.num_envs))
    rewards = torch.zeros((args.num_steps, args.num_envs))
    dones = torch.zeros((args.num_steps, args.num_envs))
    values = torch.zeros((args.num_steps, args.num_envs))

    returns_f = open(f"{run_dir}/returns.csv", "w", newline="")
    returns_w = csv.writer(returns_f)
    returns_w.writerow(["global_step", "episodic_return", "episodic_length"])
    losses_f = open(f"{run_dir}/losses.csv", "w", newline="")
    losses_w = csv.writer(losses_f)
    losses_w.writerow(["global_step", "pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "lr"])

    global_step = 0
    start_time = time.time()
    next_obs = torch.as_tensor(envs.reset(seed=args.seed)[0])
    next_done = torch.zeros(args.num_envs)

    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ===================================================================
        # [ROLLOUT] jugar num_steps pasos con la política actual (sin gradientes)
        # ===================================================================
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob          # log pi_vieja(a|s), lo necesitamos para el ratio
            values[step] = value.flatten()

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.numpy())
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32)
            # Simplificación de CleanRL: trata truncated igual que terminated (no hace bootstrap
            # en el corte por tiempo). En Acrobot casi no importa una vez que aprende.
            next_done = torch.as_tensor(np.logical_or(terminated, truncated), dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np)

            # Con SAME_STEP, las estadísticas del episodio que terminó quedan en infos["final_info"]
            if "final_info" in infos and "episode" in infos["final_info"]:
                fi = infos["final_info"]
                for i in np.where(fi["_episode"])[0]:
                    returns_w.writerow([global_step, float(fi["episode"]["r"][i]), int(fi["episode"]["l"][i])])
            returns_f.flush()

        # ===================================================================
        # [GAE] ventaja, recorriendo el rollout de atrás para adelante
        #   delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
        #   A_t     = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}
        # ===================================================================
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
            # Target del crítico: returns = A + V  (estimación lambda-return)
            returns = advantages + values

        # Aplanar (num_steps, num_envs) -> (batch_size,)
        b_obs = obs.reshape(-1, obs_dim)
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # ===================================================================
        # Optimización: update_epochs pasadas sobre el mismo rollout, en minibatches
        # ===================================================================
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start:start + minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])

                # [RATIO] pi_nueva / pi_vieja = exp(log pi_nueva - log pi_vieja)
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()  # estimador de KL(vieja || nueva)
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # [CLIP] L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
                # Lo escribimos como max de los negativos porque minimizamos.
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # [CRITICO] regresión de V(s) contra los returns
                v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                # Pérdida total: política - bonus de entropía + crítico
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        losses_w.writerow([global_step, pg_loss.item(), v_loss.item(), entropy_loss.item(),
                           approx_kl.item(), float(np.mean(clipfracs)), optimizer.param_groups[0]["lr"]])
        losses_f.flush()
        if iteration % 20 == 0:
            print(f"step={global_step:>7}  entropy={entropy_loss.item():.3f}  kl={approx_kl.item():.4f}  "
                  f"clipfrac={np.mean(clipfracs):.3f}  sps={int(global_step / (time.time() - start_time))}")

    torch.save({"model": agent.state_dict(), "args": vars(args)}, f"{run_dir}/model.pt")
    returns_f.close()
    losses_f.close()
    envs.close()
    print(f"Listo en {time.time() - start_time:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
