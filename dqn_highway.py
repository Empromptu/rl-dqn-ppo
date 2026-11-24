"""
DQN sobre highway-fast-v0 (grupo impar, "un pasito más").

Es EXACTAMENTE el mismo algoritmo que dqn_acrobot.py. Lo que cambia:
    - el entorno: conducción en autopista (highway-env); la observación es una matriz
      5x5 (el auto propio + 4 vecinos, cada uno [presence, x, y, vx, vy]) que aplanamos a 25
    - 5 meta-acciones: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER
    - los hiperparámetros (sacados del quickstart de highway-env)
    - registramos si el episodio terminó en choque
Los cambios están marcados con  # >>> HIGHWAY

Script single-file, estilo CleanRL. Se lee de arriba a abajo:
    (1) Argumentos / hiperparámetros
    (2) Red Q
    (3) Replay buffer
    (4) Schedule de epsilon
    (5) Loop principal: actuar (epsilon-greedy) -> guardar transición -> entrenar
    (6) Update de la red Q (ecuación de Bellman) y de la target network

Uso:
    uv run python dqn_acrobot.py --seed 1
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
import torch.nn.functional as F
import tyro

import highway_env  # noqa: F401  >>> HIGHWAY: registra los entornos highway-*


# ---------------------------------------------------------------------------
# (1) Hiperparámetros
#     >>> HIGHWAY: hiperparámetros de highway-env.farama.org/quickstart (DQN de SB3)
# ---------------------------------------------------------------------------
@dataclass
class Args:
    env_id: str = "highway-fast-v0"
    seed: int = 1
    total_timesteps: int = 20_000
    learning_rate: float = 5e-4
    buffer_size: int = 15_000
    gamma: float = 0.8                   # >>> HIGHWAY: horizonte corto (recompensa densa, episodios de 30 pasos)
    batch_size: int = 32
    learning_starts: int = 200          # pasos al azar antes de empezar a entrenar
    train_freq: int = 1                  # cada cuántos pasos de entorno hacemos un update
    target_network_frequency: int = 50   # cada cuántos pasos copiamos Q -> Q_target
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.3    # fracción del entrenamiento en la que epsilon baja de start_e a end_e
    hidden: int = 256
    run_dir: str = ""                    # por defecto: runs/dqn_highway__s<seed>


# ---------------------------------------------------------------------------
# (2) Red Q: entra el estado (6 números), salen |A| = 3 valores Q(s, a)
#     Una sola pasada calcula Q para todas las acciones a la vez.
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# (3) Replay buffer: memoria circular de transiciones (s, a, r, s', terminated)
#     Samplear al azar rompe la correlación entre pasos consecutivos y
#     permite reutilizar cada experiencia muchas veces.
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, size: int, obs_dim: int):
        self.size = size
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.terminated = np.zeros(size, dtype=np.float32)
        self.ptr = 0      # dónde escribo la próxima transición
        self.full = False

    def add(self, obs, action, reward, next_obs, terminated):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.terminated[self.ptr] = terminated
        self.ptr = (self.ptr + 1) % self.size  # cuando se llena, pisa lo más viejo
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size: int):
        high = self.size if self.full else self.ptr
        idx = np.random.randint(0, high, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.actions[idx]),
            torch.as_tensor(self.rewards[idx]),
            torch.as_tensor(self.next_obs[idx]),
            torch.as_tensor(self.terminated[idx]),
        )


# ---------------------------------------------------------------------------
# (4) Epsilon decae linealmente de start_e a end_e y después queda fijo
# ---------------------------------------------------------------------------
def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main(args: Args):
    run_dir = args.run_dir or f"runs/dqn_highway__s{args.seed}"
    os.makedirs(run_dir, exist_ok=True)

    # Semillas para reproducibilidad
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)  # redes chicas: 1 hilo es más rápido y permite correr seeds en paralelo

    # RecordEpisodeStatistics agrega info["episode"] = {"r": retorno, "l": largo} al final de cada episodio
    env = gym.make(args.env_id)
    # >>> HIGHWAY: (5, 5) -> (25,) para poder usar el mismo MLP
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_network = QNetwork(obs_dim, n_actions, args.hidden)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=args.learning_rate)
    # Target network: copia "congelada" de Q que se usa para calcular el target
    target_network = QNetwork(obs_dim, n_actions, args.hidden)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(args.buffer_size, obs_dim)

    returns_f = open(f"{run_dir}/returns.csv", "w", newline="")
    returns_w = csv.writer(returns_f)
    returns_w.writerow(["global_step", "episodic_return", "episodic_length", "epsilon", "crashed"])
    losses_f = open(f"{run_dir}/losses.csv", "w", newline="")
    losses_w = csv.writer(losses_f)
    losses_w.writerow(["global_step", "td_loss", "q_mean"])

    start_time = time.time()
    obs, _ = env.reset(seed=args.seed)

    # -----------------------------------------------------------------------
    # (5) Loop principal
    # -----------------------------------------------------------------------
    for global_step in range(args.total_timesteps):
        # --- Actuar: epsilon-greedy ---
        epsilon = linear_schedule(args.start_e, args.end_e,
                                  int(args.exploration_fraction * args.total_timesteps), global_step)
        if random.random() < epsilon:
            action = env.action_space.sample()           # explorar
        else:
            with torch.no_grad():
                q_values = q_network(torch.as_tensor(obs).unsqueeze(0))
            action = int(q_values.argmax(dim=1).item())  # explotar: argmax_a Q(s, a)

        next_obs, reward, terminated, truncated, info = env.step(action)

        # --- Guardar transición ---
        # Importante: guardamos `terminated` y NO `truncated`.
        #   terminated -> >>> HIGHWAY: el auto chocó: no hay futuro, V(s') = 0
        #   truncated  -> se cumplieron los 30 pasos sin chocar: el futuro existe, hay que hacer bootstrap
        rb.add(obs, action, reward, next_obs, float(terminated))

        obs = next_obs
        if terminated or truncated:
            ep = info["episode"]
            crashed = int(info["crashed"])  # >>> HIGHWAY
            returns_w.writerow([global_step, float(ep["r"]), int(ep["l"]), round(epsilon, 4), crashed])
            returns_f.flush()
            print(f"step={global_step:>7}  return={float(ep['r']):>6.2f}  len={int(ep['l']):>2}  "
                  f"crashed={crashed}  eps={epsilon:.3f}")
            obs, _ = env.reset()

        # -------------------------------------------------------------------
        # (6) Entrenar
        # -------------------------------------------------------------------
        if global_step > args.learning_starts and global_step % args.train_freq == 0:
            b_obs, b_actions, b_rewards, b_next_obs, b_term = rb.sample(args.batch_size)

            # Target de Bellman: r + gamma * max_a' Q_target(s', a') * (1 - terminated)
            with torch.no_grad():
                target_max = target_network(b_next_obs).max(dim=1).values
                td_target = b_rewards + args.gamma * target_max * (1.0 - b_term)

            # Q(s, a) de las acciones que efectivamente se tomaron
            q_sa = q_network(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)

            # Huber loss (smooth L1): como MSE para errores chicos, lineal para grandes -> más estable
            loss = F.smooth_l1_loss(q_sa, td_target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
            optimizer.step()

            if global_step % 100 == 0:
                losses_w.writerow([global_step, loss.item(), q_sa.mean().item()])

        # Actualización "dura" de la target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    torch.save({"model": q_network.state_dict(), "args": vars(args)}, f"{run_dir}/model.pt")
    returns_f.close()
    losses_f.close()
    env.close()
    print(f"Listo en {time.time() - start_time:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
