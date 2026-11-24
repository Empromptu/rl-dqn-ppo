# TP final RL: DQN y PPO (grupo impar)

| Ejercicio | Script | Entorno |
|---|---|---|
| DQN mínimo | `dqn_acrobot.py` | Acrobot-v1 |
| DQN pasito más | `dqn_highway.py` | highway-fast-v0 |
| PPO mínimo | `ppo_acrobot.py` | Acrobot-v1 |
| PPO pasito más | `ppo_pendulum.py` | Pendulum-v1 (acción continua) |

## Cómo correr
```bash
uv sync                                          # crea .venv con python 3.12 + torch CPU
uv run python dqn_acrobot.py --seed 1            # entrena (cualquier hiperparámetro: --help)
uv run python plot.py                            # curvas en plots/
uv run python evaluate.py runs/dqn_acrobot__s1   # evalúa sin exploración
uv run python evaluate.py runs/dqn_highway__s1 --render   # ver al agente en vivo
uv run python evaluate.py runs/dqn_highway__s1 --random   # baseline aleatorio
```
Cada run deja en `runs/<script>__s<seed>/` los archivos `returns.csv` (retorno por episodio), `losses.csv` (diagnósticos) y `model.pt`.
Todo corre en CPU. Tiempos por seed: DQN Acrobot ~2 min, PPO Acrobot ~3 min, PPO Pendulum ~3 min, DQN highway ~19 min (el simulador corre a ~25 pasos/s).

---

## 1. DQN en Acrobot-v1 (`dqn_acrobot.py`)

**Bloques del script** (numerados igual que en el código):
1. **Hiperparámetros**: rl-baselines3-zoo para Acrobot.
2. **QNetwork**: MLP 6 → 256 → 256 → 3. Una sola pasada da Q(s,a) para las 3 acciones.
3. **ReplayBuffer**: arrays circulares de numpy; `sample` toma índices al azar.
4. **Schedule de ε**: lineal de 1.0 a 0.1 durante el primer 12% del entrenamiento.
5. **Loop**: ε-greedy → `env.step` → se guarda `(s, a, r, s', terminated)`.
6. **Update**: `target = r + γ·max_a' Q_target(s',a')·(1−terminated)`, Huber loss sobre `Q(s)[a]`. Cada 250 pasos `Q_target ← Q`.

**Detalles para explicar:**
- **terminated vs truncated.** En Acrobot, `terminated` significa que la punta llegó arriba (no hay futuro, V(s')=0). `truncated` significa que se cortó a los 500 pasos (el futuro existe y hay que hacer bootstrap). Si se mezclaran, la red aprendería que cortar por tiempo vale 0, y eso es falso.
- **Target network.** Sin ella, el target cambia en cada update y el entrenamiento oscila. Para probarlo: `--target-network-frequency 1`.
- **Replay buffer.** Con `--buffer-size 1000` (prácticamente datos on-line y correlacionados) el aprendizaje empeora.
- **Huber y clip de gradiente.** Evitan que un TD-error grande produzca un update enorme.

| | **Resultados** (100k pasos, 3 seeds) |
|---|---|
| Política aleatoria | −500 (nunca llega) |
| Últimos 20 episodios de entrenamiento (con ε=0.1) | −87 / −98 / −86 |
| Evaluación greedy (20 ep) | **−87 ± 11 / −75 ± 9 / −73 ± 9** |
| Referencia CleanRL/SB3 | ≈ −80 |

## 2. PPO en Acrobot-v1 (`ppo_acrobot.py`)

Las 5 componentes que pide el TP están marcadas en el código:
- **[ROLLOUT]**: 4 entornos en paralelo × 128 pasos, lo que da batches de 512 transiciones. Se guarda `log π_vieja(a|s)` y `V(s)`. Después del update los datos se descartan (on-policy, sin buffer).
- **[GAE]**: `δ_t = r_t + γV(s_{t+1})(1−done) − V(s_t)`, y después `A_t = δ_t + γλ(1−done)A_{t+1}`, recorriendo el rollout de atrás para adelante. `returns = A + V` es el target del crítico.
- **[RATIO]**: `exp(log π_nueva − log π_vieja)`.
- **[CLIP]**: `min(ratio·A, clip(ratio, 0.8, 1.2)·A)`. Como el cambio de la política queda acotado, se pueden hacer 4 epochs sobre el mismo rollout.
- **[CRITICO]**: MSE de V(s) contra los returns.
- Pérdida total: `pg_loss − 0.01·entropía + 0.5·v_loss`.

Detalles de implementación (de "The 37 implementation details"): init ortogonal, normalizar las ventajas por minibatch, clip de gradiente en 0.5, annealing del lr, Adam con eps=1e-5.

**Diagnósticos** (`plots/ppo_acrobot_diagnosticos.png`): la entropía baja desde ln 3 ≈ 1.10 a medida que la política se vuelve más segura. approx_kl se mantiene chico y clipfrac indica en qué fracción de las muestras actuó el clip.

**Simplificación consciente**, igual que en CleanRL: `done = terminated or truncated`, así que no se hace bootstrap en el corte por tiempo. Una vez que el agente aprende, los episodios terminan bastante antes de los 500 pasos y esto casi no influye.

| | **Resultados** (500k pasos, 3 seeds) |
|---|---|
| Evaluación determinística (20 ep) | **−89 ± 31 / −78 ± 6 / −81 ± 11** |

### DQN vs PPO en Acrobot (`plots/acrobot_dqn_vs_ppo.png`)
- **Eficiencia de muestras:** DQN llega a ≈ −100 con muchos menos pasos de entorno, porque reutiliza cada transición muchas veces gracias al replay. PPO usa cada dato solo durante 4 epochs y después lo descarta.
- **Tiempo de reloj:** son comparables. PPO necesita más pasos, pero cada paso es más barato (no entrena en cada paso y vectoriza los entornos).
- **Exploración:** DQN explora con ε-greedy. PPO explora porque samplea de una política estocástica, más el bonus de entropía.

---

## 3. DQN en highway-fast-v0 (`dqn_highway.py`)

Es **el mismo algoritmo** (el script es una copia de `dqn_acrobot.py`). Los cambios están marcados con `>>> HIGHWAY`:
- **Observación:** matriz 5×5 (el auto propio más los 4 vecinos más cercanos, cada uno con `[presence, x, y, vx, vy]`, normalizados y relativos al ego). Se aplana a 25 con `FlattenObservation`.
- **Acciones:** 5 meta-acciones (LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER). Un controlador de bajo nivel se encarga de girar el volante y acelerar.
- **Recompensa:** premia la velocidad (en el rango 20–30 m/s) y el carril derecho, y castiga chocar. Está normalizada a [0,1] por paso.
- **Episodio:** termina (`terminated`) si choca, y se trunca a los 30 pasos.
- **Hiperparámetros** (quickstart de highway-env): 20k pasos, batch 32, buffer 15k, target cada 50, **γ = 0.8**. Un γ bajo alcanza porque la recompensa es densa y lo que importa es no chocar en los próximos segundos: el horizonte efectivo es 1/(1−γ) = 5 pasos.
- El simulador es lento (unos 25 pasos/s) y por eso el presupuesto es de solo 20k pasos.

| **Resultados** (20k pasos, 3 seeds; evaluación greedy, 30 episodios) | Retorno | Largo medio (máx 30) | Tasa de choque |
|---|---|---|---|
| Política aleatoria | 8.8 ± 5.5 | 11.7 | **97%** |
| DQN seed 1 | 21.5 ± 5.3 | 27.0 | **20%** |
| DQN seed 2 | 18.9 ± 6.7 | 21.9 | 60% |
| DQN seed 3 | 19.2 ± 8.0 | 22.2 | 60% |

Las 3 seeds mejoran mucho respecto de la política aleatoria. Duplican el retorno y sobreviven el doble de tiempo, pero no aprendieron a no chocar nunca. Lo que aprende el agente: acelerar (la recompensa por velocidad) y cambiar de carril cuando tiene un auto adelante. 20k pasos son pocos (es el presupuesto del quickstart). Con más pasos o con Double DQN (ver el PDF de highway-env) debería bajar la tasa de choque. Para verlo manejar: `uv run python evaluate.py runs/dqn_highway__s1 --render`.

---

## 4. PPO con acción continua en Pendulum-v1 (`ppo_pendulum.py`)

Es el mismo PPO que en Acrobot. Lo que cambia de fondo es **la política** (marcado con `>>> CONTINUO`):
- **Acrobot:** `π(a|s) = Categorical(softmax(logits(s)))`.
- **Pendulum:** `π(a|s) = Normal(μ(s), σ)`. El actor devuelve μ, y **log σ es un parámetro aprendible** que no depende del estado. Arranca en σ = 1 (exploración amplia) y el entrenamiento lo achica (ver `plots/ppo_pendulum_diagnosticos.png`).
- Con varias dimensiones de acción, `log π(a|s)` es la suma de los log-densidades de cada dimensión. En Pendulum hay una sola.
- **Clip de la acción:** la gaussiana puede samplear un torque de 3.7, pero el entorno solo acepta valores en [−2, 2]. `ClipAction` recorta solo lo que se manda al entorno, y en el buffer queda la acción original, para que el ratio se calcule sobre la misma acción que se sampleó.
- **Normalización:** se usa `NormalizeReward`, porque las recompensas de Pendulum llegan a −16 por paso. `NormalizeObservation` está **apagada**, y la razón está en la sección de debugging de abajo.
- **Bootstrap en truncation:** Pendulum *nunca* termina, solo se trunca a los 200 pasos. En el paso donde se corta, se suma `γ·V(s_final)` a la recompensa, para no enseñarle al crítico que "el paso 200 vale 0".
- **Por qué DQN no sirve acá:** necesita `max_a Q(s,a)` tanto para actuar como para el target. Con a ∈ [−2, 2] continuo, eso es un problema de optimización en cada paso. Lo único que puede hacer DQN es discretizar el torque, por ejemplo en 5 valores, y así pierde precisión.
- **γ = 0.9** (como en rl-zoo): la recompensa es densa y lo que importa es el futuro cercano.

| **Resultados** (300k pasos, 3 seeds) | |
|---|---|
| Al inicio (σ=1, política casi aleatoria) | ≈ −900 a −1200 |
| Evaluación determinística (μ(s), 20 ep) | **−182 ± 99 / −175 ± 98 / −173 ± 97** |
| Referencia sb3 (con gSDE) | −230 ± 142 |

El ± grande viene del estado inicial: si el péndulo arranca colgando, tarda varios pasos en subir y acumula castigo. σ baja de 1.0 a ≈ 0.3–0.4 y converge en unos 100k pasos.

### Debugging: lo que no funcionó al principio (para contar en la presentación)
La primera versión, con hiperparámetros tipo CleanRL y `NormalizeObservation` más `done = terminated or truncated`, **no convergía**. Mejoraba hasta ≈ −600 durante los primeros 100k pasos y **después empeoraba** a ≈ −900. Al mirar la política, el péndulo giraba sin parar en lugar de frenarse arriba. Se corrieron ablaciones con 300k pasos y 2 a 3 seeds cada una (en `runs/exp_pend/`):

| Variante | Retorno final (por seed) |
|---|---|
| Versión original (norm obs, sin bootstrap) | −844, −877 |
| + bootstrap en truncation | −189, −816, −660 |
| + bootstrap, lr 3e-4 | −195, −611, −595 |
| + bootstrap, bonus de entropía 0.01 | −609, −627 |
| + bootstrap, γ 0.95 / 8 envs / 4 epochs | entre −530 y −710 |
| **Sin normalizar obs**, + bootstrap (versión final) | **−171, −179, −176** |
| Sin normalizar obs, sin bootstrap | −183, −183, −175 |

**Conclusión:** el problema era `NormalizeObservation`. Sus estadísticas (media y varianza) se siguen actualizando durante todo el entrenamiento. Cuando la política empieza a hacer girar el péndulo, la varianza de la velocidad angular crece y la normalización cambia, así que el mismo estado físico le llega a la red como un input distinto. La política se entrena sobre un input que se mueve, se degrada y se traba girando. Como las observaciones de Pendulum ya vienen en una escala razonable, sacarla lo resolvió en todas las seeds. El bootstrap en truncation es la forma correcta de tratar el corte por tiempo y en la versión rota fue lo que más ayudó, pero una vez resuelto lo otro su efecto es chico (converge un poco antes).
