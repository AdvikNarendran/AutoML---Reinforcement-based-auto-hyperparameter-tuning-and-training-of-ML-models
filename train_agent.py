# train_agent.py
import os
import json
import time
import random
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from tqdm import trange

# Gymnasium API
import gymnasium as gym
from gymnasium import spaces

# RL
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Surrogate
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------
# Configuration / Hyperparams
# ---------------------------
MODEL_DIR = "Automan_rl/models"
LOG_DIR = "Automan_rl/logs"
SURROGATE_PATH = "Automan_rl/models/surrogate_rf.joblib"
HISTORICAL_TRIALS_DIR = "Automan_rl/trials"   # optional historical runs for surrogate training
DATASETS_DIR = "Automan_rl/data"              # your 18 datasets (used only for meta-features)
RANDOM_SEED = 42
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Agent / training hyperparams
TOTAL_TIMESTEPS = 20_000_000
PPO_PARAMS = {
    "policy": "MlpPolicy",
    "verbose": 1,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "tensorboard_log": "Automan_rl/tensorboard_logs/",
}

# Reward weights (tunable)
W_PERF = 1.0
W_COST = 0.3
W_PENALTY = 0.2
STEP_PENALTY = 0.01

# Environment params
LAST_K = 5
STATE_DIM = LAST_K + 1 + 1 + 1 + 1 + 1 + 1  # approximate; see state builder
MAX_BUDGET_UNIT = 1.0  # normalize costs to 0-1
MAX_MODEL_AGE = 365.0   # days normalization

# Action mapping (discrete)
ACTION_MEANINGS = {
    0: "wait",
    1: "retrain_full",
    2: "tune_low",
    3: "tune_med",
    4: "tune_high",
    5: "feature_transform_x",
    6: "feature_transform_y",
    7: "rollback",
    8: "save_snapshot",
}
N_ACTIONS = len(ACTION_MEANINGS)

# For reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ---------------------------
# Utilities
# ---------------------------
def safe_normalize(x: float, min_v: float, max_v: float) -> float:
    """Normalize to [0,1] with clipping (handles constant ranges)."""
    if max_v <= min_v:
        return 0.0
    return float(np.clip((x - min_v) / (max_v - min_v), 0.0, 1.0))


def metric_to_normalized(metric: float, baseline: float = 0.0, clip_range: Tuple[float, float] = (0.0, 1.0)) -> float:
    """Assume metrics in [0,1] already (e.g., F1_macro). Clip just in case."""
    return float(np.clip(metric, clip_range[0], clip_range[1]))


# ---------------------------
# Surrogate model (optional)
# ---------------------------
class SurrogateModel:
    """
    RandomForest surrogate that predicts metric outcome and runtime/cost
    given dataset meta-features + action type + hyperparam-embedding.
    If historical trials exist, train it. Otherwise use a fallback noisy model.
    """

    def __init__(self, model_path: str = SURROGATE_PATH):
        self.model_path = model_path
        self.reg_metric = None
        self.reg_cost = None
        if os.path.exists(model_path):
            try:
                data = joblib.load(model_path)
                self.reg_metric = data["metric"]
                self.reg_cost = data["cost"]
                print("Loaded surrogate from", model_path)
            except Exception as e:
                print("Failed to load surrogate:", e)

    def train_from_trials_dir(self, trials_dir: str):
        # Expect CSVs of historical trials with columns:
        # dataset_name, action_type, meta_feature_1..n, metric_after, runtime
        rows = []
        for fname in os.listdir(trials_dir) if os.path.exists(trials_dir) else []:
            if not fname.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(trials_dir, fname))
            rows.append(df)
        if not rows:
            print("No historical trials found for surrogate training.")
            return False
        df = pd.concat(rows, ignore_index=True)
        # Build X, y_metric, y_cost
        # Keep it simple: encode action_type as int, use meta features columns automatically
        if "metric_after" not in df.columns or "runtime" not in df.columns:
            print("Historical trials missing required columns 'metric_after' or 'runtime'.")
            return False
        feature_cols = [c for c in df.columns if c not in ("dataset_name", "metric_after", "runtime")]
        X = df[feature_cols].fillna(0).values
        y_metric = df["metric_after"].values
        y_cost = df["runtime"].values
        X_train, X_val, ym_train, ym_val = train_test_split(X, y_metric, test_size=0.2, random_state=RANDOM_SEED)
        _, _, yc_train, yc_val = train_test_split(X, y_cost, test_size=0.2, random_state=RANDOM_SEED)
        self.reg_metric = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
        self.reg_cost = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
        self.reg_metric.fit(X_train, ym_train)
        self.reg_cost.fit(X_train, yc_train)
        joblib.dump({"metric": self.reg_metric, "cost": self.reg_cost}, self.model_path)
        print("Trained surrogate and saved to", self.model_path)
        return True

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        features: 1D array with correct ordering used in training.
        returns (pred_metric, pred_cost)
        """
        if self.reg_metric is None or self.reg_cost is None:
            # fallback: small improvement/noisy cost
            pred_metric = 0.01 * np.random.randn() + 0.5
            pred_cost = 0.05 + 0.1 * np.abs(np.random.randn())
            return float(np.clip(pred_metric, 0.0, 1.0)), float(max(pred_cost, 0.0))
        x = features.reshape(1, -1)
        pm = float(self.reg_metric.predict(x)[0])
        pc = float(self.reg_cost.predict(x)[0])
        return float(np.clip(pm, 0.0, 1.0)), float(max(pc, 1e-6))


# ---------------------------
# Environment
# ---------------------------

class AutoMLControllerEnv(gym.Env):
    """
    Gym-like environment that simulates AutoML controller decisions.

    Observation (continuous Box, shape=(STATE_LEN,)):
      - last_k_metrics: last K validation metrics normalized [0,1]
      - recent_train_delta: normalized [-1,1] mapped to [0,1] or just [0,1]
      - best_metric_so_far: [0,1]
      - remaining_budget: [0,1]
      - last_action: scalar index normalized
      - time_since_last_retrain: normalized [0,1]
      - avg_inference_loss_window: [0,1]
      - feature_drift_score: [0,1]
      - model_age: [0,1]
      - model_size_latency: [0,1] (optional)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 dataset_name: str = "dataset_1",
                 surrogate: Optional[SurrogateModel] = None,
                 max_episode_steps: int = 100,
                 use_surrogate: bool = True):
        super().__init__()
        self.dataset_name = dataset_name
        self.surrogate = surrogate
        self.max_episode_steps = max_episode_steps
        self.use_surrogate = use_surrogate
        self.step_count = 0

        # Action and observation spaces
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(LAST_K + 9,), dtype=np.float32)

        # internal history
        self.history_metrics = [0.5] * LAST_K  # start with baseline
        self.best_metric = max(self.history_metrics)
        self.remaining_budget = 1.0
        self.last_action = 0
        self.time_since_last_retrain = 0.0
        self.avg_inference_loss = 0.0
        self.feature_drift = 0.0
        self.model_age_days = 0.0
        self.model_latency = 0.0

        self.episode_reward = 0.0
        self.current_metric = self.history_metrics[-1]
        self.done = False

        # store logs
        self.run_log: List[Dict[str, Any]] = []

        # dataset meta-features (simple)
        self.dataset_meta = self._load_dataset_meta()

    def _load_dataset_meta(self) -> Dict[str, float]:
        # attempt to compute or load some meta features from dataset file if present
        meta = {}
        dataset_path_csv = os.path.join(DATASETS_DIR, f"{self.dataset_name}.csv")
        if os.path.exists(dataset_path_csv):
            try:
                df = pd.read_csv(dataset_path_csv, nrows=1000)
                meta["nrows"] = df.shape[0]
                meta["ncols"] = df.shape[1]
                meta["class_imbalance"] = float(df.iloc[:, -1].value_counts(normalize=True).max())
            except Exception:
                meta["nrows"] = 1000
                meta["ncols"] = 10
                meta["class_imbalance"] = 0.6
        else:
            meta["nrows"] = 1000
            meta["ncols"] = 10
            meta["class_imbalance"] = 0.6
        # normalize meta features roughly
        meta_norm = {
            "nrows_norm": safe_normalize(meta["nrows"], 100, 100000),
            "ncols_norm": safe_normalize(meta["ncols"], 1, 1000),
            "imbalance": safe_normalize(meta["class_imbalance"], 0.5, 1.0),
        }
        return meta_norm

    def _construct_state(self) -> np.ndarray:
        # build vector
        last_k = list(self.history_metrics[-LAST_K:])
        # ensure length LAST_K
        if len(last_k) < LAST_K:
            last_k = [self.history_metrics[0]] * (LAST_K - len(last_k)) + last_k

        # recent_train_delta: use difference last - prev averaged (normalized to [-1,1], then map)
        recent_train_delta = 0.0
        if len(self.history_metrics) >= 2:
            recent_train_delta = (self.history_metrics[-1] - self.history_metrics[-2])
        recent_train_delta_norm = (recent_train_delta + 1.0) / 2.0  # map -1..1 -> 0..1

        best_metric_norm = metric_to_normalized(self.best_metric)
        remaining_budget_norm = np.clip(self.remaining_budget, 0.0, 1.0)
        last_action_norm = safe_normalize(self.last_action, 0, max(1, N_ACTIONS - 1))
        tslr_norm = safe_normalize(self.time_since_last_retrain, 0.0, MAX_MODEL_AGE)
        avg_infer_loss_norm = np.clip(self.avg_inference_loss, 0.0, 1.0)
        feat_drift_norm = np.clip(self.feature_drift, 0.0, 1.0)
        model_age_norm = safe_normalize(self.model_age_days, 0.0, MAX_MODEL_AGE)
        model_latency_norm = np.clip(self.model_latency, 0.0, 1.0)

        vec = np.array(
            last_k
            + [recent_train_delta_norm]
            + [best_metric_norm]
            + [remaining_budget_norm]
            + [last_action_norm]
            + [tslr_norm]
            + [avg_infer_loss_norm]
            + [feat_drift_norm]
            + [model_age_norm]
            + [model_latency_norm],
            dtype=np.float32,
        )
        return vec

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.step_count = 0
        self.history_metrics = [0.5 + 0.02 * np.random.randn() for _ in range(LAST_K)]
        self.best_metric = max(self.history_metrics)
        self.remaining_budget = 1.0
        self.last_action = 0
        self.time_since_last_retrain = 0.0
        self.avg_inference_loss = 0.1
        self.feature_drift = 0.0
        self.model_age_days = 0.0
        self.model_latency = 0.1
        self.episode_reward = 0.0
        self.done = False
        self.run_log = []
        return self._construct_state(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.step_count += 1
        prev_metric = self.history_metrics[-1]
        action_name = ACTION_MEANINGS[int(action)]

        # Simulate the action outcome
        result = self._execute_action_sim(action)

        metric_after = float(result["metric_after"])
        cost = float(result["cost"])
        time_spent = float(result["time"])

        # update budget & time-based variables
        self.remaining_budget = max(0.0, self.remaining_budget - cost / MAX_BUDGET_UNIT)
        self.time_since_last_retrain += time_spent / (24.0 * 3600.0)  # convert seconds to days
        self.model_age_days += time_spent / (24.0 * 3600.0)
        self.model_latency = np.clip(self.model_latency + (0.01 if action_name.startswith("retrain") else 0.0), 0.0, 1.0)
        self.feature_drift = max(0.0, self.feature_drift + 0.02 * np.random.randn())  # random drift
        self.avg_inference_loss = np.clip(self.avg_inference_loss + 0.01 * np.random.randn(), 0.0, 1.0)

        # update metrics history and best
        self.history_metrics.append(metric_after)
        if metric_after > self.best_metric:
            self.best_metric = metric_after

        # reward calculation
        delta = metric_after - prev_metric
        delta_norm = float(np.clip(delta, -1.0, 1.0))  # metrics in [-1,1] clamp
        normalized_cost = cost / (MAX_BUDGET_UNIT + 1e-8)
        retrain_penalty = 0.0
        if action_name in ("retrain_full", "tune_high"):
            retrain_penalty = 1.0

        reward = W_PERF * delta_norm - W_COST * normalized_cost - W_PENALTY * retrain_penalty - STEP_PENALTY
        # optional smoothing or scaling
        reward = float(np.clip(reward, -10.0, 10.0))

        self.episode_reward += reward
        self.last_action = int(action)

        # done logic
        done = False
        if self.step_count >= self.max_episode_steps or self.remaining_budget <= 0.0:
            done = True
        self.done = done

        # log
        log_entry = {
            "step": self.step_count,
            "dataset": self.dataset_name,
            "action": action_name,
            "metric_before": prev_metric,
            "metric_after": metric_after,
            "delta": delta,
            "cost": cost,
            "time": time_spent,
            "reward": reward,
            "remaining_budget": self.remaining_budget,
        }
        self.run_log.append(log_entry)

        obs = self._construct_state()
        info = {"raw_result": result}
        return obs, reward, done, False, info

    def _execute_action_sim(self, action: int) -> Dict[str, Any]:
        """
        Execute the action using surrogate (fast) or a random/mock fallback.
        Return dict: metric_after, cost, time
        """
        action_name = ACTION_MEANINGS[int(action)]
        # assemble features for surrogate: dataset meta + action int
        meta = self.dataset_meta
        feat = np.array([meta["nrows_norm"], meta["ncols_norm"], meta["imbalance"], float(action)], dtype=np.float32)

        if self.use_surrogate and self.surrogate is not None:
            pred_metric, pred_cost = self.surrogate.predict(feat)
            # shape predicted metric to be relative to current (simulate improvement or drop)
            # Add a small dependency on drift and age (makes env interesting)
            drift_factor = -0.05 * self.feature_drift
            age_penalty = -0.01 * (self.model_age_days / MAX_MODEL_AGE)
            if action_name == "wait":
                metric_after = max(0.0, self.history_metrics[-1] + drift_factor + 0.005 * np.random.randn())
            elif action_name == "rollback":
                metric_after = max(0.0, self.best_metric - 0.01 * np.random.randn())
            else:
                # treat surrogate metric as absolute target; combine with current for realism
                metric_after = float(np.clip(0.6 * self.history_metrics[-1] + 0.4 * pred_metric + drift_factor + age_penalty, 0.0, 1.0))
            cost = float(pred_cost)
            time_spent = float(cost * 3600.0)  # cost maps to hours -> seconds
        else:
            # fallback simplistic simulator
            if action_name == "wait":
                metric_after = max(0.0, self.history_metrics[-1] - 0.001 * np.random.randn())
                cost = 0.001
            elif action_name == "retrain_full":
                metric_after = float(np.clip(self.history_metrics[-1] + 0.02 + 0.03 * np.random.randn(), 0.0, 1.0))
                cost = 0.2
            elif action_name.startswith("tune"):
                # fidelity mapping
                fidelity = {"tune_low": 0.05, "tune_med": 0.12, "tune_high": 0.4}.get(action_name, 0.05)
                metric_after = float(np.clip(self.history_metrics[-1] + fidelity * 0.5 + 0.02 * np.random.randn(), 0.0, 1.0))
                cost = fidelity
            elif action_name == "rollback":
                metric_after = self.best_metric - 0.001 * np.random.randn()
                cost = 0.02
            elif action_name.startswith("feature_transform"):
                metric_after = float(np.clip(self.history_metrics[-1] + 0.01 * np.random.randn() + 0.01, 0.0, 1.0))
                cost = 0.03
            else:
                metric_after = self.history_metrics[-1]
                cost = 0.001
            time_spent = cost * 3600.0

        return {"metric_after": float(metric_after), "cost": float(cost), "time": float(time_spent)}

    def render(self, mode="human"):
        print(f"Step {self.step_count}, last_metric={self.history_metrics[-1]:.4f}, best_metric={self.best_metric:.4f}, budget={self.remaining_budget:.3f}")

    def save_run_log(self, fname: Optional[str] = None):
        if fname is None:
            fname = os.path.join(LOG_DIR, f"runlog_{self.dataset_name}_{int(time.time())}.json")
        with open(fname, "w") as f:
            json.dump(self.run_log, f, indent=2)
        print("Saved run log to", fname)


# ---------------------------
# Helper: Run training harness
# ---------------------------
def train_agent_on_datasets(datasets: List[str],
                            total_timesteps: int = TOTAL_TIMESTEPS,
                            use_surrogate=True,
                            save_name="ppo_automan_v1"):
    # Prepare surrogate model (train from trials if available)
    surrogate = SurrogateModel()
    surrogate_trained = False
    if os.path.exists(HISTORICAL_TRIALS_DIR):
        surrogate_trained = surrogate.train_from_trials_dir(HISTORICAL_TRIALS_DIR)
        if surrogate_trained:
            print("Surrogate trained from historical trials.")
    if not surrogate_trained:
        print("Proceeding with default/fallback surrogate behavior (untrained).")

    # Create a single vectorized environment that cycles datasets per episode
    # For simplicity we will train on one environment that randomizes dataset per reset
    def make_env():
        ds = random.choice(datasets)
        return AutoMLControllerEnv(dataset_name=ds, surrogate=surrogate, use_surrogate=use_surrogate)

    # Stable-baselines3 requires gym-like env object; use a simple wrapper
    env = make_env()

    # Initialize agent
    print("Initializing PPO agent...")
    policy_kwargs = dict()
    agent = PPO(**PPO_PARAMS, env=env, policy_kwargs=policy_kwargs)

    # optional callbacks for checkpointing and eval
    checkpoint_cb = CheckpointCallback(save_freq=50_000, save_path=MODEL_DIR, name_prefix=save_name)
    # evaluate on small environment periodically: here we reuse make_env()
    eval_env = make_env()
    eval_cb = EvalCallback(eval_env, best_model_save_path=MODEL_DIR, log_path=LOG_DIR, eval_freq=50_000, n_eval_episodes=5, deterministic=True)

    # Train
    print("Starting training for", total_timesteps, "timesteps")
    agent.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])

    # Save agent
    timestamp = int(time.time())
    model_path = os.path.join(MODEL_DIR, f"{save_name}_{timestamp}.zip")
    agent.save(model_path)
    print("Saved trained agent to", model_path)

    # Save metadata
    meta = {
        "name": save_name,
        "model_path": model_path,
        "trained_on": datasets,
        "total_timesteps": total_timesteps,
        "ppo_params": PPO_PARAMS,
        "reward_weights": {"w_perf": W_PERF, "w_cost": W_COST, "w_penalty": W_PENALTY},
        "timestamp": timestamp,
    }
    meta_path = os.path.join(MODEL_DIR, f"{save_name}_{timestamp}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved model metadata to", meta_path)

    return agent, model_path, meta_path


# ---------------------------
# Evaluation helper
# ---------------------------
def evaluate_agent(agent: PPO, datasets: List[str], n_episodes: int = 10):
    results = []
    for i in range(n_episodes):
        ds = random.choice(datasets)
        env = AutoMLControllerEnv(dataset_name=ds, surrogate=None, use_surrogate=False)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            ep_reward += reward
        results.append(ep_reward)
    avg = float(np.mean(results))
    print(f"Evaluation over {n_episodes} episodes: avg_reward={avg:.4f}")
    return results


# ---------------------------
# Main CLI
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (basename without extension) located in data/ folder. If not provided, defaults to mock list.")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--use_surrogate", action="store_true", default=True)
    parser.add_argument("--save_name", type=str, default="ppo_automan_v1")
    args = parser.parse_args()

    if args.datasets:
        ds_list = [d.strip() for d in args.datasets.split(",")]
    else:
        # create a mock list of 18 dataset names if not provided
        ds_list = [f"dataset_{i+1}" for i in range(18)]
    print("Datasets used for training:", ds_list)

    agent, model_path, meta_path = train_agent_on_datasets(ds_list, total_timesteps=args.timesteps, use_surrogate=args.use_surrogate, save_name=args.save_name)

    # quick evaluation
    eval_results = evaluate_agent(agent, ds_list, n_episodes=10)
    # save evaluation
    eval_path = os.path.join(MODEL_DIR, f"eval_{os.path.basename(model_path)}.json")
    with open(eval_path, "w") as f:
        json.dump({"eval_rewards": eval_results}, f, indent=2)
    print("Saved evaluation to", eval_path)
