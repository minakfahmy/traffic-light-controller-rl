import argparse
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from sumo_rl import SumoEnvironment

from agents.reinforce_agent import ReinforceAgent
from rewards import get_mean_queued


NETWORK = Path(__file__).resolve().parent / "networks" / "two-way-single-intersection" / "single-intersection.net.xml"


def run_episode(agent, env, seed, training):
    state, _ = env.reset(seed=seed)
    observations, actions, rewards = [], [], []
    done = False
    while not done:
        action = agent.make_action(state, test=not training)
        next_state, reward, terminated, truncated, _ = env.step(action)
        observations.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        done = terminated or truncated

    losses = agent.update(np.stack(observations), actions, rewards) if training else {}
    return float(np.mean(rewards)), len(rewards), losses


def main():
    parser = argparse.ArgumentParser(description="Train REINFORCE with a state-value baseline")
    parser.add_argument("--route-file", type=Path, required=True, help="Path to the PPO PR's scenario-normal.rou.xml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--num-seconds", type=int, default=3600)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.episodes, args.num_seconds, args.eval_interval) < 1:
        parser.error("episodes, num-seconds, and eval-interval must be positive")
    if not args.route_file.is_file():
        parser.error(f"route file does not exist: {args.route_file}")
    flow_ends = [float(flow.attrib["end"]) for flow in ET.parse(args.route_file).findall("flow")]
    if not flow_ends or min(flow_ends) < args.num_seconds:
        parser.error("route demand must cover the full episode. Use the normal route from Owen's PPO PR")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args), "route_file": str(args.route_file.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "route_sha256": hashlib.sha256(args.route_file.read_bytes()).hexdigest(),
        "evaluation_seeds": [601, 602, 603], "gamma": 0.95,
        "policy_lr": 5e-4, "value_lr": 1e-3,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    env = SumoEnvironment(
        net_file=str(NETWORK),
        route_file=str(args.route_file.resolve()),
        single_agent=True,
        use_gui=False,
        num_seconds=args.num_seconds,
        reward_fn=get_mean_queued,
        additional_sumo_cmd="--no-step-log true",
    )
    try:
        agent = ReinforceAgent(env)
        metrics = {key: [] for key in (
            "episode", "train_reward", "episode_len", "policy_loss", "value_loss",
            "entropy", "eval_episode", "eval_reward",
        )}
        best_eval_reward = float("-inf")
        for episode in range(1, args.episodes + 1):
            reward, length, losses = run_episode(agent, env, seed=args.seed, training=True)
            for key, value in {"episode": episode, "train_reward": reward, "episode_len": length, **losses}.items():
                metrics[key].append(value)
            if episode % args.eval_interval == 0 or episode == args.episodes:
                eval_reward = float(np.mean([
                    run_episode(agent, env, seed=seed, training=False)[0] for seed in (601, 602, 603)
                ]))
                metrics["eval_episode"].append(episode)
                metrics["eval_reward"].append(eval_reward)
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    torch.save(agent.state_dict(), args.output_dir / "best_model.pth")
                print(f"Episode {episode}: train {reward:.5f}, evaluation {eval_reward:.5f}", flush=True)
                (args.output_dir / "training_results.json").write_text(json.dumps(metrics, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
