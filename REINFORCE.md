# REINFORCE with a learned baseline

This draft implements episodic Monte Carlo REINFORCE with separate policy and state-value networks. The policy uses discounted returns minus a detached value prediction. The value network fits those returns with mean squared error. Both networks have two 256-unit tanh layers. Gamma is 0.95, policy learning rate is 0.0005, and value learning rate is 0.001.

## Run

Use Python 3.11, SUMO 1.28.0, and the dependencies below.

```sh
python -m pip install torch==2.15.0 sumo-rl==1.4.5 sumolib==1.28.0 traci==1.28.0
export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/1.28.0/EclipseSUMO/share/sumo"
mkdir -p outputs
git fetch origin refs/pull/6/head
git show FETCH_HEAD:networks/two-way-single-intersection/scenario-normal.rou.xml > outputs/ppo-scenario-normal.rou.xml
python train_reinforce.py --route-file outputs/ppo-scenario-normal.rou.xml --output-dir outputs/reinforce-run
python evaluate_reinforce.py --route-file outputs/ppo-scenario-normal.rou.xml --checkpoint outputs/reinforce-run/best_model.pth --output outputs/reinforce-evaluation.json
```

The normal route in PPO PR #6 extends demand to 100,000 seconds. The route on main ends at 1,000 seconds. Use the PR route for 3,600-second episodes to match Owen's saved PPO runs of 720 decisions. Its SHA-256 at PR commit 88fe5be is 432605ec720116d354d522aa0c3b9c05d82cb0a5b245870159fc4c2e1318fdfa.

Training writes the configuration and route hash, learning-curve metrics, and best deterministic validation checkpoint. Each episode uses SUMO seed 0 by default, matching the PPO training code. Validation uses seeds 601–603. Evaluation defaults to held-out seeds 701–703 and resets the action-sampling seed for each episode.

The reward is Owen's negative capacity-weighted average of normalized lane queues, ranging from -1 to 0. Higher is better.

## Preliminary results and open work

A 30-episode pilot used SUMO 1.28.0, sumo-rl 1.4.5, torch 2.15.0, and the PR route. Training used sampled actions. Deterministic validation selected the episode-5 checkpoint.

On held-out SUMO seeds 701, 702, and 703, using that same unchanged checkpoint:

| Action selection | Mean reward per step |
| --- | --- |
| Highest-probability action | -0.380173 |
| Sample from policy probabilities | -0.155687 |

For each deterministic episode, requested action counts were [623, 1, 0, 96]. Observed green phase counts were [528, 0, 0, 192]. Sampling activated all four green phases. This demonstrates a substantial evaluation-mode effect on this checkpoint.

The same setup gave random -0.173647, greedy -0.021694, and Owen's saved PPO run-2 policy -0.020477. Three seeds and one training run are preliminary evidence. The PPO checkpoint came from 300 training episodes.

The learning curve's first and last five training episodes averaged -0.177143 and -0.064423. Evaluation still needs improvement. Investigate the long episode's interaction with the gamma-to-the-timestep policy weighting, baseline fit, and training duration before declaring the agent ready. The evaluation command supports both action-selection modes without updating model weights.

## Validation

- Completed the 30-episode, 720-decision pilot with finite losses and rewards.
- Completed six full episodes comparing action selection on one checkpoint.
- Confirmed a fresh short run writes configuration, metrics, and a loadable checkpoint.
- Confirmed deterministic held-out results reproduce the earlier evaluation.
