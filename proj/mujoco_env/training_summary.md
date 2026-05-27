# Cart-Pole Balance Training Summary

## All Versions Tested

| Version | Algorithm | Environment | Network | Steps | Mean Steps | Max Steps | Notes |
|---------|-----------|-------------|---------|-------|------------|-----------|-------|
| v1 (original) | PPO | CartPoleEnv | default | 2M | ~154 | ~403 | Best recorded in eval logs |
| v2 | PPO | CartPoleEnv (modified reward) | default | 500K | ~93 | - | Higher angle penalty |
| v3 | PPO | CartPoleEnvV3 | 256x256 | 1M | ~131 | ~350 | Positive reward design |
| v4 | PPO | CartPoleEnvV4 | 512x512 | 2M | ~100 | ~265 | Larger network |
| v5 | PPO | CartPoleEnvV5 | 128x128 | 3M | ~118 | ~279 | Exp+linear angle penalty |
| v6 | PPO | CartPoleEnv | default | 1M | ~96 | ~173 | Standard PPO |
| v7 | PPO | CartPoleEnvV5 | 512x512 | 5M | ~135 | ~257 | Largest network tested |
| v8 | PPO | CartPoleEnvV8 (fixed sensor) | 256x256 | 3M | ~160 | ~350 | Sensor extraction fixed |
| v9 | SAC | CartPoleEnvV9 | 256x256 | 2M | ~110 | ~231 | Off-policy, state normalization |
| v10 | SAC | CartPoleEnvV10 | 256x256 | 2M | ~89 | ~130 | Survival bonus + penalties |
| v11 | PPO | CartPoleEnvV11 | default | 280K | ~100 | ~142 | Curriculum + LR schedule |
| TD3 | TD3 | CartPoleEnv | 256x256 | 5M | ~85 | ~114 | Twin Delayed DDPG |
| Extended PPO | PPO | CartPoleEnv | default | 10M+ | ~100 | ~152 | Extended training |

## Key Findings

1. **All RL algorithms (PPO, SAC, TD3) achieve similar performance: ~100-200 steps average**
2. **No algorithm consistently achieved 1000+ steps**
3. **v1 original model had best recorded max: ~403 steps in eval, ~300 in our tests**
4. **Larger networks and more training steps did NOT significantly improve results**
5. **Reward function changes had minimal impact on final performance**

## Environment Analysis

- **Termination**: 45 degrees (0.785 rad) - relatively generous
- **Max episode**: 1000-1200 steps (5-6 seconds at 200Hz simulation)
- **Action space**: Continuous torque [-2, 2] Nm
- **Observation**: 8-dim (wheel angles, pole angles, angular velocities)

## Technical Observations

1. **Sensor extraction**: Original code used Y-component of frameangvel, which may not be optimal
2. **Action scaling**: Network outputs [-1, 1] scaled to physical torque
3. **No artificial delays found**: Only `time.sleep(0.005)` in visualization, which is for display only

## Why 1000 Steps Is Difficult

1. **Double pendulum dynamics**: pole2 is cantilevered, errors accumulate rapidly
2. **Continuous torque control**: Requires precise balancing vs. simple push/no-push
3. **Limited sensing**: Only angular positions/velocities, no direct position feedback
4. **Noise accumulation**: Small errors compound over time

## Recommendations for Achieving 1000+ Steps

1. **Behavior Cloning**: Use LQR controller as expert, train neural network to mimic
2. **Simplify Model**: Single pole instead of double would be much easier
3. **Hybrid Approach**: Use LQR for actual control, neural network for higher-level decisions
4. **Accept Limits**: Current physics may inherently limit achievable balance time