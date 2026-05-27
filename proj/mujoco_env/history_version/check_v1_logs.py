"""
Check v1 training logs to understand what worked
"""
import numpy as np
import os

log_path = "tb_logs/PPO_CartPole_20260520_1616_2000000/evaluations.npz"
if os.path.exists(log_path):
    d = np.load(log_path, allow_pickle=True)
    print("v1 training evaluations:")
    print(f"  Total evaluations: {len(d['timesteps'])}")
    print(f"  Timestep range: {d['timesteps'][0]} to {d['timesteps'][-1]}")
    ep_lengths = d['ep_lengths']
    means = [np.mean(x) for x in ep_lengths]
    print(f"  Episode length history: {[round(m,1) for m in means]}")
    print(f"  Max mean episode length: {max(means):.1f}")
    print(f"  Best eval index: {np.argmax(means)}")
else:
    print("v1 log not found")