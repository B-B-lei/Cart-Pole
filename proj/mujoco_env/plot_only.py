"""
单独绘图脚本 - 从已有训练数据生成图表
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import load_results

# 使用最新的训练日志目录
log_dir = "./tb_logs/PPO_CartPole_20260527_0909_1000000/"

print(f"Reading from: {log_dir}")
df = load_results(log_dir)

print(f"Columns: {df.columns.tolist()}")
print(f"Episodes: {len(df)}")
print(df.head())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'PPO CartPole Training Results', fontsize=14)

window = min(50, len(df))

# 1. Episode Reward
ax1 = axes[0, 0]
if 'r' in df.columns:
    rolling_mean = df['r'].rolling(window=window).mean()
    ax1.plot(df.index, df['r'], alpha=0.3, color='blue', label='Episode Reward')
    ax1.plot(df.index, rolling_mean, color='blue', linewidth=2, label=f'Rolling Mean ({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# 2. Episode Length
ax2 = axes[0, 1]
if 'l' in df.columns:
    rolling_len = df['l'].rolling(window=window).mean()
    ax2.plot(df.index, df['l'], alpha=0.3, color='green', label='Episode Length')
    ax2.plot(df.index, rolling_len, color='green', linewidth=2, label=f'Rolling Mean ({window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# 3. Training Progress (timesteps vs reward)
ax3 = axes[1, 0]
if 't' in df.columns and 'r' in df.columns:
    ax3.plot(df['t'], df['r'], alpha=0.5, color='red')
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward over Timesteps')
    ax3.grid(True, alpha=0.3)

# 4. Statistics Box
ax4 = axes[1, 1]
ax4.axis('off')
stats_text = f"""
Training Statistics:
─────────────────────
Total Timesteps: {int(df['t'].max()) if 't' in df.columns else 'N/A':,}
Episodes: {len(df)}

Reward Statistics:
  Mean: {df['r'].mean():.2f}
  Std:  {df['r'].std():.2f}
  Min:  {df['r'].min():.2f}
  Max:  {df['r'].max():.2f}

Length Statistics:
  Mean: {df['l'].mean():.2f}
  Std:  {df['l'].std():.2f}
  Min:  {df['l'].min():.2f}
  Max:  {df['l'].max():.2f}
"""
ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
        fontsize=11, verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plot_path = os.path.join(log_dir, "training_results.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to: {plot_path}")