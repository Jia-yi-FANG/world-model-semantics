import gymnasium as gym
import minigrid
import numpy as np


def test_env():
    env = gym.make("MiniGrid-FourRooms-v0")
    obs, _ = env.reset(seed=42)

    print("=== 环境测试 ===")
    print(f"观测 keys: {list(obs.keys())}")
    print(f"图像 shape: {obs['image'].shape}")
    print(f"方向: {obs['direction']}")
    print(f"任务描述: {obs['mission']}")
    print(f"动作空间: {env.action_space}")
    print(f"智能体位置: {env.unwrapped.agent_pos}")
    print(f"网格大小: {env.unwrapped.grid.width} x {env.unwrapped.grid.height}")

    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            print(f"Episode 结束，步数={step+1}, 总奖励={total_reward:.3f}")
            break

    env.close()
    print("\n环境测试通过 OK")


if __name__ == "__main__":
    test_env()
