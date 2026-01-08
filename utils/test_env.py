import gymnasium as gym

def test_cartpole():
    
    # step1.创建环境
    env = gym.make("CartPole-v0", render_mode='human')
    # step2.重置环境
    observation, info = env.reset()

    print(f'初始状态维度： {observation.shape}')
    
    # step3.action采样
    for _ in range(100):
        action = env.action_space.sample()
        
        # 执行动作，获得反馈
        # Gymnasium的step返回5个值： observation, reward, terminated, truncated, info
        next_obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print('环境测试成功')

if __name__ == "__main__":
    test_cartpole()
