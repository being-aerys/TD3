# Created by Aashish Adhikari at 3:42 PM 4/15/2021
import gym
import TD3
import numpy as np

def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            eval_env.render('rgb_array')
            avg_reward += reward

    avg_reward /= eval_episodes
    eval_env.close() #VVI

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward




env_name = "Pendulum-v0"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
model = TD3.TD3(state_dim, action_dim, max_action)
path = "models\TD3_Pendulum-v0"
model.load(path)

eval_policy(model,env_name, 0, 1)