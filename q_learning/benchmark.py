import torch

def play_episodes(env, q_table, episodes=1, render=False):
    rewards = torch.zeros((episodes,))

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if render:
                env.render()

            action = torch.argmax(q_table[state]).item()

            state, reward, done, _ = env.step(action)
            total_reward += reward

        rewards[ep] = total_reward

    # render last state
    if render:
        env.render()

    return rewards
