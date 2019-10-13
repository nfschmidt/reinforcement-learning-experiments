import torch
import benchmark

def fit(env,
        episodes=10_000,
        validate_n=1000,
        validation_episodes=100,
        learning_rate=0.1,
        epsilon=1.0, epsilon_decay=0.99995, epsilon_min=0.1,
        discount_factor=0.99,
        verbose=True):

    q_table = torch.zeros((env.observation_space.n, env.action_space.n))

    best_q_table = q_table.clone()
    best_score = 0.0

    for ep in range(1, episodes+1):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Determine action via exploration or
            # explotation according to random value
            if torch.rand(1).item() < epsilon:
                action = env.action_space.sample()
            else:
                action = torch.argmax(q_table[state]).item()

            new_state, reward, done, _ = env.step(action)
            total_reward += reward

            # update q table using bellman's equation
            target_value = torch.max(q_table[new_state])
            q_table[state, action] += \
                    learning_rate*(reward
                                   + discount_factor*target_value
                                   - q_table[state, action])   

            state = new_state

        # update exploration probability
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay    

        if ep % validate_n == 0:
            rewards = benchmark.play_episodes(env,
                                               q_table,
                                               episodes=validation_episodes)
            mean_reward = rewards.mean().item()

            if mean_reward > best_score:
                best_score = mean_reward
                best_q_table = q_table.clone()
                if verbose:
                    print(f'Episode {ep}: New best score! {best_score}')
                    
    return best_score, best_q_table
