import torch
from tqdm.notebook import tqdm

def evaluate_agent(env, model, n_episodes):
    episode_rewards = []

    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        state = torch.from_numpy(state)

        total_rewards_ep = 0

        while True:
            predicted_action = model(state)
            action = torch.argmax(predicted_action, dim=-1)

            new_state, reward, done, truncated, info = env.step(action.item())
            total_rewards_ep += reward

            if done: break
            state = torch.from_numpy(new_state)

        episode_rewards.append(total_rewards_ep)

    episode_rewards = torch.tensor(episode_rewards)
    mean_reward = torch.mean(episode_rewards)
    std_reward = torch.std(episode_rewards)

    return mean_reward, std_reward