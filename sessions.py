import numpy as np
from collections import deque


def _run_train_episode(agent, env, brain_name, epsilon):
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        action = agent.act(state, epsilon)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]
        agent.save_experiences(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        score += reward
        if done:
            break
    return score


def train(agent, env, config):
    """Deep Q-Learning session"""

    # get the default brain
    brain_name = env.brain_names[0]

    scores = []
    scores_window = deque(maxlen=100)
    epsilon = config["train"]["epsilon_high"]
    for i_episode in range(1, config["train"]["nb_episodes"] + 1):
        score = _run_train_episode(agent, env, brain_name, epsilon)
        scores_window.append(score)
        scores.append(score)
        epsilon = max(config["train"]["epsilon_low"], config["train"]["epsilon_decay"] * epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= config["general"]["average_score_for_solving"]:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            break
    return scores


def test(agent, env, num_test_runs=3):
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]

    for episode in range(num_test_runs):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state)
            print("Current Action: {}".format(action))
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break

        print("Score at Episode {}: {}".format(episode, score))
