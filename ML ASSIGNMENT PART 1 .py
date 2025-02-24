#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import gym

# Define Q-Learning Algorithm
def q_learning(env, alpha, gamma, epsilon, min_epsilon, epsilon_decay, num_episodes, max_steps_per_episode):
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    q_table = np.zeros((state_space_size, action_space_size))
    rewards = []
    frames = []  # WE store frames here

    for episode in range(num_episodes):
        state = env.reset()[0]  
        total_reward = 0
        done = False

        if episode == 0:
            frames.append(env.render())  # we are capturing initial frame at starting episode

        for step in range(max_steps_per_episode):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  
            else:
                action = np.argmax(q_table[state, :])  

            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # Q-Update
            best_next_action = np.argmax(q_table[next_state, :])
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
            )
            state = next_state

            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)

        # here we are Printing the policy and rewards for every 2000 episodes
        if episode % 2000 == 0 and episode != 0:
            print(f"Episode {episode} - Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")
            print("Policy after this episode:")
            render_policy(env, q_table)
            frames.append(env.render())

    frames.append(env.render())

    return q_table, rewards, frames

# Function to render policy as arrows
def render_policy(env, data):
    directions = ['←', '↓', '→', '↑']  # we are setting arrow directins of agent 
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow']  
    
    if isinstance(data, np.ndarray) and data.ndim == 2:
        policy = np.argmax(data, axis=1).reshape((4, 4))
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        policy = data.reshape((4, 4))
    else:
        raise ValueError("Invalid data type for policy rendering.")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)

    for row in range(4):
        for col in range(4):
            action = policy[row, col]
            direction = directions[action]
            color = colors[action]
            
            # Create colored box for the action
            ax.add_patch(plt.Rectangle((col, 3 - row), 1, 1, color=color, alpha=0.5))
            ax.annotate(direction, xy=(col + 0.5, 3 - row + 0.5), 
                        xytext=(col + 0.5, 3 - row + 0.5),
                        ha='center', va='center', fontsize=20, color='black')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    
    plt.show()

# policy iteration algorithm
def policy_iteration(env, gamma, max_iterations=1000):
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    policy = np.zeros(state_space_size, dtype=int)  
    value_function = np.zeros(state_space_size)  

    for i in range(max_iterations):
        # Policy Evaluation
        for state in range(state_space_size):
            action = policy[state]
            value_function[state] = sum(
                [prob * (reward + gamma * value_function[next_state])
                 for prob, next_state, reward, done in env.P[state][action]]
            )

        # Policy Improvement
        new_policy = np.zeros(state_space_size, dtype=int)
        for state in range(state_space_size):
            action_values = np.zeros(action_space_size)
            for action in range(action_space_size):
                action_values[action] = sum(
                    [prob * (reward + gamma * value_function[next_state])
                     for prob, next_state, reward, done in env.P[state][action]]
                )
            new_policy[state] = np.argmax(action_values)

        # we are checking convergence
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy

    return policy, value_function

# we are intialiazing function to evaluate policy and calculate success rate of algorithm
def evaluate_policy(env, policy, num_episodes=1000, max_steps_per_episode=100):
    total_success = 0

    for _ in range(num_episodes):
        state = env.reset()[0]  
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            action = policy[state]  
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            step += 1

        if done and reward > 0:
            total_success += 1

    success_rate = (total_success / num_episodes)*100   
    return success_rate

# Main code for testing different parameter combinations
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='rgb_array') 

# Different combinations of alpha, gamma, and epsilon values to test
alpha_values = [0.1, 0.3, 0.5]
gamma_values = [0.8, 0.9, 0.99]
epsilon_values = [0.1, 0.5, 0.9]

# Initialize variables to track the best success rate and parameters
best_success_rate = 0
best_params = {}

# Run Q-learning algorithm with different parameter combinations
for alpha in alpha_values:
    for gamma in gamma_values:
        for epsilon in epsilon_values:
            print(f"Running Q-learning with alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
            min_epsilon = 0.01
            epsilon_decay = 0.995
            num_episodes = 20000
            max_steps_per_episode = 100

            q_table, q_rewards, captured_frames = q_learning(env, alpha, gamma, epsilon, min_epsilon, epsilon_decay, num_episodes, max_steps_per_episode)

            # Plot rewards per episode
            plt.figure(figsize=(14, 7))
            plt.plot(q_rewards, label='Total Reward per Episode')
            highlight_episodes = [2000, 4000, 6000, 8000, 10000] 
            for ep in highlight_episodes:
                if ep < len(q_rewards):
                    plt.scatter(ep, q_rewards[ep], color='red')  
                    plt.text(ep, q_rewards[ep], f'Ep {ep}', color='red', verticalalignment='bottom')

            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title(f'Total Reward per Episode (alpha={alpha}, gamma={gamma}, epsilon={epsilon})')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Histogram plot of total rewards per episode
            plt.figure(figsize=(12, 6))
            plt.hist(q_rewards, bins=50, color='skyblue', edgecolor='black')
            plt.xlabel('Total Reward')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Total Rewards per Episode (alpha={alpha}, gamma={gamma}, epsilon={epsilon})')
            plt.grid(axis='y', alpha=0.75)
            plt.show()

            # Display captured frames: initial, random middle, and last frame
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Initial frame
            axes[0].imshow(captured_frames[0])
            axes[0].axis('off')
            axes[0].set_title('Initial Frame')

            # Random middle frame
            middle_index = len(captured_frames) // 2
            axes[1].imshow(captured_frames[middle_index])
            axes[1].axis('off')
            axes[1].set_title(f'Middle Frame ({middle_index} episodes)')

            # Last frame
            axes[2].imshow(captured_frames[-1])
            axes[2].axis('off')
            axes[2].set_title('Last Frame')

            plt.tight_layout()
            plt.show()

            # Evaluate policy and compare success rates
            success_rate = evaluate_policy(env, q_table.argmax(axis=1))
            print(f"Success Rate: {success_rate:.2f}")

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_params = {
                    'alpha': alpha,
                    'gamma': gamma,
                    'epsilon': epsilon
                }

# Print the best success rate and their parameters
print(f"Best Success Rate: {best_success_rate}%")
print(f"Best Parameters: {best_params}")

# Display policy iteration result
print("Running Policy Iteration...")
optimal_policy_policy_iteration, _ = policy_iteration(env, gamma=best_params['gamma'])
render_policy(env, optimal_policy_policy_iteration) 

policy_iteration_success_rate = evaluate_policy(env, optimal_policy_policy_iteration)
print(f"Policy Iteration Success Rate: {policy_iteration_success_rate}%")


# Render the optimal policy from Q-learning
print("Optimal Policy from Q-learning:")
render_policy(env, q_table)

# Render the optimal policy from Policy Iteration
print("Optimal Policy from Policy Iteration:")
render_policy(env, optimal_policy_policy_iteration)



# In[ ]:




