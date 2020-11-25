import numpy as np
np.random.seed(0)


# Newly defined episode T
def T(state, direction, actions):
        return [(0.8, go(state, actions[direction])),
                (0.1, go(state, actions[(direction + 1) % 4])),
                (0.1, go(state, actions[(direction - 1) % 4]))]


# The function to change the state of the state
def go(state, direction):
    return [s+d for s, d in zip(state, direction)]

# Determine next action by the epsilon-greedy
def get_action(t_state, episode):
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[t_state])
    else:
        next_action = np.random.choice(len(actions))
    return next_action


# implement a function to take single action 
def take_single_action(state, direction, actions):
    x = np.random.uniform(0, 1)
    cumulative_probability = 0.0
    for probability_state in T(state, direction, actions):
        probability, next_state = probability_state
        cumulative_probability += probability
        if x < cumulative_probability:
            break
    reward = situation[next_state[0], next_state[1]]
    if reward is None:
        return state, -0.04
    else:
        return next_state, reward


# Implement a function to update the action value function
def update_Qtable(q_table, t_state, action, reward, t_next_state, next_action):
    gamma = 0.8
    alpha = 0.4
    q_table[t_state, action] += alpha * (reward + gamma * q_table[t_next_state, next_action] - q_table[t_state, action])
    return q_table

def trans_state(state):
    return sum([n*(10**i) for i, n in enumerate(state)])

q_table = np.random.uniform(
    low=-0.01, high=0.01, size=(10 ** 2, 4))

num_episodes = 500
max_number_of_steps = 1000
total_reward_vec = np.zeros(5)
goal_average_reward = 0.7

# Define the environment
situation = np.array([[None, None, None, None, None, None],
                      [None, -0.04, -0.04, -0.04, -0.04, None],
                      [None, -0.04, None, -0.04, -1, None],
                      [None, -0.04, -0.04, -0.04, +1, None],
                      [None, None, None, None, None, None]])


terminals=[[2, 4], [3, 4]]
init = [1,1]
actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
state = [n*(10**i) for i,n in enumerate(init)]


# Define the repetition of an episode
for episode in range(num_episodes):
    state = init
    t_state = trans_state(state)
    action = np.argmax(q_table[t_state])
    episode_reward = 0

    # Define a time step loop
    for t in range(max_number_of_steps):
        next_state, reward = take_single_action(state, action, actions)
        episode_reward += reward
        t_next_state = trans_state(next_state)
        next_action = get_action(t_next_state, episode)
        # Update the behavioral value function
        q_table = update_Qtable(q_table, t_state, action, reward, t_next_state, next_action)
        # q_table = update_QtableQ(q_table, t_state, action, reward, t_next_state, next_action)
        state = next_state
        t_state = trans_state(state)
        action = next_action

        if state in terminals :
            break

    # Record the reward  
    total_reward_vec = np.hstack((total_reward_vec[1:],episode_reward))
    print(total_reward_vec)
    print("Episode %d has finished. t=%d" %(episode+1, t+1))
    print(min(total_reward_vec),goal_average_reward)

    # Success if the last 100 episodes are more than the stipulated reward
    if (min(total_reward_vec) >= goal_average_reward):
        print('Episode %d train agent successfuly! t=%d' %(episode, t))
        break
