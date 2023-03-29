######### Q- Learning #####################
import numpy as np
import random

# Define the states and actions
states = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
actions = ['support', 'charge', 'partial support']

# Define the transition probabilities
P = {'s1': {'support': {'s1': 0.1, 's2': 0.7, 's3': 0.2, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0}, 
             'charge': {'s1': 0.8, 's2': 0.2, 's3': 0, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0},
             'partial support': {'s1': 0.5, 's2': 0.3, 's3': 0.2, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0}},
     's2': {'support': {'s1': 0, 's2': 0.2, 's3': 0.8, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0},
             'charge': {'s1': 0.6, 's2': 0.4, 's3': 0, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0},
             'partial support': {'s1': 0, 's2': 0.4, 's3': 0.6, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0}},
     's3': {'support': {'s1': 0, 's2': 0, 's3': 0.2, 's4': 0.6, 's5': 0, 's6': 0, 's7': 0, 's8': 0.2},
             'charge': {'s1': 0, 's2': 0.5, 's3': 0.1, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0.4},
             'partial support': {'s1': 0, 's2': 0, 's3': 0.4, 's4': 0.4, 's5': 0, 's6': 0, 's7': 0, 's8': 0.2}},
     's4': {'support': {'s1': 0, 's2': 0, 's3': 0, 's4': 0.2, 's5': 0.7, 's6': 0, 's7': 0, 's8': 0.1},
             'charge': {'s1': 0, 's2': 0, 's3': 0.5, 's4': 0.2, 's5': 0, 's6': 0, 's7': 0, 's8': 0.3},
             'partial support': {'s1': 0, 's2': 0, 's3': 0, 's4': 0.3, 's5': 0.6, 's6': 0, 's7': 0, 's8': 0.1}},
     's5': {'support': {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0.3, 's6': 0.5, 's7': 0, 's8': 0.2},
             'charge': {'s1': 0, 's2': 0, 's3': 0, 's4': 0.2, 's5': 0.4, 's6': 0, 's7': 0, 's8': 0.4},
             'partial support': {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0.8, 's6': 0.1, 's7': 0, 's8': 0.1}},
     's6': {'support': {'s1': 0, 's2': 0.6, 's3': 0, 's4': 0, 's5': 0, 's6': 0.1, 's7': 0.1, 's8': 0.2},
             'charge': {'s1': 0, 's2': 0.1, 's3': 0, 's4': 0, 's5': 0, 's6': 0.2, 's7': 0.2, 's8': 0.5},
             'partial support': {'s1': 0, 's2': 0.3, 's3': 0, 's4': 0, 's5': 0, 's6': 0.3, 's7': 0.2, 's8': 0.2}},
     's7': {'support': {'s1': 0, 's2': 0.3, 's3': 0.5, 's4': 0, 's5': 0, 's6': 0, 's7': 0.2, 's8': 0},
             'charge': {'s1': 0, 's2': 0.4, 's3': 0.1, 's4': 0, 's5': 0, 's6': 0, 's7': 0.5, 's8': 0},
             'partial support': {'s1': 0, 's2': 0.4, 's3': 0.3, 's4': 0, 's5': 0, 's6': 0, 's7': 0.3, 's8': 0}},
     's8': {'support': {'s1': 0, 's2': 0.4, 's3': 0.2, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0.4},
             'charge': {'s1': 0, 's2': 0.5, 's3': 0, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0.5},
             'partial support': {'s1': 0, 's2': 0.5, 's3': 0.1, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0.4}}}

# Define the rewards
rewards =  {'s1': {'support': {'s1': 0, 's2': 200, 's3': 200, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0}, 
                  'charge': {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0},
                   'partial support': {'s1': 0, 's2': 50, 's3': 100, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0}},
     's2': {'support': {'s1': 0, 's2': 0, 's3': 200, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0}, 
            'charge': {'s1': 0, 's2': 50, 's3': 0, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0},
             'partial support': {'s1': 0, 's2': 0, 's3': 50, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0}},
     's3': {'support': {'s1': 0, 's2': 0, 's3': 0, 's4': 250, 's5': 0, 's6': 0, 's7': 0, 's8': -50}, 
            'charge': {'s1': 0, 's2': 50, 's3': 25, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': -50},
             'partial support': {'s1': 0, 's2': 0, 's3': 200, 's4': 200, 's5': 0, 's6': 0, 's7': 0, 's8': -50}},
     's4': {'support': {'s1': 0, 's2': 0, 's3': 0, 's4': 100, 's5': 100, 's6': 0, 's7': 0, 's8': -25}, 
            'charge': {'s1': 0, 's2': 0, 's3': -25, 's4': 100, 's5': 0, 's6': 0, 's7': 0, 's8': -50},
             'partial support': {'s1': 0, 's2': 0, 's3': 0, 's4': 100, 's5': 200, 's6': 0, 's7': 0, 's8': -50}},
     's5': {'support': {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 50, 's6': -10, 's7': 0, 's8': -50}, 
            'charge': {'s1': 0, 's2': 0, 's3': 0, 's4': 50, 's5': 100, 's6': 0, 's7': 0, 's8': -50},
             'partial support': {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 50, 's6': -50, 's7': 0, 's8': -50}},
     's6': {'support': {'s1': 0, 's2': 50, 's3': 0, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': -10}, 
            'charge': {'s1': 0, 's2': 10, 's3': 0, 's4': 0, 's5': 0, 's6': 0, 's7': -10, 's8': -50},
             'partial support': {'s1': 0, 's2': 60, 's3': 0, 's4': 0, 's5': 0, 's6': -10, 's7': 0, 's8': -10}},
     's7': {'support': {'s1': 0, 's2': 50, 's3': 100, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0}, 
            'charge': {'s1': 0, 's2': 50, 's3': 50, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0},
             'partial support': {'s1': 0, 's2': 50, 's3': 50, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0}},
     's8': {'support': {'s1': 0, 's2': 100, 's3': 50, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': -25}, 
            'charge': {'s1': 0, 's2': 50, 's3': 0, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0},
             'partial support': {'s1': 0, 's2': 50, 's3': 50, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': -25}}}

# Initialize the Q-values for each state-action pair to 0
q_values = np.zeros((len(states), len(actions)))


# Define the learning rate and discount factor
alpha = 0.1
gamma = 0.9

# Perform Q-learning
for _ in range(10000):
    state = np.random.choice(states)  ##randomly selects a state from a list of states.
    action = np.random.choice(actions)  ##randomly selects an action from a list of possible actions.
    next_state_prob = P[state][action]  ##selects the next state probabilities for the current state and action from a transition probability matrix P.
    next_state = np.random.choice(list(next_state_prob.keys()), p=list(next_state_prob.values())) ##randomly selects the next state based on the transition probabilities for the current state and action.
    reward = rewards[state][action][next_state]  ##reward for taking the selected action in the current state and moving to the selected next state.
    # Update the Q-value for the current state and action based on the reward obtained and the maximum Q-value for the next state
    q_values[states.index(state)][actions.index(action)] = q_values[states.index(state)][actions.index(action)] + alpha * (reward + gamma * np.max(q_values[states.index(next_state)]) - q_values[states.index(state)][actions.index(action)])

print(q_values)
