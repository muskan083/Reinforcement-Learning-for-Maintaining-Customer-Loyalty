import numpy
############### Value Iteration #########################
# Initialize value function for each state
V = {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0}

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
R = {'s1': {'support': {'s1': 0, 's2': 200, 's3': 200, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0}, 
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
# Define the discount factor
gamma = 0.9

# Perform value iteration
for i in range(1000):
    for s in V.keys():
        V[s] = max([sum([P[s][a][s1] * (R[s][a][s1] + gamma * V[s1]) for s1 in V.keys()]) for a in ['support','charge', 'partial support']])
print("Value of state: ", V)

# Initialize the action-value function
Q = {s: {a: sum([P[s][a][s1] * (R[s][a][s1] + gamma * V[s1]) for s1 in V.keys()]) for a in ['support', 'charge', 'partial support']} for s in V.keys()}
print(Q)

     
