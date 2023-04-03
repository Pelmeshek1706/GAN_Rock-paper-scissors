import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv1D
# from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist


# Define state evaluation function using Minimax algorithm
def evaluate_state(state, agent):
    # Convert state to binary representation
    binary_state = np.zeros((3,3))
    for i in range(3):
        if state[0] == i:
            binary_state[i][0] = 1
        if state[1] == i:
            binary_state[i][1] = 1
    # Generate fake state using GAN
    noise = np.random.normal(0, 1, (1, 100))
    fake_state = generator.predict(noise)[0]
    # Evaluate state using Minimax algorithm for agent 1
    if agent == 0:
        values = []
        for i in range(3):
            action = np.zeros(3)
            action[i] = 1
            reward = np.sum(action * binary_state) - np.sum(action * fake_state)
            values.append(reward)
        action = np.argmax(values)
    # Evaluate state using Regret Matching for agent 2
    else:
        # Calculate regret for each action
        regrets = []
        for i in range(3):
            action = np.zeros(3)
            action[i] = 1
            reward = np.sum(action * binary_state) - np.sum(action * fake_state)
            regrets.append(reward)
        # Update probabilities using regret matching
        probabilities = np.zeros(3)
        total_regret = np.sum([r for r in regrets if r > 0])
        if total_regret > 0:
            for i in range(3):
                if regrets[i] > 0:
                    probabilities[i] = regrets[i] / total_regret
        else:
            probabilities = np.ones(3) / 3
        # Choose action based on probabilities
        action = np.random.choice(3, p=probabilities)
    return action

# Define generator network
generator = Sequential()
generator.add(Dense(128 * 3, input_dim=100))
generator.add(LeakyReLU())
generator.add(Reshape((3, 128)))
generator.add(Conv1D(64, kernel_size=3, padding='same'))
generator.add(LeakyReLU())
generator.add(Conv1D(1, kernel_size=3, padding='same', activation='tanh'))

# Initialize state, score, and tie count
state = [0, 0]
score = [0, 0]
ties = 0
round_number = 1

for _ in range(10000):
    # Choose action for second agent using MiniMax
    action1 = evaluate_state(state, 0)
    # Choose action for second agent using Regret Matching
    action2 = evaluate_state(state, 1)
    # Determine winner of round and update score
    if action1 == action2:
        ties += 1
        print('Round {}: Agent 1 chose {} and Agent 2 chose {}. Tie. Total score: {}:{}, Ties: {}.'.format(sum(score)+ties, ['Rock', 'Paper', 'Scissors'][action1], ['Rock', 'Paper', 'Scissors'][action2], score[0], score[1], ties))
    elif (action1 == 0 and action2 == 1) or (action1 == 1 and action2 == 2) or (action1 == 2 and action2 == 0):
        score[1] += 1
        print('Round {}: Agent 1 chose {} and Agent 2 chose {}. Agent 2 wins. Total score: {}:{}, Ties: {}.'.format(sum(score)+ties, ['Rock', 'Paper', 'Scissors'][action1], ['Rock', 'Paper', 'Scissors'][action2], score[0], score[1], ties))
    else:
        score[0] += 1
        print('Round {}: Agent 1 chose {} and Agent 2 chose {}. Agent 1 wins. Total score: {}:{}, Ties: {}.'.format(sum(score)+ties, ['Rock', 'Paper', 'Scissors'][action1], ['Rock', 'Paper', 'Scissors'][action2], score[0], score[1], ties))

