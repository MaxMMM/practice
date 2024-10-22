import torch
import random

class TabularSARSA:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount=0.8, epsilon=0.1):
        self.q_table = torch.rand((num_states, num_actions))
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.states = []

    def print_q_table(self):
        print(self.q_table)

    def state_to_index(self, current_state):
        """
        Receives a state, returns the index on q_table that corresponds to that state
        """
        for i, state in enumerate(self.states):
            if torch.equal(current_state, state):
                return i

        if len(self.states) < self.q_table.size(0):
            self.states.append(current_state)
            return len(self.states) - 1
        else:
            print("IF THIS IS BEING PRINTED, THEN SOMETHING WENT WRONG...")

    def select_action(self, state_index):
        if random.random() < self.epsilon:
            action = torch.randint(0, self.q_table.size(1), (1,)).item()
        else:
            action = torch.argmax(self.q_table[state_index]).item()
        return action

    def update(self, state_index, action_index, reward, next_state_index, next_action_index):
        target = reward + self.discount * self.q_table[next_state_index, next_action_index]
        current = self.q_table[state_index, action_index]
        new_value = current + self.learning_rate * (target - current)
        self.q_table[state_index, action_index] = new_value

class EnvironmentStub:
    def __init__(self, num_states, num_actions, observation_space=8):
        self.action_space = torch.arange(num_actions)
        self.state_space = torch.rand(num_states, observation_space)
        self.current_state_index = None
        self.num_states = num_states
        self.num_actions = num_actions

    def reset(self):
        self.current_state_index = torch.randint(0, self.num_states, (1,)).item()
        initial_state = self.state_space[self.current_state_index, :]
        return initial_state

    def step(self, action):
        # Simulate state transition based on action
        self.current_state_index = (self.current_state_index + action) % self.num_states
        next_state = self.state_space[self.current_state_index, :]
        # Define reward arbitrarily
        reward = 1.0 if self.current_state_index == self.num_states - 1 else 0.0
        done = self.current_state_index == self.num_states - 1
        info = {}
        return next_state, reward, done, info

if __name__ == '__main__':
    num_states = 4
    num_actions = 3

    model = TabularSARSA(num_states, num_actions)
    model.print_q_table()

    env = EnvironmentStub(num_states, num_actions)

    for episode in range(100000):
        done = False
        state = env.reset()
        state_index = model.state_to_index(state)
        action = model.select_action(state_index)
        while not done:
            next_state, reward, done, info = env.step(action)
            next_state_index = model.state_to_index(next_state)
            next_action = model.select_action(next_state_index)
            model.update(state_index, action, reward, next_state_index, next_action)
            state_index = next_state_index
            action = next_action
        if episode % 100 == 0:
            print(f"Episode {episode}")
            model.print_q_table()
