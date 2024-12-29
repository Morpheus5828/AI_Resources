import random
import config as cfg
from collections import defaultdict


class ExpectedSarsa:
    def __init__(self, actions, alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon):
        self.q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    def learn(self, state1, action1, state2, reward):
        old_utility = self.q[(state1, action1)]
        expected_utility = 0.0

        action_probabilities = self.get_action_probabilities(state2)
        for action in self.actions:
            expected_utility += action_probabilities[action] * self.q[(state2, action)]

        self.q[(state1, action1)] = old_utility + self.alpha * (reward + self.gamma * expected_utility - old_utility)

    def get_utility(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.get_utility(state, act) for act in self.actions]
            max_utility = max(q)
            if q.count(max_utility) > 1:
                best_actions = [self.actions[i] for i in range(len(self.actions)) if q[i] == max_utility]
                action = random.choice(best_actions)
            else:
                action = self.actions[q.index(max_utility)]
        return action

    def get_action_probabilities(self, state):
        q_values = [self.q[(state, action)] for action in self.actions]
        max_q = max(q_values)
        optimal_actions = [action for action, q in zip(self.actions, q_values) if q == max_q]
        n_optimal = len(optimal_actions)

        action_probabilities = {}
        for action in self.actions:
            if action in optimal_actions:
                action_probabilities[action] = (1 - self.epsilon) / n_optimal + self.epsilon / len(self.actions)
            else:
                action_probabilities[action] = self.epsilon / len(self.actions)

        return action_probabilities
