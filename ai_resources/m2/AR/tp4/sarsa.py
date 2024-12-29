

import random
import config as cfg


class Sarsa:
    def __init__(self, actions, alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.epsilon = epsilon

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

    def learn(self, state1, action1, state2, action2, reward):
        old_utility = self.q.get((state1, action1), 0.0)
        next_utility = self.q.get((state2, action2), 0.0)
        self.q[(state1, action1)] = old_utility + self.alpha * (reward + self.gamma * next_utility - old_utility)

