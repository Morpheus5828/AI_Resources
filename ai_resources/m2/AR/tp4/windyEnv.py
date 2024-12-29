import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class WindyEnv:
    def __init__(self):
        self.start = 0
        self.goal = 0
        self.rows = 7
        self.cols = 10
        self.x_max = self.cols - 1
        self.y_max = self.rows - 1
        self.wind_1 = [3, 4, 5, 8]
        self.wind_2 = [6, 7]
        self.actions = ["N", "S", "E", "W"]

    def set_terminal(self, start_state, goal_state):
        self.start = self.cell(start_state)
        self.goal = self.cell(goal_state)

    def cell(self, pos):
        #return pos[1] + self.cols * pos[0]
        row, col = pos
        return col + self.cols * row

    def step(self, s, a):
        x = s % self.cols
        y = s // self.cols

        del_x = 0
        del_y = 0
        if a == 'E':
            del_x = 1
        elif a == 'W':
            del_x = -1
        elif a == 'N':
            del_y = -1
        elif a == 'S':
            del_y = 1
        new_x = max(0, min(x + del_x, self.x_max))
        new_y = max(0, min(y + del_y, self.y_max))
        if new_x in self.wind_1:
            new_y = max(0, new_y - 1)
        if new_x in self.wind_2:
            new_y = max(0, new_y - 2)
        return self.cell((new_y, new_x))


class QlearningAgent:
    def __init__(self, actions, alpha, gamma, epsilon):
        self.q = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_utility(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, s):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_vals = [self.get_utility(s, a) for a in self.actions]
            max_q = max(q_vals)
            candidates = [a for a, qv in zip(self.actions, q_vals) if qv == max_q]
            action = random.choice(candidates)
        return action

    def fit_step(self, s, a, r, s_new):
        old_utility = self.q.get((s, a), None)
        if old_utility is None:
            self.q[(s, a)] = r
        else:
            next_max_utility = max(self.get_utility(s_new, a2) for a2 in self.actions)
            self.q[(s, a)] = old_utility + self.alpha * (r + self.gamma * next_max_utility - old_utility)

class SarsaAgent:
    def __init__(self, actions, alpha, gamma, epsilon):
        self.q = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_utility(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, s):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.get_utility(s, act) for act in self.actions]
            max_utility = max(q)
            if q.count(max_utility) > 1:
                best_actions = [self.actions[i] for i in range(len(self.actions)) if q[i] == max_utility]
                action = random.choice(best_actions)
            else:
                action = self.actions[q.index(max_utility)]
        return action

    def fit_step(self, s1, a1, s2, a2, r):
        old_utility = self.q.get((s1, a1), 0.0)
        next_utility = self.q.get((s2, a2), 0.0)
        self.q[(s1, a1)] = old_utility + self.alpha * (r + self.gamma * next_utility - old_utility)


if __name__ == "__main__":
    start_state = (3, 0)
    goal_state = (3, 7)
    alpha = 0.5
    gamma = 0.9
    epsilon = 0.1

    world = WindyEnv()
    world.set_terminal(start_state, goal_state)

    #agent = QlearningAgent(world.actions, alpha, gamma, epsilon)
    agent = SarsaAgent(world.actions, alpha, gamma, epsilon)

    n_episodes = 200
    max_steps = 1000


    timesteps_history = []
    episodes_completed_history = []
    global_timestep = 0

    episodes_completed = 0

    last_action = None

    for episode in tqdm(range(n_episodes)):
        s = world.start
        a = agent.choose_action(s)
        steps = 0
        while s != world.goal and steps < max_steps:
            s_new = world.step(s, a)
            r = -1 if s_new != world.goal else 0
            a_new = agent.choose_action(s_new)
            agent.fit_step(s, a, s_new, a_new, r)

            s = s_new
            a = a_new

            global_timestep += 1
            steps += 1
        episodes_completed += 1
        timesteps_history.append(global_timestep)
        episodes_completed_history.append(episodes_completed)

    plt.plot(timesteps_history, episodes_completed_history)
    plt.xlabel("Time steps")
    plt.ylabel("Episodes completed")
    plt.title("Windy Gridworld: nombre d’épisodes terminés au cours du temps")
    plt.show()
