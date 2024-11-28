
"""
.. moduleauthor:: Valentin Emiya, Marius THORRE
"""

import numpy as np
import matplotlib.pyplot as plt
import math


class LinearBandits:
    """
    Linear bandit problem

    Parameters
    ----------
    n_arms : int
        Number of arms or actions
    n_features : int
        Number of features
    """

    def __init__(self, n_arms, n_features):
        self._theta = np.random.randn(n_features, n_arms)

    @property
    def n_arms(self):
        return self._theta.shape[1]

    @property
    def n_features(self):
        return self._theta.shape[0]

    def step(self, a, x):
        """
        Parameters
        ----------
        a : int
            Index of action/arm
        x : ndarray
            Context (1D array)

        Returns
        -------
        float
            Reward
        """

        assert 0 <= a
        assert a < self.n_arms
        return np.vdot(x, self._theta[:, a]) + np.random.randn()

    def get_context(self):
        """
        Returns
        -------
        ndarray
            Context (1D array)
        """
        return np.random.randn(self.n_features)

    def __str__(self):
        return '{}-arms linear bandit in dimension {}'.format(self.n_arms,
                                                              self.n_features)


class LinUCBAlgorithm:
    """
    Parameters
    ----------
    n_arms : int
        Number of arms
    n_features : int
        Number of features
    delta : float
        Confidence level in [0, 1]
    """

    def __init__(self, n_arms, n_features, delta):
        self.A = np.ones((n_features, n_features, n_arms))
        self.B = np.zeros((n_features, n_arms))

        self.alpha = 1 + math.sqrt(1/2 + math.log(2/delta))
        self.theta = np.linalg.inv(self.A) @ self.B
        self.nb_arm = n_arms

    @property
    def n_arms(self):
        return self.A.shape[2]

    @property
    def n_features(self):
        return self.A.shape[0]

    def get_action(self, x):
        """
        Choose an action

        Parameters
        ----------
        x : ndarray
            Context

        Returns
        -------
        int
            The chosen action
        """
        u_k = []
        for k in range(self.n_arms):
           u_k.append(x.T @ self.theta[k] + self.alpha * np.sqrt(x.T @ np.linalg.inv(self.A[:, :, k]) @ x))
        u_k = np.array(u_k)
        return np.argmax(u_k)

    def fit_step(self, action, reward, x):
        """
        Update current value estimates with an (action, reward) pair

        Parameters
        ----------
        action : int
        reward : float
        x : ndarray
        """
        self.A[:, :, action] += np.outer(x, x.T)
        self.B[:, action] += reward * x
        self.theta[action] = np.linalg.inv(self.A[:, :, action]) @ self.B[:, action]


if __name__ == "__main__":
    K = 100
    iteration = 1000
    d = 10

    linear_b = LinearBandits(n_arms=K, n_features=d)
    agent = LinUCBAlgorithm(n_arms=K, n_features=d, delta=0.3)
    for _ in range(iteration):
        action = agent.get_action(linear_b.get_context())
        agent.fit_step(action=action, reward=linear_b.step())
