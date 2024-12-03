
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
    def __init__(self, n_arms, n_features, delta):
        self.n_arms = n_arms
        self.n_features = n_features
        self.delta = delta

        # Initialisation de A et B pour chaque bras
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.B = [np.zeros(n_features) for _ in range(n_arms)]

        self.alpha = 1 + math.sqrt(1/2 * math.log(2/delta))
        self.theta = [np.zeros(n_features) for _ in range(n_arms)]

    def get_action(self, x):
        ucb_values = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.B[a]
            self.theta[a] = theta_a  # Mise à jour de θ pour le bras courant

            # Calcul de la borne supérieure de confiance
            ucb = theta_a @ x + self.alpha * np.sqrt(x.T @ A_inv @ x)
            ucb_values.append(ucb)
        return np.argmax(ucb_values)

    def fit_step(self, action, reward, x):
        x = x.reshape(-1, 1)  # Assurez-vous que x est un vecteur colonne
        self.A[action] += x @ x.T
        self.B[action] += (reward * x.flatten())



