"""Random Agent that just tries out random actions and takes the best one"""

import numpy as np

from ..dataset.meta_dataset import MetaDataset


class RandomAgent:
    """A random agent for evaluating the combination of moves"""

    def __init__(
        self,
        num_actions: int,
        evaluator: MetaDataset,
    ) -> None:
        """Initialize a new random searcher object.
        This initializes an agent which will randomly try out different
        states to try out and evaluate which states will be the best.
        :param num_actions: Maximum number of actions that can be taken
        :param evaluator: Loss function which takes the state and given evaluation
        """
        self.evaluator = evaluator
        self.num_actions = num_actions
        self.best_state: int = 0
        self.best_reward: float = 0.0

    def search(self, n_trials: int) -> None:
        """Perform the random search for best value
        :param n_trials: Number of iterations of random samples to run
        """
        for _ in range(n_trials):
            state = np.random.randint(2 ** self.num_actions)
            reward = self.evaluator(state)
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_state = state

    def act(self) -> int:
        """Searches and returns the best result
        :return: The value of the best state found
        """
        self.search(100)
        return self.best_state
