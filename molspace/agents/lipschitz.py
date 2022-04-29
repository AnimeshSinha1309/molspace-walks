"""The idea of implementing MCTS inspired from the Lipschitz bandit problem"""

import typing as ty
import numpy as np
import tqdm.auto as tqdm


class BiasingSamplerAgent:
    """A lipschitz sample agent for evaluating the combination of moves
    Uses adaptive discretization for avoiding normal MCTS.
    """

    def __init__(
        self,
        num_actions: int,
        evaluator,
    ) -> None:
        """Initialize a new lipschitz sampler object.
        This maintains the mean rewards of all the clusters and their variances,
        activates and eliminates while performing adaptive discretization and
        optimism under uncertainty.
        :param num_actions: Maximum number of actions that can be taken
        :param evaluator: Loss function which takes the state and given evaluation
        """
        self.evaluator = evaluator
        self.num_actions = num_actions
        self.best_state: int = 0
        self.best_reward: float = 0.0
        self.cached_evaluations: ty.Dict[int, float] = {}

    @staticmethod
    def distance(n1: int, n2: int) -> int:
        """Computes the edit distance between two states
        :param n1: One of the two elements to find the distance between
        :param n2: One of the two elements to find the distance between
        :return: The edit distance
        """
        return sum([i == "1" for i in bin(n1 ^ n2)])

    def cached_evaluator(self, x: int) -> float:
        """Caches results from evaluator so the redundant queries don't get sent in
        :param x: The state to be evaluated
        :return: The value of the evaluation
        """
        if x not in self.cached_evaluations:
            self.cached_evaluations[x] = self.evaluator(x)
        return self.cached_evaluations[x]

    def _generate_neighborhood(
        self, start_values: ty.List[int], max_distance: int, min_distance: int
    ) -> ty.List[int]:
        """Generates samples in neighborhood so that each number in the
        max-distance neighborhood of start value has a sampled neighbor
        in min-distance of itself.
        :param start_values: The value in whose neighborhood we are searching
        :param max_distance: The distance of neighborhood we need to cover
        :param min_distance: The sparseness/density of sampling
        :return: the generated samples
        """
        # Generate the set to sample from
        neighborhood_as_set = set()
        for start_value in start_values:
            for i in range(2 ** self.num_actions):
                if self.distance(i, start_value) <= max_distance:
                    neighborhood_as_set.add(i)
        neighborhood = list(neighborhood_as_set)
        # Generate the actual samples
        samples: ty.List[int] = []
        while len(neighborhood) > 0:
            result = np.random.choice(neighborhood)
            is_already_sampled = False
            for sample in samples:
                if self.distance(sample, result) < min_distance:
                    is_already_sampled = True
                    break
            neighborhood.remove(result)
            if not is_already_sampled:
                samples.append(result)
        return samples

    def _run_selection_round(
        self, samples: ty.List[int]
    ) -> ty.Tuple[ty.List[int], ty.List[float]]:
        """Runs the evaluator, sorts the input states by their
        evaluations and returns all of this.
        :param samples: The set of samples in the previous evaluation
        :return: The sorted list of states and their evaluations
        """
        samples = sorted(samples, key=lambda x: self.evaluator(x), reverse=True)
        values = list(map(self.evaluator, samples))
        return samples, values

    def search(self, _n_trials: int) -> None:
        """Searches the space with adaptive discretization
        :param _n_trials: Dummy variable
        """
        samples = [0]
        for sample_radius, sample_reduce_to in tqdm.tqdm(
            [
                (self.num_actions, self.num_actions // 2),
                (self.num_actions // 2, 3),
                (3, 1),
            ]
        ):
            samples = self._generate_neighborhood(
                samples, sample_radius, sample_reduce_to
            )
            states, values = self._run_selection_round(samples)
            if values[0] > self.best_reward:
                self.best_state = states[0]
                self.best_reward = values[0]
            samples = states[:3]

    def act(self) -> int:
        """Searches and returns the best result
        :return: The value of the best state found
        """
        self.search(0)
        return self.best_state
