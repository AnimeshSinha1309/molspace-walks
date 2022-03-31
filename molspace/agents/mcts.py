"""Implements Monte Carlo Tree Search over all possible subsets"""

import typing as ty


class MCTSAgent:
    """Monte Carlo Tree Search object for evaluating the combination of moves"""

    HYPER_PARAMETER_NOISE_ALPHA = 0.2
    HYPER_PARAMETER_EXPLORATION = 1.0
    HYPER_PARAM_DISCOUNT_FACTOR = 0.95
    HYPER_PARAM_POLICY_TEMPERATURE = 0

    def __init__(
        self,
        num_actions: int,
        evaluator,
        value: int = 0,
        parent: ty.Optional[ty.Tuple["MCTSAgent", int, float]] = None,
    ) -> None:
        """Initialize a new state object.
        This state object will have a visit counter and a mean reward estimate for
        each of it's children. Each state will contain an integer value representing
        the set. Any bit added to that set will be a valid action if and only if it
        was not already present in the set.
        :param value: The value stored in the state
        :param num_actions: Maximum number of actions that can be taken
        :param evaluator: Loss function which takes the state and given evaluation
        :param parent: The parent element in the tree search and the action to get there
        """

    def update_q(self, reward: float, index: int) -> None:
        """Updates the q-value for the state
        n_value is the number of times a node visited
        q_value is the q function
        n += 1, w += reward, q = w / n -> this is being implicitly computed
        using the weighted average
        :param reward: The obtained total reward from this state
        :param index: The index of the action chosen for which the reward was provided
        """

    def select(self) -> ty.Optional[int]:
        """Select one of the child actions based on UCT rule
        :returns: The selected id of the move, none if no move should be taken
        """

    def expand(self, action_index: int) -> "MCTSAgent":
        """Expand the nodes given the action
        :param action_index: The action from current node which should lead to expansion
        :returns: The new node in the MCTS object
        """

    def rollout(self):
        """Performs a random rollout
        The total reward in each rollout is computed.
        :returns: mean across the R random rollouts.
        """

    def backup(self, reward) -> None:
        """Backs-up the rewards at the terminal node into the tree
        Propagates up the values to all the ancestors
        :param reward: reward obtained from the rollout
        """

    def search(self, n_mcts) -> None:
        """Perform the MCTS search from the root
        :param n_mcts: Number of iterations of MCTS to run
        """

    def act(self) -> int:
        """Explore nodes greedily and return the optimal state resulting from this process
        :return: The value of the best state found
        """
