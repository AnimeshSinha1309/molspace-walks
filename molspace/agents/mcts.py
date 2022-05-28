"""Implements Monte Carlo Tree Search over all possible subsets"""

import typing as ty
import numpy as np
import tqdm.auto as tqdm


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
        self.value = value
        self.evaluator = evaluator

        self.n_value = np.full(num_actions, fill_value=0.001)
        self.w_value = np.zeros(num_actions, dtype=np.float128)
        for i in range(num_actions):
            if (1 << i) & value != 0:
                self.w_value[i] = -1e8
                self.n_value[i] = 1e8
        # noinspection PyTypeChecker
        self.child_states: ty.List[ty.Optional["MCTSAgent"]] = [
            None for _ in range(num_actions)
        ]
        self.parent = parent

    def update_q(self, reward: float, index: int) -> None:
        """Updates the q-value for the state
        n_value is the number of times a node visited
        q_value is the q function
        n += 1, w += reward, q = w / n -> this is being implicitly computed
        using the weighted average
        :param reward: The obtained total reward from this state
        :param index: The index of the action chosen for which the reward was provided
        """
        self.w_value[index] += reward
        self.n_value[index] += 1

    def select(self) -> ty.Optional[int]:
        """Select one of the child actions based on UCT rule
        :returns: The selected id of the move, none if no move should be taken
        """
        n_visits = np.sum(self.n_value) + 1
        uct = self.w_value / self.n_value + self.HYPER_PARAMETER_EXPLORATION * np.sqrt(
            np.log(n_visits) / self.n_value
        )
        best_val = np.max(uct)
        if best_val < 0:
            return None
        best_move_indices = np.where(np.equal(best_val, uct))[0]
        winner: int = np.random.choice(best_move_indices)
        return winner

    def expand(self, action_index: int) -> "MCTSAgent":
        """Expand the nodes given the action
        :param action_index: The action from current node which should lead to expansion
        :returns: The new node in the MCTS object
        """
        next_state = self.evaluator.step(self.value, action_index)
        next_reward = self.evaluator(next_state) - self.evaluator(self.value)
        self.child_states[action_index] = MCTSAgent(
            num_actions=len(self.n_value),
            evaluator=self.evaluator,
            value=next_state,
            parent=(self, action_index, next_reward),
        )
        mcts_state = self.child_states[action_index]
        return ty.cast(MCTSAgent, mcts_state)

    def rollout(self):
        """Performs a random rollout
        The total reward in each rollout is computed.
        :returns: mean across the R random rollouts.
        """
        return self.evaluator(self.value)

    def backup(self, reward) -> None:
        """Backs-up the rewards at the terminal node into the tree
        Propagates up the values to all the ancestors
        :param reward: reward obtained from the rollout
        """
        mcts_state = self
        while mcts_state.parent is not None:
            reward = mcts_state.parent[2] + self.HYPER_PARAM_DISCOUNT_FACTOR * reward
            mcts_state.parent[0].update_q(reward, mcts_state.parent[1])
            mcts_state = mcts_state.parent[0]

    def search(self, n_mcts) -> None:
        """Perform the MCTS search from the root
        :param n_mcts: Number of iterations of MCTS to run
        """
        for _ in range(n_mcts):
            mcts_state = self
            while True:
                action_index: ty.Optional[int] = mcts_state.select()
                if action_index is None:
                    break
                elif mcts_state.child_states[action_index] is not None:
                    mcts_state = ty.cast(
                        MCTSAgent, mcts_state.child_states[action_index]
                    )
                else:
                    mcts_state = mcts_state.expand(action_index)
                    break

            total_reward = mcts_state.rollout()
            mcts_state.backup(total_reward)

    def act(self) -> int:
        """Explore nodes greedily and return the optimal state resulting from this process
        :return: The value of the best state found
        """
        mcts_agent = self
        with tqdm.tqdm() as progress_bar:
            while True:
                progress_bar.update(1)
                progress_bar.set_postfix(value=mcts_agent.value)
                mcts_agent.search(1000)
                move = mcts_agent.select()
                if move is None:
                    progress_bar.close()
                    return mcts_agent.value
                assert mcts_agent.child_states[move] is not None
                mcts_agent = ty.cast(MCTSAgent, mcts_agent.child_states[move])

    def replay(self):
        self.evaluator.model.train()
        value_losses = []
        policy_losses = []
        for state, v, p in self.evaluator:
            loss_v, loss_p = self.evaluator.model.fit(state, v, p)
            value_losses.append(loss_v)
            policy_losses.append(loss_p)
        self.evaluator.memory.clear()
        return np.mean(value_losses), np.mean(policy_losses)
