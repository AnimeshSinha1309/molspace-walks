"""The idea of implementing MCTS inspired from the Lipschitz bandit problem"""

import typing as ty
import numpy as np
import torch

from molspace.predictors.drug_likeness import predict_log_p
from molspace.models.molclr import MoleculeComparator
from molspace.dataspace.featurizers import featurize_molecule
from molspace.memory.simple_memory import SimpleMemory


class MolspaceAgent:
    """The final agent which does the full processing of random walks
    with the uncertainty approximations
    """

    def __init__(
        self,
        graph,
        evaluator: MoleculeComparator,
    ) -> None:
        """Initialize a new lipschitz sampler object.
        This maintains the mean rewards of all the clusters and their variances,
        activates and eliminates while performing adaptive discretization and
        optimism under uncertainty.
        :param evaluator: Loss function which takes the state and given evaluation
        """
        self.evaluator = evaluator
        self.graph = graph
        self.best_state: int = -1
        self.best_reward: float = -99999999999.0
        self.cached_evaluations: ty.Dict[ty.Tuple[int, int], float] = {}
        self.memory = SimpleMemory()

    def distance(self, n1: int, n2: int) -> float:
        """Computes the edit distance between two states
        :param n1: One of the two elements to find the distance between
        :param n2: One of the two elements to find the distance between
        :return: The edit distance
        """
        with torch.no_grad():
            mol_1 = self.graph.data.get_rdkit(self.graph.data[n1])
            mol_2 = self.graph.data.get_rdkit(self.graph.data[n2])
            data_1 = featurize_molecule(mol_1)
            data_2 = featurize_molecule(mol_2)
            transition_value = self.evaluator(data_1.edge_index, data_1.x, data_1.edge_attr,
                                              data_2.edge_index, data_2.x, data_2.edge_attr)
            return transition_value

    def cached_evaluator(self, x: int, y: int) -> float:
        """Caches results from evaluator so the redundant queries don't get sent in
        :param x: The state to be evaluated to
        :param y: The state to be evaluated from
        :return: The value of the evaluation
        """
        if (x, y) not in self.cached_evaluations:
            self.cached_evaluations[x, y] = self.evaluator(x)
        return self.cached_evaluations[x, y]

    def act(self, state):
        """Searches and returns the best result
        :return: The value of the best state found
        """
        self.best_reward: float = -99999999999.0
        if self.best_state == -1:
            self.memory.store(state, None)

        actions = self.graph.actions(state)
        action_values = []
        for action in actions:
            next_state = action
            _sim_score = self.distance(state, next_state)
            reward_score = predict_log_p(self.graph.data.get_rdkit(self.graph.data[next_state]))
            action_values.append(reward_score)

        if len(action_values) > 0:
            probabilities = np.array(action_values)
            probabilities = np.exp(probabilities / 10)
            probabilities = probabilities / np.sum(probabilities)
            best_action = np.random.choice(np.arange(len(probabilities)))
            self.best_state = self.graph.step(state, best_action)
        else:
            self.best_state = np.random.randint(len(self.graph))

        self.memory.store(self.best_state, self.best_reward)
        self.best_reward = predict_log_p(self.graph.data.get_rdkit(self.graph.data[self.best_state]))
        return self.best_state, self.best_reward, False, {}

    def replay(self):
        loss_fn = torch.nn.HuberLoss()
        optimizer = torch.optim.Adam(self.evaluator.parameters(), lr=1e-3)
        for time in range(len(self.memory)):
            optimizer.zero_grad()
            s1, r, s2 = self.memory.retrieve(time)
            s1 = (featurize_molecule(self.graph.data.get_rdkit(self.graph.data[s1[0]])),
                  featurize_molecule(self.graph.data.get_rdkit(self.graph.data[s1[1]])))
            s2 = (featurize_molecule(self.graph.data.get_rdkit(self.graph.data[s2[0]])),
                  featurize_molecule(self.graph.data.get_rdkit(self.graph.data[s2[1]])))
            with torch.no_grad():
                self.evaluator.eval()
                v2 = self.evaluator(s2[0].edge_index, s2[0].x, s2[0].edge_attr,
                                    s2[1].edge_index, s2[1].x, s2[1].edge_attr)
            self.evaluator.train()
            v1 = self.evaluator(s1[0].edge_index, s1[0].x, s1[0].edge_attr,
                                s1[1].edge_index, s1[1].x, s1[1].edge_attr)
            loss = loss_fn(v1, v2 + r)
            loss.backward()
            optimizer.step()
            # print(loss.item())
