"""
This module is responsible for generation of the molecular space graph
that we are constructing our random walks on.

It will provide whatever data interface we have to get the molecules,
their neighbors on the graph, the graph based representation of said molecules
in PyTorch and RDKit, and more.
"""

import networkx as nx
import tqdm.auto as tqdm

from molspace.dataspace.gdbloader import GDBMoleculesDataset
from molspace.dataspace.similarity import molecule_similarity


class MolecularSpace:

    def __init__(self, data: GDBMoleculesDataset, threshold: int = 0.7):
        self.graph = nx.Graph()
        self.data = data

        for i in range(len(data)):
            self.graph.add_node(i)

        for idx in tqdm.trange(len(data) * len(data)):
            i, j = idx // len(data), idx % len(data)
            if i == j:
                continue
            mol_i = data.get_rdkit(data[i])
            mol_j = data.get_rdkit(data[j])
            similarity = molecule_similarity(mol_i, mol_j)
            if similarity > threshold:
                self.graph.add_edge(i, j, w=similarity)

    def actions(self, node):
        return list(self.graph.neighbors(node))

    def step(self, state, action):
        return list(self.graph.neighbors(state))[action]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.graph)
