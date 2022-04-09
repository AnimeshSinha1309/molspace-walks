"""
This module is responsible for generation of the molecular space graph
that we are constructing our random walks on.

It will provide whatever data interface we have to get the molecules,
their neighbors on the graph, the graph based representation of said molecules
in PyTorch and RDKit, and more.
"""
import networkx as nx

from molspace.dataspace.gdbloader import GDBMoleculesDataset
from molspace.dataspace.featurizers import molecule_similarity


class MolecularSpace:

    def __init__(self, data: GDBMoleculesDataset):
        graph = nx.Graph()
        for i in range(len(data)):
            mol_i = data.get_rdkit(data[i])

        for i in range(len(data)):
            for j in range(len(data)):
                mol_i = data.get_rdkit(data[i])
                mol_j = data.get_rdkit(data[j])
                similarity = molecule_similarity(mol_i, mol_j)
                graph.add_edge(mol_i, mol_j, w=similarity)

    def __str__(self):
        pass
