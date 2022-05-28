import torch
from molspace.models.molfeat import GraphEncoder


class MoleculeComparator(torch.nn.Module):
    """
    A class to compare two molecules and pull them closer or
    push them farther in the latent space. Designed to be the interface
    which helps in the contrastive learning pipeline.
    """

    def __init__(self, input_dim, output_dim, num_conv_layers=2, num_linear_layers=2):
        """
        Create the 2 molecule comparator network using the same featurizer network
        as a base.
        Note that `self.feat` is a neural network which featurizes and can be used in isolation.
        :param input_dim: size of the input features
        :param output_dim: latent space vector dimensionality
        :param num_conv_layers: number of graph convolutional layers
        :param num_linear_layers: number of linear layers in the prediction head
        """
        super(MoleculeComparator, self).__init__()
        self.feat = GraphEncoder(
            input_dim, output_dim, num_conv_layers, num_linear_layers
        )
        self.linear_1 = torch.nn.Linear(3 * output_dim, 10)
        self.linear_2 = torch.nn.Linear(10, 10)
        self.linear_3 = torch.nn.Linear(10, 1)

    def forward(self, mol_1_graph, mol_1_nodes, _mol_1_edges, mol_2_graph, mol_2_nodes, _mol_2_edges):
        molecule_1 = self.feat(mol_1_graph, mol_1_nodes)
        molecule_2 = self.feat(mol_2_graph, mol_2_nodes)

        result = torch.cat([molecule_1, molecule_2, torch.mul(molecule_1, molecule_2)], dim=-1)
        result = torch.relu(self.linear_1(result))
        result = torch.relu(self.linear_2(result))
        result = torch.sigmoid(self.linear_3(result))
        return result
