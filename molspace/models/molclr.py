import torch
import torch_geometric


class GraphEncoder(torch.nn.Module):
    """
    The class to encode molecules from graph to vectors to
    generate their similarity estimates.
    """

    def __init__(self, input_dim, output_size, num_conv_layers=10, num_linear_layers=4):
        """
        Create the molecule featuriser neural network
        :param input_dim: size of the input features
        :param output_size: latent space vector dimensionality
        :param num_conv_layers: number of graph convolutional layers
        :param num_linear_layers: number of linear layers in the prediction head
        """
        super(GraphEncoder, self).__init__()
        self.conv_input = torch_geometric.nn.GraphConv(input_dim, 20)
        self.conv_internal = torch.nn.ModuleList([
            torch_geometric.nn.GraphConv(20, 20) for _ in range(num_conv_layers)])
        self.conv_output = torch_geometric.nn.GraphConv(20, output_size)
        self.linear_output = torch.nn.ModuleList(
            [torch.nn.Linear(output_size, output_size) for _ in range(num_linear_layers)])

    def forward(self, graph: torch.Tensor, n_feat: torch.Tensor):
        """
        Run the model to get the features
        :param graph: graph as an edge list
        :param n_feat: node feature tensor
        :return: the latent space representation tensor
        """
        n_hid = torch.relu(self.conv_input(n_feat, graph))
        for conv_layer in self.conv_internal:
            n_hid = torch.relu(conv_layer(n_hid, graph))
        n_hid = torch.relu(self.conv_output(n_hid, graph))
        n_hid = torch.sum(n_hid, dim=0)
        for linear_layer in self.linear_output:
            n_hid = torch.relu(linear_layer(n_hid))
        return n_hid


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
        Note that self.feat is a neural network which featurizes and can be used in isolation.
        :param input_dim: size of the input features
        :param output_dim: latent space vector dimensionality
        :param num_conv_layers: number of graph convolutional layers
        :param num_linear_layers: number of linear layers in the prediction head
        """
        super(MoleculeComparator, self).__init__()
        self.feat = GraphEncoder(input_dim, output_dim, num_conv_layers, num_linear_layers)
        self.linear_1 = torch.nn.Linear(2 * output_dim, 10)
        self.linear_2 = torch.nn.Linear(10, 10)
        self.linear_3 = torch.nn.Linear(10, 1)

    def forward(self, mol_1_graph, mol_1_nodes, mol_2_graph, mol_2_nodes):
        molecule_1 = self.feat(mol_1_graph, mol_1_nodes)
        molecule_2 = self.feat(mol_2_graph, mol_2_nodes)

        result = torch.cat([molecule_1, molecule_2], dim=-1)
        result = torch.relu(self.linear_1(result))
        result = torch.relu(self.linear_2(result))
        result = torch.sigmoid(self.linear_3(result))
        return result
