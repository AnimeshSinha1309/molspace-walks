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
        self.conv_internal = torch.nn.ModuleList(
            [torch_geometric.nn.GraphConv(20, 20) for _ in range(num_conv_layers)]
        )
        self.conv_output = torch_geometric.nn.GraphConv(20, output_size)
        self.linear_output = torch.nn.ModuleList(
            [
                torch.nn.Linear(output_size, output_size)
                for _ in range(num_linear_layers)
            ]
        )

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
