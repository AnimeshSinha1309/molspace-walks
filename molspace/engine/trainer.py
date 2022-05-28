import torch, torch.utils.data
import tqdm


def train_loop(model, dataset, steps=10000):
    """
    Implements a training loop for the model in classical loss optimization
    over dataset setting
    :type model: torch.nn.Module
    :param model: The model being trained
    :type dataset: torch.utils.data.Dataloader
    :param dataset: The data we are training on as a dataloader
    :type steps: int
    :param steps: The number of steps we want to train for

    The function simultaneously provides access to
    """
    model.train()
    optimizer = torch.optim.Adam(lr=1e-3)
    loss_fn = torch.nn.HuberLoss()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    for _step in tqdm.trange(steps):
        for x, y in dataloader:
            optimizer.zero_grad()
            outputs = model(x.node_attr, x.edge_index, x.edge_attr, x.batch)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
