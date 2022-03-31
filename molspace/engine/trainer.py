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
