import torch
import pandas as pd

from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
import dgl

from cigin.featurize import get_graph_from_smile
from cigin.utils import get_len_matrix


def collate(samples):
    solute_graphs, solvent_graphs, labels = map(list, zip(*samples))
    solute_graphs = dgl.batch(solute_graphs)
    solvent_graphs = dgl.batch(solvent_graphs)
    solute_len_matrix = torch.from_numpy(get_len_matrix(solute_graphs.batch_num_nodes()))
    solvent_len_matrix = torch.from_numpy(get_len_matrix(solvent_graphs.batch_num_nodes()))
    return solute_graphs, solvent_graphs, solute_len_matrix, solvent_len_matrix, labels


class Dataclass(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        solute = self.dataset.loc[idx]['SoluteSMILES']
        mol = Chem.MolFromSmiles(solute)
        mol = Chem.AddHs(mol)
        solute = Chem.MolToSmiles(mol)
        solute_graph = get_graph_from_smile(solute)

        solvent = self.dataset.loc[idx]['SolventSMILES']
        mol = Chem.MolFromSmiles(solvent)
        mol = Chem.AddHs(mol)
        solvent = Chem.MolToSmiles(mol)
        solvent_graph = get_graph_from_smile(solvent)

        delta_g = self.dataset.loc[idx]['DeltaGsolv']

        return [solute_graph, solvent_graph, [delta_g]]


def csv_to_loader(csv_path, batch_size=32, shuffle=False):
    dataframe = pd.read_csv(csv_path, sep=";")
    dataset = Dataclass(dataframe)
    loader = DataLoader(dataset, collate_fn=collate, batch_size=batch_size, shuffle=shuffle)
    return loader


def data_to_loader(solute_smiles, solvent_smiles):
    solute = solute_smiles
    mol = Chem.MolFromSmiles(solute)
    mol = Chem.AddHs(mol)
    solute = Chem.MolToSmiles(mol)
    solute_graph = get_graph_from_smile(solute)

    solvent = solvent_smiles
    mol = Chem.MolFromSmiles(solvent)
    mol = Chem.AddHs(mol)
    solvent = Chem.MolToSmiles(mol)
    solvent_graph = get_graph_from_smile(solvent)

    solute_batch = torch.ones(size=(1, solute_graph.ndata['x'].shape[0]))
    solvent_batch = torch.ones(size=(1, solvent_graph.ndata['x'].shape[0]))

    return solute_graph, solvent_graph, solute_batch, solvent_batch
