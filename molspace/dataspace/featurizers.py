import numpy as np
import rdkit

import torch
import torch_geometric


def featurize_atom(atom: rdkit.Chem.Atom) -> np.ndarray:
    """
    Generates a Feature Vector for the given atom
    :param atom: atom object in a rdkit molecule
    :return: np.array, the feature vector for the atom
    """
    possible_atom_labels = np.array(
        ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Si"]
    )
    atom_label = atom.GetSymbol() == possible_atom_labels

    feature_vector = np.concatenate([atom_label]).astype(np.float64)
    return feature_vector


def featurize_bond(bond: rdkit.Chem.Bond) -> np.ndarray:
    """
    Generates a Feature Vector for the given bond
    :param bond: bond object in a rdkit molecule
    :return: np.array, the feature vector for the atom
    """
    possible_bond_types = np.array(
        [
            rdkit.Chem.rdchem.BondType.SINGLE,
            rdkit.Chem.rdchem.BondType.DOUBLE,
            rdkit.Chem.rdchem.BondType.TRIPLE,
            rdkit.Chem.rdchem.BondType.AROMATIC,
        ]
    )
    bond_type = bond.GetBondType() == possible_bond_types

    feature_vector = np.concatenate([bond_type]).astype(np.float64)
    return feature_vector


def featurize_molecule(molecule, permute=True) -> torch_geometric.data.Data:
    """
    Method that constructs a molecular graph with nodes being the atoms
    and bonds being the edges.
    :param molecule: RD-kit molecule object to featurize
    :param permute: permute the atoms if True, otherwise leave in order
    :return: PyG graph object, Node features and Edge features
    """
    num_atoms = molecule.GetNumAtoms()
    permute = np.random.permutation(num_atoms) if permute else np.arange(num_atoms)

    node_features, edge_features, edge_list = [], [], []
    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(int(permute[i]))
        node_features.append(featurize_atom(atom_i))
        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(int(permute[i]), int(permute[j]))
            if bond_ij is not None:
                edge_list.append([permute[i], permute[j]])
                edge_features.append(featurize_bond(bond_ij))

    if len(edge_list) == 0:
        return torch_geometric.data.Data(
            x=torch.from_numpy(np.stack(node_features)).float(),
        )

    return torch_geometric.data.Data(
        x=torch.from_numpy(np.stack(node_features)).float(),
        edge_index=torch.from_numpy(np.stack(edge_list, axis=1)).long(),
        edge_attr=torch.from_numpy(np.stack(edge_features)).float(),
    )
