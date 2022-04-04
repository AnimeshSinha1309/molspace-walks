import os
import typing

import numpy as np
import torch
import torch_geometric
import tqdm.auto as tqdm

import rdkit
from rdkit.Chem.Draw import MolsToGridImage


class GDBMoleculesDataset(torch_geometric.data.InMemoryDataset):

    _url = "https://zenodo.org/record/5172018/files/gdb11.tgz?download=1"
    _molecules_per_file = 200_000

    def __init__(self, root: str, name: str, min_size: int = 1, max_size: int = 11):
        self.name = name
        self._size_range_in_atoms = (min_size, max_size)
        self._num_atoms_per_file = [
            4,
            9,
            20,
            80,
            352,
            1850,
            10568,
            66706,
            444313,
            3114041,
            22796628,
        ]
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name)

    @property
    def raw_file_names(self) -> typing.List[str]:
        return [
            f"gdb11_size{i:02}.smi"
            for i in range(
                self._size_range_in_atoms[0], self._size_range_in_atoms[1] + 1
            )
        ]

    @property
    def processed_file_names(self) -> typing.List[str]:
        total_atoms = sum(
            self._num_atoms_per_file[
                self._size_range_in_atoms[0] - 1 : self._size_range_in_atoms[1]
            ]
        )
        number_of_files = (
            total_atoms + self._molecules_per_file - 1
        ) // self._molecules_per_file
        return [f"data_{i}.pt" for i in range(1, number_of_files + 1)]

    def download(self):
        torch_geometric.data.download_url(self._url, self.raw_dir)
        os.system(
            f"tar -xvf {os.path.join(self.raw_dir, 'gdb11.tgz')} -C {self.raw_dir}"
        )

    def process(self):
        molecules: typing.List[torch_geometric.data.Data] = []
        file_number = 1
        total_atoms = sum(
            self._num_atoms_per_file[
                self._size_range_in_atoms[0] - 1 : self._size_range_in_atoms[1]
            ]
        )
        progress_bar = tqdm.tqdm(total=total_atoms)
        for size in range(
            self._size_range_in_atoms[0], self._size_range_in_atoms[1] + 1
        ):
            with open(os.path.join(self.raw_dir, "gdb11_size%02d.smi" % size)) as f:
                for line in f.readlines():
                    progress_bar.update(1)
                    molecules.append(
                        self.featurize_molecule(
                            rdkit.Chem.MolFromSmiles(line.split()[0])
                        )
                    )
                    if len(molecules) >= self._molecules_per_file:
                        torch.save(
                            self.collate(molecules),
                            os.path.join(self.processed_dir, f"data_{file_number}.pt"),
                        )
                        file_number += 1
                        molecules = []
        if len(molecules) > 0:
            torch.save(
                self.collate(molecules),
                os.path.join(self.processed_dir, f"data_{file_number}.pt"),
            )
        progress_bar.close()

    def __repr__(self) -> str:
        return f"{self.name}()"

    @property
    def node_features_shape(self):
        mol = rdkit.Chem.MolFromSmiles("C")
        return self.featurize_atom(mol.GetAtomWithIdx(0)).shape

    @property
    def edge_features_shape(self):
        mol = rdkit.Chem.MolFromSmiles("CC")
        return self.featurize_atom(mol.GetBondBetweenAtoms(0, 1)).shape

    @staticmethod
    def featurize_atom(atom):
        """
        Generates a Feature Vector for the given atom
        :param atom: atom object in an rdkit molecule
        :return: np.array, the feature vector for the atom
        """
        possible_atom_labels = np.array(
            ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Si"]
        )
        atom_label = atom.GetSymbol() == possible_atom_labels

        feature_vector = np.concatenate([atom_label]).astype(np.float)
        return feature_vector

    @staticmethod
    def featurize_bond(bond):
        """
        Generates a Feature Vector for the given bond
        :param bond: bond object in an rdkit molecule
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

        feature_vector = np.concatenate([bond_type]).astype(np.float)
        return feature_vector

    @classmethod
    def featurize_molecule(cls, molecule, permute=True) -> torch_geometric.data.Data:
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
            node_features.append(cls.featurize_atom(atom_i))
            for j in range(molecule.GetNumAtoms()):
                bond_ij = molecule.GetBondBetweenAtoms(int(permute[i]), int(permute[j]))
                if bond_ij is not None:
                    edge_list.append([permute[i], permute[j]])
                    edge_features.append(cls.featurize_bond(bond_ij))

        if len(edge_list) == 0:
            return torch_geometric.data.Data(
                x=torch.from_numpy(np.stack(node_features)).float(),
            )

        return torch_geometric.data.Data(
            x=torch.from_numpy(np.stack(node_features)).float(),
            edge_index=torch.from_numpy(np.stack(edge_list, axis=1)).long(),
            edge_attr=torch.from_numpy(np.stack(edge_features)).float(),
        )

    @staticmethod
    def draw_molecule(graph_edge_list, node_features, _bond_features):
        possible_atom_labels = np.array(
            ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Si"]
        )
        molecule = rdkit.Chem.RWMol()
        # TODO: Complete this RWMol problem
        pass
