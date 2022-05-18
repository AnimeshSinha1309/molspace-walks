import os
import typing

import torch
import torch_geometric
import tqdm.auto as tqdm

import rdkit
from rdkit.Chem.Draw import MolsToGridImage

from molspace.dataspace.featurizers import featurize_molecule, featurize_atom, featurize_bond


class GDBMoleculesDataset(torch_geometric.data.InMemoryDataset):
    """
    A loader for downloading and loading GDB molecule lists upto arbitrary sizes.
    """

    _url = "https://zenodo.org/record/5172018/files/gdb11.tgz?download=1"
    _molecules_per_file = 200_000

    def __init__(self, root: str, name: str, min_size: int = 1, max_size: int = 11):
        """
        Create the GDB molecule dataset class.
        Downloads and Processes the data if prepared files do not exist
        Delete processed files to regenerate dataset
        :type root: str
        :param root: The directory in which all datasets will be placed
        :type name: str
        :param name: The nome of the dataset being downloaded
        :type min_size: int
        :param min_size: Smallest molecules to load (in number of atoms)
        :type max_size: int
        :param max_size: Largest molecules to load (in number of atoms)
        """
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
        """
        Directory in which the zenodo download files are stored
        :rtype: str
        :return: path to the directory where raw data is being housed
        """
        return os.path.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        """
        Directory in which the processed graphs are stored, including
        data_i.pt for `i` depending on number of molecules, pre_filter.pt and
        post_transform.pt.
        :rtype: str
        :return: path to the directory where processed data is being housed
        """
        return os.path.join(self.root, self.name)

    @property
    def raw_file_names(self) -> typing.List[str]:
        """
        List of raw data file names being downloaded and extracted from zenodo
        :rtype: List[str]
        :return: List of file names of the smiles lists
        """
        return [
            f"gdb11_size{i:02}.smi"
            for i in range(
                self._size_range_in_atoms[0], self._size_range_in_atoms[1] + 1
            )
        ]

    @property
    def processed_file_names(self) -> typing.List[str]:
        """
        List of processed data file names which have been generated from processing
        raw data in zenodo
        :rtype: List[str]
        :return: List of file names which house torch_geometric graph data
        """
        total_molecules = sum(
            self._num_atoms_per_file[
                (self._size_range_in_atoms[0] - 1):(self._size_range_in_atoms[1])
            ]
        )
        number_of_files = (
            total_molecules + self._molecules_per_file - 1
        ) // self._molecules_per_file
        return [f"data_{i}.pt" for i in range(1, number_of_files + 1)]

    def __len__(self):
        """
        Gives the total number of molecules in the dataset
        WARNING: If you have changed the number of atoms in the constructor,
        but not deleted the processed files, this information will be wrong and
        carried forward from the older processing step.
        :return:
        """
        total_molecules = sum(
            self._num_atoms_per_file[
                (self._size_range_in_atoms[0] - 1):(self._size_range_in_atoms[1])
            ]
        )
        return total_molecules

    def download(self) -> None:
        torch_geometric.data.download_url(self._url, self.raw_dir)
        os.system(
            f"tar -xvf {os.path.join(self.raw_dir, 'gdb11.tgz')} -C {self.raw_dir}"
        )

    def process(self) -> None:
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
                        featurize_molecule(
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
    def node_features_shape(self) -> typing.Tuple[int]:
        mol = rdkit.Chem.MolFromSmiles("C")
        return featurize_atom(mol.GetAtomWithIdx(0)).shape

    @property
    def edge_features_shape(self) -> typing.Tuple[int]:
        mol = rdkit.Chem.MolFromSmiles("CC")
        return featurize_bond(mol.GetBondBetweenAtoms(0, 1)).shape

    @classmethod
    def draw_molecule(cls, molecule: torch_geometric.data.Data):
        """
        Generates a RDKit molecule object from the PyTorch Graph
        :type molecule: pyg.data.Data
        :param molecule: The RDKit molecule object
        :rtype: PIL.image
        :return: An image object for the graph
        Call `im.show()` on the image object to see the image rendered.
        """
        molecule = cls.get_rdkit(molecule)
        image = rdkit.Chem.Draw.MolToImage(molecule,)
        return image

    @staticmethod
    def get_rdkit(molecule: torch_geometric.data.Data) -> rdkit.Chem.Mol:
        """
        Generates a RDKit molecule object from the PyTorch Graph
        :type molecule: pyg.data.Data
        :param molecule: The RDKit molecule object
        :rtype: rdkit.Chem.Mol
        :return: The RDKit molecule corresponding to that graph
        """
        possible_atom_labels = [
            "C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Si"
        ]
        possible_bond_types = [
            rdkit.Chem.rdchem.BondType.SINGLE,
            rdkit.Chem.rdchem.BondType.DOUBLE,
            rdkit.Chem.rdchem.BondType.TRIPLE,
            rdkit.Chem.rdchem.BondType.AROMATIC,
        ]
        # noinspection PyArgumentList
        rw_molecule = rdkit.Chem.RWMol()
        for atom_idx in range(len(molecule.x)):
            atom = rdkit.Chem.Atom(possible_atom_labels[torch.where(molecule.x[atom_idx])[0]])
            rw_molecule.AddAtom(atom)
        list_of_bond_pairs = set()
        for bond_idx in range(molecule.edge_index.shape[1]):
            u, v = molecule.edge_index[0, bond_idx].item(), molecule.edge_index[1, bond_idx].item()
            if (u, v) not in list_of_bond_pairs:
                bond = possible_bond_types[torch.where(molecule.edge_attr[bond_idx])[0]]
                rw_molecule.AddBond(u, v, bond)
                list_of_bond_pairs.add((u, v))
                list_of_bond_pairs.add((v, u))
        return rw_molecule.GetMol()


if __name__ == "__main__":
    data = GDBMoleculesDataset("data", "gdb11", 3, 8)
    idx = -35
    print(data[idx].x)
    print(data[idx].edge_index)
    print(data[idx].edge_attr)
    mol = GDBMoleculesDataset.get_rdkit(data[idx])
    smiles = rdkit.Chem.MolToSmiles(mol)
    print(smiles)
    GDBMoleculesDataset.draw_molecule(data[idx]).show()
