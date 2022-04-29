import rdkit.Chem.Draw


def visualize_molecule_match(mol, pattern):
    hit_atoms = list(mol.GetSubstructMatch(pattern))
    hit_bonds = []
    for bond in pattern.GetBonds():
        atom_id_1 = hit_atoms[bond.GetBeginAtomIdx()]
        atom_id_2 = hit_atoms[bond.GetEndAtomIdx()]
        hit_bonds.append(mol.GetBondBetweenAtoms(atom_id_1, atom_id_2).GetIdx())

    d = rdkit.Chem.Draw.rdMolDraw2D.MolDraw2DSVG(500, 500)
    rdkit.Chem.Draw.rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        mol,
        highlightAtoms=hit_atoms,
        highlightBonds=hit_bonds
    )
    d.FinishDrawing()
    return d
