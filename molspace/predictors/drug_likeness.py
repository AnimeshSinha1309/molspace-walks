import rdkit, rdkit.Chem.Crippen


def predict_log_p(mol):
    try:
        mol.UpdatePropertyCache()
        rdkit.Chem.AddHs(mol)
        current_value = rdkit.Chem.Crippen.MolLogP(mol, includeHs=True)
        return current_value
    except rdkit.Chem.rdchem.AtomValenceException:
        return -999999999999.
