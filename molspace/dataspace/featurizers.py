import rdkit.DataStructs, rdkit.Chem.AllChem


def molecule_similarity(mol_1, mol_2):
    sim = rdkit.DataStructs.FingerprintSimilarity(
        rdkit.Chem.AllChem.RDKFingerprint(mol_1),
        rdkit.Chem.AllChem.RDKFingerprint(mol_2),
        metric=rdkit.DataStructs.TanimotoSimilarity
    )
    return sim


def single_removes(mol_1, mol_2):
    patt = rdkit.Chem.MolFromSmarts(rdkit.Chem.MolToSmiles(mol_2))
    matches = list(mol_1.GetSubstructMatch(patt))
    return len(matches) > 0
