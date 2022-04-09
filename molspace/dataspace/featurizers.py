import rdkit.DataStructs


def molecule_similarity(mol_1, mol_2):
    sim = rdkit.DataStructs.FingerprintSimilarity(
        mol_1,
        mol_2,
        metric=rdkit.DataStructs.TanimotoSimilarity
    )
    return sim
