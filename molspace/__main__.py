import numpy as np
import rdkit.Chem.Crippen, rdkit.RDLogger

import molspace

rdkit.RDLogger.logger().setLevel(rdkit.RDLogger.CRITICAL)

gdb = molspace.dataspace.gdbloader.GDBMoleculesDataset("data", "gdb11", min_size=3, max_size=4)
graph = molspace.dataspace.generate.MolecularSpace(gdb)

# molspace.visualizers.curvature_vis.curvature_histogram(graph)

best_property = -999999999999.
for i in range(len(gdb)):
    molecule = gdb.get_rdkit(gdb[i])
    try:
        molecule.UpdatePropertyCache()
        rdkit.Chem.AddHs(molecule)
        current_property = rdkit.Chem.Crippen.MolLogP(molecule, includeHs=True)
        if current_property > best_property:
            best_property = current_property
            best_molecule = molecule
    except rdkit.Chem.rdchem.AtomValenceException:
        pass
print(best_property, rdkit.Chem.MolToSmiles(best_molecule))

state = np.random.randint(len(graph))
for i in range(100):
    actions = graph.actions(state)
    best_action, best_value = None, -999999999.
    for idx, action in enumerate(actions):
        next_state = graph.data[action]
        next_molecule = graph.data.get_rdkit(next_state)
        try:
            next_molecule.UpdatePropertyCache()
            rdkit.Chem.AddHs(next_molecule)
            current_value = rdkit.Chem.Crippen.MolLogP(next_molecule, includeHs=True)
            if current_value > best_value:
                best_action = idx
                best_value = rdkit.Chem.Crippen.MolLogP(next_molecule)
        except rdkit.Chem.rdchem.AtomValenceException:
            pass
    state = graph.step(state, best_action) if best_action is not None else np.random.randint(len(graph))
    print(best_value, rdkit.Chem.MolToSmiles(graph.data.get_rdkit(graph.data[state])))
