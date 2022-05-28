import numpy as np
import rdkit.Chem.Crippen, rdkit.RDLogger, rdkit.Chem.Draw

import molspace

rdkit.RDLogger.logger().setLevel(rdkit.RDLogger.CRITICAL)

gdb = molspace.dataspace.gdbloader.GDBMoleculesDataset("data", "gdb11", min_size=3, max_size=6)
graph = molspace.environment.tanimoto.MolecularSpace(gdb)
molspace.visualizers.curvature_vis.curvature_histogram(graph)

best_property = -999999999999.
best_molecule = None
for i in range(len(gdb)):
    molecule = gdb.get_rdkit(gdb[i])
    current_property = molspace.predictors.drug_likeness.predict_log_p(molecule)
    if current_property > best_property:
        best_property = current_property
        best_molecule = molecule
print(best_property, rdkit.Chem.MolToSmiles(best_molecule))


molecules_encountered = []
for i in range(10):
    state = np.random.randint(len(graph))
    for j in range(10):
        actions = graph.actions(state)
        action_values = []
        for action in actions:
            next_state = graph.data[action]
            next_molecule = graph.data.get_rdkit(next_state)
            action_values.append(molspace.predictors.drug_likeness.predict_log_p(next_molecule))

        if len(action_values) > 0:
            probabilities = np.array(action_values)
            probabilities = np.exp(probabilities / 10)
            probabilities = probabilities / np.sum(probabilities)
            best_action = np.random.choice(np.arange(len(probabilities)))
            state = graph.step(state, best_action)
        else:
            state = np.random.randint(len(graph))

        molecule = graph.data.get_rdkit(graph.data[state])
        molecules_encountered.append(molecule)
        print(molspace.predictors.drug_likeness.predict_log_p(molecule), rdkit.Chem.MolToSmiles(molecule))

rdkit.Chem.Draw.MolsToGridImage(molecules_encountered, 10).show()
