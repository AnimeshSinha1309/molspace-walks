import molspace

gdb = molspace.dataspace.gdbloader.GDBMoleculesDataset("data", "gdb11", min_size=3, max_size=4)
graph = molspace.dataspace.generate.MolecularSpace(gdb)

molspace.visualizers.curvature_vis.curvature_histogram(graph)
