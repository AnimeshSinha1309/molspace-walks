"""
This module is responsible for generation of the molecular space graph
that we are constructing our random walks on.

It will provide whatever data interface we have to get the molecules,
their neighbors on the graph, the graph based representation of said molecules
in PyTorch and RDKit, and more.
"""

import molspace.dataspace.gdbloader
import environment.tanimoto
import molspace.dataspace.featurizers
