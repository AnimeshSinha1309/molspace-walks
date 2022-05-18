import time

import torch
import GraphRicciCurvature.OllivierRicci

import plotly.express as px

from molspace.environment.tanimoto import MolecularSpace


def curvature_histogram(graph: MolecularSpace) -> None:
    curvature_finder = GraphRicciCurvature.OllivierRicci.OllivierRicci(graph.graph)
    curvature_computation_start_time = time.time()
    curvature_finder.compute_ricci_curvature()
    curvature_computation_end_time = time.time()
    print(
        f"Curvature Computation complete: type={curvature_finder.__class__.__name__}, "
        f"duration={curvature_computation_end_time - curvature_computation_start_time}"
    )

    curvature = torch.Tensor(
        [
            curvature_finder.G[i][j]['ricciCurvature'] if i != j else 0
            for i, j in graph.graph.edges
        ]
    )
    fig = px.histogram(curvature)
    fig.show()
    print(curvature)
