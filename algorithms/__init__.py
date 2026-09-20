"""Semi-external graph algorithms: vertex state stays in RAM, edges stream out of the store.

NetworkX walks one vertex at a time, which costs a round trip per step against an external store.
Every algorithm here drives the store instead — scattering over an edge stream, or gathering a page
of vertices at a time — and holds a fixed number of bytes per vertex rather than a dict of floats.

Each one is a class declaring what it costs, reached through `on`, which picks the variant the store
would rather serve. The plain function beside it runs that variant to completion with its defaults.
"""

from __future__ import annotations

from algorithms.base import REGISTRY, Algorithm, Bounds, EdgeOriented, Sweeping, VertexOriented
from algorithms.betweenness import BetweennessCentrality, betweenness_centrality
from algorithms.breadth_first import BreadthFirstLayers, breadth_first_layers
from algorithms.clustering import ClusteringCoefficients, clustering_coefficients
from algorithms.connected_components import (
    ConnectedComponents,
    WeaklyConnectedComponents,
    connected_components,
    weakly_connected_components,
)
from algorithms.core_numbers import CoreNumbers, core_numbers
from algorithms.degree_histogram import DegreeHistogram, degree_histogram
from algorithms.delta_stepping import DeltaSteppingLengths, delta_stepping_lengths
from algorithms.dijkstra import DijkstraLengths, dijkstra_lengths
from algorithms.hits import HITS, GatheredHITS, ScatteredHITS, hits
from algorithms.k_truss import KTruss, k_truss
from algorithms.label_propagation import (
    GatheredLabelPropagation,
    LabelPropagation,
    ScatteredLabelPropagation,
    label_propagation,
)
from algorithms.neighbors_of_neighbors import NeighborsOfNeighbors, Reach, neighbors_of_neighbors
from algorithms.pagerank import GatheredPageRank, PageRank, ScatteredPageRank, pagerank
from algorithms.personalized_pagerank import (
    GatheredPersonalizedPageRank,
    PersonalizedPageRank,
    ScatteredPersonalizedPageRank,
    personalized_pagerank,
)
from algorithms.results import DenseIndex, VertexMap
from algorithms.sampling import SampleEdges, SampleNodes, sample_edges, sample_nodes
from algorithms.shortest_path_lengths import ShortestPathLengths, shortest_path_lengths
from algorithms.streams import adjacency, reservoir, scan_arcs, weighted_adjacency
from algorithms.strongly_connected_components import StronglyConnectedComponents, strongly_connected_components
from algorithms.topological_order import TopologicalOrder, topological_order
from algorithms.triangles import TriangleCounts, triangle_counts
from networkxternal.base_api import Orientation

BOUNDS = {name: kind.BOUNDS for name, kind in REGISTRY.items()}
"""What each algorithm costs, derived from the classes so the two can never disagree."""

__all__ = [
    "BOUNDS",
    "HITS",
    "REGISTRY",
    "Algorithm",
    "BetweennessCentrality",
    "Bounds",
    "BreadthFirstLayers",
    "ClusteringCoefficients",
    "ConnectedComponents",
    "CoreNumbers",
    "DegreeHistogram",
    "DeltaSteppingLengths",
    "DenseIndex",
    "DijkstraLengths",
    "EdgeOriented",
    "GatheredHITS",
    "GatheredLabelPropagation",
    "GatheredPageRank",
    "GatheredPersonalizedPageRank",
    "KTruss",
    "LabelPropagation",
    "NeighborsOfNeighbors",
    "Orientation",
    "PageRank",
    "PersonalizedPageRank",
    "Reach",
    "SampleEdges",
    "SampleNodes",
    "ScatteredHITS",
    "ScatteredLabelPropagation",
    "ScatteredPageRank",
    "ScatteredPersonalizedPageRank",
    "ShortestPathLengths",
    "StronglyConnectedComponents",
    "Sweeping",
    "TopologicalOrder",
    "TriangleCounts",
    "VertexMap",
    "VertexOriented",
    "WeaklyConnectedComponents",
    "adjacency",
    "betweenness_centrality",
    "breadth_first_layers",
    "clustering_coefficients",
    "connected_components",
    "core_numbers",
    "degree_histogram",
    "delta_stepping_lengths",
    "dijkstra_lengths",
    "hits",
    "k_truss",
    "label_propagation",
    "neighbors_of_neighbors",
    "pagerank",
    "personalized_pagerank",
    "reservoir",
    "sample_edges",
    "sample_nodes",
    "scan_arcs",
    "shortest_path_lengths",
    "strongly_connected_components",
    "topological_order",
    "triangle_counts",
    "weakly_connected_components",
    "weighted_adjacency",
]
