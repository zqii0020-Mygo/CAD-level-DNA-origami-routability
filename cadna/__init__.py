"""CAD-DNA: CAD-level DNA origami design generator + routability dataset toolkit."""

from .lattice import HONEYCOMB, SQUARE, Lattice, get_lattice, RISE_PER_BP
from .params import CADParams, sample_params
from .model import Feature, Bundle, Cylinder, Adjacency, CandidateCrossover, Mate, Design
from .generator import generate
from .io import save_design, load_design, design_to_dict, design_from_dict
from .linkgraph import LinkGraph, Link, build_link_graph, port_of, opposite
from .precheck import FAILURE_CLASSES, PrecheckResult, precheck
from .routing import DesignLabel, RoutingResult, evaluate, find_scaffold_route, find_scaffold_path
from .graph import NODE_TYPES, RELATIONS, HeteroGraph, build_graph, targets_from_label

__all__ = [
    "HONEYCOMB", "SQUARE", "Lattice", "get_lattice", "RISE_PER_BP",
    "CADParams", "sample_params",
    "Feature", "Bundle", "Cylinder", "Adjacency", "CandidateCrossover", "Mate", "Design",
    "generate", "save_design", "load_design", "design_to_dict", "design_from_dict",
    "LinkGraph", "Link", "build_link_graph", "port_of", "opposite",
    "FAILURE_CLASSES", "PrecheckResult", "precheck",
    "DesignLabel", "RoutingResult", "evaluate", "find_scaffold_route", "find_scaffold_path",
    "NODE_TYPES", "RELATIONS", "HeteroGraph", "build_graph", "targets_from_label",
]
