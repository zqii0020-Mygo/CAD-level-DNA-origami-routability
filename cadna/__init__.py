"""CAD-DNA: CAD-level DNA origami design generator + routability dataset toolkit."""

from .lattice import HONEYCOMB, SQUARE, Lattice, get_lattice, RISE_PER_BP
from .params import CADParams, sample_params
from .model import Feature, Bundle, Cylinder, Adjacency, CandidateCrossover, Mate, Design
from .generator import generate
from .io import save_design, load_design, design_to_dict, design_from_dict

__all__ = [
    "HONEYCOMB", "SQUARE", "Lattice", "get_lattice", "RISE_PER_BP",
    "CADParams", "sample_params",
    "Feature", "Bundle", "Cylinder", "Adjacency", "CandidateCrossover", "Mate", "Design",
    "generate", "save_design", "load_design", "design_to_dict", "design_from_dict",
]
