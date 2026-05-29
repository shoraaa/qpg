from dataclasses import dataclass
from networkx import Graph
from numpy import ndarray
from enum import Enum


class Solver(Enum):
    DWAVE = 'dwave'
    MQLIB = 'mqlib'
    GUROBI = 'gurobi'
    EXACT = 'exact'
    LOCAL = 'local'
    GREEDY_RESIDUAL = 'greedy_residual'
    RANDOM_RESIDUAL = 'random_residual_walk'
    BEAM = 'beam_search'
    ACO = 'aco'
    NEURAL_ACO = 'neural_aco'
    ASTAR = 'astar'
    SEEA = 'seea'

COVERAGE_SUFFIX = "coverage"


@dataclass
class QuboDescription:
    filename: str
    data_dir: str
    graph: Graph
    time_limits: list[int]
    jobs: int
    Q: ndarray
    offset: int
    T: int
    V: int
    solver: Solver
