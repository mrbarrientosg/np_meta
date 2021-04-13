from typing import List, Union
from src.problem import Problem
import tsplib95


class TravellingSalesmanProblem(Problem):
    def __init__(self, path: str):
        self.Q: float = 1.0
        self.problem = tsplib95.load(path)

    def delta_tau(self, fitness: float, i: int, j: int):
        return self.Q / fitness

    def heuristic(self, i: int, j: int) -> Union[float, int]:
        nodes = list(self.problem.get_nodes())
        weight = self.problem.get_weight(nodes[i], nodes[j])
        return 1.0 / (weight or 1.0)

    def init_possibles_moves(self, start_move: int) -> List[int]:
        moves = []
        for i in range(self.size):
            if i != start_move:
                moves.append(i)
        return moves

    def update_possibles_moves(self, solution: List[int],
                               actual_moves: List[int]) -> List[int]:
        if not actual_moves:
            if solution[0] != solution[len(solution) - 1]:
                actual_moves.append(solution[0])

        return actual_moves

    def fitness(self, solution: List[int]) -> Union[float, int]:
        nodes = list(self.problem.get_nodes())
        tour = [[nodes[i] for i in solution]]
        return self.problem.trace_tours(tour)[0]

    @property
    def size(self) -> int:
        return self.problem.dimension