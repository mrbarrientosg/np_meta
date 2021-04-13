from src import problem
from src.problem.knapsack import Knapsack
from src.problem.tsp import TravellingSalesmanProblem
from src.metaheuristic.aco import ACO

#problem = TravellingSalesmanProblem("C:\\Users\\Matias\\Downloads\\a280.tsp")
problem = Knapsack()
problem.read_file(
    "C:\\Users\\Matias\\Downloads\\instances_01_KP\\large_scale\\knapPI_1_100_1000_1"
)

agent = ACO(problem, num_ants=5, num_iterations=50)

best = agent.run()

print(best)