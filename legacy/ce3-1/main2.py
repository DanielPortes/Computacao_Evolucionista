import random, time
import matplotlib.pyplot as plt
import operator
import os
import numpy as np
from deap import base, creator, tools, algorithms
import tsplib95




def evaluation(individual):
    return (total_distance(create_tour(individual)),)


def create_tour(individual):
    return [list(cities)[e] for e in individual]


def distance(A, B):
    a = np.asarray(A[1])
    b = np.asarray(B[1])
    x = np.linalg.norm(a - b)
    return np.linalg.norm(a - b)


def distance_matrix(A, B):
    weight = graph.edges[A[0], B[0]]['weight']
    return weight



def total_distance(tour):
    if problem.type == 'TSP':
        return sum(distance(tour[i], tour[i - 1]) for i in range(len(tour)))
    elif problem.type == 'ATSP':
        return sum(distance_matrix(tour[i], tour[i - 1]) for i in range(len(tour)))


def plotline(points, style='bo-', alpha=1, size=7, color=None):
    X, Y = XY(points)

    if color:
        plt.plot(X, Y, style, alpha=alpha, markersize=size, color=color)
    else:
        plt.plot(X, Y, style, alpha=alpha, markersize=size)


def XY(points):
    return [p[1][0] for p in points], [p[1][1] for p in points]


def plot_tour(tour, alpha=1, color=None):
    plotline(list(tour) + [tour[0]], alpha=alpha, color=color)
    plotline([tour[0]], style='gD', alpha=alpha, size=10)
    plt.show()


input_dir = 'input/'
files = os.listdir(input_dir)

for filename in files:
    if not (filename.endswith('.tsp') or not filename.endswith('.atsp')):
        continue

    print('\n\n' + '-' * 80)
    print('File: ', filename)

    problem = tsplib95.load(input_dir + filename)

    graph = problem.get_graph()
    num_cities = problem.dimension
    nodes = list(problem.get_nodes())

    if problem.type == 'TSP':
        coordinates = [problem.node_coords[i] for i in nodes]
    else:
        coordinates = [problem.edge_weights[i] for i in nodes]
    if min(nodes) == 1:
        nodes = [x - 1 for x in nodes]
    cities = list(zip(nodes, coordinates))

    creator.create("minTourLength", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.minTourLength)



    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(0, num_cities), num_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluation)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=100)
    gens = 20000 if num_cities < 50 else 70000
    hof = tools.HallOfFame(1)

    statistics = tools.Statistics(key=operator.attrgetter("fitness.values"))
    statistics.register('mean', np.mean)
    statistics.register('min', np.min)
    statistics.register('max', np.max)

    result, log = algorithms.eaSimple(pop, toolbox,
                                      cxpb=0.7, mutpb=0.05,
                                      ngen=gens, verbose=False,
                                      stats=statistics, halloffame=hof)

    best_individual = tools.selBest(result, k=1)[0]
    best_possible_values = {'berlin52.tsp': 7542, 'ch130.tsp': 6110, 'br17.atsp': 39, 'ftv70.atsp': 1950}
    print('Best tour found: ', evaluation(best_individual)[0])
    print('Best tour existent: ', best_possible_values[filename])
    print('Best tour: ', best_individual)
    print('Average fitness: ', np.mean(log.select('mean')))
    # print('Execution time: ', execution_time, 'seconds')
    print('Relative error: ',
          (evaluation(best_individual)[0] - best_possible_values[filename]) / best_possible_values[filename] * 100, '%')

    plot_tour(create_tour(best_individual))
    plt.figure(figsize=(11, 4))
    plots = plt.plot(log.select('min'), 'c-', log.select('mean'), 'b-', log.select('max'), 'r-')
    plt.legend(plots, ('Minimum fitness', 'Mean fitness', 'Max fitness'), frameon=True)
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    plt.show()
    del creator.Individual, creator.minTourLength, toolbox, pop, hof, result, log, best_individual, best_possible_values, plots, coordinates
