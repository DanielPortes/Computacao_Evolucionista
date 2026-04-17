import numpy as np
from platypus import *
from cec2017 import functions
import matplotlib.pyplot as plt

functions_list = [2, 3, 4, 5, 6, 7, 8]  # functions to be tested
f_index = 0
dims = [2, 10]
d_index = 0
budget_multiplier = 10000  # 10000
n_runs = 51  # 51


def alternate_dimensions():
    global d_index
    d_index = (d_index + 1) % len(dims)
    return dims[d_index]


def test_fun(x):
    x = np.array(x).reshape(1, dims[d_index])
    i = functions_list[f_index]
    for xi in enumerate(x):
        f = functions.all_functions[i](x)
    return f


if __name__ == '__main__':

    for f_index in range(len(functions_list)):
        for d_index in range(len(dims)):
            print("\nFunction: f", (functions_list[f_index] + 1), "D = ", dims[d_index])

            size_problem = int(dims[d_index])
            problem = Problem(size_problem, 1)
            problem.types[:] = Real(-100, 100)
            problem.function = test_fun
            problem.directions[:] = Problem.MINIMIZE
            algorithm = NSGAII
            algorithm.population_size = 50

            solutions_list = []
            fx_values = []
            f = functions.all_functions[functions_list[f_index]]  # encontrando o endereco da funcao

            for i in range(n_runs):
                value = experiment(
                    algorithm,
                    problem,
                    nfe=(dims[d_index] * budget_multiplier),
                    seeds=1)

                solutions_list.append(value.get("NSGAII").get("Problem")[0][i].variables)
                m = np.array(solutions_list[i]).reshape(-1, 2)  # transformando a lista para bi dimensional
                fx_values.append(f(m))  # chamando funcao f com a matriz m

            maior = max(max(sublist) for sublist in fx_values)
            menor = min(min(sublist) for sublist in fx_values)

            mean = np.mean(np.mean(fx_values, axis=0), axis=0)
            median = np.median(np.median(fx_values, axis=0), axis=0)
            std = np.std(np.std(fx_values, axis=0), axis=0)
            print("D =" + str(dims[d_index]))
            print("maior: ", maior)
            print("menor: ", menor)
            print("media: ", mean)
            print("mediana: ", median)
            print("desvio padrao: ", std)

            alternate_dimensions()
        print("\n")
        print("-------------------------------------")
        print("\n")
