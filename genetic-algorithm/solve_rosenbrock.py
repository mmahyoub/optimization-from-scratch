'''
Solve the Rosenbrock function minimization problem. It can be run on any number of dimensions. 

We will utilize the GA class in genetic_algorithm.py module.

Here we illustrate on a 2-D example. 
'''
import os 
import logging
from genetic_algorithm import GA
from matplotlib import pyplot as plt
import numpy as np 

def main():
    # Initiate logging 
    set_logging()

    # 2D problem 
    logging.info('Solving the Rosenbrock function in 2D: f(x,y)...')
    try:
        # Edit hyperparameters accordingly if you change the number of dimensions
        ga_solver = GA(dimensions=2,
                       lower_bound = -2.0,
                       upper_bound = 2.0,
                       population_size = 500,
                       n_generations = 200,
                       crossover_rate = 0.7,
                       mutation_rate = 0.2,
                       )
        
        ga_solver.optimize()

        logging.info(f'Best value: {round(ga_solver.best_fitness, 4)} | Actual optimal value is 0')
        logging.info(f'Best Solution: {[round(g, 4) for g in ga_solver.best_chromosome]} | Actual optimal solution is (1,1)')
        plot_rosenbrock(lower_bound=-2.0, upper_bound=2.0, best_solution = ga_solver.best_chromosome)

    except Exception:
        logging.error('Failed to run GA() on Rosenbrock function 2D: ', exc_info = True)

def rosenbrock(x, y):
    '''2D Rosenbrock function'''
    return 100 * (y - x**2)**2 + (1 - x)**2

def plot_rosenbrock(lower_bound:float, upper_bound:float, best_solution:list):
    '''Plot Rosenbrock function with global and GA minima'''
    x_min, x_max = lower_bound, upper_bound
    y_min, y_max = lower_bound, upper_bound

    X, Y = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = rosenbrock(X, Y)
    plt.contour(X, Y, Z, levels=100)
    
    # Global minimum 
    plt.scatter(1, 1, s=50, c='red', marker='x', label = 'global minimum')

    # Best solution
    plt.scatter(best_solution[0], best_solution[1], s=50, c='blue', marker='o', label = 'GA minimum')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rosenbrock Function (2D)')
    plt.legend()
    plt.savefig('ga_rosenbrock.jpg', dpi = 600)
    plt.show()

def set_logging():
    if 'logs.log' in os.listdir(os.getcwd()):
        os.remove('logs.log')

    logging.basicConfig(level=logging.INFO)
    file_handler =  logging.FileHandler(filename = './logs.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

if __name__ == "__main__":
    main()