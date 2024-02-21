''''
Solve a 0-1 knapsack problem.
We are going to use the classes defined in knapsack_optimization.py.

. Dynamic programming 
. Genetic algorithm (To be added)
'''
import os 
import numpy as np 
from knapsack_optimization import dynamic_programming, genetic_algorithm
import logging

def main():
    set_logging() # Initiate logging 
    logging.info('Defining 0-1 knapsack problem...')
    items = [(60, 20), (100, 20), (120, 30), (100, 20)]
    item_names = ['A', 'B', 'C', 'D']
    capacity = 70
    logging.info(f'Items (value, weight): {items} | capacity = {capacity} | Item labels = {item_names}')

    logging.info('Optimize using dynamic programming...')
    try:
        dp_solver = dynamic_programming(items, capacity)
        dp_solver.optimize()

        logging.info(f'Optimal value: {dp_solver.optimal_value}')
        logging.info(f'Selected items: {sorted([item_names[i] for i in dp_solver.selected_items])}')
    except Exception:
        logging.error('Dynamic programming failed :', exc_info = True)
    

    logging.info('Optimizing using genetic algorithm...')
    try:
        ga_solver = genetic_algorithm(items,
                                   capacity,
                                   population_size=20,
                                   n_generations = 100,
                                   crossover_rate = 0.85,
                                   mutation_rate = 0.2)
        ga_solver.optimize()

        logging.info(f'Optimal value: {ga_solver.best_fitness}')
        logging.info(f'Selected items: {[name for name, selected in zip(item_names, ga_solver.best_chromosome) if selected == 1 ]}')
    except Exception:
        logging.error('Genetic algorithm failed :', exc_info = True)

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