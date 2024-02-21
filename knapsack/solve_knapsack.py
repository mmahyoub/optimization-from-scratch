''''
Solve a 0-1 knapsack problem.
We are going to use the classes defined in knapsack_optimization.py.

. Dynamic programming 
. Genetic algorithm (To be added)
'''
import os 
from knapsack_optimization import dynamic_programming
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
        solver = dynamic_programming(items, capacity)
        solver.optimize()

        logging.info(f'Optimal value: {solver.optimal_value}')
        logging.info(f'Selected items: {sorted([item_names[i] for i in solver.selected_items])}')
    except Exception:
        logging.error('Dynamic programming failed :', exc_info = True)

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