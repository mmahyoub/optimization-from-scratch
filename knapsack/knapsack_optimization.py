'''
Author: Mohammed Mahyoub 

Explore several methods to solve the 0-1 knapsack problem. 

What is the 0-1 knapsack problem?
A problem in which we need to choose a set of items that will maximize the 
    overall value while respecting the limit (.e.g, maximum capacity or weight).
    An item is selected (1) or not selected 0.
    
    x_i: a binary variable (decision variable)
    v_i: item value 
    w_i: item weight 

    maximize sum(v_i * x_i for all items)
    s.t. 
        sum(w_i * x_i for all items) <= capacity_limit 
        x_i is 0 or 1 

The 0-1 knapsack problem can be solved in several ways with varying solution quality.
. Dynamic programming  
. Branch and bound
. Genetic algorithm 

Here, I will create a class for each algorithm. 
'''

class dp():
    '''Dynamic programming for solving the knapsack problem'''
    def __init__(self, items:dict, capacity):
        '''
        Optimization problem data:
            - items: dictionary for story item:(value, weight). Item can be id or name of item.
            - capacity: knapsack capacity
        '''
        assert isinstance(items, dict), 'items must be a dict object'

        self.items = items
        self.capacity = capacity
        self.n = len(items.keys())

