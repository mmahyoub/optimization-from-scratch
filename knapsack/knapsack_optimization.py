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
. Genetic algorithm 

Here, I will create a class for each algorithm. 
'''
import numpy as np 

class dynamic_programming():
    '''Dynamic programming for solving the knapsack problem'''
    def __init__(self, items: list, capacity: int):
        '''
        Optimization problem data:
            - items: a list of value-weight pairs: (value, weight). 
            - capacity: knapsack capacity
        '''
        assert isinstance(items, list), 'items must be a dict object'
        assert isinstance(capacity, int), "capacity must be integer"

        # Initialize dynamic programming data/parameters 
        self.items = items
        self.capacity = capacity
        self.n = len(items)
        self.values = np.array([items[i][0] for i in range(self.n)])
        self.weights =  np.array([items[i][1] for i in range(self.n)])

        # Initialize dynamic programming table 
        self.dp = np.zeros((self.capacity + 1, self.n + 1))
        self.dp[0, :] = 0
        self.dp[:, 0] = 0
    
    def optimize(self):
        # Iterate and build on the dynamic programming table 
        self.iterate()

        # Backtrack to find the selected items
        self.backtrack()

    def iterate(self):
        # Iterate through each remaining capacity and item
        for i in range(1, self.capacity + 1):
            for j in range(1, self.n + 1):
                if self.weights[j - 1] <= i:
                    # 1. Value remains the same as previous capacity
                    option1 = self.dp[i, j - 1]
                    # 2. Add its value to the value obtained by using remaining capacity (i - weight) and previous items (j - 1)
                    option2 = self.dp[i - self.weights[j - 1], j - 1] + self.values[j - 1]
                    # Choose the option with the maximum value
                    self.dp[i, j] = max(option1, option2)
                else:
                    # If the item's weight is too large, exclude it
                    self.dp[i, j] = self.dp[i, j - 1]

    def backtrack(self):
        # Optimal value is the last element of the table
        self.optimal_value = self.dp[self.capacity, len(self.weights)]

        # Backtrack to find the selected items
        self.selected_items = []
        i = self.capacity
        j = self.n
        while i > 0 and j > 0:
            if self.dp[i, j] != self.dp[i, j - 1]:
                # If the value changed, it means the current item was included
                self.selected_items.append(j - 1)
                i -= self.weights[j - 1]
            j -= 1
    
