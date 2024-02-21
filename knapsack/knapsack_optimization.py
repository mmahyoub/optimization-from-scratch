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
import random

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

###################################################
            
class genetic_algorithm():
    '''Genetic algorithm for solving the knapsack problem'''
    def __init__(self,
                items:list,
                capacity: int,
                population_size: int,
                n_generations: int,
                crossover_rate:float,
                mutation_rate:float):
        
        # Assert 
        assert isinstance(items, list)
        assert isinstance(capacity, int)
        assert isinstance(population_size, int)
        assert isinstance(n_generations, int)
        assert crossover_rate >= 0 and crossover_rate <= 1, "crossover_rate must be in [0,1]"
        assert mutation_rate >= 0 and mutation_rate <= 1, "muation_rate must be in [0,1]"

        # Initialize optimization problem data/parameters 
        self.items = items
        self.capacity = capacity
        self.n = len(items)
        self.values = np.array([items[i][0] for i in range(self.n)])
        self.weights =  np.array([items[i][1] for i in range(self.n)])

        # Genetic algorithm paramters
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def optimize(self):
        '''Run genetic algorithm to solve the 0-1 knapsack problem'''
        self.generate_population()

        self.best_chromosome = None 
        self.best_fitness = 0

        for _ in range(self.n_generations):
            self.fitness_scores = [self.fitness(chromosome) for chromosome in self.population]
            best_index = np.argmax(self.fitness_scores)

            if self.fitness_scores[best_index] > self.best_fitness:
                self.best_chromosome = self.population[best_index]
                self.best_fitness = self.fitness_scores[best_index]
            
            new_population = []
            for _ in range(int(self.population_size/2)):
                parent1, parent2 = self.selection()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_population.append(child1)
                new_population.append(child2)

            self.population = new_population

    def generate_population(self):
        '''Generate initial population randomly: a population is a set of chromosomes'''
        self.population = []
        for _ in range(self.population_size):
            chromosome = np.random.choice([0,1], size = self.n)  # binary chromosome
            self.population.append(chromosome)
    
    def fitness(self, chromosome):
        '''Calculate the fitness score of a single chromosome'''
        total_weight= 0
        total_value = 0

        for i, gene in enumerate(chromosome):
            if gene == 1:
                total_weight += self.weights[i]
                total_value += self.values[i]
        if total_weight > self.capacity:
            return 0
        else:
            return total_value

    def selection(self):
        '''Use roulete wheel selection to select individuals for crossover, fittest wins'''

        total_fitness = sum(self.fitness_scores)
        probabilities = self.fitness_scores / total_fitness 
        selected_indices = np.random.choice(len(self.population), size = 2, p = probabilities)
        return self.population[selected_indices[0]], self.population[selected_indices[1]]

    def crossover(self, parent1, parent2):
        ''' Crossover on two parent chromosomes'''
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        
        return child1, child2

    def mutation(self, chromosome):
        '''Bit flipping with a given probability'''
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]  # if 1 becomes 0 and if 0 becomes 1
        return chromosome    