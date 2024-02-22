'''
Author: Mohammed Mahyoub 

Genetic algorithm for solving the Rosenbrock function minimization problem in any number of dimensions.

'''

import numpy as np 
import random 

class GA():
    '''Genetic algorithm for solving the Rosenbrock function'''
    def __init__(self,
                dimensions: int,
                lower_bound: float,
                upper_bound: float,
                population_size: int,
                n_generations: int,
                crossover_rate:float,
                mutation_rate:float):
        
        # Assert 
        assert isinstance(dimensions, int)
        assert isinstance(population_size, int)
        assert isinstance(n_generations, int)
        assert dimensions >= 2, "dimensions must be 2 or more"
        assert crossover_rate >= 0 and crossover_rate <= 1, "crossover_rate must be in [0,1]"
        assert mutation_rate >= 0 and mutation_rate <= 1, "muation_rate must be in [0,1]"

        # Initialize optimization problem data/parameters 
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Genetic algorithm paramters
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def optimize(self):
        '''Run genetic algorithm to solve the Rosenbrock function'''
        self.generate_population()

        self.best_chromosome = None 
        self.best_fitness = float('inf')  # Goal is to minimize the Rosenbrock function 

        for _ in range(self.n_generations):
            self.fitness_scores = [self.fitness(chromosome) for chromosome in self.population]
            best_index = np.argmin(self.fitness_scores)

            if self.fitness_scores[best_index] < self.best_fitness:
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
        self.population = np.random.uniform(self.lower_bound,
                                             self.upper_bound,
                                             size = (self.population_size, self.dimensions))
    
    def fitness(self, chromosome):
        '''Calculate the fitness score of a single chromosome'''
        return sum((100 * (x_next - x**2)**2) + (x - 1)**2 for x, x_next in zip(chromosome[0:self.dimensions-1], chromosome[1:]))

    def selection(self):
        '''Use roulete wheel selection to select individuals for crossover, fittest wins'''
        total_fitness = sum(self.fitness_scores)
        probabilities = self.fitness_scores / total_fitness 
        selected_indices = np.random.choice(len(self.population), size = 2, p = probabilities)
        return self.population[selected_indices[0]], self.population[selected_indices[1]]

    def crossover(self, parent1, parent2):
        ''' Crossover on two parent chromosomes -- uniform '''
        
        if random.random() < self.crossover_rate:
            child1 = []
            child2 = []
            for i in range(len(parent1)):
                if random.random() < 0.5:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
                else:
                    child1.append(parent2[i])
                    child2.append(parent1[i])    
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        return child1, child2

    def mutation(self, chromosome):
        '''Uniform mutation'''
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                mutation_step = random.uniform(self.lower_bound, self.upper_bound)
                chromosome[i] += mutation_step
                chromosome[i] = np.clip(chromosome[i], self.lower_bound, self.upper_bound)
        return chromosome    