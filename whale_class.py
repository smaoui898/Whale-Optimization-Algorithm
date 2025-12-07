import random
import math
 
 # Whale class to represent each search agent in population
class Whale:
     def __init__(self, fitness_function, dimensions, min_bound, max_bound, seed=None):
         # Initialize random number generator with seed (for reproducibility)
         self.rnd = random.Random(seed)
         
         # Initialize position vector in search space
         self.position = [0.0 for i in range(dimensions)]
         
         # Randomly position whale within the bounds
         for i in range(dimensions):
             self.position[i] = ((max_bound - min_bound) * self.rnd.random() + min_bound)
         
         # Calculate initial fitness value
         self.fitness = fitness_function(self.position)
         
         # Additional attributes can be added (e.g., velocity, best position)
         self.best_position = self.position.copy()
         self.best_fitness = self.fitness
 
     def update_position(self, new_position, fitness_function):
         # Update whale position and recalculate fitness
         self.position = new_position
         self.fitness = fitness_function(self.position)
         
         # Update personal best if current position is better
         if self.fitness < self.best_fitness:
             self.best_position = self.position.copy()
             self.best_fitness = self.fitness
             
         return self.fitness