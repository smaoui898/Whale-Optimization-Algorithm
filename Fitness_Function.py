import math
 
 # Rastrigin function
def fitness_rastrigin(position):
     """Rastrigin function: f(x) = 10n + Σ(x_i² - 10cos(2πx_i))
     Global minimum at f(0,...,0) = 0
     Bounds: [-5.12, 5.12] (typical)
     """
     fitness_value = 0.0
     n = len(position)
     fitness_value = 10 * n
     for i in range(n):
         xi = position[i]
         fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi))
     return fitness_value

 # Sphere function
def fitness_sphere(position):
     """Sphere function: f(x) = Σ(x_i²)
     Global minimum at f(0,...,0) = 0
     Bounds: [-5.12, 5.12] (typical)
     """
     fitness_value = 0.0
     for i in range(len(position)):
         xi = position[i]
         fitness_value += (xi * xi)
     return fitness_value

 # Example usage with WOA
def run_optimizer(fitness_func=fitness_sphere, dim=3, bounds=[-10, 10]):
     num_whales = 50
     max_iter = 100
     return woa(fitness_func, max_iter, num_whales, dim, bounds[0], bounds[1])