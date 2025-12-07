import math
import random


def woa(fitness, max_iter, n, dim, minx, maxx):
     # Initialize whale population and find initial best solution
     whales = [initialize_whale(dim, minx, maxx, i) for i in range(n)] # type: ignore
     Xbest, Fbest = find_best_solution(whales) # type: ignore
     
     # Main loop of WOA algorithm
     Iter = 0
     while Iter < max_iter:
         # a decreases linearly from 2 to 0
         a = 2 * (1 - Iter / max_iter)
         
         for i in range(n):
             # Update coefficient vectors
             A = 2 * a * random() - a   # Controls step size/direction
             C = 2 * random()            # Weights the influence of reference
             b = 1                       # Defines spiral shape
             l = random() * 2 - 1      # Random in [-1,1] for spiral
             p = random()                # Mechanism selection probability
             
             # Mechanism selection based on probability
             if p < 0.5:  # Encircling mechanism
                 if abs(A) < 1:  # Exploitation - Move toward best
                     D = [abs(C * Xbest[j] - whales[i].position[j]) for j in range(dim)]
                     whales[i].position = [Xbest[j] - A * D[j] for j in range(dim)]
                 else:  # Exploration - Move away from random whale
                     random_whale_idx = random.randint(0, n-1)
                     Xrand = whales[random_whale_idx].position
                     D = [abs(C * Xrand[j] - whales[i].position[j]) for j in range(dim)]
                     whales[i].position = [Xrand[j] - A * D[j] for j in range(dim)]
             else:  # Spiral update mechanism
                 D1 = [abs(Xbest[j] - whales[i].position[j]) for j in range(dim)]
                 whales[i].position = [D1[j] * math.exp(b*l) * math.cos(2*math.pi*l) + Xbest[j] for j in range(dim)]
             
             # Apply bounds and update fitness
             whales[i].position = [max(min(whales[i].position[j], maxx), minx) for j in range(dim)]
             whales[i].fitness = fitness(whales[i].position)
             
             # Update best solution if improved
             if whales[i].fitness < Fbest:
                 Xbest = whales[i].position.copy()
                 Fbest = whales[i].fitness
                 
         Iter += 1
     
     return Xbest, Fbest  # Return the best solution found