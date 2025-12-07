# Whale-Optimization-Algorithm
Whale Optimization Algorithm

Overview

This repository contains a Python implementation and visualization of the Whale Optimization Algorithm (WOA), a metaheuristic inspired by the bubble-net hunting strategy of humpback whales. The project demonstrates:

  Mathematical formulation of WOA
  Exploration and exploitation mechanisms
  Shrinking encircling and spiral update rules
  Convergence on benchmark functions (Sphere, Rastrigin, Ackley, Rosenbrock)
  Dynamic 2D and 3D visualizations with HD animation

Files :
  File	              Description
  whale_class.py	    Class definition for whale agents and basic operations.
  woa_fitness.py	    Benchmark fitness functions for Sphere, Rastrigin, Ackley, and Rosenbrock.
  woa_mainloop.py	    Core WOA iteration loop, including position updates and coefficient calculation.
  woa_visual_full.py	Full visualization script: heatmap, 3D surface, and MP4 animation.
  Features:

2D heatmap and 3D surface visualizations of search space
Real-time animation of whale positions and convergence curve
Supports multiple benchmark functions
Adjustable population size, iterations, and dimensionality
Shows exploitation (shrinking encircling), exploration, and spiral movements

Installation:
Clone the repository:
   git clone https://github.com/smaoui898/Whale-Optimization-Algorithm.git
   Whale-Optimization-Algorithm

Install required Python libraries:

   pip install numpy matplotlib
   Optional: Install ffmpeg to save animations as MP4. GIF fallback is available if ffmpeg is not installed.


Python Libraries: numpy, matplotlib

