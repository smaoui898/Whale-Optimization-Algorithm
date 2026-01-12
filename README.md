🐋 Whale Optimization Algorithm (WOA)
Overview : 

This repository contains a Python implementation and visualization of the Whale Optimization Algorithm (WOA), a metaheuristic inspired by the bubble-net hunting strategy of humpback whales.

The project demonstrates:

Mathematical formulation of WOA
Exploration and exploitation mechanisms
Shrinking encircling and spiral update rules
Convergence on benchmark functions:

Sphere
Rastrigin
Ackley
Rosenbrock
Dynamic 2D and 3D visualizations with HD animation

Repository Structure : 
File	                  Description
whale_class.py	        Class definition for whale agents and basic operations
woa_fitness.py	        Benchmark fitness functions (Sphere, Rastrigin, Ackley, Rosenbrock)
woa_mainloop.py	        Core WOA iteration loop, including position updates and coefficient calculation
woa_visual_full.py	    Full visualization script: heatmap, 3D surface, and MP4 animation

Features :  

2D heatmap visualization of the search space
3D surface visualization of benchmark functions
Real-time animation of whale positions
Convergence curve visualization
Supports multiple benchmark functions

Adjustable:

Population size
Number of iterations
Problem dimensionality
Visual demonstration of:
Exploitation (shrinking encircling)
Exploration
Spiral bubble-net movement

Installation : 
1. Clone the repository
git clone https://github.com/smaoui898/Whale-Optimization-Algorithm.git
cd Whale-Optimization-Algorithm

2. Install required Python libraries
pip install numpy matplotlib

3. (Optional) Enable MP4 animation export

Install ffmpeg to save animations as MP4.
If ffmpeg is not installed, a GIF fallback will be used.

Dependencies : 

Python 3.x
numpy
matplotlib
(Optional) ffmpeg for MP4 export
