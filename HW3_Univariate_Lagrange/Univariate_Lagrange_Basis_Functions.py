#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 06:48:53 2024

@author: kendrickshepherd
"""

import sys
import numpy as np
from matplotlib import pyplot as plt

# Given a set of points, pts, to interpolate
# and a polynomial degree, this function will
# evaluate the a-th basis function at the 
# location xi
def LagrangeBasisEvaluation(p,pts,xi,a):
    # ensure valid input
    if (p+1 != len(pts)):
        sys.exit("The number of input points for interpolating must be the same as one plus the polynomial degree for interpolating")

    # complete this code and return an appropriate value
    # as given in Hughes Equation 3.6.1

# Plot the Lagrange polynomial basis functions
# COMMENT THIS CODE
def PlotLagrangeBasisFunctions(p,pts,n_samples = 101):
    xis = np.linspace(min(pts),max(pts),n_samples)
    fig, ax = plt.subplots()
    for a in range(0,p+1):
        vals = []
        for xi in xis:
            vals.append(LagrangeBasisEvaluation(p, pts, xi, a))
                
        plt.plot(xis,vals)
    ax.grid(linestyle='--')
        
# Interpolate two-dimensional data
def InterpolateFunction(p,pts2D,n_samples = 101):
    # Insert appropriate text here, as described
    # in the homework prompt
    
    return
    

def PlotInterpolateFunction(p,pts2D,n_samples = 101):
    xis, ys = InterpolateFunction(p, pts2D, n_samples)
    fig, ax = plt.subplots()    
    plt.plot(xis,ys)
    plt.scatter([x[0] for x in pts2D], [x[1] for x in pts2D],color='r')
    
mypts = np.linspace(-1,1,4)
myaltpts = [-1,0,.5,1]
p = 3
PlotLagrangeBasisFunctions(p,mypts)

my2Dpts = [[0,1],[4,6],[6,2],[7,11]]
PlotInterpolateFunction(p,my2Dpts)