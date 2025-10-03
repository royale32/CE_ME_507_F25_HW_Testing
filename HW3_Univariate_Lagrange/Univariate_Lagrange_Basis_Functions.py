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
    
    quotient = 1
    
    for i in range(0,p+1):
        if i == a:                          # at i = a, undefined
            continue                        # go back to top of loop
        
        else:
            num = xi - pts[i]
            den = pts[a] - pts[i]
            
            quotient *= num/den
            
    return quotient   # gives us the y-axis value
    # we give it an x-value, tell it what basis function we want
    # at x = 0, tell me what the first basis function value should be
    # 4 points, 4 basis functions
        # first point you give it, the first basis function evaluates to 1 at that point; 2nd point 2nd basis function evaluates to 1
    # feed it an x, tell it which basis function you want
        # at x = something, tell me the y-value of basis function 3
    # does it for one basis function; the next one does it for all of the ones we're interested in
    # make separate basis functions (1-however many you need), plot each one separately

# Plot the Lagrange polynomial basis functions
# COMMENT THIS CODE
def PlotLagrangeBasisFunctions(p,pts,n_samples = 101):
    xis = np.linspace(min(pts),max(pts),n_samples)                  # Looking at the points to be evaluated, it creates n_samples evenly spaced
    
    # We have 100 intervals between the whole interval, 101 points for the total set of intervals
    # in linspace (100 intervals in linspace from start to end of points)
                                                                    # sample points that are evenly spaced from the interval
                                                                    # beginning and ending at the first and last points on the 
                                                                    # coordinate plane
    fig, ax = plt.subplots()                                        # Creates a figure to hold all plot elements and a single set of axes                                   # 
    for a in range(0,p+1):                                          # For a values from 0 to "the polynomial degree" (ath basis function)
                                                                    # second value in range is excluded, so 3rd degree is 0, 1, 2, 3 (for a total of 4)
        vals = []                                                   # The values of the function start as an empty list
        for xi in xis:                                              # For each point to be evaluated in the list of points
            vals.append(LagrangeBasisEvaluation(p, pts, xi, a))     # Add the result of the function LagrangeBasisEvaluation for 
                                                                    # the input values p, pts, xi, and a to the list of values (append to a list)
                
        plt.plot(xis,vals)                                          # Plot the points to be evaluated on the horizontal axis and output values (result of first function) on the vertical axis
    ax.grid(linestyle='--')                                         # Changes the grid's line style
    # The plot shows the different basis functions, each line is a different basis function
    # the vals list has a list of points for the basis function (specifically the y-values for the corresponding x values)       


# Interpolate two-dimensional data
def InterpolateFunction(p,pts2D,n_samples = 101):
    # Insert appropriate text here, as described
    # in the homework prompt
    
    pts = [x for x,y in pts2D]                                  # Convert x-coords of pts2D into list called pts
    coeffs = [y for x,y in pts2D]                               # convert y-coords of pts2D into list called coeffs
    
    xis = np.linspace(min(pts),max(pts),n_samples)              # create evenly-dist list of pts btwn min of pts and max of pts with n_samples points, call it xis
    
    ys = np.zeros(n_samples)                                    # create array of zeroes same dimension of xis, to fill with y-vals of polynomial; call array ys
                                                                # n_samples is the length of xis
    for i in range(0,len(xis)):                                 # goes thru every index in list xis; extract the i-th value of xis and call it xi
        xi = xis[i]
        func_y=0                                                # set an initial value for the y-value of interpolated function to add on later in loop
        for a in range(0,p+1):                                  # evalues every p+1 of the lagrange polynomial basis functions at the point xi (each a value is a basis func, 0 to degree value (for total of p+1 basis functions))
            bf_y = LagrangeBasisEvaluation(p, pts, xi, a)       # gets the value of the basis function (the a-th basis function at the point xi)
            func_y += bf_y * coeffs[a]                          # after get bf_y for that basis func, mult it by the coeff corresponding to that basis function and add to the total function value at that point
        ys[i]=(func_y)                                          # the index of the array of y values of the polynomial at that specific x-value is equal to what you calculated the total function's y-value to be at that point
    
    fig, ax = plt.subplots()    
    plt.plot(xis,ys)
    plt.scatter(pts, coeffs,color='r')

def PlotInterpolateFunction(p,pts2D,n_samples = 101):
    xis, ys = InterpolateFunction(p, pts2D, n_samples)
    fig, ax = plt.subplots()    
    plt.plot(xis,ys)
    plt.scatter([x[0] for x in pts2D], [x[1] for x in pts2D],color='r')

mypts = np.linspace(-1,1,4)
myaltpts = [-1,0,1/2,1]
p = 3
PlotLagrangeBasisFunctions(p,myaltpts) # used to say mypts, i think it wasn't working for me
# you can use mypts if the points are equally spaced (ex: four points equally spaced from -1 to 1 like [-1,-1/3,1/3,1])

my2Dpts = [[0,1],[4,6],[6,2],[7,11]]
InterpolateFunction(p,my2Dpts)
