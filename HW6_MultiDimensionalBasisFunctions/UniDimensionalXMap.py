# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 10:02:39 2025

@author: Owner
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

# IMPORT (or copy) your code from HW3 here
# which evaluated a Lagrane polynomial basis function
sys.path.append('../HW3_Univariate_Lagrange/')
import Univariate_Lagrange_Basis_Functions as ulbf

# x(\xi) = \sum_{a=0}^p x_a * N_a(\xi)
def XMap(deg,spatial_pts,interp_pts,xi):
    # complete this function
    xe = 0
    for a in range(0,deg+1):                              # goes to p (polynomial degree)
        xe += spatial_pts[a]*ulbf.LagrangeBasisEvaluation(deg, interp_pts, xi, a)

    return xe

def PlotXMap(deg,spatial_pts,interp_pts, npts=101, contours=True):
    # parametric points to evaluate in Lagrange basis function
    xi_vals = np.linspace(interp_pts[0],interp_pts[-1],npts)
    
    # evaluate and plot as a line
    if not contours:
        xs = []
        for i in range(0,len(xi_vals)):
            xi = xi_vals[i]
            xs.append(XMap(deg,spatial_pts,interp_pts,xi))
        plt.plot(xi_vals,xs)
        plt.show()

    # evaluate as a contour plot
    else:
        Xi,Xi2 = np.meshgrid(xi_vals,[-0.2,0.2])
        Z = np.zeros(Xi.shape)
        for i in range(0,len(xi_vals)):
            xi = xi_vals[i]
            x = XMap(deg,spatial_pts,interp_pts,xi)
            for j in range(0,2):
                Z[j,i] = x

        fig, ax = plt.subplots()
        surf = ax.contourf(Z,Xi2,Xi,levels=100,cmap=matplotlib.cm.binary)
        ax.set_xlabel(r"$x$")
        ax.yaxis.set_ticklabels([])
        fig.colorbar(surf)
        plt.show()
        
def Input1():
    deg = 1
    spatial_pts = [0,1]
    interp_pts = [-1,1]
    PlotXMap(deg,spatial_pts,interp_pts)

def Input2():
    deg = 2
    spatial_pts = [0.5,1,1.5]
    interp_pts = [-1,0,1]
    PlotXMap(deg,spatial_pts,interp_pts)
    
def Input3():
    deg = 2
    spatial_pts = [0.5,0.7,1.5]
    interp_pts = [-1,-0.6,1]
    PlotXMap(deg,spatial_pts,interp_pts)
    
def Input4():
    deg = 2
    spatial_pts = [0.5,0.7,1.5]
    interp_pts = [-1,0,1]
    PlotXMap(deg,spatial_pts,interp_pts)
    
Input1()
Input2()
Input3()
Input4()