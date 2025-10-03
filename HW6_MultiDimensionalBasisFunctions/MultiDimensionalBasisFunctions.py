# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 10:18:05 2025

@author: Owner
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:45:50 2024

@author: kendrickshepherd
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

# IMPORT (or copy) your code from HW3 here
# which evaluated a Lagrane polynomial basis function
from CE_ME_507_Lagrange_Basis_Function_Code import LagrangeBasisEvaluation

# idxs = a (a is a list of indexes)
# degs = p (polynomial degrees in each direction)
# interp_pts = set of interpolation points in each direction
# xis = coordinates you want to evaluate

# higher-dimensional basis function with multi-index
# this function takes lists; these correspond to the arguments taken by the
# LagrangeBasisEvaluation function from HW 3
# LBE takes a basis function index (a)
        # point (xi)
        # set of interpolation points (pts)
# MDBFI takes lists of each of the above
# The expression we apply the product operator is N_{a_{i}}(xi_{i}); this is a single evaluation of the Lagrange basis
# loop should take each a list item at each iteration
def MultiDimensionalBasisFunctionIdxs(idxs,degs,interp_pts,xis):
    Ni = 1              # initial value of basis function before loop
    
    for i in range(len(idxs)):
        p = degs[i]
        pts = interp_pts[i]
        xi = xis[i]
        a = idxs[i]
       
        Ni *= LagrangeBasisEvaluation(p, pts, xi, a)
    return Ni
    
# higher-dimensional basis function with single index
def MultiDimensionalBasisFunction(A,degs,interp_pts,xis):
    a_list = []
    denom = degs[0]+1
    ai = A % (degs[0]+1)
    a_list.append(ai)
    for i in range(1,len(degs)):
        a_list.append(A//denom)
        deg = degs[i]             # since p0 starts at i = 1
        denom *= (deg+1)
        """
        if i == 0:
            ai = A % (degs[0]+1)
            a_list.append(ai)
        else:
            ai = A//denom
            # print(A,ai)
            a_list.append(ai)
        """
        # the code commented out didn't work; i changed it
    result = MultiDimensionalBasisFunctionIdxs(a_list,degs,interp_pts,xis)
    return result
    
# plot of 2D basis functions with A a single index
def PlotTwoDimensionalParentBasisFunction(A,degs,npts = 101,contours = False):
    interp_pts = [np.linspace(-1,1,degs[i]+1) for i in range(0,len(degs))]
    xivals = np.linspace(-1,1,npts)
    etavals = np.linspace(-1,1,npts)
    
    Xi,Eta = np.meshgrid(xivals,etavals)
    Z = np.zeros(Xi.shape)

    for i in range(0,len(xivals)):
        for j in range(0,len(etavals)):
            Z[i,j] = MultiDimensionalBasisFunction(A, degs, interp_pts, [xivals[i],etavals[j]])
    
    # contour plot
    if contours:
        fig, ax = plt.subplots()
        surf = ax.contourf(Eta,Xi,Z,levels=100,cmap=matplotlib.cm.jet)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(surf)
        plt.show()
    # 3D surface plot
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(Eta, Xi, Z, cmap=matplotlib.cm.jet,
                       linewidth=0, antialiased=False)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        ax.set_zlabel(r"$N(\xi,\eta)$")
        plt.show()

def Input1():
    # value for range = (p_0 + 1)*(p_1 + 1) = (1+1)*(1+1)=4
    degs = [1,1]
    for A in range(4):          # range 4 = 0,1,2,3 (total of 4)
        PlotTwoDimensionalParentBasisFunction(A,degs)

def Input2():
    # value for range = (p_0 + 1)*(p_1 + 1) = (2+1)*(1+1)=6
    degs = [2,1]
    for A in range(6):
        PlotTwoDimensionalParentBasisFunction(A,degs)

def Input3():
    # value for range = (p_0 + 1)*(p_1 + 1) = (2+1)*(2+1)=9
    degs = [2,2]
    for A in range(9):
        PlotTwoDimensionalParentBasisFunction(A,degs)

def Input4():
    # value for range = (p_0 + 1)*(p_1 + 1) = (3+1)*(3+1)=16
    degs = [3,3]
    for A in range(16):
        PlotTwoDimensionalParentBasisFunction(A,degs)
    
# Inputs for Problem 1:
# Input1()
# Input2()
# Input3()
# Input4()

# change contour to false to see 3D