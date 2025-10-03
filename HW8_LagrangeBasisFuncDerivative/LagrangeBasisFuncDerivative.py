# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 15:33:58 2025

@author: Owner
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

# Depending on your approach, 
# You may need to import information from HW3 and/or
# HW6 to load in your unidimensional and multidimensional
# Lagrange basis function code 
sys.path.append('../HW3_Univariate_Lagrange/')
import Univariate_Lagrange_Basis_Functions as ulbf
# sys.path.insert(0, "../HW6_MultiDimensionalBasisFunctions/")
# import MultiDimensionalBasisFunctions as mdbf

# input polynomial degree, p,
#       interpolation points, pts,
#       the point of evaluation, xi,
#   and the basis fnction index, a
def LagrangeBasisParamDervEvaluation(p,pts,xi,a):
    # print(p, pts, xi, a)
    added = 0
    for j in range(p+1):
        if j == a:
            continue
        else:
            # print(j)
            # print(pts[j])
            # p_pts = pts[0:j]+pts[j+1:]
            # p_pts = [pts[j] for p in pts if p != j]
            new_pts = []
            for p in range(len(pts)):
                if p != j:
                    new_pts.append(pts[p])
            # print(new_pts)
            # print(len(p_pts))
            new_a = a if j > a else a-1 
            # mult = 1/(p_pts[a]-p_pts[j])
            lbf = ulbf.LagrangeBasisEvaluation(p-1, new_pts, xi, new_a)
            added += (1/(pts[a]-pts[j]))*lbf
    return added

# you may want to create a function here that 
# converts from a single index, A, and a set of 
# polynomial degrees into a multi-index indicating
# the indices of the univariate lagrange polynomials
# 
# you will need this functionality, though you do
# not need to complete this function for full credit
def GlobalToLocalIdxs(A,degs):
    a_list = []
    denom = degs[0]+1
    ai = A % (degs[0]+1)
    a_list.append(ai)
    for i in range(1,len(degs)):
        a_list.append(A//denom)
        deg = degs[i]             # since p0 starts at i = 1
        denom *= (deg+1)
    return a_list

# evaluate the partial derivative of a nD lagrange 
# basis function of index A in the "dim" dimension
# (e.g. derivative in xi is 0, in eta is 1)
def LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim):
    val = 1
    # length of degrees is the number of dimensions
    # loop over 0 and 1 (length of degrees is 2)
    # dim is the index of the direction (either 0 or 1)
    # take derivative when i matches dim; when doesn't match, compute the regular
    
    a_list = GlobalToLocalIdxs(A, degs)
    for i in range(len(degs)):
        p = degs[i]
        pts = interp_pts[i]
        xi = xis[i]
        a = a_list[i]
        
        if i == dim: 
            val *= LagrangeBasisParamDervEvaluation(p,pts,xi,a)
            # print(f"deriv = {val}")
        else:
            #  non_deriv = MultiDimensionalBasisFunction(A, degs, interp_pts, xis) wrong because does both eta and xi
            val *= ulbf.LagrangeBasisEvaluation(p,pts,xi,a)
            # print(f"non_deriv = {val}")
        # print(f"final = {val}")
    return val

# Plot the Lagrange polynomial basis function
# derivatives
def PlotLagrangeBasisDerivatives(p,pts,n_samples = 101):
    xis = np.linspace(min(pts),max(pts),n_samples)
    fig, ax = plt.subplots()
    for a in range(0,p+1):
        vals = []
        for xi in xis:
            vals.append(LagrangeBasisParamDervEvaluation(p, pts, xi, a))
                
        plt.plot(xis,vals)
    ax.grid(linestyle='--')

# plot a basis function defined on a parent
# domain; this is similar to what was
# in a previous homework, but slightly generalized                
def PlotTwoDBasisFunctionParentDomain(A,degs,interp_pts,dim,npts=21,contours=False):
    xivals = np.linspace(interp_pts[0][0],interp_pts[0][-1],npts+1)
    etavals = np.linspace(interp_pts[1][0],interp_pts[1][-1],npts)
    
    Xi,Eta = np.meshgrid(xivals,etavals)
    Z = np.zeros(Xi.shape)
    
    for i in range(0,len(xivals)):
        for j in range(0,len(etavals)):
            xi_vals = [xivals[i],etavals[j]]
            if dim < 0:
                # if you imported your multidimensional
                # lagrange basis function code as
                # module "m_basis", uncomment the line below
                #Z[j,i] = m_basis.MultiDimensionalBasisFunction(A,degs,interp_pts,xi_vals)
                continue
            else:
                Z[j,i] = LagrangeBasisDervParamMultiD(A,degs,interp_pts,xi_vals,dim) # xi_vals is an (xi,eta) coordinate on grid

    if contours:
        fig, ax = plt.subplots()
        surf = ax.contourf(Xi,Eta,Z,levels=100,cmap=matplotlib.cm.jet)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(surf)
        plt.show()
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(Xi, Eta, Z, cmap=matplotlib.cm.jet,
                       linewidth=0, antialiased=False)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        ax.set_zlabel(r"$\frac{\partial N_A}{\partial xi_%d}$" % dim)
        plt.show()

def Problem3():
    p = 2
    pts = np.linspace(-1,1,p+1) # interpolation points
                                # 101 is the sampling points
    PlotLagrangeBasisDerivatives(p,pts)
    
def Problem4xi():
    degs = [2,2]
    deg_xi = degs[0]
    deg_eta = degs[1]
    interp_pts_xi = np.linspace(-1,1,deg_xi+1)
    interp_pts_eta = np.linspace(-1,1,deg_eta+1)
    interp_pts = np.vstack([interp_pts_xi,interp_pts_eta])
    # for deg in range(len(degs)):
    dim = 0
    for A in range((degs[0]+1)*(degs[1]+1)):
        PlotTwoDBasisFunctionParentDomain(A, degs, interp_pts, dim)
        
def Problem4eta():
    degs = [2,2]
    deg_xi = degs[0]
    deg_eta = degs[1]
    interp_pts_xi = np.linspace(-1,1,deg_xi+1)
    interp_pts_eta = np.linspace(-1,1,deg_eta+1)
    interp_pts = np.vstack([interp_pts_xi,interp_pts_eta])
    # for deg in range(len(degs)):
    dim = 1
    for A in range((degs[0]+1)*(degs[1]+1)):
        PlotTwoDBasisFunctionParentDomain(A, degs, interp_pts, dim)
        
Problem3()
Problem4xi()
Problem4eta()

# To compare values/check
def Test():
    A = 4
    degs = [2,2]
    interp_pts = [np.linspace(-1,1,3),np.linspace(-1,1,3)]
    xis = [.33,-.25]
    dim = 1
    
    LagrangeBasisDervParamMultiD(A,degs,interp_pts,xis,dim)
    
Test()