# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 13:49:50 2025

@author: Owner
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

# You will need the functionality from your 
# previous homework assignment that evaluates
# a two-dimensional basis function at a point
# Either copy that here or import the 
# functionality using
sys.path.insert(0, "../HW6_MultiDimensionalBasisFunctions/")
import MultiDimensionalBasisFunctions as mdbf

# This is a class that describes a Lagrange basis
# in two dimensions
class LagrangeBasis2D:
    
    # initializor
    def __init__(self,degx,degy,interp_pts_x,interp_pts_y):
        self.degs = [degx,degy]
        self.interp_pts = [interp_pts_x,interp_pts_y]
        self.n_bf = (self.degs[0]+1)*(self.degs[1]+1) # so I can refer to n_bf elsewhere in the code
        
    # the number of basis functions is the 
    # product of basis functions in the x (xi)
    # and y (eta) directions
    def NBasisFuncs(self):
        n_bf = (self.degs[0]+1)*(self.degs[1]+1)
        # (number of Lagrange basis functions in xi direction)*(number of basis functions in eta direction)
        # (p+1)*(q+1)
        return n_bf         # total number of 2D basis functions for basis
        
    # basis function evaluation code from 
    # previous homework assignment
    # this should be imported from that assignment
    # or copied before this class is defined
    def EvalBasisFunction(self,A,xi_vals):
        return mdbf.MultiDimensionalBasisFunction(A, self.degs, self.interp_pts, xi_vals)       

    # Evaluate a sum of basis functions times 
    # coefficients on the parent domain
    def EvaluateFunctionParentDomain(self, d_coeffs, xi_vals):
        # self.NBasisFuncs()
        uh = 0
        for A in range(self.n_bf):
            uh += d_coeffs[A]*self.EvalBasisFunction(A,xi_vals)
        return uh
        
    # Evaluate the spatial mapping from xi and eta
    # into x and y coordinates
    def EvaluateSpatialMapping(self, x_pts, xi_vals):
        x_y = []
        x = 0
        y = 0
        for A in range(self.n_bf):
            x += x_pts[A][0]*self.EvalBasisFunction(A,xi_vals)
            y += x_pts[A][1]*self.EvalBasisFunction(A,xi_vals)
        x_y.append(x)
        x_y.append(y)
        return x_y      # vector
    
    # Grid plotting functionality that is used
    # in all other plotting functions
    def PlotGridData(self,X,Y,Z,npts=21,contours=False,xlabel=r"$x$",ylabel=r"$y$",zlabel=r"$z$"):
        if contours:
            fig, ax = plt.subplots()
            surf = ax.contourf(X,Y,Z,levels=100,cmap=matplotlib.cm.jet)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(surf)
            plt.show()
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.jet,
                           linewidth=0, antialiased=False)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            plt.show()

            
    # plot the mapping from parent domain to 
    # spatial domain
    def PlotSpatialMapping(self,x_pts,npts=21,contours=False):
        dim = len(x_pts[0])
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)

        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
        
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[i,j] = pt[0]
                Y[i,j] = pt[1]
                if dim == 3:
                    Z[i,j] = pt[2] 
        
        self.PlotGridData(X,Y,Z,contours=contours,)

    # plot a basis function defined on a parent
    # domain; this is similar to what was
    # in a previous homework, but slightly generalized
    def PlotBasisFunctionParentDomain(self,A,npts=21,contours=False):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
        
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                Z[i,j] = self.EvalBasisFunction(A, xi_vals)

        self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$N(\xi,\eta)$")

    # plot a basis function defined on a spatial
    # domain
    def PlotBasisFunctionSpatialDomain(self,A,x_pts,npts=21,contours=False,on_parent_domain=True):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)

        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[i,j] = pt[0]
                Y[i,j] = pt[1]
                Z[i,j] = self.EvalBasisFunction(A, xi_vals)
        
        self.PlotGridData(X,Y,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$N(\xi,\eta)$")

    # plot a solution field defined on a parent
    # domain
    def PlotParentSolutionField(self,d_coeffs,npts=21,contours = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                Z[i,j] = self.EvaluateFunctionParentDomain(d_coeffs,[xivals[i],etavals[j]])
    
        self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$u_h^e(\xi,\eta)$")

    # define a solution field mapped into the
    # spatial domain for an element
    def PlotSpatialSolutionField(self,d_coeffs,x_pts,npts=21,contours = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[i,j] = pt[0]
                Y[i,j] = pt[1]
                Z[i,j] = self.EvaluateFunctionParentDomain(d_coeffs,[xivals[i],etavals[j]])
    
        self.PlotGridData(X,Y,Z,contours=contours,zlabel=r"$u_h^e(x,y)$")
        

    
def Problem5Plotting():
    # class: when I make an instance of something, I'll put certain things in
    degx = 2
    degy = 2
    x_pts = np.array([[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]])
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
        
    d_coeffs = [-1,2,3,5,6,7,2,1,3]
        
    Prob5Object = LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    # Prob5Object.EvaluateFunctionParentDomain(d_coeffs, xi_vals)
    
        # Spatial to Parametric Mapping:
    Prob5Object.PlotSpatialMapping(x_pts)
    
        # Solution field on the parent domain
    Prob5Object.PlotParentSolutionField(d_coeffs)
    
        # 9 basis functions in both the parent and spatial domains
    for i in range (9):             # loops through the A values; gives you a specific basis function for that A value
        Prob5Object.PlotBasisFunctionParentDomain(i)
        Prob5Object.PlotBasisFunctionSpatialDomain(i, x_pts)
    
        # Solution field on the spatial domain of the element's domain:
    Prob5Object.PlotSpatialSolutionField(d_coeffs,x_pts)
    
Problem5Plotting()