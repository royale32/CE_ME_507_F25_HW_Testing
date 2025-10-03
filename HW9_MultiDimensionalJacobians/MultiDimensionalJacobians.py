# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 13:21:51 2025

@author: Owner
"""

import sys
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt

# Copy or import functionality that you created 
# in previous homework assignments to complete
# this homework and minimize the amount of 
# work you have to repeat
sys.path.insert(0, "../HW6_MultiDimensionalBasisFunctions")
from MultiDimensionalBasisFunctions import MultiDimensionalBasisFunction

# degx = 2
# degy = 2
# interp_pts_x = np.linspace(-1,1,degx+1)
# interp_pts_y = np.linspace(-1,1,degy+1)
# MDM_obj = mdm.LagrangeBasis2D(degx,degy,interp_pts_x,interp_pts_y)

sys.path.insert(0, "../HW8_LagrangeBasisFuncDerivative/")
import LagrangeBasisFuncDerivative as lbfd


# this class was created earlier in a previous
# assignment, but has been extended to cope with
# derivatives of basis functions and to plot
# Jacobians

# This is a class that describes a Lagrange basis
# in two dimensions
class LagrangeBasis2D:
    
    # initializor
    def __init__(self,degx,degy,interp_pts_x,interp_pts_y):
        self.degs = [degx,degy]
        self.interp_pts = [interp_pts_x,interp_pts_y]
        
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
        return MultiDimensionalBasisFunction(A, self.degs, self.interp_pts, xi_vals)       

    # Evaluate a sum of basis functions times 
    # coefficients on the parent domain
    def EvaluateFunctionParentDomain(self, d_coeffs, xi_vals):
        # self.NBasisFuncs()
        uh = 0
        for A in range(self.NBasisFuncs()):
            uh += d_coeffs[A]*self.EvalBasisFunction(A,xi_vals)
        return uh
        
    # Evaluate the spatial mapping from xi and eta
    # into x and y coordinates
    def EvaluateSpatialMapping(self, x_pts, xi_vals):
        x_y = []
        x = 0
        y = 0
        for A in range(self.NBasisFuncs()):
            x += x_pts[A][0]*self.EvalBasisFunction(A,xi_vals)
            y += x_pts[A][1]*self.EvalBasisFunction(A,xi_vals)
        x_y.append(x)
        x_y.append(y)
        return x_y      # vector
    
    # Evaluate the Deformation Gradient (i.e.
    # the Jacobian matrix)
    def EvaluateDeformationGradient(self, x_pts, xi_vals):
        # COMPLETE THIS TIME
        DF = [[],[]]
        x_xi = 0
        y_xi = 0
        x_eta = 0
        y_eta = 0
        
        for A in range((self.degs[0]+1)*(self.degs[1]+1)):
            dim = 0
            x_xi += x_pts[A][0]*lbfd.LagrangeBasisDervParamMultiD(A,self.degs,self.interp_pts,xi_vals,dim)
            y_xi += x_pts[A][1]*lbfd.LagrangeBasisDervParamMultiD(A,self.degs,self.interp_pts,xi_vals,dim)

            dim = 1
            x_eta += x_pts[A][0]*lbfd.LagrangeBasisDervParamMultiD(A,self.degs,self.interp_pts,xi_vals,dim)
            y_eta += x_pts[A][1]*lbfd.LagrangeBasisDervParamMultiD(A,self.degs,self.interp_pts,xi_vals,dim)
            
        DF[0].append(x_xi)
        DF[1].append(y_xi)
        DF[0].append(x_eta)
        DF[1].append(y_eta)

        return DF
    
    # Evaluate the jacobian (or the determinant
    # of the deformation gradient)
    def EvaluateJacobian(self, x_pts, xi_vals):
        # COMPLETE THIS TIME
        DF = self.EvaluateDeformationGradient(x_pts, xi_vals)
        J = DF[0][0]*DF[1][1] - DF[0][1]*DF[1][0]
        return J
    
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


    # plot Jacobians defined on the spatial 
    # or parent domain
    def PlotJacobian(self,x_pts,npts=21,contours = False, parent_domain = False):
        # See contour plot by setting contours = True
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                if not parent_domain:
                    pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                    X[j,i] = pt[0]
                    Y[j,i] = pt[1]
                Z[j,i] = self.EvaluateJacobian(x_pts,xi_vals)
                # print(Z[j,i])
    
        if parent_domain:
            self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$J^e(\xi,\eta)$")
        else:
            self.PlotGridData(X,Y,Z,contours=contours,zlabel=r"$J^e(x,y)$")

# SCP = spatial control points
def Problem4_1():
    degx = 2
    degy = 2
    
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
        
    obj = LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    
    # Keep only one input uncommented
    
        # Input 1:
    x_pts = np.array([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]])

        # Run the uncommented input
    obj.PlotJacobian(x_pts)

def Problem4_2():
    degx = 2
    degy = 2
    
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
        
    obj = LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    
    # Keep only one input uncommented
    
        # Input 2:
    x_pts = np.array([[0,0],[0.5,0],[1,0],[0,1],[0.5,1],[1,1],[0,2],[0.5,2],[1,2]])

        # Run the uncommented input
    obj.PlotJacobian(x_pts)
    
def Problem4_3():
    degx = 2
    degy = 2
    
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
        
    obj = LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    
    # Keep only one input uncommented
    
        # Input 3:
    x_pts = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])

        # Run the uncommented input
    obj.PlotJacobian(x_pts)
    
def Problem4_4():
    degx = 2
    degy = 2
    
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
        
    obj = LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    
    # Keep only one input uncommented
    
        # Input 4:
    x_pts = np.array([[0,2],[0,1],[0,0],[1,2],[1,1],[1,0],[2,2],[2,1],[2,0]])
    
        # Run the uncommented input
    obj.PlotJacobian(x_pts)

def Problem4_5():
    degx = 2
    degy = 2
    
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
        
    obj = LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    
    # Keep only one input uncommented
    
        # Input 5:
    x_pts = np.array([[0,0],[1,0],[2,0],[1,1],[1,1],[1,1],[2,2],[1,2],[0,2]])
    
        # Run the uncommented input
    obj.PlotJacobian(x_pts)

def Problem4_6():
    degx = 2
    degy = 2
    
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
        
    obj = LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    
    # Keep only one input uncommented
    
        # Input 6:
    x_pts = np.array([[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]])
    
        # Run the uncommented input
    obj.PlotJacobian(x_pts)
    
Problem4_1()
Problem4_2()
Problem4_3()
Problem4_4()
Problem4_5()
Problem4_6()