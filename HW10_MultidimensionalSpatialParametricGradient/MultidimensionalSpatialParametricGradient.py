# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:11:26 2025

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

sys.path.insert(0, "../HW6_MultiDimensionalBasisFunctions/")
from MultiDimensionalBasisFunctions import MultiDimensionalBasisFunction

# degx = 2
# degy = 2
# interp_pts_x = np.linspace(-1,1,degx+1)
# interp_pts_y = np.linspace(-1,1,degy+1)
# MDM_obj = lb2D(degx,degy,interp_pts_x,interp_pts_y)

sys.path.insert(0, "../HW8_LagrangeBasisFuncDerivative/")
import LagrangeBasisFuncDerivative as lbfd

sys.path.insert(0, "../HW9_MultiDimensionalJacobians/")
import MultiDimensionalJacobians as mdj

# degx = 2
# degy = 2
# interp_pts_x = np.linspace(-1,1,degx+1)
# interp_pts_y = np.linspace(-1,1,degy+1)
# MDJ_obj = lb2D_2(degx,degy,interp_pts_x,interp_pts_y)

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

    # derivative of basis function code
    # from previous homework
    def EvalBasisDerivative(self,A,xis,dim):
        # IMPORT/COPY THIS FROM RECENT HOMEWORK
        return lbfd.LagrangeBasisDervParamMultiD(A,self.degs,self.interp_pts,xis,dim)


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

    # Evaluate the parametric gradient of a basis
    # function
    def EvaluateBasisParametricGradient(self,A, xi_vals):
        # COMPLETE THIS TIME
        # looking at one of the basis funcs (one A, where there are 9 possible combinations)
        # ask for the val of the deriv at some xi
            # look at the derivative of the basis function at the xi and the deriv of other basis func at eta
        # make empty vector/array
        p_grad = np.zeros(2)
        xi_derv = self.EvalBasisDerivative(A, xi_vals, dim=0)
        eta_derv = self.EvalBasisDerivative(A, xi_vals, dim=1)
        p_grad[0] += xi_derv
        p_grad[1] += eta_derv
        return p_grad

    # Evaluate the parametric gradient of a basis
    # function
    def EvaluateBasisSpatialGradient(self,A, x_pts, xi_vals):
        # COMPLETE THIS TIME
        p_grad = self.EvaluateBasisParametricGradient(A, xi_vals)
        DF = self.EvaluateDeformationGradient(x_pts, xi_vals)
        inv_DF = np.linalg.inv(DF)
        inv_t_DF = inv_DF.T
        s_grad = np.linalg.solve(inv_t_DF,p_grad)
        return s_grad

    # Grid plotting functionality that is used
    # in all other plotting functions
    def PlotGridData(self,X,Y,Z,npts=21,contours=False,xlabel=r"$x$",ylabel=r"$y$",zlabel=r"$z$", show_plot = True):
        if contours:
            fig, ax = plt.subplots()
            surf = ax.contourf(X,Y,Z,levels=100,cmap=matplotlib.cm.jet)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(surf)
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.jet,
                           linewidth=0, antialiased=False)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
        if show_plot:
            plt.show()
        
        return fig,ax

    # plot the mapping from parent domain to 
    # spatial domain            
    def PlotSpatialMapping(self,x_pts,npts=21,contours=False):
        dim = len(x_pts[0])
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)

        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
        
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[j,i] = pt[0]
                Y[j,i] = pt[1]
                if dim == 3:
                    Z[i,j] = pt[2] 
        
        self.PlotGridData(X,Y,Z,contours=contours,)

    # plot a basis function defined on a parent
    # domain; this is similar to what was
    # in a previous homework, but slightly generalized                
    def PlotBasisFunctionParentDomain(self,A,npts=21,contours=False):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
        
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                Z[j,i] = self.EvalBasisFunction(A, xi_vals)

        self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$N(\xi,\eta)$")

    # plot a basis function defined on a spatial
    # domain
    def PlotBasisFunctionSpatialDomain(self,A,x_pts,npts=21,contours=False,on_parent_domain=True):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)

        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                X[j,i] = pt[0]
                Y[j,i] = pt[1]
                Z[j,i] = self.EvalBasisFunction(A, xi_vals)
        
        self.PlotGridData(X,Y,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$N(\xi,\eta)$")


    # plot a solution field defined on a parent
    # domain
    def PlotParentSolutionField(self,d_coeffs,npts=21,contours = False):
        
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        Z = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                Z[j,i] = self.EvaluateFunctionParentDomain(d_coeffs,[xivals[i],etavals[j]])
    
        self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$u_h^e(\xi,\eta)$")

    # define a solution field mapped into the
    # spatial domain for an element
    def PlotSpatialSolutionField(self,d_coeffs,x_pts,npts=21,contours = False):
        
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
                X[j,i] = pt[0]
                Y[j,i] = pt[1]
                Z[j,i] = self.EvaluateFunctionParentDomain(d_coeffs,[xivals[i],etavals[j]])
    
        self.PlotGridData(X,Y,Z,contours=contours,zlabel=r"$u_h^e(x,y)$")


    # plot Jacobians defined on the spatial 
    # or parent domain
    def PlotJacobian(self,x_pts,npts=21,contours = False, parent_domain = False):
        
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
    
        if parent_domain:
            self.PlotGridData(Xi,Eta,Z,contours=contours,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$J^e(\xi,\eta)$")
        else:
            self.PlotGridData(X,Y,Z,contours=contours,zlabel=r"$J^e(x,y)$")

    def PlotBasisFunctionGradient(self,A,x_pts,npts=21, parent_domain = True, parent_gradient = True):
        xivals = np.linspace(self.interp_pts[0][0],self.interp_pts[0][-1],npts+1)
        etavals = np.linspace(self.interp_pts[1][0],self.interp_pts[1][-1],npts)
        
        Xi,Eta = np.meshgrid(xivals,etavals)
        X = np.zeros(Xi.shape)
        Y = np.zeros(Xi.shape)
        Z = np.zeros(Xi.shape)
        U = np.zeros(Xi.shape)
        V = np.zeros(Xi.shape)
    
        for i in range(0,len(xivals)):
            for j in range(0,len(etavals)):
                xi_vals = [xivals[i],etavals[j]]
                if not parent_domain:
                    pt = self.EvaluateSpatialMapping(x_pts, xi_vals)
                    X[j,i] = pt[0]
                    Y[j,i] = pt[1]
                if parent_gradient:
                    grad = self.EvaluateBasisParametricGradient(A, xi_vals)
                else:
                    grad = self.EvaluateBasisSpatialGradient(A, x_pts, xi_vals)
                U[j,i] = grad[0]
                V[j,i] = grad[1]
                Z[j,i] = self.EvalBasisFunction(A, xi_vals)

        if parent_domain:
            fig,ax = self.PlotGridData(Xi,Eta,Z,contours=True,xlabel=r"$\xi$",ylabel=r"$\eta$",zlabel=r"$J^e(\xi,\eta)$",show_plot = False)
            ax.quiver(Xi,Eta,U,V)
        else:
            fig,ax = self.PlotGridData(X,Y,Z,contours=True,zlabel=r"$J^e(x,y)$",show_plot = False)
            ax.quiver(X,Y,U,V)
        plt.show()

# Parametric = parent (xi,eta)
# Spatial: element (x, y)

def Plot_para_para():
    # Plot the parametric gradient on the parametric domain
    # Polynomial degree 2x2
    degx = 2
    degy = 2
    
    # Parametric/parent domain
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
    
    obj = LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    
    # Spatial control points
    x_pts = np.array([[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]])
    
    # Plot all of them
    for A in range((obj.degs[0]+1)*(obj.degs[1]+1)):
        obj.PlotBasisFunctionGradient(A,x_pts,npts=21, parent_domain = True, parent_gradient = True)
    
    # Plot the first one
    # A = 0
    # obj.PlotBasisFunctionGradient(A,x_pts,npts=21, parent_domain = True, parent_gradient = True)
    
    return

def Plot_para_spat():
    # Plot the parametric gradient on the spatial domain
    # Polynomial degree 2x2
    degx = 2
    degy = 2
    
    # Parametric/parent domain
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
    
    obj = LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    
    # Spatial control points
    x_pts = np.array([[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]])
    
    # Plot all of them
    for A in range((obj.degs[0]+1)*(obj.degs[1]+1)):
        obj.PlotBasisFunctionGradient(A,x_pts,npts=21, parent_domain = False, parent_gradient = True)
    
    # Plot the first one
    # A = 0
    # obj.PlotBasisFunctionGradient(A,x_pts,npts=21, parent_domain = False, parent_gradient = True)
    
    return
    
def Plot_spat_para():
    # Plot the spatial gradient on the parametric domain
    # Polynomial degree 2x2
    degx = 2
    degy = 2
    
    # Parametric/parent domain
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
    
    obj = LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    
    # Spatial control points
    x_pts = np.array([[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]])
    
    # Plot all of them
    for A in range((obj.degs[0]+1)*(obj.degs[1]+1)):
        obj.PlotBasisFunctionGradient(A,x_pts,npts=21, parent_domain = True, parent_gradient = False)
    
    # Plot the first one
    # A = 0
    # obj.PlotBasisFunctionGradient(A,x_pts,npts=21, parent_domain = True, parent_gradient = False)
    
    return

def Plot_spat_spat():
    # Plot the spatial gradient on the spatial domain
    # Polynomial degree 2x2
    degx = 2
    degy = 2
    
    # Parametric/parent domain
    interp_pts_x = np.linspace(-1,1,degx+1)
    interp_pts_y = np.linspace(-1,1,degy+1)
    
    obj = LagrangeBasis2D(degx, degy, interp_pts_x, interp_pts_y)
    
    # Spatial control points
    x_pts = np.array([[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]])
    
    # Plot all of them
    for A in range((obj.degs[0]+1)*(obj.degs[1]+1)):
        obj.PlotBasisFunctionGradient(A,x_pts,npts=21, parent_domain = False, parent_gradient = False)
    
    # Plot the first one
    # A = 0
    # obj.PlotBasisFunctionGradient(A,x_pts,npts=21, parent_domain = False, parent_gradient = False)
    
    return

# Plot_para_para()
# Plot_para_spat()
# Plot_spat_para()

# Plot_spat_spat()
