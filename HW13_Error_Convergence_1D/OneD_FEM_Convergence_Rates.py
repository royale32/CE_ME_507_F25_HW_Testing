# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 14:00:38 2025

@author: Owner
"""

import sys
import numpy as np
from matplotlib import pyplot as plt

# Import appropriate functionality to complete this task
# here is some example text that could inform you how to do this
sys.path.append('../HW3_Univariate_Lagrange/')
sys.path.append('../HW8_LagrangeBasisFuncDerivative/')
sys.path.append('../HW6_MultiDimensionalBasisFunctions/')

import Univariate_Lagrange_Basis_Functions as uni
import LagrangeBasisFuncDerivative as derv
import UniDimensionalXMap as xmap
import Gaussian_Quadrature as gq

def DeformationGradient1D(deg,x_bnds,xi):
    val = 0
    x_vals = np.linspace(x_bnds[0], x_bnds[-1], deg+1)
    xi_vals = np.linspace(-1,1,deg+1)
    for a in range(0,deg+1):
        val += x_vals[a] * derv.LagrangeBasisParamDervEvaluation(deg, xi_vals, xi, a)
    return val

# a is local basis function index
# e is element index
# output A which is global basis function index
def IEN(deg,a,e):
    return deg*e + a
    
# A is a basis function index with A >= 0
def ID(A, n_basis_functions):
    if A == n_basis_functions-1:
        return -1
    else:
        return A
    
    
def LocalForceMatrix(deg,x_bnds,f,gaussian):
    fe = np.zeros(deg+1)
    xi_pts = np.linspace(-1, 1, deg+1)
    x_vals = np.linspace(x_bnds[0], x_bnds[-1], deg+1)
    quad_wts = gaussian.quad_wts
    quad_pts = gaussian.quad_pts
    for g in range(0, gaussian.n_quad):
        xi = quad_pts[g]
        w = quad_wts[g]
        x_g = xmap.XMap(deg, x_vals, xi_pts, xi)
        jac = DeformationGradient1D(deg, x_bnds, xi)
        f_g = f(x_g)
        for a in range(0,deg+1):
            Na = uni.LagrangeBasisEvaluation(deg, xi_pts, xi, a)
            fe[a] += w * Na * f_g * jac
    
    return fe

def LocalForceBdryContrib(n_elems,e,h,g,ke):
    fe_bdry = np.zeros(ke.shape[0])
    if e == 0:
        fe_bdry[0] += h
    if e == n_elems-1:
        fe_bdry -= g*ke[:,-1]
    return fe_bdry

def LocalStiffnessMatrix(deg,x_bnds,gaussian):
    ke = np.zeros((deg+1,deg+1))
    xi_pts = np.linspace(-1, 1, deg+1)
    x_vals = np.linspace(x_bnds[0], x_bnds[-1], deg+1)
    quad_wts = gaussian.quad_wts
    quad_pts = gaussian.quad_pts
    for g in range(0, gaussian.n_quad):
        xi = quad_pts[g]
        w = quad_wts[g]
        jac = DeformationGradient1D(deg, x_bnds, xi)
        for a in range(0,deg+1):
            Na_x = derv.LagrangeBasisParamDervEvaluation(deg, xi_pts, xi, a)
            for b in range(0,deg+1):
                Nb_x = derv.LagrangeBasisParamDervEvaluation(deg, xi_pts, xi, b)
                ke[a,b] += w * (Na_x/jac) * (Nb_x/jac) * jac
    return ke
        

# solve the 1D finite element problem
# deg is polynomial degree
# x_nodes are a set of nodes 
def OneDFEM(deg,x_nodes,f,g,h,gaussian):
    
    n_elems = len(x_nodes)-1
    n_basis_funcs = deg * n_elems + 1
    n_unknowns = n_basis_funcs - 1
    
    F = np.zeros(n_unknowns)
    K = np.zeros((n_unknowns,n_unknowns))
    
    for e in range(0,n_elems):
        x_bnds = [x_nodes[e],x_nodes[e+1]]
        ke = LocalStiffnessMatrix(deg, x_bnds, gaussian)
        fe = LocalForceMatrix(deg, x_bnds, f, gaussian)
        fe_bdry = LocalForceBdryContrib(n_elems,e,h,g,ke)
        fe += fe_bdry
        for a in range(0,deg+1):
            A = IEN(deg,a,e)
            P = ID(A,n_basis_funcs)
            
            if P == -1:
                continue
            
            F[P] += fe[a]
            
            for b in range(0,deg+1):
                B = IEN(deg,b,e)
                Q = ID(B,n_basis_funcs)
                
                if Q == -1:
                    continue
                else:
                    K[P,Q] += ke[a,b]
                 
    # Right now we are not using h and g, which means we are
    # assuming that they are 0
    
    
                 
    d = np.linalg.solve(K,F)
    
    d_total = [val for val in d]
    d_total.append(g)
    
    return np.array(d_total)
    
def EvaluateSolution(e,deg,x_pts,xi_basis,xi,dtotal,deriv=0,derv_multiplier=1):
    uhval = 0
    for a in range(0,deg+1):
        A = IEN(deg,a,e)
        dA = dtotal[A]
        if deriv == 0:
            uhval += dA*uni.LagrangeBasisEvaluation(deg, xi_basis, xi, a)
        elif deriv == 1:
            J = DeformationGradient1D(deg,[x_pts[0],x_pts[-1]],xi)
            uhval += derv_multiplier*dA*derv.LagrangeBasisParamDervEvaluation(deg, xi_basis, xi, a) / J
        else:
            sys.exit("Cannot compute on high-order Lagrange derivatives at this time")
    return uhval

def PlotSolutionCurve(deg,x_nodes,dtotal,n_samples = 5,deriv = 0, derv_multiplier = 1):
    uh = []
    x = []
    n_elems = len(x_nodes) - 1
    xi_basis = np.linspace(-1, 1, deg+1)
    xi_samples = np.linspace(-1,1,n_samples)
    for e in range(0,n_elems):
        x_pts = np.linspace(x_nodes[e],x_nodes[e+1],deg+1)
        for xi in xi_samples:
            xval = xmap.XMap(deg, x_pts, xi_basis, xi)
            uhval = EvaluateSolution(e,deg,x_pts,xi_basis,xi,dtotal,deriv,derv_multiplier)
            x.append(xval)
            uh.append(uhval)
    
    plt.plot(x,uh)
    
def EvaluateConvergence(deg,num_intervals,L,f,g,h,gaussian,num_iterations,exact_sol,normtype=0,exact_sol_derv=lambda x:0):
    error_list = []
    h_list = []
    xi_basis = np.linspace(-1, 1, deg+1)
    
    def make_xvals(num_iterations_index):
        num_elems = 2**(num_iterations_index + 1)
        discretzn = np.linspace(0,L,num_elems+1) # global coords
        return discretzn # discretization
    
    if num_iterations < 2:
        sys.exit("Cannot compute convergence rates on fewer than two data points")
    
    for i in range(0,num_iterations): # num_iterations = number of refinements; loop through the mesh elements
        ## create a discretization with num_intervals equal elements
        discretzn = make_xvals(i) # splitting up the domain (same domain, discretize differently)
        num_elems = 2**(i + 1)
        ## solve for the solution vector, d
        dtotal = OneDFEM(deg, discretzn, f, g, h, gaussian) # total solution vector with boundary condition added
            # coefficients for basis function 
        # discretization is the x values we're looking at
        # len(nodes-1) is number of elements
        
        ## compute the error in the solution from a given input solution
        for e in range(0,num_elems+1):
            x_nodes = np.linspace(discretzn[e],discretzn[e+1],deg+1)
                # inside the node, split up for basis funcs
            for g in range(0,gq.n_quad):
                quad_pt = gq.quad_pts[g] # get the gth quadrature point out
                quad_wt = gq.quad_wts[g] # get the gth quadrature weight
                
                # map the values from parametric domain to spatial domain
                x_loc = xmap.XMap(deg, x_nodes, xi_basis, quad_pt)
                # the xi points are the quadrature points; map into spatial domain, evaluate what it will be based on given soln
                
                # J = XMapDeriv(deg,x_nodes,xi_basis,quad_pt)
                J = DeformationGradient1D(deg,[x_nodes[0],x_nodes[-1]],quad_pt)
                
                u_val = exact_sol(x_loc) # evaluate what the solution value is in the current x position
                ux_val = exact_sol_derv(x_loc) # H1
                
                uhval = EvaluateSolution(e, deg, x_nodes, xi_basis, quad_pt, dtotal,0)
                uhxval = EvaluateSolution(e, deg, x_nodes, xi_basis, quad_pt, dtotal,1)
                
                u_sub = u_val - uhval
                ux_sub = ux_val - uhxval
                
                e_2 = quad_wt * (u_sub)**2 * J # equal to the square of the norm of the error
                
                # H1 things if the norm you're asking for is equal to 1
                if normtype == 1:
                    # db same, take deriv for Nb, divide by jacobian (what you do for the derivative of x stuff with basis function)
                    e_2 = quad_wt * (ux_sub)**2 * J
                    
            elem_len = x_nodes[e+1] - x_nodes[e]
            # element length: look at what the error does the more you refine (hopefully goes down)
    
        # log(e0_2) = 2log(e0)
        # 0.5*log(e0_2) = log(e0)
        # if log base 10: norm of error = 10^(0.5*log(e0_2))
        e = 10**(0.5*np.log(e_2,10))
            
        ## store the error and the element length in error_list and h_list, respectively
        # append at each iteration
        error_list.append(e)
        h_list.append(elem_len)
        

    ## take the log of h_list and error_list values
        # take log of the length of the discretization (if 2 elements, log of 1/2 (divide the total length by the number of elements))
    # can do list comprehension to convert
    # h list and error list should be same length
    for i in range(len(h_list)):
        # go to each list, get the log values
        # log_h = np.log(h_list[i])
        log_h_list = np.log(x for x in h_list)
        log_e_list = np.array(np.log(x for x in error_list))
        # multiply by 1/2 for only the error
    
    ## find the slope of the log-h (in x) and log-error (in y) line
    # if curved, is a vector of slopes
    beta = (log_e_list[-1]-log_e_list[-2])/(log_h_list[-1]-log_h_list[-2])
    
    return beta
            
    
# Example software to use this code
def EvaluateAndPlotExample():
    deg = 1
    num_nodes = 10
    L = 1
    f = lambda x:np.exp(x)
    g = 1
    h = 1
    quadrature = gq.GaussQuadrature(deg)
    x_nodes = np.linspace(0,L,num_nodes)
    d = OneDFEM(deg, x_nodes, f, g, h, quadrature)
    print(d)
    
    # not sure about the inputs, change later
    num_intervals = num_nodes - 1
    num_iterations = 2
    exact_sol = lambda x:np.exp(x)
    
    # Normtype 0 (L2)
    beta0 = EvaluateConvergence(deg,num_intervals,L,f,g,h,gq,num_iterations,exact_sol,normtype=0,exact_sol_derv=lambda x:0)
    print(f"Beta for L2 = {beta0}")
    
    # Normtype 1 (H1)
    beta1 = EvaluateConvergence(deg,num_intervals,L,f,g,h,gq,num_iterations,exact_sol,normtype=1,exact_sol_derv=lambda x:0)
    print(f"Beta for L2 = {beta1}")
    
    PlotSolutionCurve(deg,x_nodes,d,n_samples=100,deriv=1)

EvaluateAndPlotExample()