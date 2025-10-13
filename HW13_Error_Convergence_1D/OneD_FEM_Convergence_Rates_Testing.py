#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:37:32 2025

@author: kendrickshepherd
"""

import numpy as np
import Gaussian_Quadrature as gq
# import OneD_FEM_Convergence_Rates_Solution as fem
import OneD_FEM_Convergence_Rates as fem

import unittest

class TestOneDConvergenceRates(unittest.TestCase):

    def test_OneDConvergence_Linear_L2(self):
        deg=1
        normtype = 0
        quadplus = 0
        num_nodes = 11
        num_elems = num_nodes-1
        L = 1
        f = lambda x:np.exp(x)
        g = 1
        h = 1
        quadrature = gq.GaussQuadrature(deg+quadplus)
        num_iterations = 4
        decimal_place = 3

        exact_sol = lambda x: (g+np.exp(L)-L) + (L-x)*h + x - np.exp(x)
        exact_derv = lambda x: -h + 1 - np.exp(x)

        slope = fem.EvaluateConvergence(deg,num_elems,L,f,g,h,quadrature,num_iterations,exact_sol,normtype=normtype,exact_sol_derv=exact_derv)
        self.assertAlmostEqual(deg+1, slope, decimal_place)

    def test_OneDConvergence_Linear_H1(self):
        deg=1
        normtype = 1
        quadplus = 1
        num_nodes = 11
        num_elems = num_nodes-1
        L = 1
        f = lambda x:np.exp(x)
        g = 1
        h = 1
        quadrature = gq.GaussQuadrature(deg+quadplus)
        num_iterations = 4
        decimal_place = 3

        exact_sol = lambda x: (g+np.exp(L)-L) + (L-x)*h + x - np.exp(x)
        exact_derv = lambda x: -h + 1 - np.exp(x)

        slope = fem.EvaluateConvergence(deg,num_elems,L,f,g,h,quadrature,num_iterations,exact_sol,normtype=normtype,exact_sol_derv=exact_derv)
        self.assertAlmostEqual(deg, slope, decimal_place)

    def test_OneDConvergence_Quadratic_L2(self):
        deg=2
        normtype = 0
        quadplus = 1
        num_nodes = 11
        num_elems = num_nodes-1
        L = 1
        f = lambda x:np.exp(x)
        g = 1
        h = 1
        quadrature = gq.GaussQuadrature(deg+quadplus)
        num_iterations = 3
        decimal_place = 3

        exact_sol = lambda x: (g+np.exp(L)-L) + (L-x)*h + x - np.exp(x)
        exact_derv = lambda x: -h + 1 - np.exp(x)

        slope = fem.EvaluateConvergence(deg,num_elems,L,f,g,h,quadrature,num_iterations,exact_sol,normtype=normtype,exact_sol_derv=exact_derv)
        self.assertAlmostEqual(deg+1, slope, decimal_place)

    def test_OneDConvergence_Quadratic_H1(self):
        deg=2
        normtype = 1
        quadplus = 1
        num_nodes = 11
        num_elems = num_nodes-1
        L = 1
        f = lambda x:np.exp(x)
        g = 1
        h = 1
        quadrature = gq.GaussQuadrature(deg+quadplus)
        num_iterations = 3
        decimal_place = 3

        exact_sol = lambda x: (g+np.exp(L)-L) + (L-x)*h + x - np.exp(x)
        exact_derv = lambda x: -h + 1 - np.exp(x)

        slope = fem.EvaluateConvergence(deg,num_elems,L,f,g,h,quadrature,num_iterations,exact_sol,normtype=normtype,exact_sol_derv=exact_derv)
        self.assertAlmostEqual(deg, slope, decimal_place)


    def test_OneDConvergence_Quintic_L2(self):
        deg=5
        normtype = 0
        quadplus = deg+1
        num_nodes = 3
        num_elems = num_nodes-1
        L = 1
        f = lambda x:np.exp(x)
        g = 1
        h = 1
        quadrature = gq.GaussQuadrature(deg+quadplus)
        num_iterations = 3
        decimal_place = 1

        exact_sol = lambda x: (g+np.exp(L)-L) + (L-x)*h + x - np.exp(x)
        exact_derv = lambda x: -h + 1 - np.exp(x)

        slope = fem.EvaluateConvergence(deg,num_elems,L,f,g,h,quadrature,num_iterations,exact_sol,normtype=normtype,exact_sol_derv=exact_derv)
        self.assertAlmostEqual(deg+1, slope, decimal_place)


    def test_OneDConvergence_Quintic_H1(self):
        deg=5
        normtype = 1
        quadplus = 5
        num_nodes = 3
        num_elems = num_nodes-1
        L = 1
        f = lambda x:np.exp(x)
        g = 1
        h = 1
        quadrature = gq.GaussQuadrature(deg+quadplus)
        num_iterations = 3
        decimal_place = 1

        exact_sol = lambda x: (g+np.exp(L)-L) + (L-x)*h + x - np.exp(x)
        exact_derv = lambda x: -h + 1 - np.exp(x)

        slope = fem.EvaluateConvergence(deg,num_elems,L,f,g,h,quadrature,num_iterations,exact_sol,normtype=normtype,exact_sol_derv=exact_derv)
        self.assertAlmostEqual(deg, slope, decimal_place)




if __name__ == '__main__':
    unittest.main()