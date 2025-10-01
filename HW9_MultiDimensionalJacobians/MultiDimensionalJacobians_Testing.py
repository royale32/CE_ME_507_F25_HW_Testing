#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:21:14 2025

@author: kendrickshepherd
"""

import numpy as np
import MultiDimensionalJacobians as Lagrange
# import MultiDimensionalJacobians_Solutions as Lagrange

import unittest


class TestLagrangeBasisFuncDerivative(unittest.TestCase):

    def test_EvaluateFunctionParentDomain_Degree1x1_Uniform(self):
        degs = [1,1]
        single_pts = [-1,1]
        full_pts = [single_pts for i in range(len(degs))]
        decimal_place = 4
        
        basis = Lagrange.LagrangeBasis2D(degs[0], degs[1], full_pts[0], full_pts[1])

        d_coeffs = [3,7,2,1]

        self.assertAlmostEqual(3, basis.EvaluateFunctionParentDomain(d_coeffs, [-1,-1]), decimal_place)
        self.assertAlmostEqual(7, basis.EvaluateFunctionParentDomain(d_coeffs, [1,-1]), decimal_place)
        self.assertAlmostEqual(2, basis.EvaluateFunctionParentDomain(d_coeffs, [-1,1]), decimal_place)
        self.assertAlmostEqual(1, basis.EvaluateFunctionParentDomain(d_coeffs, [1,1]), decimal_place)
        self.assertAlmostEqual(13./4, basis.EvaluateFunctionParentDomain(d_coeffs, [0,0]), decimal_place)


    def test_EvaluateFunctionParentDomain_Degree2x2_Uniform(self):
        degs = [2,2]
        single_pts = [-1,0,1]
        full_pts = [single_pts for i in range(len(degs))]
        decimal_place = 4
        
        basis = Lagrange.LagrangeBasis2D(degs[0], degs[1], full_pts[0], full_pts[1])

        d_coeffs = [3,7,2,1,8,34,2,1,-5]

        self.assertAlmostEqual(3, basis.EvaluateFunctionParentDomain(d_coeffs, [-1,-1]), decimal_place)
        self.assertAlmostEqual(2, basis.EvaluateFunctionParentDomain(d_coeffs, [1,-1]), decimal_place)
        self.assertAlmostEqual(2, basis.EvaluateFunctionParentDomain(d_coeffs, [-1,1]), decimal_place)
        self.assertAlmostEqual(-5, basis.EvaluateFunctionParentDomain(d_coeffs, [1,1]), decimal_place)
        self.assertAlmostEqual(8, basis.EvaluateFunctionParentDomain(d_coeffs, [0,0]), decimal_place)
        self.assertAlmostEqual(34, basis.EvaluateFunctionParentDomain(d_coeffs, [1,0]), decimal_place)

    def test_EvaluateFunctionParentDomain_Degree2x2_Problem5(self):
        degs = [2,2]
        single_pts = [-1,0,1]
        full_pts = [single_pts for i in range(len(degs))]
        decimal_place = 4
        
        basis = Lagrange.LagrangeBasis2D(degs[0], degs[1], full_pts[0], full_pts[1])

        d_coeffs = [-1,2,3,5,6,7,2,1,3]

        self.assertAlmostEqual(-1, basis.EvaluateFunctionParentDomain(d_coeffs, [-1,-1]), decimal_place)
        self.assertAlmostEqual(3, basis.EvaluateFunctionParentDomain(d_coeffs, [1,-1]), decimal_place)
        self.assertAlmostEqual(2, basis.EvaluateFunctionParentDomain(d_coeffs, [-1,1]), decimal_place)
        self.assertAlmostEqual(3, basis.EvaluateFunctionParentDomain(d_coeffs, [1,1]), decimal_place)
        self.assertAlmostEqual(6, basis.EvaluateFunctionParentDomain(d_coeffs, [0,0]), decimal_place)
        self.assertAlmostEqual(7, basis.EvaluateFunctionParentDomain(d_coeffs, [1,0]), decimal_place)
        
    def test_EvaluateXMapping_Degree2x2(self):
        degs = [2,2]
        single_pts = [-1,0,1]
        full_pts = [single_pts for i in range(len(degs))]
        decimal_place = 4
        
        basis = Lagrange.LagrangeBasis2D(degs[0], degs[1], full_pts[0], full_pts[1])

        cpts = np.array([[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]])

        val0 = basis.EvaluateSpatialMapping(cpts, [-1,-1])
        val2 = basis.EvaluateSpatialMapping(cpts, [1,-1])
        val6 = basis.EvaluateSpatialMapping(cpts, [-1,1])
        val8 = basis.EvaluateSpatialMapping(cpts, [1,1])
        val4 = basis.EvaluateSpatialMapping(cpts, [0,0])
        val5 = basis.EvaluateSpatialMapping(cpts, [1,0])
        
        self.assertAlmostEqual(0, val0[0], decimal_place)
        self.assertAlmostEqual(1, val2[0], decimal_place)
        self.assertAlmostEqual(-2, val6[0], decimal_place)
        self.assertAlmostEqual(1, val8[0], decimal_place)
        self.assertAlmostEqual(-1, val4[0], decimal_place)
        self.assertAlmostEqual(1, val5[0], decimal_place)

        self.assertAlmostEqual(0, val0[1], decimal_place)
        self.assertAlmostEqual(1, val2[1], decimal_place)
        self.assertAlmostEqual(0, val6[1], decimal_place)
        self.assertAlmostEqual(3, val8[1], decimal_place)
        self.assertAlmostEqual(2, val4[1], decimal_place)
        self.assertAlmostEqual(2, val5[1], decimal_place)

    def test_EvaluateDeformationGradient_Degree2x2(self):
        degs = [2,2]
        single_pts = [-1,0,1]
        full_pts = [single_pts for i in range(len(degs))]
        decimal_place = 4
        
        basis = Lagrange.LagrangeBasis2D(degs[0], degs[1], full_pts[0], full_pts[1])

        cpts = np.array([[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]])

        vals = [[-1,-1],
               [-0.5,-1],
               [0,-1],
               [0.5,-1],
               [1,-1],
               [-1,-0.5],
               [-0.5,-0.5],
               [0,-0.5],
               [0.5,-0.5],
               [1,-0.5],
                [-1,0],
                [-0.5,0],
                [0,0],
                [0.5,0],
                [1,0],
                [-1,0.5],
                [-0.5,0.5],
                [0,0.5],
                [0.5,0.5],
                [1,0.5],
                [-1,1],
                [-0.5,1],
                [0,1],
                [0.5,1],
                [1,1],
                [1./3.,-1./3.]
              ]
        
        dfs = [
               [[-0.5,-1],[1.5,0]],
               [[ 0.,    -1.125],         [ 1.,     0.625]],
               [[ 0.5, -1. ],         [ 0.5,  1. ]],
               [[ 1.,    -0.625],         [ 0.,     1.125]],
               [[ 1.5,  0. ],         [-0.5,  1. ]],
               [[-0.75, -1.  ],         [ 2.25,  0.  ]],
               [[ 0.,    -1.125],         [ 1.5,    0.625]],
               [[ 0.75, -1.  ],         [ 0.75,  1.  ]],
               [[ 1.5 , -0.625 ],[ 0.0 , 1.125 ]],
               [[ 2.25 , 0.0 ],[ -0.75 , 1.0 ]],
               [[ -1.0 , -1.0 ],[ 3.0 , 0.0 ]],
               [[ 0.0 , -1.125 ],[ 2.0 , 0.625 ]],
               [[ 1.0 , -1.0 ],[ 1.0 , 1.0 ]],
               [[ 2.0 , -0.625 ],[ 0.0 , 1.125 ]],
               [[ 3.0 , 0.0 ],[ -1.0 , 1.0 ]],
               [[ -1.25 , -1.0 ],[ 3.75 , 0.0 ]],
               [[ 0.0 , -1.125 ],[ 2.5 , 0.625 ]],
               [[ 1.25 , -1.0 ],[ 1.25 , 1.0 ]],
               [[ 2.5 , -0.625 ],[ 0.0 , 1.125 ]],
               [[ 3.75 , 0.0 ],[ -1.25 , 1.0 ]],
               [[ -1.5 , -1.0 ],[ 4.5 , 0.0 ]],
               [[ 0.0 , -1.125 ],[ 3.0 , 0.625 ]],
               [[ 1.5 , -1.0 ],[ 1.5 , 1.0 ]],
               [[ 3.0 , -0.625 ],[ 0.0 , 1.125 ]],
               [[ 4.5 , 0.0 ],[ -1.5 , 1.0 ]],
               [[ 1.3888888888888888 , -0.7777777777777776 ],[ 0.2777777777777778 , 1.1111111111111112 ]]
              ]
                
        for k in range(0,len(vals)):
            for j in range(0,2):
                for i in range(0,2):
                    self.assertAlmostEqual(dfs[k][i][j], basis.EvaluateDeformationGradient(cpts, [vals[k][0],vals[k][1]])[i][j], decimal_place)



    def test_EvaluateJacobians_Degree2x2(self):
        degs = [2,2]
        single_pts = [-1,0,1]
        full_pts = [single_pts for i in range(len(degs))]
        decimal_place = 4
        
        basis = Lagrange.LagrangeBasis2D(degs[0], degs[1], full_pts[0], full_pts[1])

        cpts = np.array([[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]])

        vals = [[-1,-1],
               [-0.5,-1],
               [0,-1],
               [0.5,-1],
               [1,-1],
               [-1,-0.5],
               [-0.5,-0.5],
               [0,-0.5],
               [0.5,-0.5],
               [1,-0.5],
                [-1,0],
                [-0.5,0],
                [0,0],
                [0.5,0],
                [1,0],
                [-1,0.5],
                [-0.5,0.5],
                [0,0.5],
                [0.5,0.5],
                [1,0.5],
                [-1,1],
                [-0.5,1],
                [0,1],
                [0.5,1],
                [1,1]]
        
        def det(twomat):
            return twomat[0][0]*twomat[1][1]-twomat[0][1]*twomat[1][0]
        
        dfs = [
               [[-0.5,-1],[1.5,0]],
               [[ 0.,    -1.125],         [ 1.,     0.625]],
               [[ 0.5, -1. ],         [ 0.5,  1. ]],
               [[ 1.,    -0.625],         [ 0.,     1.125]],
               [[ 1.5,  0. ],         [-0.5,  1. ]],
               [[-0.75, -1.  ],         [ 2.25,  0.  ]],
               [[ 0.,    -1.125],         [ 1.5,    0.625]],
               [[ 0.75, -1.  ],         [ 0.75,  1.  ]],
               [[ 1.5 , -0.625 ],[ 0.0 , 1.125 ]],
               [[ 2.25 , 0.0 ],[ -0.75 , 1.0 ]],
               [[ -1.0 , -1.0 ],[ 3.0 , 0.0 ]],
               [[ 0.0 , -1.125 ],[ 2.0 , 0.625 ]],
               [[ 1.0 , -1.0 ],[ 1.0 , 1.0 ]],
               [[ 2.0 , -0.625 ],[ 0.0 , 1.125 ]],
               [[ 3.0 , 0.0 ],[ -1.0 , 1.0 ]],
               [[ -1.25 , -1.0 ],[ 3.75 , 0.0 ]],
               [[ 0.0 , -1.125 ],[ 2.5 , 0.625 ]],
               [[ 1.25 , -1.0 ],[ 1.25 , 1.0 ]],
               [[ 2.5 , -0.625 ],[ 0.0 , 1.125 ]],
               [[ 3.75 , 0.0 ],[ -1.25 , 1.0 ]],
               [[ -1.5 , -1.0 ],[ 4.5 , 0.0 ]],
               [[ 0.0 , -1.125 ],[ 3.0 , 0.625 ]],
               [[ 1.5 , -1.0 ],[ 1.5 , 1.0 ]],
               [[ 3.0 , -0.625 ],[ 0.0 , 1.125 ]],
               [[ 4.5 , 0.0 ],[ -1.5 , 1.0 ]]
              ]
                
        for k in range(0,len(vals)):
            for j in range(0,2):
                for i in range(0,2):
                    self.assertAlmostEqual(det(dfs[k]), basis.EvaluateJacobian(cpts, [vals[k][0],vals[k][1]]), decimal_place)




if __name__ == '__main__':
    unittest.main()