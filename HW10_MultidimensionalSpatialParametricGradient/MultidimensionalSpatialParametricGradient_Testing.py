#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:21:14 2025

@author: kendrickshepherd
"""

import numpy as np
import MultidimensionalSpatialParametricGradient as Lagrange
# import MultidimensionalSpatialParametricGradient_Solutions as Lagrange

import unittest

class TestLagrangeBasisFuncDerivative(unittest.TestCase):

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


    def test_EvaluateBasisParametricGradient_Degree1x1(self):
        
        degs = [1,1]
        # single_pts = [-1,0,1]
        full_pts = [[-1,1],[-1,1]]
        decimal_place = 4
        
        basis = Lagrange.LagrangeBasis2D(degs[0],degs[1],full_pts[0],full_pts[1])
        
        # Derivative in ksi
        self.assertAlmostEqual(-0.5, basis.EvaluateBasisParametricGradient(0, [-1,-1])[0], decimal_place)
        self.assertAlmostEqual(-0.25,  basis.EvaluateBasisParametricGradient(0, [-1,0])[0], decimal_place)
        self.assertAlmostEqual(0,  basis.EvaluateBasisParametricGradient(0, [-1,1])[0], decimal_place)
        self.assertAlmostEqual(-0.25,  basis.EvaluateBasisParametricGradient(0, [0,0])[0], decimal_place)
        self.assertAlmostEqual(0,  basis.EvaluateBasisParametricGradient(0,[0,1])[0], decimal_place)
        
        self.assertAlmostEqual(0.5,  basis.EvaluateBasisParametricGradient(1, [-1,-1])[0], decimal_place)
        self.assertAlmostEqual(0.25,  basis.EvaluateBasisParametricGradient(1, [-1,0])[0], decimal_place)
        self.assertAlmostEqual(0,  basis.EvaluateBasisParametricGradient(1, [-1,1])[0], decimal_place)
        self.assertAlmostEqual(0.25,  basis.EvaluateBasisParametricGradient(1, [0,0])[0], decimal_place)
        self.assertAlmostEqual(0,  basis.EvaluateBasisParametricGradient(1, [0,1])[0], decimal_place)

        self.assertAlmostEqual(0, basis.EvaluateBasisParametricGradient(2, [-1,-1])[0], decimal_place)
        self.assertAlmostEqual(-0.25, basis.EvaluateBasisParametricGradient(2, [-1,0])[0], decimal_place)
        self.assertAlmostEqual(-0.5, basis.EvaluateBasisParametricGradient(2, [-1,1])[0], decimal_place)
        self.assertAlmostEqual(-0.25, basis.EvaluateBasisParametricGradient(2, [0,0])[0], decimal_place)
        self.assertAlmostEqual(-0.5, basis.EvaluateBasisParametricGradient(2, [0,1])[0], decimal_place)
        
        self.assertAlmostEqual(0, basis.EvaluateBasisParametricGradient(3, [-1,-1])[0], decimal_place)
        self.assertAlmostEqual(0.25, basis.EvaluateBasisParametricGradient(3, [-1,0])[0], decimal_place)
        self.assertAlmostEqual(0.5, basis.EvaluateBasisParametricGradient(3, [-1,1])[0], decimal_place)
        self.assertAlmostEqual(0.25, basis.EvaluateBasisParametricGradient(3, [0,0])[0], decimal_place)
        self.assertAlmostEqual(0.5, basis.EvaluateBasisParametricGradient(3, [0,1])[0], decimal_place)

        # Derivative in eta
        self.assertAlmostEqual(-0.5, basis.EvaluateBasisParametricGradient(0, [-1,-1])[1], decimal_place)
        self.assertAlmostEqual(-0.5, basis.EvaluateBasisParametricGradient(0, [-1,0])[1], decimal_place)
        self.assertAlmostEqual(-0.5, basis.EvaluateBasisParametricGradient(0, [-1,1])[1], decimal_place)
        self.assertAlmostEqual(-0.25, basis.EvaluateBasisParametricGradient(0, [0,0])[1], decimal_place)
        self.assertAlmostEqual(-0.25, basis.EvaluateBasisParametricGradient(0, [0,1])[1], decimal_place)
        
        self.assertAlmostEqual(0, basis.EvaluateBasisParametricGradient(1, [-1,-1])[1], decimal_place)
        self.assertAlmostEqual(0, basis.EvaluateBasisParametricGradient(1, [-1,0])[1], decimal_place)
        self.assertAlmostEqual(0, basis.EvaluateBasisParametricGradient(1, [-1,1])[1], decimal_place)
        self.assertAlmostEqual(-0.25, basis.EvaluateBasisParametricGradient(1, [0,0])[1], decimal_place)
        self.assertAlmostEqual(-0.25, basis.EvaluateBasisParametricGradient(1, [0,1])[1], decimal_place)

        self.assertAlmostEqual(0.5, basis.EvaluateBasisParametricGradient(2, [-1,-1])[1], decimal_place)
        self.assertAlmostEqual(0.5, basis.EvaluateBasisParametricGradient(2, [-1,0])[1], decimal_place)
        self.assertAlmostEqual(0.5, basis.EvaluateBasisParametricGradient(2, [-1,1])[1], decimal_place)
        self.assertAlmostEqual(0.25, basis.EvaluateBasisParametricGradient(2, [0,0])[1], decimal_place)
        self.assertAlmostEqual(0.25, basis.EvaluateBasisParametricGradient(2, [0,1])[1], decimal_place)
        
        self.assertAlmostEqual(0, basis.EvaluateBasisParametricGradient(3, [-1,-1])[1], decimal_place)
        self.assertAlmostEqual(0, basis.EvaluateBasisParametricGradient(3, [-1,0])[1], decimal_place)
        self.assertAlmostEqual(0, basis.EvaluateBasisParametricGradient(3, [-1,1])[1], decimal_place)
        self.assertAlmostEqual(0.25, basis.EvaluateBasisParametricGradient(3, [0,0])[1], decimal_place)
        self.assertAlmostEqual(0.25, basis.EvaluateBasisParametricGradient(3, [0,1])[1], decimal_place)



    def test_EvaluateBasisParametricGradient_Degree2x2(self):
        degs = [2,2]
        full_pts = [[-1,0,1],[-1,0,1]]
        decimal_place = 4
        
        basis = Lagrange.LagrangeBasis2D(degs[0],degs[1],full_pts[0],full_pts[1])
        
        self.assertAlmostEqual(0, basis.EvaluateBasisParametricGradient(2, [1,0])[0], decimal_place)
        self.assertAlmostEqual(-0.5, basis.EvaluateBasisParametricGradient(2, [1,0])[1], decimal_place)

        self.assertAlmostEqual(-1./27., basis.EvaluateBasisParametricGradient(0, [1./3.,-1./3.])[0], decimal_place)
        self.assertAlmostEqual(5./54., basis.EvaluateBasisParametricGradient(0, [1./3.,-1./3.])[1], decimal_place)

    def test_EvaluateBasisSpatialGradient_Degree2x2(self):
        degs = [2,2]
        full_pts = [[-1,0,1],[-1,0,1]]
        decimal_place = 4
        
        cpts = np.array([[0,0],[0,1],[1,1],[-1,0],[-1,2],[1,2],[-2,0],[-2,3],[1,3]])
        
        basis = Lagrange.LagrangeBasis2D(degs[0],degs[1],full_pts[0],full_pts[1])
        
        xi1 = [1,0]
        df1 = np.array([[ 3.0 , 0.0 ],[ -1.0 , 1.0 ]])
        paramgrad1 = np.array([0,-0.5])
        dfinv1 = np.linalg.inv(df1)
        dfinv_trans1 = dfinv1.transpose()
        spatgrad1 = dfinv_trans1.dot(paramgrad1)
        

        xi2 = [1./3.,-1./3.]
        df2 = np.array([[ 1.3888888888888888 , -0.7777777777777776 ],[ 0.2777777777777778 , 1.1111111111111112 ]])
        paramgrad2 = np.array([-1./27.,5./54])
        dfinv2 = np.linalg.inv(df2)
        dfinv_trans2 = dfinv2.transpose()
        spatgrad2 = dfinv_trans2.dot(paramgrad2)

        
        self.assertAlmostEqual(spatgrad1[0], basis.EvaluateBasisSpatialGradient(2, cpts, xi1)[0], decimal_place)
        self.assertAlmostEqual(spatgrad1[1], basis.EvaluateBasisSpatialGradient(2, cpts, xi1)[1], decimal_place)

        self.assertAlmostEqual(spatgrad2[0], basis.EvaluateBasisSpatialGradient(0, cpts, xi2)[0], decimal_place)
        self.assertAlmostEqual(spatgrad2[1], basis.EvaluateBasisSpatialGradient(0, cpts, xi2)[1], decimal_place)




if __name__ == '__main__':
    unittest.main()
