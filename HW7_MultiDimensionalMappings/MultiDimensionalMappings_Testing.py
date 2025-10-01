#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:21:14 2025

@author: kendrickshepherd
"""

import numpy as np
import MultiDimensionalMappings as MultiMap
# import MultiDimensionalMappings_Solutions as MultiMap

import unittest

class TestMultiDimensionalMappings(unittest.TestCase):

    def test_NumberBasisFunctions(self):
        basis1 = MultiMap.LagrangeBasis2D(1, 1, [-1,1], [-1,1])
        self.assertEqual(4, basis1.NBasisFuncs())

        basis2 = MultiMap.LagrangeBasis2D(2, 2, [-1,0,1], [-1,0,1])
        self.assertEqual(9, basis2.NBasisFuncs())

        basis3 = MultiMap.LagrangeBasis2D(1, 2, [-1,1], [-1,0,1])
        self.assertEqual(6, basis3.NBasisFuncs())

        basis4 = MultiMap.LagrangeBasis2D(2, 1, [-1,0,1], [-1,1])
        self.assertEqual(6, basis4.NBasisFuncs())

        basis5 = MultiMap.LagrangeBasis2D(100, 73, np.linspace(-1,1,100), np.linspace(-1,1,73))
        self.assertEqual(7474, basis5.NBasisFuncs())


    def test_EvalBasisFunction_MixedDegree(self):
        degs = [2,1]
        single_pts = [-1,0,1]
        full_pts = [single_pts,[-1,1]]
        decimal_place = 4
        
        basis = MultiMap.LagrangeBasis2D(degs[0], degs[1], full_pts[0], full_pts[1])
        
        for j in range(0,degs[1]+1):
            for i in range(0,degs[0]+1):
                A = i+(degs[0]+1)*j
                
                for b in range(0,degs[1]+1):
                    for a in range(0,degs[0]+1):
                        B = a + (degs[0]+1)*b
                        if B == A:
                            self.assertAlmostEqual(1, basis.EvalBasisFunction(A,[full_pts[0][a],full_pts[1][b]]),decimal_place)
                        else:
                            self.assertAlmostEqual(0, basis.EvalBasisFunction(A,[full_pts[0][a],full_pts[1][b]]),decimal_place)

    def test_EvalBasisFunction_BiQuadratic(self):
        degs = [2,2]
        single_pts = [-1,0,1]
        full_pts = [single_pts,single_pts]
        decimal_place = 4
        
        basis = MultiMap.LagrangeBasis2D(degs[0], degs[1], full_pts[0], full_pts[1])
        
        for j in range(0,degs[1]+1):
            for i in range(0,degs[0]+1):
                A = i+(degs[0]+1)*j
                
                for b in range(0,degs[1]+1):
                    for a in range(0,degs[0]+1):
                        B = a + (degs[0]+1)*b
                        if B == A:
                            self.assertAlmostEqual(1, basis.EvalBasisFunction(A,[full_pts[0][a],full_pts[1][b]]),decimal_place)
                        else:
                            self.assertAlmostEqual(0, basis.EvalBasisFunction(A,[full_pts[0][a],full_pts[1][b]]),decimal_place)

        

    def test_EvaluateFunctionParentDomain_Degree1x1_Uniform(self):
        degs = [1,1]
        single_pts = [-1,1]
        full_pts = [single_pts for i in range(len(degs))]
        decimal_place = 4
        
        basis = MultiMap.LagrangeBasis2D(degs[0], degs[1], full_pts[0], full_pts[1])

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
        
        basis = MultiMap.LagrangeBasis2D(degs[0], degs[1], full_pts[0], full_pts[1])

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
        
        basis = MultiMap.LagrangeBasis2D(degs[0], degs[1], full_pts[0], full_pts[1])

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


        


if __name__ == '__main__':
    unittest.main()