#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:21:14 2025

@author: kendrickshepherd
"""

import Univariate_Lagrange_Basis_Functions as UniLagrange
# import Univariate_Lagrange_Basis_Functions_Solutions as UniLagrange

import unittest

class TestUnivariateLagrangeBasisFunctions(unittest.TestCase):

    def test_LagrangeBasisEvaluation_Degree1_Uniform(self):
        p = 1
        pts = [-1,1]
        decimal_place = 4
        
        self.assertAlmostEqual(1, UniLagrange.LagrangeBasisEvaluation(p,pts,-1,0), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,1,0), decimal_place)
        self.assertAlmostEqual(0.5, UniLagrange.LagrangeBasisEvaluation(p,pts,0,0), decimal_place)
        self.assertAlmostEqual(0.75, UniLagrange.LagrangeBasisEvaluation(p,pts,-0.5,0), decimal_place)

        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,-1,1), decimal_place)
        self.assertAlmostEqual(1, UniLagrange.LagrangeBasisEvaluation(p,pts,1,1), decimal_place)
        self.assertAlmostEqual(0.5, UniLagrange.LagrangeBasisEvaluation(p,pts,0,1), decimal_place)
        self.assertAlmostEqual(0.25, UniLagrange.LagrangeBasisEvaluation(p,pts,-0.5,1), decimal_place)

    def test_LagrangeBasisEvaluation_Degree3_Uniform(self):
        p = 3
        pts = [-1,-1./3.,1./3.,1]
        decimal_place = 4
        
        self.assertAlmostEqual(1, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[0],0), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[1],0), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[2],0), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[3],0), decimal_place)

        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[0],1), decimal_place)
        self.assertAlmostEqual(1, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[1],1), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[2],1), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[3],1), decimal_place)

        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[0],2), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[1],2), decimal_place)
        self.assertAlmostEqual(1, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[2],2), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[3],2), decimal_place)

        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[0],3), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[1],3), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[2],3), decimal_place)
        self.assertAlmostEqual(1, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[3],3), decimal_place)

    def test_LagrangeBasisEvaluation_Degree3_NonUniform(self):
        p = 3
        pts = [-1,0,1./2.,1]
        decimal_place = 4
        
        self.assertAlmostEqual(1, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[0],0), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[1],0), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[2],0), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[3],0), decimal_place)

        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[0],1), decimal_place)
        self.assertAlmostEqual(1, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[1],1), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[2],1), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[3],1), decimal_place)

        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[0],2), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[1],2), decimal_place)
        self.assertAlmostEqual(1, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[2],2), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[3],2), decimal_place)

        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[0],3), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[1],3), decimal_place)
        self.assertAlmostEqual(0, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[2],3), decimal_place)
        self.assertAlmostEqual(1, UniLagrange.LagrangeBasisEvaluation(p,pts,pts[3],3), decimal_place)

        
    def test_InterpolateFunction_SmallSample(self):
        p = 3
        pts = [[0,1],[4,6],[6,2],[7,11]]
        decimal_place = 4
        
        xis, ys = UniLagrange.InterpolateFunction(p,pts,4)
        
        self.assertAlmostEqual(0, xis[0], decimal_place)
        self.assertAlmostEqual(2.33333333, xis[1], decimal_place)
        self.assertAlmostEqual(4.66666667, xis[2], decimal_place)
        self.assertAlmostEqual(7, xis[3], decimal_place)

        self.assertAlmostEqual(1, ys[0], decimal_place)
        self.assertAlmostEqual(14.59567901, ys[1], decimal_place)
        self.assertAlmostEqual(2.65432099, ys[2], decimal_place)
        self.assertAlmostEqual(11, ys[3], decimal_place)

    def test_InterpolateFunction_LargeSample(self):
        p = 3
        pts = [[0,1],[4,6],[6,2],[7,11]]
        decimal_place = 4
        
        xis, ys = UniLagrange.InterpolateFunction(p,pts,101)
        
        self.assertAlmostEqual(0.7,xis[10],decimal_place)
        self.assertAlmostEqual(10.486625,ys[10],decimal_place)
        


if __name__ == '__main__':
    unittest.main()