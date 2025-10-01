#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:21:14 2025

@author: kendrickshepherd
"""

import LagrangeBasisFuncDerivative as Lagrange
# import LagrangeBasisFuncDerivative_Solutions as Lagrange

import unittest

class TestLagrangeBasisFuncDerivative(unittest.TestCase):

    def test_LagrangeBasisParamDervEvaluation_1D(self):
        deg = 1
        interp_pts = [-1,1]
        decimal_place = 4
        self.assertAlmostEqual(-0.5, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,-1,0), decimal_place)
        self.assertAlmostEqual(-0.5, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,0,0), decimal_place)
        self.assertAlmostEqual(-0.5, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,1,0), decimal_place)

        self.assertAlmostEqual(0.5, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,-1,1), decimal_place)
        self.assertAlmostEqual(0.5, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,0,1), decimal_place)
        self.assertAlmostEqual(0.5, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,1,1), decimal_place)

        deg = 2
        interp_pts = [-1,0,1]
        decimal_place = 4
        self.assertAlmostEqual(-1.5, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,-1,0), decimal_place)
        self.assertAlmostEqual(-0.5, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,0,0), decimal_place)
        self.assertAlmostEqual(0.5, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,1,0), decimal_place)

        self.assertAlmostEqual(2, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,-1,1), decimal_place)
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,0,1), decimal_place)
        self.assertAlmostEqual(-2, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,1,1), decimal_place)

        self.assertAlmostEqual(-0.5, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,-1,2), decimal_place)
        self.assertAlmostEqual(0.5, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,0,2), decimal_place)
        self.assertAlmostEqual(1.5, Lagrange.LagrangeBasisParamDervEvaluation(deg,interp_pts,1,2), decimal_place)



    def test_LagrangeBasisParamDervEvaluation_2D_Bilinear(self):
        degs = [1,1]
        # single_pts = [-1,0,1]
        full_pts = [[-1,1],[-1,1]]
        decimal_place = 4
        
        # Derivative in ksi
        self.assertAlmostEqual(-0.5, Lagrange.LagrangeBasisDervParamMultiD(0, degs, full_pts, [-1,-1], 0), decimal_place)
        self.assertAlmostEqual(-0.25, Lagrange.LagrangeBasisDervParamMultiD(0, degs, full_pts, [-1,0], 0), decimal_place)
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(0, degs, full_pts, [-1,1], 0), decimal_place)
        self.assertAlmostEqual(-0.25, Lagrange.LagrangeBasisDervParamMultiD(0, degs, full_pts, [0,0], 0), decimal_place)
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(0, degs, full_pts, [0,1], 0), decimal_place)
        
        self.assertAlmostEqual(0.5, Lagrange.LagrangeBasisDervParamMultiD(1, degs, full_pts, [-1,-1], 0), decimal_place)
        self.assertAlmostEqual(0.25, Lagrange.LagrangeBasisDervParamMultiD(1, degs, full_pts, [-1,0], 0), decimal_place)
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(1, degs, full_pts, [-1,1], 0), decimal_place)
        self.assertAlmostEqual(0.25, Lagrange.LagrangeBasisDervParamMultiD(1, degs, full_pts, [0,0], 0), decimal_place)
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(1, degs, full_pts, [0,1], 0), decimal_place)

        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(2, degs, full_pts, [-1,-1], 0), decimal_place)
        self.assertAlmostEqual(-0.25, Lagrange.LagrangeBasisDervParamMultiD(2, degs, full_pts, [-1,0], 0), decimal_place)
        self.assertAlmostEqual(-0.5, Lagrange.LagrangeBasisDervParamMultiD(2, degs, full_pts, [-1,1], 0), decimal_place)
        self.assertAlmostEqual(-0.25, Lagrange.LagrangeBasisDervParamMultiD(2, degs, full_pts, [0,0], 0), decimal_place)
        self.assertAlmostEqual(-0.5, Lagrange.LagrangeBasisDervParamMultiD(2, degs, full_pts, [0,1], 0), decimal_place)
        
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(3, degs, full_pts, [-1,-1], 0), decimal_place)
        self.assertAlmostEqual(0.25, Lagrange.LagrangeBasisDervParamMultiD(3, degs, full_pts, [-1,0], 0), decimal_place)
        self.assertAlmostEqual(0.5, Lagrange.LagrangeBasisDervParamMultiD(3, degs, full_pts, [-1,1], 0), decimal_place)
        self.assertAlmostEqual(0.25, Lagrange.LagrangeBasisDervParamMultiD(3, degs, full_pts, [0,0], 0), decimal_place)
        self.assertAlmostEqual(0.5, Lagrange.LagrangeBasisDervParamMultiD(3, degs, full_pts, [0,1], 0), decimal_place)

        # Derivative in eta
        self.assertAlmostEqual(-0.5, Lagrange.LagrangeBasisDervParamMultiD(0, degs, full_pts, [-1,-1], 1), decimal_place)
        self.assertAlmostEqual(-0.5, Lagrange.LagrangeBasisDervParamMultiD(0, degs, full_pts, [-1,0], 1), decimal_place)
        self.assertAlmostEqual(-0.5, Lagrange.LagrangeBasisDervParamMultiD(0, degs, full_pts, [-1,1], 1), decimal_place)
        self.assertAlmostEqual(-0.25, Lagrange.LagrangeBasisDervParamMultiD(0, degs, full_pts, [0,0], 1), decimal_place)
        self.assertAlmostEqual(-0.25, Lagrange.LagrangeBasisDervParamMultiD(0, degs, full_pts, [0,1], 1), decimal_place)
        
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(1, degs, full_pts, [-1,-1], 1), decimal_place)
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(1, degs, full_pts, [-1,0], 1), decimal_place)
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(1, degs, full_pts, [-1,1], 1), decimal_place)
        self.assertAlmostEqual(-0.25, Lagrange.LagrangeBasisDervParamMultiD(1, degs, full_pts, [0,0], 1), decimal_place)
        self.assertAlmostEqual(-0.25, Lagrange.LagrangeBasisDervParamMultiD(1, degs, full_pts, [0,1], 1), decimal_place)

        self.assertAlmostEqual(0.5, Lagrange.LagrangeBasisDervParamMultiD(2, degs, full_pts, [-1,-1], 1), decimal_place)
        self.assertAlmostEqual(0.5, Lagrange.LagrangeBasisDervParamMultiD(2, degs, full_pts, [-1,0], 1), decimal_place)
        self.assertAlmostEqual(0.5, Lagrange.LagrangeBasisDervParamMultiD(2, degs, full_pts, [-1,1], 1), decimal_place)
        self.assertAlmostEqual(0.25, Lagrange.LagrangeBasisDervParamMultiD(2, degs, full_pts, [0,0], 1), decimal_place)
        self.assertAlmostEqual(0.25, Lagrange.LagrangeBasisDervParamMultiD(2, degs, full_pts, [0,1], 1), decimal_place)
        
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(3, degs, full_pts, [-1,-1], 1), decimal_place)
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(3, degs, full_pts, [-1,0], 1), decimal_place)
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(3, degs, full_pts, [-1,1], 1), decimal_place)
        self.assertAlmostEqual(0.25, Lagrange.LagrangeBasisDervParamMultiD(3, degs, full_pts, [0,0], 1), decimal_place)
        self.assertAlmostEqual(0.25, Lagrange.LagrangeBasisDervParamMultiD(3, degs, full_pts, [0,1], 1), decimal_place)


    def test_LagrangeBasisParamDervEvaluation_2D_Biquadratic_Part(self):
        degs = [2,2]
        full_pts = [[-1,0,1],[-1,0,1]]
        decimal_place = 4
        
        # Derivative in ksi
        self.assertAlmostEqual(0, Lagrange.LagrangeBasisDervParamMultiD(2, degs, full_pts, [1,0], 0), decimal_place)
        self.assertAlmostEqual(-0.5, Lagrange.LagrangeBasisDervParamMultiD(2, degs, full_pts, [1,0], 1), decimal_place)

        self.assertAlmostEqual(-1./27., Lagrange.LagrangeBasisDervParamMultiD(0, degs, full_pts, [1./3.,-1./3.], 0), decimal_place)
        self.assertAlmostEqual(5./54., Lagrange.LagrangeBasisDervParamMultiD(0, degs, full_pts, [1./3.,-1./3.], 1), decimal_place)

        


if __name__ == '__main__':
    unittest.main()