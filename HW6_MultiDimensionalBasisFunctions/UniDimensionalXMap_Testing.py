#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:21:14 2025

@author: kendrickshepherd
"""

import UniDimensionalXMap as UniMap
# import UniDimensionalXMap_Solutions as UniMap

import unittest

class TestUniDimensionalXMap(unittest.TestCase):

    def test_UniDimensionalXMap_Linear(self):
        deg = 1
        interp_pts = [-1,1]
        spatial_pts = [0,1]
        decimal_place = 4
        
        self.assertAlmostEqual(0, UniMap.XMap(deg,spatial_pts,interp_pts,-1), decimal_place)
        self.assertAlmostEqual(0.5, UniMap.XMap(deg,spatial_pts,interp_pts,0), decimal_place)
        self.assertAlmostEqual(1, UniMap.XMap(deg,spatial_pts,interp_pts,1), decimal_place)


    def test_UniDimensionalXMap_Quadratic_Uniform(self):
        deg = 2
        interp_pts = [-1,0,1]
        spatial_pts = [0.5,1,1.5]
        decimal_place = 4
        
        self.assertAlmostEqual(0.5, UniMap.XMap(deg,spatial_pts,interp_pts,-1), decimal_place)
        self.assertAlmostEqual(1, UniMap.XMap(deg,spatial_pts,interp_pts,0), decimal_place)
        self.assertAlmostEqual(1.5, UniMap.XMap(deg,spatial_pts,interp_pts,1), decimal_place)

    def test_UniDimensionalXMap_Quadratic_NonUniform1(self):
        deg = 2
        interp_pts = [-1,-0.6,1]
        spatial_pts = [0.5,0.7,1.5]
        decimal_place = 4
        
        self.assertAlmostEqual(0.5, UniMap.XMap(deg,spatial_pts,interp_pts,-1), decimal_place)
        self.assertAlmostEqual(1, UniMap.XMap(deg,spatial_pts,interp_pts,0), decimal_place)
        self.assertAlmostEqual(1.5, UniMap.XMap(deg,spatial_pts,interp_pts,1), decimal_place)

        
    def test_UniDimensionalXMap_Quadratic_NonUniform2(self):
        deg = 2
        interp_pts = [-1,0,1]
        spatial_pts = [0.5,0.7,1.5]
        decimal_place = 4
        
        self.assertAlmostEqual(0.5, UniMap.XMap(deg,spatial_pts,interp_pts,-1), decimal_place)
        self.assertAlmostEqual(0.7, UniMap.XMap(deg,spatial_pts,interp_pts,0), decimal_place)
        self.assertAlmostEqual(1.5, UniMap.XMap(deg,spatial_pts,interp_pts,1), decimal_place)



        


if __name__ == '__main__':
    unittest.main()