#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:21:14 2025

@author: kendrickshepherd
"""

import MultiDimensionalBasisFunctions as MultiBFs
# import MultiDimensionalBasisFunctions_Solutions as MultiBFs

import unittest

class TestMultiDimensionalLagrangeBasisFunctions(unittest.TestCase):

    def test_MultiDimensionalBasisFunctionIdxs_Degree1x1_Uniform(self):
        degs = [1,1]
        single_pts = [-1,1]
        full_pts = [single_pts for i in range(len(degs))]
        decimal_place = 4
        
        self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunctionIdxs([0,0],degs,full_pts,[-1,-1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([0,0],degs,full_pts,[1,-1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([0,0],degs,full_pts,[-1,1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([0,0],degs,full_pts,[1,1]), decimal_place)

        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([1,0],degs,full_pts,[-1,-1]), decimal_place)
        self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunctionIdxs([1,0],degs,full_pts,[1,-1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([1,0],degs,full_pts,[-1,1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([1,0],degs,full_pts,[1,1]), decimal_place)

        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([0,1],degs,full_pts,[-1,-1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([0,1],degs,full_pts,[1,-1]), decimal_place)
        self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunctionIdxs([0,1],degs,full_pts,[-1,1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([0,1],degs,full_pts,[1,1]), decimal_place)

        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([1,1],degs,full_pts,[-1,-1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([1,1],degs,full_pts,[1,-1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([1,1],degs,full_pts,[-1,1]), decimal_place)
        self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunctionIdxs([1,1],degs,full_pts,[1,1]), decimal_place)


    def test_MultiDimensionalBasisFunctionIdxs_Degree2x1_Uniform(self):
        degs = [2,1]
        single_pts = [-1,0,1]
        full_pts = [single_pts,[-1,1]]
        decimal_place = 4
        
        for j in range(0,degs[1]+1):
            for i in range(0,degs[0]+1):
                A = i+(degs[0]+1)*j
                
                for b in range(0,degs[1]+1):
                    for a in range(0,degs[0]+1):
                        B = a + (degs[0]+1)*b
                        if B == A:
                            self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunctionIdxs([i,j],degs,full_pts,[full_pts[0][a],full_pts[1][b]]),decimal_place)
                        else:
                            self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([i,j],degs,full_pts,[full_pts[0][a],full_pts[1][b]]),decimal_place)

    def test_MultiDimensionalBasisFunctionIdxs_Degree2x2_Uniform(self):
        degs = [2,2]
        single_pts = [-1,0,1]
        full_pts = [single_pts,single_pts]
        decimal_place = 4
        
        for j in range(0,degs[1]+1):
            for i in range(0,degs[0]+1):
                A = i+(degs[0]+1)*j
                
                for b in range(0,degs[1]+1):
                    for a in range(0,degs[0]+1):
                        B = a + (degs[0]+1)*b
                        if B == A:
                            self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunctionIdxs([i,j],degs,full_pts,[full_pts[0][a],full_pts[1][b]]),decimal_place)
                        else:
                            self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([i,j],degs,full_pts,[full_pts[0][a],full_pts[1][b]]),decimal_place)

        
    def test_MultiDimensionalBasisFunctionIdxs_Degree3x3_Uniform(self):
        degs = [3,3]
        single_pts = [-1,-1./3.,1./3.,1]
        full_pts = [single_pts,single_pts]
        decimal_place = 4
        
        for j in range(0,degs[1]+1):
            for i in range(0,degs[0]+1):
                A = i+(degs[0]+1)*j
                
                for b in range(0,degs[1]+1):
                    for a in range(0,degs[0]+1):
                        B = a + (degs[0]+1)*b
                        if B == A:
                            self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunctionIdxs([i,j],degs,full_pts,[full_pts[0][a],full_pts[1][b]]),decimal_place)
                        else:
                            self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunctionIdxs([i,j],degs,full_pts,[full_pts[0][a],full_pts[1][b]]),decimal_place)


    def test_MultiDimensionalBasisFunctionSingleIdx_Degree1x1_Uniform(self):
        degs = [1,1]
        single_pts = [-1,1]
        full_pts = [single_pts for i in range(len(degs))]
        decimal_place = 4
        
        self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunction(0,degs,full_pts,[-1,-1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(0,degs,full_pts,[1,-1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(0,degs,full_pts,[-1,1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(0,degs,full_pts,[1,1]), decimal_place)

        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(1,degs,full_pts,[-1,-1]), decimal_place)
        self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunction(1,degs,full_pts,[1,-1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(1,degs,full_pts,[-1,1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(1,degs,full_pts,[1,1]), decimal_place)

        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(2,degs,full_pts,[-1,-1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(2,degs,full_pts,[1,-1]), decimal_place)
        self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunction(2,degs,full_pts,[-1,1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(2,degs,full_pts,[1,1]), decimal_place)

        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(3,degs,full_pts,[-1,-1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(3,degs,full_pts,[1,-1]), decimal_place)
        self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(3,degs,full_pts,[-1,1]), decimal_place)
        self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunction(3,degs,full_pts,[1,1]), decimal_place)


    def test_MultiDimensionalBasisFunctionSingleIdx_Degree2x1_Uniform(self):
        degs = [2,1]
        single_pts = [-1,0,1]
        full_pts = [single_pts,[-1,1]]
        decimal_place = 4
        
        for j in range(0,degs[1]+1):
            for i in range(0,degs[0]+1):
                A = i+(degs[0]+1)*j
                
                for b in range(0,degs[1]+1):
                    for a in range(0,degs[0]+1):
                        B = a + (degs[0]+1)*b
                        if B == A:
                            self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunction(A,degs,full_pts,[full_pts[0][a],full_pts[1][b]]),decimal_place)
                        else:
                            self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(A,degs,full_pts,[full_pts[0][a],full_pts[1][b]]),decimal_place)

    def test_MultiDimensionalBasisFunctionSingleIdx_Degree2x2_Uniform(self):
        degs = [2,2]
        single_pts = [-1,0,1]
        full_pts = [single_pts,single_pts]
        decimal_place = 4
        
        for j in range(0,degs[1]+1):
            for i in range(0,degs[0]+1):
                A = i+(degs[0]+1)*j
                
                for b in range(0,degs[1]+1):
                    for a in range(0,degs[0]+1):
                        B = a + (degs[0]+1)*b
                        if B == A:
                            self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunction(A,degs,full_pts,[full_pts[0][a],full_pts[1][b]]),decimal_place)
                        else:
                            self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(A,degs,full_pts,[full_pts[0][a],full_pts[1][b]]),decimal_place)

        
    def test_MultiDimensionalBasisFunctionSingleIdx_Degree3x3_Uniform(self):
        degs = [3,3]
        single_pts = [-1,-1./3.,1./3.,1]
        full_pts = [single_pts,single_pts]
        decimal_place = 4
        
        for j in range(0,degs[1]+1):
            for i in range(0,degs[0]+1):
                A = i+(degs[0]+1)*j
                
                for b in range(0,degs[1]+1):
                    for a in range(0,degs[0]+1):
                        B = a + (degs[0]+1)*b
                        if B == A:
                            self.assertAlmostEqual(1, MultiBFs.MultiDimensionalBasisFunction(A,degs,full_pts,[full_pts[0][a],full_pts[1][b]]),decimal_place)
                        else:
                            self.assertAlmostEqual(0, MultiBFs.MultiDimensionalBasisFunction(A,degs,full_pts,[full_pts[0][a],full_pts[1][b]]),decimal_place)


        


if __name__ == '__main__':
    unittest.main()