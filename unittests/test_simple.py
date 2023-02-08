# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:21:02 2023

@author: kevin
"""

import unittest
from GLMnet.GLM_RNN import NL
import numpy as np

class SimplisticTest(unittest.TestCase):

    def test(self):
        self.assertTrue(True)
    
    def function_test(self):
        self.assertTrue(NL(np.random.randn())>=0)

if __name__ == '__main__':
    unittest.main()