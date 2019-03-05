# -*- coding: utf-8 -*-
"""
Python code of Spider-Monkey Optimization (SMO)
Coded by: Mukesh Saraswat (emailid: saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar to code given at link: https://github.com/himanshuRepo/CKGSA-in-Python 
 and C++ version of the SMO at link: http://smo.scrs.in/

Reference: Jagdish Chand Bansal, Harish Sharma, Shimpi Singh Jadon, and Maurice Clerc. "Spider monkey optimization algorithm for numerical optimization." Memetic computing 6, no. 1, 31-47, 2014.
@link: http://smo.scrs.in/

-- Benchmark.py: Defining the benchmark function along its range lower bound, upper bound and dimensions

Code compatible:
 -- Python: 2.* or 3.*
"""

import numpy
import math

# define the function blocks 
def F1(x):
    s=numpy.sum(x**2);
    return s

# define the function parameters 
def getFunctionDetails(a):
    
    # [name, lb, ub, dim, acc_err, obj_val]
    param = {	0: ["F1",-100,100,30,1.0e-5,0],
            }
    return param.get(a, "nothing")



