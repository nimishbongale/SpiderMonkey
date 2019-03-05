# -*- coding: utf-8 -*-
"""
Python code of Spider-Monkey Optimization (SMO)
Coded by: Mukesh Saraswat (emailid: saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar to code given at link: https://github.com/himanshuRepo/CKGSA-in-Python 
 and C++ version of the SMO at link: http://smo.scrs.in/

Reference: Jagdish Chand Bansal, Harish Sharma, Shimpi Singh Jadon, and Maurice Clerc. "Spider monkey optimization algorithm for numerical optimization." Memetic computing 6, no. 1, 31-47, 2014.
@link: http://smo.scrs.in/

-- solution.py: Defining the solution variable for saving the output variables

Code compatible:
 -- Python: 2.* or 3.*
"""

class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence = []
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.error =0
        self.feval=0
        self.maxiers=0
