# -*- coding: utf-8 -*-
"""
Python code of Spider-Monkey Optimization (SMO)
Coded by: Mukesh Saraswat (emailid: saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar to code given at link: https://github.com/himanshuRepo/CKGSA-in-Python 
 and C++ version of the SMO at link: http://smo.scrs.in/

Reference: Jagdish Chand Bansal, Harish Sharma, Shimpi Singh Jadon, and Maurice Clerc. "Spider monkey optimization algorithm for numerical optimization." Memetic computing 6, no. 1, 31-47, 2014.
@link: http://smo.scrs.in/

-- Main.py: Calling the Spider-Monkey Optimization (SMO) Algorithm 
                for minimizing of an objective Function

Code compatible:
 -- Python: 2.* or 3.*
"""
import SMO as smo
import benchmarks
import csv
import numpy
import time
import math


def selector(algo,func_details,popSize,Iter,succ_rate,mean_feval):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    dim=func_details[3]
    acc_err=func_details[4]
    obj_val=func_details[5]
       
    if(algo==0):
        x,succ_rate,mean_feval=smo.main(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,acc_err,obj_val,succ_rate,mean_feval)       
    return x,succ_rate,mean_feval
    
    
# Select optimizers
SMO= True # Code by Himanshu Mittal


# Select benchmark function
F1=True

optimizer=[SMO]
benchmarkfunc=[F1] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns=2

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 10
Iterations= 5

#Export results ?
Export=True

#Automaticly generated name by date and time
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

# Check if it works at least once
Flag=False

# CSV Header for for the cinvergence 
CnvgHeader=[]

for l in range(0,Iterations):
	CnvgHeader.append("Iter"+str(l+1))

mean_error=0
total_feval=0
mean1=0
var=0
sd=0
mean_feval=0
succ_rate=0
GlobalMins=numpy.zeros(NumOfRuns)


for i in range (0, len(optimizer)):
    for j in range (0, len(benchmarkfunc)):
        if((optimizer[i]==True) and (benchmarkfunc[j]==True)): # start experiment if an optimizer and an objective function is selected
            for k in range (0,NumOfRuns):
                    
                func_details=benchmarks.getFunctionDetails(j)
                print("Run: {}".format(k+1))
                x,succ_rate,mean_feval=selector(i,func_details,PopulationSize,Iterations,succ_rate,mean_feval)
                mean_error=mean_error+x.error;
                mean1=mean1+x.convergence[-1]
                total_feval=total_feval+x.feval
                GlobalMins[k]=x.convergence[-1]

                if(Export==True):
                    with open(ExportToFile, 'a') as out:
                        writer = csv.writer(out,delimiter=',')
                        if (Flag==False): # just one time to write the header of the CSV file
                            header= numpy.concatenate([["Optimizer","objfname","startTime","EndTime","ExecutionTime"],CnvgHeader])
                            writer.writerow(header)
                        a=numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime],x.convergence])
                        writer.writerow(a)
                    out.close()
                    print("Results of {} run are saved in 'csv' file.".format(k+1))
                Flag=True # at least one experiment
            mean1=mean1/NumOfRuns;
            mean_error=mean_error/NumOfRuns
            if(succ_rate>0):
                mean_feval=mean_feval/succ_rate
            total_feval=total_feval/NumOfRuns
            for k in range (NumOfRuns):
                var=var + math.pow((GlobalMins[k]-mean1),2)
            var=var/NumOfRuns
            sd=math.sqrt(var)
            # print("values after executing are: \n Mean Error \t Mean Function eval \t Total Function eval \t Variance \t STD \n",(mean_error,mean_feval,total_feval,var,sd))
            print("Values after executing SMO: \n Mean Error:{} \n Mean Function eval:{} \n Total Function eval:{} \n Variance:{} \n STD:{}".format(mean_error,mean_feval,total_feval,var,sd))

if (Flag==False): # Faild to run at least one experiment
    print("No Optimizer or Cost function is selected. Check lists of available optimizers and cost functions") 