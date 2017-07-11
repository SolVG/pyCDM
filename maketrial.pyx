# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:05:07 2016

@author: solveiga
"""
import cython
import numpy as np
cimport numpy as np
from scipy import stats
from scipy.stats import beta
from joblib import Parallel, delayed

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

# creating a function to compute a single trial from conflict diffusion model
# parameters that the function takes in are: 
# betaparam:    upper threshold
# t0:           non-decision time
# st0:          standard deviation of non-decision time
# A:            amplitude of automatic activation
# vc:           drift rate for controlled process
# tau:          time to peak automatic activation
# alpha:        shape of starting point distribution
# tmax:         max length of decision process in milliseconds
# aa:           shape of automatic activation
# s:            diffusion constant
# dt:           sampling rate (e.g. 1 would mean 1 sample per millisecond, 0.1 means 10 samples per millisecond, etc)
# seed:         seed for multithreading to multiple cores, set to None if single core is used 
def maketrialcdm(float betaparam, float t0, float st0, float A, float vc,
                     float tau, float alpha, float tmax, float aa, float s,
                     float dt, int seed):
    
    # set seed to allow multicore threading 
    np.random.seed(seed)
    
    # create array of time samples
    cdef np.ndarray[DTYPE_t, ndim=1] t = np.arange(dt,tmax+dt, dt)

    # apply automatic activation and drift rate for controlled process
    cdef np.ndarray[DTYPE_t, ndim=1] mu = A*np.exp(-t/tau)*((t*np.exp(1)/
                   (tau*(aa-1)))**(aa-1))*((aa-1)/t-1/tau)+vc
    
    # apply random noise to array
    cdef np.ndarray[DTYPE_t, ndim=1] dX = mu*dt+s*np.sqrt(dt)*np.random.randn(len(t))
    
    # randomly select starting point
    cdef float start = beta.rvs(alpha, alpha, loc=-betaparam, 
                                scale=betaparam*2, random_state=None)
    
    # randonly select non-decision time
    cdef float nondestime = st0 * np.random.randn() + t0
    
    # compute cumulative sum of dX array
    cdef np.ndarray[DTYPE_t, ndim=1] X = np.cumsum(dX) + start

    # predefine variables for speed
    cdef float rt
    cdef int i
    
    # try-catch loop that finds at which point the decision has crossed threshold,
    # if threshold is not crossed, then nan is returned
    try:
            i   = np.where(abs(X) - betaparam > 0)[0][0]
            rt  = nondestime + i*dt
            rt = rt * np.sign(X[i])
        except:
            rt  = np.nan

        return (rt)
