import numpy as np
from scipy.signal import correlate
from scipy.signal import convolve
from scipy.optimize import minimize
from numpy.linalg import norm

def ApplyFilter(x, h):
    #scipy.signal.convolve might use FFT method if it is faster
    #than the direct method
    return convolve(x, h)

def FixBounds(y, d):
    if len(y) < len(d):
        temp = np.zeros(y)
        temp[:len(y)] = y
        y = temp
    elif len(y) > len(d):
        y = y[:len(d)]
    return y

def FilterWithBounds(x, h, d):
    y = ApplyFilter(x, h)
    return FixBounds(y, d)

#This function is for comparison
#Both functions output similar results
def OptimizeUsingLibraryFunction(x, d, N):
    h0 = np.zeros(N)
    h = minimize(lambda h: norm(d - FilterWithBounds(x, h, d)) ** 2, h0).x
    return h, FilterWithBounds(x, h, d)

def LMS(x, d, N, delta = 0.001, number_of_iterations = 1000):
    #Performing gradient descent
    h = np.zeros(N)
    for i in range(number_of_iterations):
        e = d - FilterWithBounds(x, h, d)
        step_direction = correlate(e, x, mode = "full", method = "auto")
        pivot = max(len(e), len(x)) - 1
        step_direction = step_direction[pivot:]
        step_direction = step_direction[:N]
        h += delta * step_direction
    return h, FilterWithBounds(x, h, d)