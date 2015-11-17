#---------------------------------
# code to replicate the S&P 500
# spline garch model of Engle and
# Rangel (2008)
#---------------------------------

#--------------------
# import packages
#--------------------
import numpy as np
import pandas as pd
from sas7bdat import SAS7BDAT as sas
from math import pi, log, isnan, isinf
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import inv
import datetime

#--------------------------
# read in the data;
#--------------------------
data = sas('sp500.sas7bdat')
data = data.to_data_frame()
data = data.set_index(data['date'])
# subset the data
sp500_rtn = np.asarray(data[datetime.date(1955,01,03):datetime.date(2004,12,31)]['return'])

#------------------------
# determine the spline
#------------------------
def spline(tau_parms, T, knots):

    trend = np.array([t for t in xrange(1,T+1)])
    tau = np.resize(np.repeat(trend,knots),(T,knots))
    pts = np.array([round(T/knots*i,0) for i in xrange(knots)])
    factors = 1.0*(tau - pts > 0)
    tau = np.hstack(((trend*tau_parms[1]/T)[np.newaxis].T, tau_parms[2:]*factors*((tau - pts)/T)**2)) # scaled spline from Oxmetrics
    #tau = np.hstack(((trend*tau_parms[1])[np.newaxis].T, tau_parms[2:]*factors*((tau - pts))**2)) # Engle-Rangel (2009) spline
    tau = tau_parms[0]*np.exp(np.sum(tau,axis=1))
    return tau

#-----------------------------------
# Alternative two-sided jacobian
# with step sizes modeled on Eviews
# documentation
#-----------------------------------
def jacobian(parms, series, knots=0, full_output=False):
    
    r = 1.49e-8 # relative step size (sq. root of machine epsilon)
    m = 10.0e-10 # minimum step size

    # set up empty vectors with steps and gradients
    s = np.zeros(len(parms),float)
    grad = np.zeros(len(parms),float)

    # calculate the gradients
    for i in xrange(len(parms)):
        s[i] = max(r*parms[i], m)
        
        loglik_plus  = __loglike__(parms+s, series=series, knots=knots, full_output = False)
                                   
        loglik_minus = __loglike__(parms-s, series=series, knots=knots, full_output = False)
                                   
        grad[i] =(loglik_plus - loglik_minus)/(2*s[i])
        s[i] = 0.0
        
    return grad

#-------------------------
# T x k gradient matrix
# from Sheppard (2014)
#-------------------------
def gradient(__loglike__, k, epsilon=1.49e-8):
    
    loglik, logliks, e, tau, gt, ht, T =__loglike__(result['x'], np.array(sp500_rtn), 7, full_output=True)
    
    step = logliks*epsilon
    scores = np.zeros((T, k))
    for i in xrange(k):
        h = step[i]
        delta = np.zeros(k)
        delta[i] = h
        loglik, logliksplus, e, tau, gt, ht, T =__loglike__(result['x'] + delta, np.array(sp500_rtn), 7, full_output=True)
        loglik, logliksminus, e, tau, gt, ht, T = __loglike__(result['x'] - delta, np.array(sp500_rtn), 7, full_output=True)
        scores[:,i] = (logliksplus - logliksminus)/(2*h)

    return scores


#-----------------------------
# 2-sided Hessian function
# from Sheppard (2014)
#------------------------------
def hessian_2sided(fun, theta, args, epsilon=1e-5):
    f = fun(theta, *args)
    h = epsilon*np.abs(theta)
    thetah = theta + h
    h = thetah - theta
    K = np.size(theta,0)
    h = np.diag(h)
    
    fp = np.zeros(K)
    fm = np.zeros(K)
    for i in xrange(K):
        fp[i] = fun(theta+h[i], *args)
        fm[i] = fun(theta-h[i], *args)
        
    fpp = np.zeros((K,K))
    fmm = np.zeros((K,K))
    for i in xrange(K):
        for j in xrange(i,K):
            fpp[i,j] = fun(theta + h[i] + h[j],  *args)
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(theta - h[i] - h[j],  *args)
            fmm[j,i] = fmm[i,j]
            
    hh = (np.diag(h))
    hh = hh.reshape((K,1))
    hh = np.dot(hh,hh.T)
    
    H = np.zeros((K,K))
    for i in xrange(K):
        for j in xrange(i,K):
            H[i,j] = (fpp[i,j] - fp[i] - fp[j] + f 
                       + f - fm[i] - fm[j] + fmm[i,j])/hh[i,j]/2
            H[j,i] = H[i,j]
    
    return H

#------------------------------
# determine the log likelihood
#------------------------------
def __loglike__(parms, series, knots, full_output=False):

    # break up the parameter vector
    tau_parms = parms[-knots-2:]
    g_parms = parms[:2]

    T = series.size

    #squared residuals
    e = series
    e2 = e**2

    # determine the spline
    tau = spline(tau_parms, T, knots)

    # determine the short run component
    alpha = g_parms[0]
    beta = g_parms[1]
    gt = np.array([tau[0]])
    for t in xrange(1,tau.size):
        gt = np.append(gt, (1-alpha-beta) + alpha*(e2[t-1]/tau[t-1]) + beta*gt[t-1])

    # conditional variance
    ht = np.multiply(tau,gt)

    # log likelihood
    logliks = 0.5*(np.log(2*pi)+np.log(ht)+(e2/ht))
    
    if isnan(logliks.sum()) or isinf(logliks.sum()):
        loglik = 1.0E10
    else:
        loglik = logliks.sum()

    if full_output == True:
        return -loglik, logliks, e, tau, gt, ht, T

    else:
        return loglik

#-----------------
# initial values
#-----------------
x0 = np.array([0.1, 0.8, np.mean(sp500_rtn**2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

#-----------------------
# estimate the model
#-----------------------
result = minimize(__loglike__, x0=x0, method='SLSQP',\
                  args = (np.array(sp500_rtn), 7)) ## Sequential Least Sq


# recover the components
loglik, logliks, e, tau, gt, ht, T = __loglike__(result['x'], np.array(sp500_rtn), 7, full_output=True)

#----------------------------
# standard errors
#----------------------------
scores = gradient(__loglike__, result['x'].size)
H = hessian_2sided(__loglike__, result['x'], (np.array(sp500_rtn), 7))

# outer product of gradient standard errors
OPG = np.dot(scores.T,scores)/T
vcv_opg = inv(OPG)/T
se_opg = np.diag(vcv_opg)**0.5
t_opg = result['x']/se_opg

# second derivatives
vcv_H = inv(H/T)/T
se_H = np.diag(vcv_H)**0.5
t_H = result['x']/se_H

# sandwhich form
vcv_bw = np.dot(inv(H/T), np.dot(OPG,inv(H/T)))/T
se_bw = np.diag(vcv_bw)**0.5
t_bw = result['x']/se_bw

#-----------------
# plot results
#-----------------
# plot the result
t = np.array([t for t in xrange(T)])
plt.plot(t, gt, 'b--', tau,'r-')
plt.show()


