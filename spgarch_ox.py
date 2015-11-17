#-----------------------------------------
# code to recreate the spline garch model
# Example is drawn from the Oxmetrics 7.0
# G@RCH package using NASDAQ returns
#-----------------------------------------

#-------------------------------------------
# import necessary packages and functions
#-------------------------------------------
import numpy as np
import pandas as pd
from math import pi, log, isnan, isinf
from matplotlib import pyplot as plt
from scipy.optimize import minimize, fmin_slsqp
from scipy.linalg import inv

#-----------------------------
# read in the data
# data is from Oxmetrics 7.0
#-----------------------------
nasdaq = pd.read_csv('nasdaq_ox_example.csv', \
                    index_col=1, parse_dates=True)
nasdaq_rtn = np.asarray(nasdaq['Nasdaq'])

#----------------------------
# lag array helper function
#----------------------------
def lag_array(series, lag, c=1):
    
    # begin buildinglags
    T = series.size

    final = np.array([])
    
    if type(lag)==int:
        for i in xrange(1,lag+1):
            if i==1:
                final = series[lag-i:-i][np.newaxis].T
            else:
                final = np.hstack((final,series[lag-i:-i][np.newaxis].T))

    elif type(lag)==list:
        count = 0
        for i in lag:
            if count==0:
                final = series[max(lag)-i:-i][np.newaxis].T
            else:
                final = np.hstack((final,series[max(lag)-i:-i][np.newaxis].T))
            count += 1
            
    if c==1 and final.size > 0:
        final = np.hstack((np.ones(len(final))[np.newaxis].T, final))

    elif c==1 and final.size == 0:
        final = np.ones(T)

    return final

#--------------------------
# determine the AR errors
#--------------------------
def AR_err(series, b, ar=[], c=0):
    X = lag_array(series, ar, c)
    T = X.shape[0]
    Y = series[-T:]
    e = Y - np.dot(X,b.T)
    return e,T

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

#-----------------------------------------------
# Two-sided jacobian
# Alternative two-sided jacobian
# step size is modeled on Eviews documentation
#-----------------------------------------------
def jacobian(parms, series, c, ar=[], knots=0, full_output=False):
    
    r = 1.49e-8 # relative step size (sq. root of machine epsilon)
    m = 10.0e-10 # minimum step size

    # set up empty vectors with steps and gradients
    s = np.zeros(len(parms),float)
    grad = np.zeros(len(parms),float)

    # calculate the gradients
    for i in xrange(len(parms)):
        s[i] = max(r*parms[i], m)
        
        loglik_plus  = __loglike__(parms+s, series=series, c=c, ar=ar, knots=knots, full_output = False)
                                   
        loglik_minus = __loglike__(parms-s, series=series, c=c, ar=ar, knots=knots, full_output = False)
                                   
        grad[i] =(loglik_plus - loglik_minus)/(2*s[i])
        s[i] = 0.0
        
    return grad

#-------------------------
# T x k gradient matrix
# from Sheppard (2014)
#-------------------------
def gradient(__loglike__, k, epsilon=1e-5):
    
    loglik, logliks, e, tau, gt, ht, T =__loglike__(result['x'], np.array(nasdaq_rtn), 1, [1], 2, full_output=True)
    
    step = logliks*epsilon
    scores = np.zeros((T, k))
    for i in xrange(k):
        h = step[i]
        delta = np.zeros(k)
        delta[i] = h
        loglik, logliksplus, e, tau, gt, ht, T =__loglike__(result['x'] + delta, np.array(nasdaq_rtn), 1, [1], 2, full_output=True)
        loglik, logliksminus, e, tau, gt, ht, T = __loglike__(result['x'] - delta, np.array(nasdaq_rtn), 1, [1], 2, full_output=True)
        scores[:,i] = (logliksplus - logliksminus)/(2*h)

    return scores


#-----------------------------
# 2-sided Hessian function
# from Sheppard (2014)
#------------------------------
def hessian_2sided(fun, theta, args, epsilon=1e-05):
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

#--------------------------------------
# define the non-negativity constraint
# Tried this; doesn't work
#--------------------------------------
def sp_constraint(parms, series, c, ar, knots, full_output=False):

    # break up the parameter vector
    b = parms[:len(ar)+c]
    tau_parms = parms[-knots-2:]
    g_parms = parms[:-knots-2][-2:]

    # AR model errors
    e, T = AR_err(series, b, ar, c)

    # determine the spline
    tau = spline(tau_parms, T, knots)

    # check non-negative values
    non_neg =  1.0*(tau <=0).sum()

    return non_neg

#------------------------------
# determine the log likelihood
#------------------------------
def __loglike__(parms, series, c, ar, knots, full_output=False):

    # break up the parameter vector
    b = parms[:len(ar)+c]
    tau_parms = parms[-knots-2:]
    g_parms = parms[:-knots-2][-2:]

    # AR model errors
    e, T = AR_err(series, b, ar, c)

    # squared residuals
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
    #loglik = logliks.sum()
    
    if isnan(logliks.sum()) or isinf(logliks.sum()):
        loglik = 1E10
    else:
        loglik = logliks.sum()

    if full_output == True:
        return -loglik, logliks, e, tau, gt, ht, T

    else:
        return loglik

#-------------------------------
# initial values from oxmetrics
#-------------------------------
x0 = np.array([0.01, 0.01, 0.1, 0.8, 1.59189, 0.0, 0.0, 0.0])

#---------------------
# estimate the model
#---------------------
# use the two-sided jacobian add jac=jacobian
result = minimize(__loglike__, x0=x0, method='SLSQP',\
                  args = (np.array(nasdaq_rtn), 1, [1], 2)) ## Sequential Least Sq

#result = fmin_slsqp(__loglike__, x0=x0, full_output=True, \
 #                 args = (np.array(nasdaq_rtn), 1, [1], 2)) ## main SLSQP function


# recover the components
loglik, logliks, e, tau, gt, ht, T = __loglike__(result['x'], np.array(nasdaq_rtn), 1, [1], 2, full_output=True)

#----------------------------
# standard errors
#----------------------------
scores = gradient(__loglike__, result['x'].size)
H = hessian_2sided(__loglike__, result['x'], (np.array(nasdaq_rtn), 1, [1], 2, False))

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

#-------------------------
# plot the NASDAQ returns
#-------------------------
plt.plot(nasdaq['Nasdaq'], 'b-')
plt.show()

#------------------
# plot the results
#------------------
t = np.array([t for t in xrange(T)])

# long run variance component
plt.plot(t, tau,'r-')
plt.show()

# short-run component
plt.plot(t, gt, 'r-')
plt.show()

# total variance
plt.plot(t, gt*tau, 'r-')
plt.show()