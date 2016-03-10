#---------------------------------------------------------------------------------------------------------
# estimates the MIDAS-GARCH model of
# Engle, Ghysels, and Sohn (2013).
# Adapted from matlab code at
# The Review of Economics and Statistics
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/27513 (accessed: Mar 4, 2016)
#---------------------------------------------------------------------------------------------------------

# import modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sas7bdat import SAS7BDAT as sas
from scipy.optimize import minimize
from math import pi, isnan, isinf, sqrt
from scipy.linalg import inv
from scipy import stats

# read in a date helper function
# modeled on sas intnx function
execfile("intnx.py")

#------------------
# read in the data
#------------------
# S&P 500 returns
daily = sas('sp500.sas7bdat')
daily = daily.to_data_frame()
daily = daily[1:]

#----------------------
# lag helper function
#----------------------
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

#----------------------------------
# function to calculate
# fixed width realized volatility
#----------------------------------
def fixed_RV(var, dtvar, freq):
    """
    takes a variable input with a matching list of dates
    and outputs a realized variance as the sum of the squares
    over the interval given by freq (month, quarter, etc)

    returns dates at end of period frequency, realized variance,
    and number of days in the period.
    """
    
    # convert to numpy arrays
    var = np.asarray(var)
    dtvar = np.asarray(dtvar)
    
    freq_dt = np.unique(intnx(freq, dtvar, 0, 'e'))
    RV = np.array([])
    ndays = np.array([])
    for dt in freq_dt:
        temp = np.nansum(var[np.where(dtvar==dt)]**2)
        count = var[np.where(dtvar==dt)].size

        # append to final vectors
        RV = np.append(RV, temp)
        ndays = np.append(ndays, count)
        
    final = np.hstack((np.hstack((freq_dt[np.newaxis].T, RV[np.newaxis].T)), ndays[np.newaxis].T))
    
    return final

#---------------------------------
# define the beta lag polynomial
#---------------------------------
def betapolyn(K, j, w):
    K = float(K); w = float(w)
    j_vec = np.arange(1,K, dtype=float)[::-1]
    N = np.dot(np.ones(K-1,dtype=float),(1.0-j_vec/K)**(w-1.0))
    s = (1.0-j_vec/K)**(w-1.0)/N
    return s

#--------------------------------------
# replicating the fixed period models
#--------------------------------------
def midas_data(data, dates, freq, K):

    """
    Inputs:

    data - array like data input matrix
    array of datetimes
    freq - length that tau should be fixed
    K - number of MIDAS lags to include
    """

    data = np.asarray(data)
    dates = np.asarray(dates)

    # determine the realized variance (fixed-window)
    rv = fixed_RV(data, dates, freq)

    # parse out the results array
    day_cnt = rv[1:,2]
    avg_days = np.mean(day_cnt)
    dates = rv[1:,0]
    rv = rv[1:,1]

    # determine the lag matrix
    ## first generate a lagged matrix of RVs
    lag_rv = lag_array(rv, K, c=0)[:,::-1][:,1:]
    
    ## now repeat each row for the number of days
    for i in xrange(lag_rv.shape[0]):
        nrows = day_cnt[K-1+i]
        if i==0:
            X = np.tile(lag_rv[i,:],(nrows,1))
        else:
            temp = np.tile(lag_rv[i,:],(nrows,1))
            X = np.vstack((X, temp))

    return rv, X

#-------------------------
# T x k gradient matrix
# from Sheppard (2014)
#-------------------------
def gradient(__loglike__, k, epsilon=1.49e-8):
    
    loglik, logliks, e, tau, gt, ht, T =__loglike__(result['x'], data, X, 36, True)
    
    step = logliks*epsilon
    scores = np.zeros((T, k))
    for i in xrange(k):
        h = step[i]
        delta = np.zeros(k)
        delta[i] = h
        loglik, logliksplus, e, tau, gt, ht, T =__loglike__(result['x'] + delta, data, X, 36, full_output=True)
        loglik, logliksminus, e, tau, gt, ht, T = __loglike__(result['x'] - delta, data, X, 36, full_output=True)
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


#-----------------
# log likelihood
#-----------------
def __loglike__(parms, data, X, K, full_output=False):

    # define the tau component
    tau = parms[-1]**2 + (parms[-3]**2)*(np.dot(X ,np.resize(betapolyn(K, np.arange(1,K), parms[-2]),(K-1,1))))
    tau = tau[:,0].astype(float)
    T = tau.size

    ## define the squared errors
    e = data[-T:] - parms[0]
    e2 = e**2

    # define g
    alpha = parms[1]
    beta = parms[2]
    gt = np.array([tau[0]])
    for t in xrange(1,tau.size):
        gt = np.append(gt, (1 - alpha - beta) + alpha*(e2[t-1]/tau[t-1]) + beta*gt[t-1])

    ht = tau*gt

    # log likelihood
    logliks = 0.5*(np.log(2*pi)+np.log(ht)+(e2/ht))
    
    if isnan(logliks.sum()) or isinf(logliks.sum()):
        loglik = 1E10
    else:
        loglik = logliks.sum()

    #print loglik
    
    if full_output == True:
        return -loglik, logliks, e, tau, gt, ht, T

    else:
        return loglik

#---------------------
# estimate the model
#---------------------
data = np.asarray(pd.DataFrame(daily['return']))[1:,0]
dates = np.asarray(pd.DataFrame(daily.date).date)[1:]

# define the RV and X matrix
rv, X = midas_data(data, dates, 'month', 36)

## initial values
# [mu, alpha, beta, theta, w, m]
x0 = np.array([0.0005, 0.08, 0.90, 0.1, 5, 0.0001])

# estimate
#result = minimize(__loglike__, x0=x0, method='SLSQP', \
#                  args = (data, X, 36, False)) ## Sequential Least squares

result = minimize(__loglike__, x0=x0, method='L-BFGS-B', \
                  args = (data, X, 36, False)) ## Limited Memory BFGS

# recover the components
loglik, logliks, e, tau, gt, ht, T = __loglike__(result['x'], data, X, 36, \
full_output=True)

#----------------------------
# standard errors
#----------------------------
scores = gradient(__loglike__, result['x'].size)
H = hessian_2sided(__loglike__, result['x'], (data, X, 36, False))

# outer product of gradient standard errors
OPG = np.dot(scores.T,scores)/T
vcv_opg = inv(OPG)/T
se_opg = np.diag(vcv_opg)**0.5
t_opg = result['x']/se_opg

# second derivatives
## Note: Inverse Hessian as used by RGS in the matlab code
vcv_H = inv(H/T)/T
se_H = np.diag(vcv_H)**0.5
t_H = result['x']/se_H
pvalues_H = stats.t.sf(np.abs(t_H),data.size-len(result['x']))

# sandwich form (Bollerslev-Wooldrige)
vcv_bw = np.dot(inv(H/T), np.dot(OPG,inv(H/T)))/T
se_bw = np.diag(vcv_bw)**0.5
t_bw = result['x']/se_bw
pvalues_bw = stats.t.sf(np.abs(t_bw),data.size-len(result['x']))

#-------------------
# plot the results
#------------------
t = np.array([t for t in xrange(T)])
plt.plot(t, np.sqrt(252*ht), 'g--', np.sqrt(252*tau),'b-')
plt.show()