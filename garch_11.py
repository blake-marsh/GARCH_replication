#-----------------------------------------------
# program to replicate the model
# of McCullough and Renfro (1998)
# using data from Bollerslev and Ghysels (1996)
# data can be found at:
# http://www.econometrics.com/intro/garchdat.htm (access: July 8, 2015)
#-----------------------------------------------

# import packages
from scipy.optimize import minimize
from math import pi, log
from scipy.linalg import inv
import numpy as np

#--------------------
# read in the data
#--------------------
rawdata = open('DMBP.txt', 'r')
DMBP = np.zeros((1,2))
i = 0
for line in rawdata:
    if i <= 1973:
        if i != 0:
            DMBP = np.vstack((DMBP, np.zeros((1,2))))
        for column in xrange(len(line.split())):
            DMBP[i,column] = line.split()[column]
        i+=1
rawdata.close()

series = DMBP[:,0]

#----------------------------------
# helper function to generate lags
#----------------------------------
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


#-------------------------
# log likelihood function
#-------------------------
def loglike(parms, series, full_output=False):

    # model errors
    e = series - parms[0]
    T = e.size

    # squared errors
    e2 = e**2
    
    # initial condition
    ht = np.array([e2.sum()/T])
    e2 = np.append(ht[0], e2)

    for t in xrange(1, T+1):
        ht = np.append(ht, parms[1] + parms[2]*e2[t-1] + parms[3]*ht[t-1])

    logliks = 0.5*(log(2*pi)+np.log(ht[1:])+(e2[1:]/ht[1:]))
    loglik = logliks.sum()

    if full_output==True:
        return -loglik, e, e2, ht, T
    
    else:
        return loglik

#------------------
# initial values
#------------------
x0 = np.array([ 0.0,  0.17206402, 0.22294213, 0.0])

#----------
# result
#----------
result = minimize(loglike, x0=x0.tolist(), method = 'SLSQP', \
                  args = (series, False), \
                  bounds = [(None,None),] + [(0,None),]*3)

#-----------------------
# analytical gradients
#-----------------------
    
def __gradient__(parms, series, dist = "normal", lamb=1.0, full_output=False, \
                ar = [], sar = [], ma = [], sma = [], x = [], c=0, \
                arch = [], garch = [], garch_x = []):

    loglik, err, err2, ht, T = loglike(parms, series, full_output=True)

    #----------------------------------
    # build various matrices for later
    #----------------------------------
    ## AR data matrix
    x_mat = lag_array(series, ar+sar, c=c) ## append the errors and the x matrix here
    x_mat = np.resize(x_mat,(x_mat.size,1))
    ## concatentate the intercept, squared errors, and conditional variance
    z = np.hstack((lag_array(err2, max(arch),c=1), lag_array(ht, max(garch),c=0)))
    err2 = err2[max(arch):][np.newaxis].T
    ## resize the error and ht matrices
    ht = ht[-x_mat.shape[0]:][np.newaxis].T
    err = err[np.newaxis].T
    
    #----------------
    # AR gradients
    #----------------
    xe = x_mat*err
    xe_mean = np.mean(xe,axis=0) 
    xe = np.vstack((np.zeros((len(arch),x_mat.shape[1])), x_mat*err))
    dhdb = np.zeros((max(garch)+x_mat.shape[0],x_mat.shape[1])) # appending zeros
    betas = np.array(parms[-max(garch):len(parms)-len(garch_x)]) # variance parameters
    alphas = parms[-(max(arch)+max(garch)):len(parms)-max(garch)-len(garch_x)]

    # find the derivative of ht with respect to the mean parameters
    for t in xrange(dhdb.shape[0]):
        lags = dhdb[:t+1,:][t-max(garch)][np.newaxis]
        indc = 1*(t - (np.arange(len(alphas))+1) > 0)
        if t ==0:
            dhdb[t] = np.multiply(-2.0,alphas*((xe[:t+1,:][-len(arch):,:])**(indc))*xe_mean**(1-indc))
        else:
            dhdb[t] = np.multiply(-2.0,alphas*((xe[:t,:][-len(arch):,:])**(indc))*xe_mean**(1-indc))+np.dot(betas,lags)

    dhdb = dhdb[max(garch):,:]
    dldb = ((x_mat*err)/ht)+0.5*(1/ht)*dhdb*(((err**2)/ht)-1)

    #-------------------
    # Variance gradients
    #--------------------
    dhdw = np.vstack((np.zeros((max(garch), z.shape[1])),z)) # appending zeros
    betas = np.array(parms[-max(garch):len(parms)-len(garch_x)]) # variance parameters

    # apply the formula for the score (gradient) vector
    # omits any mean equation variables for now
    for t in xrange(max(garch),dhdw.shape[0]):
        lags = dhdw[t-max(garch):t,:]
        dhdw[t] = dhdw[t] + np.dot(betas[::-1],lags)

    dhdw = dhdw[max(garch):,:]

    dldw = 0.5*(1/ht)*dhdw*((err2/ht)-1)

    g = np.hstack((dldb,dldw))

    ## Information matrix
    d2hdw = np.dot(dhdw.T,0.5*(1/(ht**2))*dhdw)
    d2hdb = np.dot(x_mat.T,(1/ht)*x_mat)+ np.dot(dhdb.T,0.5*(1/(ht**2))*dhdb)

    I = np.vstack((np.hstack((d2hdb, np.zeros((d2hdb.shape[0],d2hdw.shape[1])))), np.hstack((np.zeros((d2hdw.shape[0], d2hdb.shape[1])), d2hdw))))

    if full_output==True:
        return loglik, err, err2, ht, g, I, T
    else:
        return g

#----------------
# Final outputs
#----------------
# coefficients given by result['x'] object

# calculate the gradients
loglik, err, err2, ht, g, I, T = __gradient__(result['x'], series, c=1, arch=[1], garch=[1], full_output=True)

# Bollerslev-Wooldridge standard errors
OP = np.dot(g.T,g)
vcv = np.dot(inv(I), np.dot(OP,inv(I)))
se = np.diag(vcv)**0.5
t_stat = result['x']/se

