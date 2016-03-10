********************************************
* README file for replicating the various 
* GARCH models;
********************************************

This file describes how to run each of the 
GARCH models described below.

Three models are estimated:
  
  * The GARCH(1,1) model of Bollerslev(1986) using analytical derivatives 
    as shown in Fiorentini, Calzolari, and Panattoni (1996). The benchmark 
    model of McCullough and Renfro (1998) is estimated using data from 
    Bollerslev and Ghysels (1996). 

  * The Spline-GARCH model of Engle and Rangel (2008).  The benchmark models
    come from the OxMetrics 7.0 documentation and the Engle and Rangel paper.

  * The MIDAS-GARCH model of Engle, Rangel, and Sohn (2013). The benchmark is 
    the matlab code provided by the authors at the Review of Economic Statistics.

Most models calculate Bollerslev-Wooldridge (1992) standard errors unless otherwise noted.

To run the code:
 All the datasets are included. Some programs require the sas7bdat package to load the data.
 The code should be self contained with all functions and output in a single file.
 The MIDAS-GARCH package requires the intnx function included in intnx.py

garch_11.py runs the GARCH(1,1) model

spgarch_er.py runs the Spline-GARCH model of Engle and Rangel (2008) on daily S&P 500 returns 

spgarch_ox.py runs the Spline-GARCH model for Nasdaq data from Laurent (2012).

midas_garch_egs.py runs the MIDAS-GARCH model for fixed window regressions on daily S&P 500 returns



More information on the estimated models are available in the references:

Bollerslev, Tim (1986). "Generalized Autoregressive Conditional Heteroskedasticity."
   Journal of Econometrics 31(3), pp. 307-327.

Bollerslev, Tim and Eric Ghysels (1996). "Periodic Autoregressive Conditional Heteroskedasticity."
  Journal of Business & Economic Statistics 14(2), pp. 139-151.

Bollerslev, Tim and Jeffery Wooldrige (1992). "Quasi-Maximum Likelihood Estimation and Inference 
  in Dynamic Models with Time-Varying Covariances." Econometric Reviews 11(2), pp. 143-172.

Engle, Robert, Eric Ghysels and Bumjean Sohn (2013). "Stock Market Volatility and Macroeconomic 
  Fundamentals." Review of Economics and Statistics 95(3), pp. 776-797.

Engle, Robert and Jose Gonzalo Rangel (2008). "The Spline-GARCH Model for Low-Frequency Volatility 
  and Its Global Macroeconomic Causes."     Review of Financial Studies 21(3), pp. 1187-1222.

Fiorentini, Gabriele, Giorgio Calzolari and Lorenzo Panattoni (1996). "Analytic Derivatives and 
  the Computation of GARCH Estimates." Journal of Applied Econometrics 11(4), pp. 399-417.

Laurent, Sebastien (2012). "G@RCH 7.0: Estimation and Forecast of Univariate and Multivariate 
  ARCH-type Models." Oxmetrics Technologies.

McCullough, BD and Charles G. Renfro (1998). "Benchmarks and Software Standards: A Case Study 
  in GARCH Procedures." Journal of Economic and Social Measurement 25(2), 59-71.
