#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 2018
@author: alexander.winkler@mpimet.mpg.de
Title: Optimal Fingerprinting after Ribes et al., 2009
"""

## import modules
import numpy as np
import scipy.linalg as spla
import scipy.stats as sps

## define functions
##
def eigvalvec(C):
    """
    Eigenvalue / Eigenvector calculation
    """
    ## Compute eigenvalues and eigenvectors
    eigval0, eigvec0 = spla.eigh(C) 
    ## Take real part (to avoid numeric noise, e.g. small complex numbers)
    if np.max(np.imag(eigval0))/np.max(np.real(eigval0)) > 1e-12:
        print("Matrix is not symmetric")
        return
    ## Check that C is symmetric (<=> real eigen-values/-vectors)
    eigval1 = np.real(eigval0)
    eigvec1 = np.real(eigvec0)
    ## Sort in a descending order
    dorder = np.argsort(eigval1)[::-1]
    eigvec = np.flipud(eigvec1)[:, dorder]
    eigval = eigval1[dorder]
    return eigval, eigvec

##
def projfullrank(t, s):
    """
    Projection on full rank matrix
    Data array X with shape (t, s) stores observations for t years at s spatial points.
    Using this P, it can be converted into a vector that has had its temporal mean removed and then flattened.
    """
    ## M: the matrix corresponding to the temporal centering
    M = np.eye(t, t) - np.ones([t, t])/float(t)
    ## Eigen-values/-vectors of M; note that rk(M)=T-1, so M has one eigenvalue equal to 0.
    eigval, eigvec = eigvalvec(M)
    ## (T-1) first eigenvectors (i.e., the ones corresponding to non-zero eigenvalues)
    eigvec = eigvec[:, :t-1].T
    ## The projection matrix P, which consists of S replications of U.
    P = np.zeros([(t-1)*s, t*s])
    for i in range(s):
        P[i:(t-1)*s:s, i:t*s:s] = eigvec
    return P

##
def total_wave_number(n):
    """
    Calculation of wave number
    Weighting of data vectors by spherical harmonics, a common approach in climate signal detection and attribution,
    especially when dealing with global spatial data, to account for the varying area of grid cells at different latitudes.
    This code follows a weighting scheme from Stott & Tett (1998), aiming to spatially weight signal and noise
    according to the wave number of spherical harmonics.
    """
    nr = (n+1)**2
    l = np.zeros(nr)
    ir = 1
    l[:n+1] = range(1, n+2)
    ir = ir+n
    for i in range(2, n+2):
        for j in range(i, n+2):
            l[ir] = j
            l[ir+1] = j
            ir = ir+2
    return l

##
def regC(X):
    """
    Calculation of regularized covariance matrix
    Input: X (n, p), n samples, p features
    Output: returns a p×p regularized covariance matrix
    The original covariance matrix may be unstable, ill-conditioned, or non-invertible;
    the regularized covariance matrix is more stable and improves numerical stability and accuracy in subsequent steps
    such as eigendecomposition, detection, and attribution.
    The Ledoit & Wolf method is an unbiased and consistent shrinkage estimator.
    """
    # just to be sure it is a matrix object
    X = np.matrix(X)
    n, p = np.shape(X)
    # Sample covariance
    CE = X.T * X / float(n)		# standard sample covariance matrix

    Ip = np.eye(p, p)  # identity matrix
    # First estimate in L&W
    m = np.trace(CE * Ip) / float(p)	# average variance
    XP = CE - m * Ip  # deviation matrix of covariance from target matrix
    # Second estimate in L&W
    d2 = np.trace(XP * XP.T) / float(p) 	# variance of deviation matrix

    bt = []
    for i in range(n):
        Mi = X[i, :].T * X[i, :]
        bt.append(np.trace((Mi - CE) * (Mi - CE).T) / float(p))
    bb2 = 1. / n**2 * np.sum(bt)
    # Third estimate in L&W
    b2 = np.min([bb2, d2])	
    # Fourth estimate in L&W
    a2 = d2 - b2		
    Cr = b2 * m / d2 * Ip + a2 / d2 * CE
    return Cr

##
def extract_Z2(NZ, frac_Z2, sampling_name):
    """
    Z1 and Z2 based on control
    Select a subset of indices from a set of length NZ as the Z2 set, returning an index flag vector Ind_Z2
    indicating whether each position belongs to Z2; the rest belong to Z1.
    The selection method is determined by sampling_name ('segment', 'regular', 'random').
    'segment': sequentially take the first N samples as Z2
    'regular': equal-interval sampling (fixed step); if frac_Z2 = 0.5, select every 2nd sample as Z2
    'random': random sampling; generate random numbers from a normal distribution, sort them, and take the largest NZ2 indices as Z2.
    """
    Ind_Z2 = np.zeros((int(NZ), 1))
    NZ2 = int(np.floor(NZ * frac_Z2))
    if sampling_name == 'segment':
        Ind_Z2[0:NZ2] = 1
        print('Z2 : segment 1-'+str(NZ2)+', fraction ~ '+str(NZ2/NZ))
    elif sampling_name == 'regular':
        ix = []
        a = 0
        while a <= NZ - 1./frac_Z2:
            a += 1./frac_Z2
            ix.append(int(np.floor(a)) - 1)
        Ind_Z2[ix] = 1
        print('Z2 : regular, fraction ~ '+str(sum(Ind_Z2)/NZ))
    elif sampling_name == 'random':
        u = np.random.normal(0, 1, size=(NZ, 1))
        z = np.argsort(u, axis=0)[::-1]
        Ind_Z2[z[0:NZ2]] = 1
        print('Z2 : random, fraction ~ '+str(sum(Ind_Z2)/NZ))
    else:
        print('Unknown sampling_name.')
    return Ind_Z2

##
def gke(d_H0, d):
    """
    Silverman's rule of Thumb
    """
    N = len(d_H0)
    h = 1.06 * np.std(d_H0, ddof=1) * N ** (-1./5)	# Silverman's rule of Thumb
    onem = sps.norm.cdf(d, d_H0, h)
    pvi = 1 - onem
    pv = np.sum(pvi)/N

    return pv

##
def tls(X, Y, Z2, nX, PROJ, Formule_IC_TLS):
    """
    TLS routine
    X, Y, Z2 are anomaly series multiplied by the whitening covariance matrix.
    nX is the number of model/ensemble members for each fingerprint.
    nb_runs_x is the number of model/ensemble members for each fingerprint.
    proj is the initial identity matrix (length of years).
    formule_ic_tls is the method for computing confidence intervals.
    """
    n = Y.shape[1]  # time length 
    m = X.shape[0]  # number of signals 
    # Check sizes of X and Y
    if Y.shape[1] != X.shape[1]:
        print('Error in TLS: size of inputs X, Y.')
        return
    # Normalise the variance of X
    X = np.multiply(X, (np.sqrt(nX).T * np.ones((1, n)))) # Standardize X by multiplying by the standard deviation of each signal to unify scale
    if X.shape[0] == 1: # adjusted
        DnX = np.sqrt(nX).squeeze()
    else:
        DnX = np.diag(np.sqrt(nX).A1)
    # Computation of beta_hat
    #--------------------------
    # TLS fit via svd...
    M = np.vstack([X, Y]) # Stack X and Y
    U, D, V = np.linalg.svd(M) # Perform singular value decomposition on M; the TLS solution is the singular vector corresponding to the smallest singular value
    V = V.T
    
    # Consider the "smallest" singular vector
    Uk = U[:, -1]
    Uk_proj = np.vstack([PROJ * DnX * Uk[:-1], Uk[-1]])
    # Computes beta_hat
    beta_hat = - Uk_proj[:-1] / Uk_proj[-1]  # Take the direction of the smallest singular value to construct the estimate of β
    # instantiate array for beta uncertainty estimates
    beta_hat_inf = np.zeros(beta_hat.shape)
    beta_hat_sup = np.zeros(beta_hat.shape)
    # Reconstructed data
    D_tilde = np.matrix(np.zeros(M.shape))
    np.fill_diagonal(D_tilde, D)
    D_tilde[m, m] = 0
    Z_tilde = U * D_tilde * V.T
    X_tilde = Z_tilde[0:m, :] / (np.dot(np.sqrt(nX).T, np.ones((1, n))))
    Y_tilde = Z_tilde[m, :]  # Remove the influence of the smallest singular value to reconstruct X and Y, obtaining fitted data.
    # Computation of Confidence Intervals
    #--------------------------------------
    # The square singular values (denoted by lambda in AS03)
    d = D**2
    # Computation of corrected singular value (cf Eq 34 in AS03)
    d_hat = np.zeros(d.shape)
    NAZv = Z2.shape[0]
    for i in range(len(d)):
        vi = V[:, i].T
        if Formule_IC_TLS == "AS03":
            # Formula Allen & Stott (2003)
            d_hat[i] = d[i] / np.dot(np.dot(np.dot(vi, Z2.T), Z2 / NAZv), vi.T)
        elif Formule_IC_TLS == "ODP": 
             # Formula ODP (Allen, Stone, etc)
            d_hat[i] = d[i] / np.dot(np.power(vi, 2), np.sum(np.power(Z2, 2), axis=0).T / NAZv)
        else:
            print('tls_v1.sci : unknown formula for computation of TLS CI.')
    # The "last" corrected singular value will be used in the Residual Consistency Check
    d_cons = d_hat[-1]
    # Threshold of the Fisher distribution, used for CI computation (cf Eq 36-37 in AS03)
    seuil_1d = np.sqrt(sps.f.ppf(0.9, 1, NAZv))
    # In order to compute CI, we need to run through the m-sphere (cf Eq 30 in AS03)
    # Number of points on the (m-)sphere...
    npt = 1000	
    if m == 1:
        Pts = np.array([[1], [-1]])
    else:
        Pts_R = np.random.normal(0, 1, size=(npt, m))
        # The points on the sphere
        Pts = Pts_R / (np.sqrt(np.sum(Pts_R ** 2, axis=1).reshape((npt, 1)) * np.ones((1, m))))
    # delta_d_hat provides the diagonal of the matrix used in Eq 36 in AS03
    delta_d_hat = d_hat - np.min(d_hat)
    # following notation of Eq 30 in AS03
    a = seuil_1d * Pts			
    arg_min = np.nan
    arg_max = np.nan
    # Check that 0 is not reached before the last index of delta_d_hat:
    if True not in (delta_d_hat[:-1] == 0):
        b_m1 = a / np.dot(np.ones((Pts.shape[0], 1)), np.sqrt(delta_d_hat[:-1]).reshape((1, delta_d_hat[:-1].shape[0])))
        # following notation of Eq 31 in AS03
        #b_m2 = np.sqrt(1 - np.sum(b_m1**2, axis=1))	
        b_m2 = np.matrix(np.sqrt(1 - np.sum(b_m1**2, axis=1))).T
        # b_m2 need to be strictly positive, otherwise the CI will be unbounded
        if (False in np.isreal(b_m2)) | (True in (b_m2 == 0)) | (True in np.isnan(b_m2)):
            print('Unbounded CI (2)', np.max(np.imag(b_m2)))
            beta_hat_inf += np.nan
            beta_hat_sup += np.nan
        else:
            # Then to obtain CI that include +/- infinity, the computations are made in terms of angles,
            # based on complex numbers (this is a discrepancy with ODP)
            V_pts = np.dot(np.column_stack([b_m1, b_m2]), U.T)
            V_pts_proj = np.column_stack([np.dot(np.dot(V_pts[:, :-1], DnX), PROJ.T), V_pts[:, -1]])
            for i in range(m):
                Vc_2d_pts = V_pts_proj[:, i] + V_pts_proj[:, -1] * 1j 
                Vc_2d_ref = Uk_proj[i] + Uk_proj[-1] * 1j
                Vprod_2d = Vc_2d_pts / Vc_2d_ref
                arg = np.sort(np.imag(np.log(Vprod_2d)), axis=0)
                delta_arg_min = arg[0]
                delta_arg_max = arg[-1]
                Delta_max_1 = np.max(arg[1:] - arg[:-1])
                k1 = np.argmax(arg[1:] - arg[:-1])
                delta_candidate = (arg[0] - arg[-1] + 2 * np.pi).item()
                Delta_max = np.max([Delta_max_1, delta_candidate])
                k2 = np.argmax([Delta_max_1, delta_candidate])
                # Delta_max = np.max([Delta_max_1, arg[0] - arg[-1] + 2 * np.pi])
                # k2 = np.argmax([Delta_max_1, arg[0] - arg[-1] + 2 * np.pi])
                if Delta_max < np.pi:
                    beta_hat_inf[i] = np.nan
                    beta_hat_sup[i] = np.nan
                else:
                    if k2 != 1:
                        print("Warning k2")
                    arg_ref = np.imag(np.log(Vc_2d_ref))
                    arg_min = delta_arg_min + arg_ref
                    arg_max = delta_arg_max + arg_ref
                    beta_hat_inf[i] = -1 / np.tan(arg_min)
                    beta_hat_sup[i] = -1 / np.tan(arg_max)
    else:    
        # If 0 is reached before last index of delta_d_hat, the CI will be unbounded
        print('Unbounded CI (1)')
        beta_hat_inf += np.nan
        beta_hat_sup += np.nan
    return beta_hat, beta_hat_inf, beta_hat_sup, d_cons, X_tilde, Y_tilde

##
def consist_mc_tls(Sigma, X0, nb_runs_X, n1, n2, N, Formula):
    """
    Consistency check TLS
    Null hypothesis: the observation residual can be fully explained by internal variability,
    and the model-simulated responses sufficiently explain the observations.
    """
    # Check that Sigma is a square matrix
    n = Sigma.shape[0]
    if (Sigma.shape[1] != n) | (X0.shape[0] != n):
        print("Error of size in consist_mc_tls.sci")
        return
    # Number of external forcings considered
    k = X0.shape[1]
    # Initial value of beta for the Monte Carlo simulations
    beta0 = np.ones((k,1))
    # Monte Carlo simulations
    #-------------------------
    Sigma12 = spla.sqrtm(Sigma)
    d_cons_H0 = np.zeros((N,1))
    for i in range(N):
        # Virtual observations Y
        Yt = np.dot(X0, beta0)  # Observations under simulated response, a perfect linear combination of the forcings
        Y = Yt + np.dot(Sigma12, np.random.normal(0, 1, size=(n, 1))) # Yt + internal variability (CTL1) + random noise
        # Virtual noised response patterns X
        X = X0 + np.dot(Sigma12, np.random.normal(0, 1, size=(n, k)) / (np.ones(Yt.shape) * np.sqrt(nb_runs_X))) # Also add random noise to CTL1
        # This constructs an ideal world (null hypothesis world): observation Y is fully explained by X0 forcing,
        # and the remainder is simulated internal variability, creating a world where the null hypothesis is true.
        # Variance normalised X
        Xc = np.multiply((np.dot(np.ones(Yt.shape), np.sqrt(nb_runs_X))), X)
        # Virtual independent samples of pure internal variability, Z1 and Z2
        Z1 = np.dot(Sigma12, np.random.normal(0, 1, size=(n, n1))) # Simulated pure internal variability
        Z2 = np.dot(Sigma12, np.random.normal(0, 1, size=(n, n2)))
        # Virtual estimated covariance matrix (based on Z1 only)
        C1_hat = regC(Z1.T) # Estimate covariance from Z1
        C12 = spla.inv(spla.sqrtm(C1_hat))  # Whitening matrix
        # The above steps simulate multiple samples of internal variability, use them for covariance normalization,
        # and then perform total least squares regression to compute the weight in the residual direction for each simulation.
        # The following emulates the TLS algorithm and computes the variable used in the RCC (stored in d_cons_H0). See also tls_v1.sci.
        # Xc and Y are prewhitened
        M = np.dot(C12, np.column_stack([Xc, Y]))
        U, D, V = np.linalg.svd(M.T)
        V = V.T
        d = D**2
        nd = len(d)
        vi = V[:,nd].T

        if Formula == "AS03":
            # Z2 is prewhitened
            Z2w = np.dot(C12, Z2).T
            # Formula Allen & Stott (2003): numerator is the smallest singular value after TLS regression (residual strength),
            # denominator is the expected residual strength estimated from Z2 (internal variability).
            d_cons_H0[i,0] = d[nd-1] / np.dot(np.dot(np.dot(vi, Z2w.T), Z2w / n2), vi.T).item()
        elif Formula == "ODP":
            # Z2 is prewhitened
            Z2w = np.dot(C12, Z2).T
            # Formula ODP (Allen, Stone, etc)
            d_cons_H0[i,0] = d[nd-1] / np.dot(np.power(vi,2), np.sum(np.power(Z2w, 2), axis=0).T / n2).item()
        else:
            print("consist_mc_tls.sci : unknown formula for computation of RCC.")

    return d_cons_H0


## import modules
import numpy as np
import scipy.linalg as spla
import scipy.stats as sps
import PyDnA as pda

def da(y, X, nb_runs_x, ctl, reg, cons_test, formule_ic_tls, sample_extr):
    """
    main detection and attribution routine
    
    """
    #------------- Options
    # Spherical harmonics truncation
    trunc = 0  # global mean; >0 considers latitude-dependent averaging
    # -> for extracting large sample Z into two samples Z1 and Z2
    sampling_name = sample_extr # Method for dividing the CTL matrix
    # fraction of data used in Z2 (the remaining is used in Z1)
    frac_z2 = .5
    #------------- Input parameters
    y = np.matrix(y).T
    X = np.matrix(X).T
    Z = np.transpose(np.matrix(ctl))  # Transpose array
    nb_runs_x = np.matrix(nb_runs_x)  # Number of model/ensemble members for each fingerprint
    # Number of Time steps
    nbts = y.shape[0] # y is an n×1 matrix
    # Spatial dimension
    n_spa = (trunc+1)**2   # Spatial dimension
    # Spatio-temporal dimension (i.e., dimension of y)
    n_st = n_spa * nbts  # Spatio-temporal dimension
    # number of different forcings
    I = X.shape[1]  # Number of fingerprints
    if Z.shape[1] == 1:
        nle = len(Z)
        NZ = nle / n_st
        fl = int(np.floor(Z.shape[0] / y.shape[0]))
        Z = np.transpose(np.reshape(Z[:int(y.shape[0])*fl,], (fl, int(y.shape[0]))))
    else:
        NZ = Z.shape[1]
        
    ## Z1 and Z2 are taken from Z: divide CTL into two samples
    ind_z = pda.extract_Z2(NZ, frac_z2, sampling_name)
    ind_z1 = np.argwhere(ind_z == 0)[:, 0] # Extract even columns
    ind_z2 = np.argwhere(ind_z == 1)[:, 0] # Extract odd columns
    Z1 = Z[:, ind_z1] # Rows are time spans, columns are chunks
    Z2 = Z[:, ind_z2]

    #-------------  Pre-processing
    #------------- Weighting of spherical harmonics (cf Stott & Tett, 1998) for spatial covariance weighting or signal preprocessing
    l = pda.total_wave_number(trunc) 
    p = 1. / np.sqrt(2*l-1)
    p.shape = (p.shape[0], 1)
    pml = np.matrix(np.diag(np.reshape(np.transpose(p*np.ones((1, nbts))), (n_st, 1)).squeeze()))
    y = np.dot(pml, y)  # Multiplied by identity matrix effectively
    Z1 = np.dot(pml, Z1)
    Z2 = np.dot(pml, Z2)
    X = pml*X  # Apply latitude weighting

    # Removing of useless dimensions (equivalent to remove one time step;
    # see scientific documentation, Section 2)
    # Spatio-temporal dimension after reduction
    n_red = n_st - n_spa
    U = pda.projfullrank(nbts, n_spa)  

    ## Project all input data
    yc = np.dot(U, y) # Project observation y into disturbance space, removing temporal mean; yc is the processed "anomaly" data.
    Z1c = np.dot(U, Z1)
    Z2c = np.dot(U, Z2)
    Xc = np.dot(U, X)
    proj = np.identity(X.shape[1]) # Construct an identity matrix as the initial projection matrix

    #------------- Statistical estimation
    ## Regularised covariance matrix improves numerical stability and accuracy in subsequent steps such as eigendecomposition, detection, and attribution.
    Cf = pda.regC(Z1c.T)  # Z1c.T (p,n): p is the number of variables/chunks, n is the number of years
    Cf1 = np.real(spla.inv(Cf))  # Inverse of Cf, take real part
    # Matrix is singular and may not have a square root; can be ignored
    Cf12 = np.real(spla.inv(spla.sqrtm(Cf))) # Inverse square root of Cf for whitening, making noise white
    # Matrix is singular and may not have a square root; can be ignored
    Cfp12 = np.real(spla.sqrtm(Cf)) # Square root of covariance matrix

    if reg == 'OLS':
        ## OLS algorithm
        pv_consist = np.nan
        Ft = np.transpose(np.dot(np.dot(spla.inv(np.dot(np.dot(Xc.T, Cf1), Xc)), Xc.T), Cf1))
        beta_hat = np.dot(np.dot(yc.T, Ft), proj.T)
        ## 1-D confidence intervals
        NZ2 = Z2c.shape[1]
        var_valid = np.dot(Z2c, Z2c.T) / NZ2
        var_beta_hat = np.dot(np.dot(np.dot(np.dot(proj, Ft.T), var_valid), Ft), proj.T)
        beta_hat_inf = beta_hat - sps.t.ppf(0.95, NZ2) * np.sqrt(np.diag(var_beta_hat))
        beta_hat_sup = beta_hat + sps.t.ppf(0.95, NZ2) * np.sqrt(np.diag(var_beta_hat))
        ## Consistency check
        # print('Residual Consistency Check')
        epsilon = yc - np.dot(np.dot(Xc, proj.T), beta_hat.T)
        if  cons_test == "OLS_AT99":
            # Formula provided by Allen & Tett (1999)
            d_cons = np.dot(np.dot(epsilon.T, np.linalg.pinv(var_valid)), epsilon) / (n_red - I)
            pv_cons = 1 - sps.f.cdf(d_cons, n_red - I, NZ2)
        elif cons_test == "OLS_Corr":
            # Hotelling Formula
            d_cons = np.dot(np.dot(epsilon.T, np.linalg.pinv(var_valid)), epsilon)/(NZ2*(n_red-I))*(NZ2-n_red+1)
            if NZ2-n_red + 1 > 0:
                pv_cons = 1 - sps.f.cdf(d_cons, n_red - I, NZ2 - n_red + 1)
            else:
                pv_cons = np.nan
        else:
            print('Unknown Cons_test : ', cons_test)

    elif reg == 'TLS':
        ## TLS algorithm: nb_runs_x is the number of model/ensemble members for each fingerprint, proj is the initial identity matrix (length of years), formule_ic_tls is the method for confidence interval calculation.
        ## Output parameters: TLS regression coefficients, lower CI, upper CI, residual consistency check statistic, TLS reconstructed signal.
        c0, c1, c2, d_cons, x_tilde_white, y_tilde_white = pda.tls(np.dot(Xc.T, Cf12), np.dot(yc.T, Cf12), 
                                                                   np.dot(Z2c.T, Cf12), nb_runs_x, proj, formule_ic_tls)
        x_tilde = np.dot(Cfp12, x_tilde_white.T) # De-whiten (restore original covariance scale) for later analysis or plotting
        y_tilde = np.dot(Cfp12, y_tilde_white.T)
        beta_hat = c0.T
        beta_hat_inf = c1.T
        beta_hat_sup = c2.T

        # Consistency check
        print("Residual Consistency Check")
        NZ1 = Z1c.shape[1] # Number of samples in the two CTL control simulations, used for consistency test
        NZ2 = Z2c.shape[1]

        if  cons_test == 'MC':
            ## Null-distribution sampled via Monte-Carlo simulations
            ## Note: input data here need to be pre-processed, centered, etc.
            ## First, simulate random variables following the null-distribution
            N_cons_mc = 1000
            # Sigma, X0, nb_runs_X, n1, n2, N, Formula = Cf, Xc, nb_runs_x, NZ1, NZ2, N_cons_mc, formule_ic_tls
            d_H0_cons = pda.consist_mc_tls(Cf, Xc, nb_runs_x, NZ1, NZ2, N_cons_mc, formule_ic_tls) # Generate the distribution of consistency test statistic under the null hypothesis (observation is just internal variability), 1000 simulations
            ## Evaluate the p-value from the H0 sample (this uses gke = Gaussian Kernel Estimation)
            pv_cons = pda.gke(d_H0_cons, d_cons) # Use Gaussian kernel density estimation to compute p-value for the original statistic d_cons
        elif cons_test == "AS03":
            ## Formula provided by Allen & Stott (2003)
            pv_cons = 1 - sps.f.cdf(d_cons / (n_red-I), n_red-I, NZ2)  # Use the analytical formula from Allen & Stott (2003) based on F distribution
        else:
            pv_cons = np.nan

    beta = np.zeros((4, I))
    beta[:-1, :] = np.concatenate((beta_hat_inf, beta_hat, beta_hat_sup))
    beta[-1, 0] = pv_cons
    
    return beta

def reshape_to_N_rows(arr, N):
    """
    Flatten a 2D array row-wise and then fill a new array with N rows using column-major order.

    Parameters:
    - arr: numpy.ndarray, original 2D array
    - N: int, target number of rows

    Returns:
    - new_arr: numpy.ndarray, shape (N, M)
    """
    flat = arr.flatten()  # default is row-major (C order)
    total_length = len(flat)
    trimmed_length = (total_length // N) * N
    flat_trimmed = flat[:trimmed_length]
    M = trimmed_length // N
    new_arr = flat_trimmed.reshape(M, N).T  # first reshape to (M, N), then transpose
    return new_arr

def non_overlapping_2year_mean(data):
    """
    Compute two-year non-overlapping moving average for a 1D array or for each column of a 2D array.

    Parameters:
        data (np.ndarray): 1D or 2D array

    Returns:
        np.ndarray: averaged array
            - If 1D array, returns shape=(n//2,)
            - If 2D array, returns shape=(n//2, m)
    """
    data = np.asarray(data)

    if data.ndim == 1:
        # Process 1D vector
        n = data.shape[0] // 2
        return (data[0:2*n:2] + data[1:2*n:2]) / 2

    elif data.ndim == 2:
        # Average along rows for each column
        n_rows = data.shape[0]
        n = n_rows // 2
        return (data[0:2*n:2, :] + data[1:2*n:2, :]) / 2

    else:
        raise ValueError("Only 1D or 2D arrays are supported")

main_path = '.../results/Temporal_Change/3days_SPEI128/percent/'
obs_data = np.load(main_path + 'All_obs_compound_percent_temporal_change_1950_2023.npy')
# Y = obs_data[:,:-9]
Y = np.nanmean(obs_data[:,9:-9],axis=0) # 1960-2013
Y = non_overlapping_2year_mean(Y)

ALL_data = np.load(main_path + 'CMIP6_ALL_percent_temporal_change_1950_2014.npy')
NAT_data = np.load(main_path + 'CMIP6_NAT_percent_temporal_change_1950_2014.npy')
X = np.full((54, 2), np.nan)
X[:,0] = np.nanmean(ALL_data[:,9:],axis=0) - np.nanmean(NAT_data[:,9:],axis=0)
X[:,1] = np.nanmean(NAT_data[:,9:],axis=0)
X = non_overlapping_2year_mean(X)
nb_runs_x = np.array([16, 11])

CTL_data = np.load(main_path + 'CMIP6_CTL_percent_temporal_change_390days.npy')[:,64:]
CTL = reshape_to_N_rows(CTL_data, 54)
CTL = non_overlapping_2year_mean(CTL)

# y = Y
# X = X.T
# ctl = CTL.T
# reg, cons_test, formule_ic_tls, sample_extr = 'TLS', 'MC', 'ODP', 'regular'
# da(y, X, nb_runs_x, ctl, reg, cons_test, formule_ic_tls, sample_extr):
beta = da(Y, X.T, nb_runs_x, CTL.T, 'TLS', 'MC', 'ODP', 'regular')
np.save('.../results/Detection_attribution/3days_SPEI128/DA_percent.npy',beta)