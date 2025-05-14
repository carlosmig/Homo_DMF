# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:54:54 2020
@author: Carlos Coronel
"""

# Imports
import numpy as np
from numba import float64, int32, vectorize, njit, jit
from numba.core.errors import NumbaPerformanceWarning
from scipy.io import loadmat
import warnings

# Ignore Numba performance warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# Network parameters
SC = np.mean(loadmat('SCmatrices88healthy.mat')['SCmatrices'], axis=0)  # Average SC across subjects
np.fill_diagonal(SC, 0)  # Remove self-connections
nnodes = len(SC)

# Simulation parameters
tmax = 780000            # Total simulation time (mseconds)
dt = 1               # Integration step (mseconds)
seed = 0               # Random seed

# Model parameters
gE, gI = 310, 615                # Gains for excitatory and inhibitory transfer functions
IthrE, IthrI = 0.403, 0.287      # Thresholds for excitatory and inhibitory transfer functions
tauNMDA, tauGABA = 100, 10       # Time constants for NMDA and GABA synaptic gating
gamma = 0.641                    # NMDA gating decay control
dE, dI = 0.16, 0.087             # Curvature constants for the transfer functions
I0 = 0.382                       # External baseline input
WE, WI = 1, 0.7                  # Scaling of external input for E and I populations
W_plus = 1.4                     # Strength of recurrent excitation
sigma = 0.01                     # Noise strength
JNMDA = 0.15                     # Excitatory synaptic weight
G = 0                            # Global coupling

# Synaptic plasticity parameters
target = 3        # Target firing rate (Hz)
tau_p = 1         # Plasticity time constant
Jdecay = 495439   # Decay constant for JGABA term

# Toggle plasticity (1 = on, 0 = off)
model_1 = 0

@jit(int32(int32), nopython=True)
def set_seed(seed):
    """Set the NumPy random seed (Numba-compatible)."""
    np.random.seed(seed)
    return seed

@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def rE(IE, gE, IthrE, dE):
    """Transfer function for excitatory population."""
    return gE * (IE - IthrE) / (1 - np.exp(-dE * gE * (IE - IthrE)))

@vectorize([float64(float64, float64, float64, float64)], nopython=True)
def rI(II, gI, IthrI, dI):
    """Transfer function for inhibitory population."""
    return gI * (II - IthrI) / (1 - np.exp(-dI * gI * (II - IthrI)))

@njit
def mean_field(y, SC, params):
    """Compute the DMF model derivatives."""
    SE, SI, JGABA = y
    G, WE, WI, W_plus, I0, JNMDA, tauNMDA, tauGABA, gamma, target, tau_p, Jdecay, gE, IthrE, dE, gI, IthrI, dI = params

    IE_t = WE * I0 + W_plus * JNMDA * SE + G * JNMDA * SC @ SE - JGABA * SI
    II_t = WI * I0 + JNMDA * SE - SI

    rE_t = rE(IE_t, gE, IthrE, dE)
    rI_t = rI(II_t, gI, IthrI, dI)

    SE_dot = -SE / tauNMDA + (1 - SE) * gamma * rE_t / 1000
    SI_dot = -SI / tauGABA + rI_t / 1000
    JGABA_dot = (-JGABA / Jdecay + rI_t / 1000 * (rE_t / 1000 - target / 1000) / tau_p) * model_1

    return np.vstack((SE_dot, SI_dot, JGABA_dot)), rE_t

@njit
def Noise(sigma):
    """Add Gaussian noise to SE and SI; no noise for JGABA."""
    SE_dot = sigma * np.random.normal(0, 1, nnodes)
    SI_dot = sigma * np.random.normal(0, 1, nnodes)
    JGABA_dot = sigma * np.random.normal(0, 0, nnodes)  # effectively no noise

    return np.vstack((SE_dot, SI_dot, JGABA_dot))

def update():
    """Force recompilation of Numba-accelerated functions."""
    mean_field.recompile()
    Noise.recompile()

def Sim():
    """
    Run the DMF simulation using current parameters.

    Returns
    -------
    Y_t_rates : ndarray
        Time series of excitatory firing rates for each node.
    timeSim : ndarray
        Simulation time vector.
    """
    global SC

    params = np.array([
        G, WE, WI, W_plus, I0, JNMDA, tauNMDA, tauGABA, gamma,
        target, tau_p, Jdecay, gE, IthrE, dE, gI, IthrI, dI
    ])

    set_seed(seed)
    np.random.seed(seed)

    if SC.shape[0] != SC.shape[1] or SC.shape[0] != nnodes:
        raise ValueError("check SC dimensions and number of nodes")

    if SC.dtype is not np.float64:
        try:
            SC = SC.astype(np.float64)
        except:
            raise TypeError("SC must be numeric, preferably float64")

    Nsim = int(tmax / dt)
    timeSim = np.linspace(0, tmax, Nsim)

    # Initial conditions: SE, SI, JGABA set to 1
    neural_ic = np.ones((1, nnodes)) * np.array([1, 1, 1])[:, None]
    neural_Var = neural_ic
    rE_out = np.zeros(nnodes)
    Y_t_rates = np.zeros((len(timeSim), nnodes))

    for i in range(Nsim):
        Y_t_rates[i, :] = rE_out
        derivs, rE_out = mean_field(neural_Var, SC, params)
        neural_Var += derivs * dt + Noise(sigma) * np.sqrt(dt)

    return Y_t_rates, timeSim

def ParamsNode():
    """Return model parameters as a dictionary."""
    pardict = {}
    for var in (
        'gE', 'gI', 'IthrE', 'IthrI', 'tauNMDA', 'tauGABA', 'gamma',
        'dE', 'dI', 'I0', 'WE', 'WI', 'W_plus', 'sigma', 'JNMDA',
        'target', 'tau_p'
    ):
        pardict[var] = eval(var)
    return pardict
