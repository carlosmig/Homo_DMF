import numpy as np
import DMF_ISP_numba as DMF
import BOLDModel as BD       
from scipy.io import loadmat 
import matplotlib.pyplot as plt

# Set random seed for reproducibility
seed = 1
np.random.seed(seed)
DMF.seed = seed  # Also set the seed inside the DMF module

# Simulation configuration
DMF.tmax = 360000  # Total simulation time in milliseconds
DMF.dt = 1         # Integration step size in milliseconds

# Number of brain regions (nodes), assumed to match the structural connectivity (SC) matrix
nnodes = 90

# DMF model parameters
DMF.G = 2.5             # Global coupling strength
DMF.sigma = 0.01      # Noise amplitude (scaling factor)
DMF.tau_p = 1         # Plasticity time constant
DMF.target = 3        # Target firing rate or dynamic target (depending on model)
DMF.Jdecay = 495439   # Decay parameter for synaptic scaling or homeostatic plasticity
DMF.model_1 = 1       # Enable plasticity mechanism (e.g., 1 = plasticity model, 0 = static)

# Load structural connectivity (SC) matrix and configure the network
SC = np.mean(loadmat('SCmatrices88healthy.mat')['SCmatrices'], 0)  # Average across 88 subjects
np.fill_diagonal(SC, 0)  # Remove self-connections
DMF.SC = SC.copy()       # Assign SC to the model
DMF.nnodes = len(DMF.SC) # Set the number of nodes in the model

# Finalize internal model setup (e.g., precomputations or allocation)
DMF.update()

#%%

# Collect all DMF parameters into a single array (not used here but may be passed to a function later)
params = np.array([
    DMF.G, DMF.WE, DMF.WI, DMF.W_plus, DMF.I0, DMF.JNMDA, DMF.tauNMDA,
    DMF.tauGABA, DMF.gamma, DMF.target, DMF.tau_p, DMF.Jdecay, 
    DMF.gE, DMF.IthrE, DMF.dE, DMF.gI, DMF.IthrI, DMF.dI
])

# Run the DMF simulation and get firing rates
rates, _ = DMF.Sim()

# Downsample firing rates: take one every 10 ms (from 1 ms resolution to 10 ms)
rates = rates[::10, :]

# Define time step for BOLD simulation (in seconds)
dtt = 0.01

# Simulate BOLD signals from firing rates
BOLD_signals = BD.Sim(rates, DMF.nnodes, dtt)

# Discard the first 60 seconds (60000 ms = 12000 samples at 10 ms), then further downsample to 1 Hz
BOLD_signals = BOLD_signals[12000:, :][::100, :]

# Compute functional connectivity (FC) matrix from BOLD signals
# Reorder regions: even indices followed by reversed odd indices
even = np.arange(0, 90, 2)
odd = np.arange(1, 90, 2)[::-1]
idxs = np.append(even, odd)

# Pearson correlation between BOLD time series (transposed to shape [regions, time])
FC = np.corrcoef(BOLD_signals[:, idxs].T)

# Print the mean FC value
print(f'Mean FC: {np.mean(FC)}')

# Print the average firing rate across time and nodes
print(np.mean(np.mean(rates)))

#%%

# Set global font size for plots
plt.rcParams.update({'font.size': 16})

# Create a new figure (Figure 1) with a size of 6x5 inches
plt.figure(1, figsize=(6, 5))
plt.clf()  # Clear the figure to avoid overlaps if it was already used

# Define a single subplot (1 row, 1 column, first plot)
plt.subplot(1, 1, 1)

# Plot the functional connectivity matrix
plt.imshow(FC, vmin=0, vmax=1, cmap='jet')  # Color scale fixed between 0 and 1
plt.title('Functional Connectivity')        # Title of the plot
plt.xlabel('Nodes')                         # X-axis label
plt.ylabel('Nodes')                         # Y-axis label

# Optimize spacing between elements
plt.tight_layout()

# Display the plot on screen
plt.show()