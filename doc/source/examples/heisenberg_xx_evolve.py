"""
Time Evolution of Heisenberg XX Model
-------------------------------------

Time evolution of a Heisenberg XX model Hamiltonian using TensorLy-Quantum 
and the SCF algorithm for applying gate operators to TT states.

Hamiltonian: H = J * sum_i=0^N (sig_i^x sig_(i+1)^x + sig_i^y sig_(i+1)^y) 
"""

import tensorly as tl
import tlquantum as tlq
from tensorly.tt_matrix import TTMatrix
from tensorly import complex64, float64
import numpy as np
#import cupy as np
import copy
import time as timer
from scipy.linalg import expm
import seaborn as sns
import matplotlib.pyplot as plt 


########################################################################

def sample(state, nqubits, ncontraq, nsamples, filename):
    amplitudes = tlq.amplitudes(state, nqubits, ncontraq)
    fh = open("amplitudes.dat", "a")
    fh.write("\t".join(str(amp) for amp in amplitudes))
    fh.write("\n")
    fh.close()
    bitstrings = []
    for i in range(nsamples):
        bitstrings.append(tlq.sample(state, nqubits, ncontraq))
    bitstrings = np.array(bitstrings)
    cor = np.corrcoef(bitstrings, rowvar=False)
   
    plt.clf()
    sns.heatmap(np.asnumpy(cor), cmap=sns.diverging_palette(15, 240, l=40, s=85, n=200), vmin=-1, vmax=1, center=0)
    plt.title("Step: " + filename[4:7])
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)


def trotter_step(state_orig, theta, N):
    state = copy.deepcopy(state_orig)
    J = -1 # -1 or +1
    # Z gates
    for i in range(nqubits-1):
        unitary = []
        id_list = [i,i+1]
        for ind in range(nqubits):
            if ind in id_list:
                unitary.append(J * tlq.pauli_y())
            else:
                unitary.append(tlq.identity())
        unitary = tlq.InvolutoryGeneratorUnitary(nqubits, ncontraq, unitary, theta=theta/N)
        layers = [tlq.TTCircuit([unitary], ncontraq, ncontral)]
        state = tlq.apply_circuit_SCF(state, layers)
    # X gates
    for i in range(nqubits-1):
        unitary = []
        id_list = [i,i+1]
        for ind in range(nqubits):
            if ind in id_list:
                unitary.append(J * tlq.pauli_x())
            else:
                unitary.append(tlq.identity())
        unitary = tlq.InvolutoryGeneratorUnitary(nqubits, ncontraq, unitary, theta=theta/N)
        layers = [tlq.TTCircuit([unitary], ncontraq, ncontral)]
        state = tlq.apply_circuit_SCF(state, layers)
    return state


def evolve_state(state, num_steps, timestep, num_trotter): 
    for i in range(num_steps):
        for j in range(num_trotter):
            state = trotter_step(state, timestep, num_trotter)

        nsamples = 100
        filename = 'step{:03d}.png'.format(i+1)
        sample(state, nqubits, ncontraq, nsamples, filename)

        overlap = abs(tlq.overlap(state_orig, state, 2))
        fh = open("overlap.dat", "a")
        fh.write(str(overlap)+"\n")
        fh.close()
    return state

########################################################################
         
# %% Set up simulation parameters

tl.set_backend('numpy')
#tl.set_backend('cupy')
device = 'cpu'
#device = 'cuda' 

dtype = complex64

nqubits = 20 #number of qubits
ncontraq = 10  #number of qubits to pre-contract into single core
ncontral = 1  #number of layers to pre-contract into a single core

chi = 1024  # Bond dimension for TT

np.random.seed(42)


# %% Generate an input state. For each qubit, 0 --> |0> and 1 --> |1>

#state = tlq.spins_to_tt_state([0 for i in range(nqubits)], device=device, dtype=dtype) # generate generic zero state |00000>
spins = [1 for i in range(int(nqubits/2))] + [0 for i in range(int(nqubits/2))] # |111...000...> 
#spins = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0] # |010101...>

state = tlq.qubits_contract(state, ncontraq)

# Initialize a random state
#state = [np.random.randn(*node.shape) for node in state]
#state = [np.random.randn(*node.shape) + np.random.randn(*node.shape)*1.j for node in state]
#state, NRMR = tlq.orthonorm_right(state)

state = tl.tt_tensor.pad_tt_rank(state, chi-2)


# %% Run time evolution simulation and plot overlap with initial state, qubit probability amplitudes, and bitstring correlations

state_orig = copy.deepcopy(state)
nsamples = 100
filename = 'step000.png'
sample(state, nqubits, ncontraq, nsamples, filename)
overlap = abs(tlq.overlap(state, state, 2))
fh = open("overlap.dat", "a")
fh.write(str(overlap)+"\n")
fh.close()

dt = 0.1
ntrotter = 6 
nsteps = 500
state = evolve_state(state, nsteps, dt, ntrotter)

