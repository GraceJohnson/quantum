import tensorly as tl
#tl.set_backend('pytorch')
#tl.set_backend('numpy')
tl.set_backend('cupy')
from tensorly import complex64, float64
#from numpy.random import binomial
from cupy.random import binomial
import copy
import cupy

from .tt_gates import Unitary, IDENTITY, Select0, Select1
from .tt_circuit import TTCircuit
from .tt_circuit_scf import apply_U

DTYPE = complex64


# Author: K. Grace Johnson

# License: BSD 3 clause


def sample(state_orig, nqubits, ncontraq, dtype=DTYPE):
    """Samples a bitstring of the state (creating a copy of the state, so this function can be called
       multiple times)

    Parameters
    ----------
    state_orig : tt-tensor, state to be sampled
    nqubits: int, number of qubits this state represents
    ncontraq: int, number of qubits a pre-contraction was done over

    Returns
    -------
    bitstring, list of ints of length nqubits, 0s or 1s
    """
    # TODO: take in desired qubit indices to be measured, measure all by default
    state = copy.deepcopy(state_orig)

    bitstring = []
    selectors = [IDENTITY(dtype=dtype) for i in range(nqubits)]
    measured_state = []

    for i in range(nqubits):
        # 1) Apply selector operator to qubit in question
        selectors[i] = Select0(dtype=dtype)
        selector_op = Unitary(selectors, nqubits, ncontraq)
        circuit = TTCircuit([selector_op], ncontraq, 1) # ncontral is 1, we don't need to contract operator layers

        # 2) Take expectation to get probability dist of qubit
        prob = circuit.state_inner_product(state, state)

        # 3) Sample from qubit probability distribution
        prob = abs(prob) # complex numbers
        bit = binomial(1, abs(1.-prob)) # take one sample of distribution, where probability 'success' = prob state1
        if bit == 1:
            selectors[i] = Select1(dtype=dtype)
            selector_op = Unitary(selectors, nqubits, ncontraq)
            circuit = TTCircuit([selector_op], ncontraq, 1) # ncontral is 1, we don't need to contract operator layers
        
        # 4) Add measured bit to growing bitstring
        bit = cupy.asnumpy(bit)
        bitstring.append(bit[0])

        # 5) Update selector based on bit value to prepare for next measurement
        if i < nqubits-1:
            state = apply_U(state, circuit)

    return bitstring


def amplitudes(state, nqubits, ncontraq, dtype=DTYPE):
    """Computes probability amplitudes for each qubit in state

    Parameters
    ----------
    state : tt-tensor, state to be measured
    nqubits: int, number of qubits this state represents
    ncontraq: int, number of qubits a pre-contraction was done over

    Returns
    -------
    amplitudes, list of floats of length nqubits, probabilities for each qubit
    """
    # Print out individual qubit probabilities to validate before sampling bitstring

    iden = [IDENTITY(dtype=dtype) for i in range(nqubits)]
    amplitudes = []
    for i in range(nqubits):
        # 1) Apply selector operator to qubit in question
        selector1 = copy.deepcopy(iden)
        selector1[i] = Select1(dtype=dtype)
        selector_op = Unitary(selector1, nqubits, ncontraq)
        circuit = TTCircuit([selector_op], ncontraq, 1) # ncontral is 1, we don't need to contract operator layers
        prob = circuit.state_inner_product(state, state)
        prob = abs(prob) # complex numbers
        amplitudes.append(cupy.asnumpy(prob)[0])
    return amplitudes
