import tensorly as tl
#tl.set_backend('pytorch')
tl.set_backend('numpy')
from torch.nn import Module, ModuleList
# Methods above should automatically come from backend (except randint, take from torch)
#from torch import transpose, randint, complex64
from tensorly import transpose, complex64
from torch import randint
from itertools import chain
from opt_einsum import contract, contract_expression
from numpy import ceil
from numpy.random import binomial

from .density_tensor import DensityTensor
from .tt_precontraction import qubits_contract, layers_contract
from .tt_contraction import contraction_eq, overlap_eq

#TODO: remove unnecessary dependencies
from .tt_gates import RotY, build_binary_gates_unitary, exp_pauli_y, UnaryGatesUnitary, BinaryGatesUnitary, o4_phases, so4, cnot, cz, SO4LR, CNOTL, CNOTR, CZL, CZR, Unitary, IDENTITY, Hadamard, PauliY, Select0, Select1
from .tt_state import spins_to_tt_state, tt_norm
from .tt_circuit import TTCircuit, tt_dagger
from .tt_operators import unary_hamiltonian, binary_hamiltonian, pauli_z, pauli_y, pauli_x, identity, select0, select1

import copy

def sample(state, nqubits, ncontraq, dtype=complex64):
    """ Write doc """
    # TODO: take in desired qubit indices to be measured

    check_qubits(state, nqubits, ncontraq, dtype)

    bitstring = []
    selectors = [IDENTITY(dtype=dtype) for i in range(nqubits)]
    print(selectors)
    measured_state = []

    for i in range(nqubits):
        # 1) Apply selector operator to qubit in question
        print("State before:")
        for st in state:
            print(st)
        selectors[i] = Select0(dtype=dtype)
        print(selectors)
        selector_op = Unitary(selectors, nqubits, ncontraq)
        circuit = TTCircuit([selector_op], ncontraq, 1) # ncontral is 1, we don't need to contract operator layers

        # 2) Take expectation to get probability dist of qubit
        prob = circuit.state_inner_product(state, state)

        # 3) Sample from qubit probability distribution
        print("SAMPLING")
        print(prob.real)
        bit = binomial(1, (1-prob.real)) # take one sample of distribution, where probability 'success' = prob state1
        print(bit)
        if bit == 1:
            selectors[i] = Select1(dtype=dtype)
            selector_op = Unitary(selectors, nqubits, ncontraq)
            circuit = TTCircuit([selector_op], ncontraq, 1) # ncontral is 1, we don't need to contract operator layers
        print("Updated selector: ", selectors)
        
        # 4) Update selector based on bit value
        # TODO: don't bother to update the state if you're on last bit
        state = circuit.apply_circuit_SCF(state)
        print("State after selecting:")
        for st in state:
            print(st)

        # 5) Add measured bit to growing bitstring
        bitstring.append(bit)

    print("FINAL BITSTRING: ", bitstring)
    return bitstring


def check_qubits(state, nqubits, ncontraq, dtype):
    """ Write doc """
    # Print out individual qubit probabilities to validate before sampling bitstring

    iden = [IDENTITY(dtype=dtype) for i in range(nqubits)]
    for i in range(nqubits):
        # 1) Apply selector operator to qubit in question
        selector1 = copy.deepcopy(iden)
        selector1[i] = Select1(dtype=dtype)
        selector_op = Unitary(selector1, nqubits, ncontraq)
        circuit = TTCircuit([selector_op], ncontraq, 1) # ncontral is 1, we don't need to contract operator layers
        prob = circuit.state_inner_product(state, state)
        print("TEST PROB {} : {} ".format(i, prob))
