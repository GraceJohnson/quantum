import tensorly as tl
#tl.set_backend('pytorch')
tl.set_backend('numpy')
from tensorly import complex64
from numpy.random import binomial

from .tt_gates import Unitary, IDENTITY, Select0, Select1
from .tt_circuit import TTCircuit
from .tt_circuit_scf import apply_U

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
        print("\nSAMPLING")
        print(prob.real)
        bit = binomial(1, abs(1.-prob.real)) # take one sample of distribution, where probability 'success' = prob state1
        print(bit)
        if bit == 1:
            selectors[i] = Select1(dtype=dtype)
            selector_op = Unitary(selectors, nqubits, ncontraq)
            circuit = TTCircuit([selector_op], ncontraq, 1) # ncontral is 1, we don't need to contract operator layers
        print("Updated selector: ", selectors)
        
        # 4) Update selector based on bit value
        # TODO: don't bother to update the state if you're on last bit
        state = apply_U(state, circuit)
        print("State after selecting:")
        for st in state:
            print(st)

        # 5) Add measured bit to growing bitstring
        bitstring.append(bit[0])

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
