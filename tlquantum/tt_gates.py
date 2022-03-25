import tensorly as tl
#tl.set_backend('pytorch')
#tl.set_backend('numpy')
tl.set_backend('cupy')
from tensorly import cos, sin, float32, complex64
from torch import randn, exp
#from numpy.random import randn as nprandn
from cupy.random import randn as nprandn

from torch.nn import Module, ModuleList, ParameterList, Parameter
from tensorly.tt_matrix import TTMatrix
from copy import deepcopy

from .tt_operators import identity, hadamard, pauli_y, pauli_x, pauli_z, select0, select1
from .tt_precontraction import qubits_contract, _get_contrsets
from .tt_sum import tt_matrix_sum


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>
# Author: Jean Kossaifi <jkossaifi@nvidia.com>

# License: BSD 3 clause


class Unitary(Module):
    """A unitary for all qubits in a TTCircuit, using tensor ring tensors
    with PyTorch Autograd support.
    Can be defined with arbitrary gates or used as a base-class for set circuit
    types.

    Parameters
    ----------
    gates : list of TT gate classes, each qubit in the unitary
            to be involved in one gate.
    nqubits : int, number of qubits
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    contrsets : list of lists of ints, the indices of qubit cores to
                merge in the pre-contraction path.
    device : string, device on which to run the computation.

    Returns
    -------
    Unitary
    """
    def __init__(self, gates, nqubits, ncontraq, contrsets=None, dtype=complex64, device=None):
        super().__init__()
        if contrsets is None:
            contrsets = _get_contrsets(nqubits, ncontraq)
        self.nqubits, self.ncontraq, self.contrsets, self.dtype, self.device = nqubits, ncontraq, contrsets, dtype, device
        self._set_gates(gates)


    def _set_gates(self, gates):
        """Sets the gate class instances as a PyTorch ModuleList for Unitary.

        """
        self.gates = ModuleList(gates)


    def forward(self):
        """Prepares the tensors of Unitary for forward contraction by calling the gate instances'
        forward method and doing qubit-wise (horizonal) pre-contraction.

        Returns
        -------
        List of pre-contracted gate tensors for general forward pass.
        """
	#return qubits_contract([gate.forward() for gate in self.gates], self.ncontraq, contrsets=self.contrsets)
        # TODO: involutory generator sends back a list for a single gate, resulting list of lists causes an error
        #       below is quick fix but could use a better solution
        gates = [gate.forward() for gate in self.gates]
        if isinstance(gates[0], list):
            [gates] = gates
        return qubits_contract(gates, self.ncontraq, contrsets=self.contrsets)


class BinaryGatesUnitary(Unitary):
    """A Unitary sub-class that generates a layer of a single two-qubit gates accross
    all qubits in a TTCircuit.

    Parameters
    ----------
    nqubits : int, number of qubits
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    q2gate : tuple of two gate instances, one for each qubit in gate.
    contrsets : list of lists of ints, the indices of qubit cores to
                merge in the pre-contraction path.
    device : string, device on which to run the computation.

    Returns
    -------
    BinaryGatesUnitary
    """
    def __init__(self, nqubits, ncontraq, q2gate, parity, contrsets=None, random_initialization=True):
        dtype, device = q2gate[0].dtype, q2gate[0].device
        super().__init__([], nqubits, ncontraq, contrsets=contrsets, dtype=dtype, device=device)
        self._set_gates(build_binary_gates_unitary(self.nqubits, q2gate, parity, dtype=dtype, random_initialization=random_initialization))


class UnaryGatesUnitary(Unitary):
    """A Unitary sub-class that generates a layer of unitary, single-qubit rotations.
    As simulation occurs in real-space, these rotations are about the Y-axis.

    Parameters
    ----------
    nqubits : int, number of qubits
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    contrsets : list of lists of ints, the indices of qubit cores to
                merge in the pre-contraction path.
    device : string, device on which to run the computation.

    Returns
    -------
    UnaryGatesUnitary
    """
    def __init__(self, nqubits, ncontraq, axis='y', contrsets=None, dtype=complex64, device=None):
        super().__init__([], nqubits, ncontraq, contrsets=contrsets, dtype=dtype, device=device)
        if axis == 'y':
            self._set_gates([RotY(dtype=dtype, device=device) for i in range(self.nqubits)])
        elif axis == 'x':
            self._set_gates([RotX(dtype=dtype, device=device) for i in range(self.nqubits)])
        elif axis == 'z':
            self._set_gates([RotZ(dtype=dtype, device=device) for i in range(self.nqubits)])
        else:
            raise IndexError('UnaryGatesUnitary has no rotation axis {}.\n'
                             'UnaryGatesUnitary has 3 rotation axes: x, y, and z. The y-axis is default.'.format(index))


class InvolutoryGeneratorUnitary(Unitary):
    """A Unitary sub-class that generates a unitary layer with a generator that is involutory (its own inverse).

    Parameters
    ----------
    nqubits : int, number of qubits
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    involutory_generator : list of tensors, involutory operator to use as generator
    contrsets : list of lists of ints, the indices of qubit cores to
                merge in the pre-contraction path.

    Returns
    -------
    UnaryGatesUnitary
    """
    #def __init__(self, nqubits, ncontraq, involutory_generator, contrsets=None):
    def __init__(self, nqubits, ncontraq, involutory_generator, contrsets=None, device=None, theta=None):
        #dtype, device = involutory_generator[0].dtype, involutory_generator[0].device
        dtype = involutory_generator[0].dtype
        super().__init__([], nqubits, ncontraq, contrsets=contrsets, dtype=dtype, device=device)
        self._set_gates([InvolutoryGenerator(involutory_generator, nqubits, dtype=dtype, device=device, theta=theta)])


def build_binary_gates_unitary(nqubits, q2gate, parity, random_initialization=True, dtype=complex64):
    """Generate a layer of two-qubit gates.

    Parameters
    ----------
    nqubits : int, number of qubits
    q2gate : tt-tensor, 2-core, 2-qubit gates to use in layer
    parity : int, if even, apply first q2gate core to even qubits, if odd, to odd qubits.

    Returns
    -------
    Layer of two-qubit gates as list of tt-tensors
    """
    def clone_gates(gate0, gate1, random_initialization):
        clone0, clone1 = deepcopy(gate0), deepcopy(gate1)
        if random_initialization:
            clone0.reinitialize(), clone1.reinitialize()
        return [clone0, clone1]

    q2gate0, q2gate1 = q2gate[0].type(dtype), q2gate[1].type(dtype)
    layer, device = [], q2gate0.device
    for i in range(nqubits//2 - 1):
        layer += clone_gates(q2gate0, q2gate1, random_initialization)
    if nqubits%2 == 0:
        temp = clone_gates(q2gate0, q2gate1, random_initialization)
        if parity%2 == 0:
            return layer+temp
        return [temp[1]]+layer+[temp[0]]
    temp = clone_gates(q2gate0, q2gate1, random_initialization)
    if parity%2 == 0:
        return layer+temp+[IDENTITY(dtype=dtype, device=device)]
    return [IDENTITY(dtype=dtype, device=device)]+layer+temp


class InvolutoryGenerator(Module):
    """Qubit rotations about the involutory generator.

    Parameters
    ----------
    nqubits : int, number of qubits.
    involutory_generator : list of tensors, involutory operator to use as generator.
    device : string, device on which to run the computation.

    Returns
    -------
    InvolutoryGenerator
    """
    def __init__(self, involutory_generator, nqubits, dtype=complex64, device=None, theta=None):
        super().__init__()
        #self.theta = Parameter(randn(1, device=device))
        #self.theta = nprandn(1) #TODO switch case for numpy
        self.theta = theta # TODO: implement setter
        #self.iden, self.involutory_generator = [identity(dtype=dtype, device=self.theta.device) for i in range(nqubits)], involutory_generator
        self.iden, self.involutory_generator = [identity(dtype=dtype, device=device) for i in range(nqubits)], involutory_generator


    def forward(self):
        """Prepares the RotY gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of rotation matrix depending on theta (which is
        typically updated every epoch through backprop via PyTorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        temp_iden, temp_involutory_generator = [self.iden[0]*cos(self.theta)] + self.iden[1::], [self.involutory_generator[0]*1j*sin(self.theta)] + self.involutory_generator[1::]
        return tt_matrix_sum(temp_iden, temp_involutory_generator).factors


class RotY(Module):
    """Qubit rotations about the Y-axis with randomly initiated theta.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    RotY
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        #self.theta = Parameter(randn(1, device=device))
        self.theta = nprandn(1) #TODO switch case for numpy
        #self.iden, self.epy = identity(dtype=dtype, device=self.theta.device), exp_pauli_y(dtype=dtype, device=self.theta.device)
        self.iden, self.epy = identity(dtype=dtype, device=device), exp_pauli_y(dtype=dtype, device=device)


    def forward(self):
        """Prepares the RotY gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of rotation matrix depending on theta (which is
        typically updated every epoch through backprop via PyTorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.iden*cos(self.theta/2)+self.epy*sin(self.theta/2)


class RotX(Module):
    """Qubit rotations about the X-axis with randomly initiated theta.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    RotX
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        #self.theta = Parameter(randn(1, device=device))
        self.theta = nprandn(1) #TODO switch case for numpy
        #self.iden, self.epx = identity(dtype=dtype, device=self.theta.device), exp_pauli_x(dtype=dtype, device=self.theta.device)
        self.iden, self.epx = identity(dtype=dtype, device=device), exp_pauli_x(dtype=dtype, device=device)


    def forward(self):
        """Prepares the RotX gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of rotation matrix depending on theta (which is
        typically updated every epoch through backprop via PyTorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.iden*cos(self.theta/2)+self.epx*sin(self.theta/2)


class RotZ(Module):
    """Qubit rotations about the Z-axis with randomly initiated theta.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    RotZ
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        #self.theta, self.dtype, self.device = Parameter(randn(1, device=device)), dtype, device
        self.theta, self.dtype, self.device = nprandn(1), dtype, device #TODO switch case for numpy


    def forward(self):
        """Prepares the RotZ gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of rotation matrix depending on theta (which is
        typically updated every epoch through backprop via PyTorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return tl.tensor([[[[exp(-1j*self.theta/2)],[0]],[[0],[exp(1j*self.theta/2)]]]], dtype=self.dtype, device=self.device)


class IDENTITY(Module):
    """Identity gate (does not change the state of the qubit on which it acts).

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    IDENTITY
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        self.core, self.dtype, self.device = identity(dtype=dtype, device=device), dtype, device


    def forward(self):
        """Prepares the left qubit of the IDENTITY gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


class Hadamard(Module):
    """Hadamard gate (creates superposition of |0> and |1>).

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Hadamard
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        self.core, self.dtype, self.device = hadamard(dtype=dtype, device=device), dtype, device


    def forward(self):
        """Prepares the left qubit of the Hadamard gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


class Select0(Module):
    """Selects state |0>.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Select0
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        self.core, self.dtype, self.device = select0(dtype=dtype, device=device), dtype, device


    def forward(self):
        """Prepares the left qubit of the Select0 gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


class Select1(Module):
    """Selects state |1>.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Select1
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        self.core, self.dtype, self.device = select1(dtype=dtype, device=device), dtype, device


    def forward(self):
        """Prepares the left qubit of the Select1 gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


class PauliY(Module):
    """Pauli-Y gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    PauliY
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        self.core, self.dtype, self.device = pauli_y(dtype=dtype, device=device), dtype, device


    def forward(self):
        """Prepares the left qubit of the Pauli-Y gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


class PauliX(Module):
    """Pauli-X gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    PauliX
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        self.core, self.dtype, self.device = pauli_x(dtype=dtype, device=device), dtype, device


    def forward(self):
        """Prepares the left qubit of the Pauli-X gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


class PauliZ(Module):
    """Pauli-Z gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    PauliZ
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        self.core, self.dtype, self.device = pauli_z(dtype=dtype, device=device), dtype, device


    def forward(self):
        """Prepares the left qubit of the Pauli-Z gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


def cnot(dtype=complex64, device=None):
    """Pair of CNOT class instances, one left (control) and one right (transformed).

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    (CNOTL, CNOTR)
    """
    return CNOTL(dtype=dtype, device=device), CNOTR(dtype=dtype, device=device)


class CNOTL(Module):
    """Left (control-qubit) core of a CNOT gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Left core of CNOT gate.
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        #core, self.dtype, self.device = tl.zeros((1,2,2,2), dtype=dtype, device=device), dtype, device
        core, self.dtype, self.device = tl.zeros((1,2,2,2), dtype=dtype), dtype, device
        core[0,0,0,0] = core[0,1,1,1] = 1.
        self.core = core


    def forward(self):
        """Prepares the left qubit of the CNOT gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


    def reinitialize(self):
        pass


class CNOTR(Module):
    """Right (transformed qubit) core of a CNOT gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Right core of CNOT gate.
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        #core, self.dtype, self.device = tl.zeros((2,2,2,1), dtype=dtype, device=device), dtype, device
        core, self.dtype, self.device = tl.zeros((2,2,2,1), dtype=dtype), dtype, device
        core[0,0,0,0] = core[0,1,1,0] = 1.
        core[1,0,1,0] = core[1,1,0,0] = 1.
        self.core =  core


    def forward(self):
        """Prepares the right qubit of the CNOT gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


    def reinitialize(self):
        pass


class CYL(Module):
    """Left (control-qubit) core of a CY gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Left core of CY gate.
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        #core, self.dtype, self.device = tl.zeros((1,2,2,2), dtype=dtype, device=device), dtype, device
        core, self.dtype, self.device = tl.zeros((1,2,2,2), dtype=dtype), dtype, device
        core[0,0,0,0] = core[0,1,1,1] = 1.
        self.core = core


    def forward(self):
        """Prepares the left qubit of the CY gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


    def reinitialize(self):
        pass


class CYR(Module):
    """Right (transformed qubit) core of a CY gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Right core of CY gate.
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        #core, self.dtype, self.device = tl.zeros((2,2,2,1), dtype=dtype, device=device), dtype, device
        core, self.dtype, self.device = tl.zeros((2,2,2,1), dtype=dtype), dtype, device
        core[0,0,0,0] = core[0,1,1,0] = -1.j
        core[1,0,1,0] = core[1,1,0,0] = 1.j
        self.core =  core


    def forward(self):
        """Prepares the right qubit of the CY gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


    def reinitialize(self):
        pass


def cz(dtype=complex64, device=None):
    """Pair of CZ class instances, one left (control) and one right (transformed).

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    (CZL, CZR)
    """
    return CZL(dtype=dtype, device=device), CZR(dtype=dtype, device=device)


class CZL(Module):
    """Left (control-qubit) core of a CZ gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Left core of CZ gate.
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        #core, self.dtype, self.device = tl.zeros((1,2,2,2), dtype=dtype, device=device), dtype, device
        core, self.dtype, self.device = tl.zeros((1,2,2,2), dtype=dtype), dtype, device
        core[0,0,0,0] = core[0,1,1,1] = 1.
        self.core = core


    def forward(self):
        """Prepares the left qubit of the CZ gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


    def reinitialize(self):
        pass


class CZR(Module):
    """Right (transformed qubit) core of a CZ gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Right core of CZ gate.
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        #core, self.dtype, self.device = tl.zeros((2,2,2,1), dtype=dtype, device=device), dtype, device
        core, self.dtype, self.device = tl.zeros((2,2,2,1), dtype=dtype), dtype, device
        core[0,0,0,0] = core[0,1,1,0] = core[1,0,0,0]  = 1.
        core[1,1,1,0] = -1.
        self.core = core

    def forward(self):
        """Prepares the right qubit of the CZ gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


    def reinitialize(self):
        pass


def so4(state1, state2, dtype=complex64, device=None):
    """Pair of SO4 two-qubit rotation class instances, with rotations over
    different states.

    Parameters
    ----------
    state1 : int, the first of 4 quantum states to undergo the 2-qubit rotations
    state2 : int, the second of 4 quantum states to undergo the 2-qubit rotations
    device : string, device on which to run the computation.

    Returns
    -------
    (SO4L, SO4R)
    """        
    R = SO4LR(state1, state2, 0, dtype=dtype, device=device)
    return R, SO4LR(state1, state2, 1, theta=R.theta, dtype=dtype, device=device)


class SO4LR(Module):
    """Left or right core of the two-qubit SO4 rotations gate.

    Parameters
    ----------
    state1 : int, the first of 4 quantum states to undergo the 2-qubit rotations
    state2 : int, the second of 4 quantum states to undergo the 2-qubit rotations
    position : int, if 0, then left core, if 1, then right core.
    device : string, device on which to run the computation.

    Returns
    -------
    if position == 0 --> SO4L
    if position == 1 --> SO4R
    """
    def __init__(self, state1, state2, position, theta=None, dtype=complex64, device=None):
        super().__init__()
        #self.theta, self.position, self.dtype, self.device = Parameter(randn(1, device=device)), position, dtype, device
        self.theta, self.position, self.dtype, self.device = nprandn(1), position, dtype, device #TODO switch case for numpy
        if theta is not None:
            self.theta.data = theta.data
        ind1, ind2 = min(state1, state2), max(state1, state2)
        if (ind1, ind2) == (0,1):
            self.core_generator =  _so4_01
        elif (ind1, ind2) == (1,2):
            self.core_generator =  _so4_12
        elif (ind1, ind2) == (2,3):
            self.core_generator =  _so4_23
        else:
            raise IndexError('SO4 Rotation Gates have no state interaction pairs {}.\n'
                             'Valid state interactions pairs are (0,1), (1,2), and (2,3)'.format((state1, state2)))


    def forward(self):
        """Prepares the left or right qubit of the SO4 two-qubit rotation gate for forward contraction
        by calling the forward method and preparing the tt-factorized form of matrix representation.
        Update is based on theta (which is typically updated every epoch through backprop via Pytorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core_generator(self.theta, dtype=self.dtype, device=self.device)[self.position]


    def reinitialize(self):
        #self.theta.data = randn(1, device=self.device)
        self.theta = nprandn(1) #TODO switch case for numpy


def _so4_01(theta, dtype=complex64, device=None):
    """Two-qubit SO4 gates in tt-tensor form with rotations along zeroth and first
    qubit states.

    Parameters
    ----------
    theta : PyTorch parameter, angle about which to rotate qubit, optimizable with PyTorch Autograd
    device : string, device on which to run the computation.

    Returns
    -------
    (SO4_01_L, SO4_01_R)
    """
    core1, core2 = tl.zeros((1,2,2,1), dtype=dtype, device=device), tl.zeros((1,2,2,1), dtype=dtype, device=device)
    core1[0,0,0,0] = core2[0,0,0,0] = core2[0,1,1,0] = 1
    T01I = [core1, core2]
    core1, core2 = tl.zeros((1,2,2,1), dtype=dtype, device=device), tl.zeros((1,2,2,1), dtype=dtype, device=device)
    core1[0,1,1,0] = core2[0,0,0,0] = core2[0,1,1,0] = 1
    T23I = [core1*cos(theta), core2]
    core1, core2 = tl.zeros((1,2,2,1), dtype=dtype, device=device), tl.zeros((1,2,2,1), dtype=dtype, device=device)
    core1[0,1,1,0] = core2[0,1,0,0] = 1
    core2[0,0,1,0] = -1
    R23I = [core1*sin(theta), core2]
    #return [*tt_matrix_sum(TTMatrix(T01I), tt_matrix_sum(TTMatrix(T23I), TTMatrix(R23I)))]
    return [*tt_matrix_sum(TTMatrix(T01I), tt_matrix_sum(TTMatrix(T23I), TTMatrix(R23I), device=device), device=device)]


def _so4_12(theta, dtype=complex64, device=None):
    """Two-qubit SO4 gates in tt-tensor form with rotations along first and second
    qubit states.

    Parameters
    ----------
    theta : PyTorch parameter, angle about which to rotate qubit, optimizable with PyTorch Autograd
    device : string, device on which to run the computation.

    Returns
    -------
    (SO4_12_L, SO4_12_R)
    """
    core1, core2 = tl.zeros((1,2,2,2), dtype=dtype, device=device), tl.zeros((2,2,2,1), dtype=dtype, device=device)
    core1[0,0,0,0] = core1[0,1,1,1] = core2[0,0,0,0] = core2[1,1,1,0] = 1
    T03I = [core1, core2]
    core1, core2 = tl.zeros((1,2,2,2), dtype=dtype, device=device), tl.zeros((2,2,2,1), dtype=dtype, device=device)
    core1[0,1,1,0] = core1[0,0,0,1] = core2[0,0,0,0] = core2[1,1,1,0] = 1
    T12I = [core1*cos(theta), core2]
    core1, core2 = tl.zeros((1,2,2,2), dtype=dtype, device=device), tl.zeros((2,2,2,1), dtype=dtype, device=device)
    core1[0,1,0,0] = core1[0,0,1,1] = core2[0,0,1,0] = 1
    core2[1,1,0,0] = -1
    R12I = [core1*sin(theta), core2]
    #return [*tt_matrix_sum(TTMatrix(T03I), tt_matrix_sum(TTMatrix(T12I), TTMatrix(R12I)))]
    return [*tt_matrix_sum(TTMatrix(T03I), tt_matrix_sum(TTMatrix(T12I), TTMatrix(R12I), device=device), device=device)]


def _so4_23(theta, dtype=complex64, device=None):
    """Two-qubit SO4 gates in tt-tensor form with rotations along second and third
    qubit states.

    Parameters
    ----------
    theta : PyTorch parameter, angle about which to rotate qubit, optimizable with PyTorch Autograd
    device : string, device on which to run the computation.

    Returns
    -------
    (SO4_23_L, SO4_23_R)
    """
    core1, core2 = tl.zeros((1,2,2,1), dtype=dtype, device=device), tl.zeros((1,2,2,1), dtype=dtype, device=device)
    core1[0,1,1,0] = core2[0,0,0,0] = core2[0,1,1,0] = 1
    T23I = [core1, core2]
    core1, core2 = tl.zeros((1,2,2,1), dtype=dtype, device=device), tl.zeros((1,2,2,1), dtype=dtype, device=device)
    core1[0,0,0,0] = core2[0,0,0,0] = core2[0,1,1,0] = 1
    T01I = [core1*cos(theta), core2]
    core1, core2 = tl.zeros((1,2,2,1), dtype=dtype, device=device), tl.zeros((1,2,2,1), dtype=dtype, device=device)
    core1[0,0,0,0] = core2[0,1,0,0] = 1
    core2[0,0,1,0] = -1
    R01I = [core1*sin(theta), core2]
    #return [*tt_matrix_sum(TTMatrix(T23I), tt_matrix_sum(TTMatrix(T01I), TTMatrix(R01I)))]
    return [*tt_matrix_sum(TTMatrix(T23I), tt_matrix_sum(TTMatrix(T01I), TTMatrix(R01I), device=device), device=device)]


def o4_phases(phases=None, dtype=complex64, device=None):
    """Pair of O4 phase rotations class instances. Each of four phases
    is imparted to each of the 4 states of O4.

    Parameters
    ----------
    phases : list of floats, the four phases to be imparted to the quantum states
    device : string, device on which to run the computation.

    Returns
    -------
    (O4L, O4R)
    """
    L = O4LR(0, phases=phases, dtype=dtype, device=device)
    phases = L.phases
    return [L, O4LR(1, phases=phases, dtype=dtype, device=device)]


class O4LR(Module):
    """Left and right core of the two-qubit O4 phase gate.

    Parameters
    ----------
    phases : list of floats, the four phases to be imparted to the quantum states
    device : string, device on which to run the computation.

    Returns
    -------
    Two-qubit unitary with general phase rotations for O4.
    """
    def __init__(self, position, phases=None, dtype=complex64, device=None):
        super().__init__()
        #self.phases = [Parameter(randn(1, device=device)), Parameter(randn(1, device=device)), Parameter(randn(1, device=device)), Parameter(randn(1, device=device))]
        # TODO: switch case for numpy
        self.phases = [nprandn(1), nprandn(1), nprandn(1), nprandn(1)]
        self.position, self.dtype, self.device = position, dtype, device
        if phases is not None:
            self.phases = [phases[0], phases[1], phases[2], phases[3]]


    def forward(self):
        """Prepares the left or right qubit of the SO4 two-qubit rotation gate for forward contraction
        by calling the forward method and preparing the tt-factorized form of matrix representation.
        Update is based on theta (which is typically updated every epoch through backprop via Pytorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        core1, core2 = tl.zeros((1,2,2,1), dtype=self.dtype, device=self.device), tl.zeros((1,2,2,1), dtype=self.dtype, device=self.device)
        core1[0,0,0,0] = 1
        core2[0,0,0,0] = exp(1j*self.phases[0])
        core2[0,1,1,0] = exp(1j*self.phases[1])
        d0 = [core1, core2]
        core1, core2 = tl.zeros((1,2,2,1), dtype=self.dtype, device=self.device), tl.zeros((1,2,2,1), dtype=self.dtype, device=self.device)
        core1[0,1,1,0] = 1
        core2[0,0,0,0] = exp(1j*self.phases[2])
        core2[0,1,1,0] = exp(1j*self.phases[3])
        d1 = [core1, core2]
        #return tt_matrix_sum(d0, d1)[self.position]
        return tt_matrix_sum(d0, d1, device=self.device)[self.position]


    def reinitialize(self):
        for phase in self.phases:
            #phase.data = randn(1, device=self.device)
            phase = nprandn(1) #TODO switch case for numpy


def exp_pauli_y(dtype=complex64, device=None):
    """Matrix for sin(theta) component of Y-axis rotation in tt-tensor form.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    tt-tensor core, sin(theta) Y-rotation component.
    """
    return tl.tensor([[[[0],[-1]],[[1],[0]]]], dtype=dtype, device=device)


def exp_pauli_x(dtype=complex64, device=None):
    """Matrix for sin(theta) component of X-axis rotation in tt-tensor form.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    tt-tensor core, sin(theta) X-rotation component.
    """
    return tl.tensor([[[[0],[-1j]],[[-1j],[0]]]], dtype=dtype, device=device)
