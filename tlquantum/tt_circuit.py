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

from .density_tensor import DensityTensor
from .tt_precontraction import qubits_contract, layers_contract
from .tt_contraction import contraction_eq, overlap_eq


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


class TTCircuit(Module):
    """A simulator for variational quantum circuits using tensor ring tensors
    with PyTorch Autograd support.
    Can be used to compute: 1) the expectation value of an operator, 2) the single-qubit
    measurements of circuit's qubits; specifically for Multi-Basis Encoding [1], and 
    3) the partial trace of the circuit - all with Autograd support.
    [1] T. L. Patti, J. Kossaifi, A. Anandkumar, and S. F. Yelin, "Variational Quantum Optimization with Multi-Basis Encodings," (2021), arXiv:2106.13304.

    Parameters
    ----------
    unitaries : list of TT Unitaries, circuit operations
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    ncontral  : int, number of unitaries to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    equations : dictionary, accepts pre-computed/recycled equations for
               operator expectation values, single-qubit measurements,
               and partial traces.
    contractions : dictionary, accepts pre-computed/recycled paths for
               operator expectation values, single-qubit measurements,
               and partial traces.
    max_partial_trace_size : int, the maximum number of cores to keep in
               a single partial trace for single-qubit measurements.
    device : string, device on which to run the computation.

    Returns
    -------
    TTCircuit
    """
    def __init__(self, unitaries, ncontraq, ncontral, equations=None, contractions=None, max_partial_trace_size=4, device='cpu'): 
        super().__init__()
        self.nqubits, self.nlayers, self.ncontraq, self.ncontral = unitaries[0].nqubits, len(unitaries), ncontraq, ncontral
        self.nqsystems, self.nlsystems, self.layer_rep = int(ceil(self.nqubits/self.ncontraq)), int(ceil(self.nlayers/self.ncontral)), 2
        if equations is None:
            equations = {'expectation_value_equation': None, 'partial_trace_equation': None, 'partial_trace_equation_set': None}
        if contractions is None:
            contractions = {'expectation_value_contraction': None, 'partial_trace_contraction': None, 'partial_trace_contraction_set': None}
        self.equations, self.contractions = equations, contractions
        self.device, self.nparam_layers, contrsets = device, int(ceil(self.nlayers/self.layer_rep)), list(range(self.nqubits))
        self.contrsets = [contrsets[i:i+self.ncontraq] for i in range(0, self.nqubits, self.ncontraq)]
        self.max_partial_trace_size, segments = max_partial_trace_size, list(range(self.nqsystems))
        self.segments = [segments[0:self.max_partial_trace_size]] + [segments[i:i+self.max_partial_trace_size] for i in range(self.max_partial_trace_size, self.nqsystems, self.max_partial_trace_size)]
        self.unitaries = ModuleList(unitaries)


    def forward_expectation_value(self, state, operator, precontract_operator=True):
        """Full expectation value of self.measurement of the unitary evolved state.

        Parameters
        ----------
        state : tt-tensor, input state to be evolved by unitary
        operator: tt-tensor, operator of which to get expectation value
        precontract_operator: bool, if true, the operator must be precontracted before main contraction pass

        Returns
        -------
        float, expectation value of self.measurement with unitary evolved state
        """
        if precontract_operator:
            operator = qubits_contract(operator, self.ncontraq, contrsets=self.contrsets)
        circuit = self._build_circuit(state, operator=operator)
        if self.contractions['expectation_value_contraction'] is None:
            if self.equations['expectation_value_equation'] is None:
                self.equations['expectation_value_equation'] = contraction_eq(self.nqsystems, 2*self.nlsystems+1)
            self.contractions['expectation_value_contraction'] = contract_expression(self.equations['expectation_value_equation'], *[core.shape for core in circuit])
        return self.contractions['expectation_value_contraction'](*circuit)


    def forward_single_qubit(self, state, op1, op2):
        """Expectation values of op for each qubit of state. Takes partial trace of subset of qubits and then
        takes single-operator measurements of these qubits.
        Specifically useful for Multi-Basis Encoding [1] (MBE).
        [1] T. L. Patti, J. Kossaifi, A. Anandkumar, and S. F. Yelin, "Variational Quantum Optimization with Multi-Basis Encodings," (2021), arXiv:2106.13304.

        Parameters
        ----------
        state : tt-tensor, input state to be evolved by unitary
        op1 : tt-tensor, first single-measurement operator
        op2 : tt-tensor, second single-measurement operator

        Returns
        -------
        float, expectation value of self.measurement with unitary evolved state
        """
        circuit, expvals1, expvals2, count = self._build_circuit(state), tl.zeros((self.nqubits,), device=op1.device), tl.zeros((self.nqubits,), device=op1.device), 0
        if self.contractions['partial_trace_contraction_set'] is None:
            self._generate_partial_trace_contraction_set([core.shape for core in self._build_circuit(state)])
        for ind in range(len(self.segments)):
            partial = self.contractions['partial_trace_contraction_set'][ind](*circuit)
            partial_nqubits = int(tl.log2(tl.prod(tl.tensor(partial.shape)))/2)
            dims = [2 for i in range(partial_nqubits)]
            dims = [dims, dims]
            partial = DensityTensor(partial.reshape(sum(dims, [])), dims)
            for qubit_ind in range(partial_nqubits):
                qubit = partial.partial_trace(list(range(qubit_ind, qubit_ind+1)))[0].reshape(2,2)
                expvals1[count], expvals2[count], count = tl.sum(tl.diag(tl.dot(qubit, op1))), tl.sum(tl.diag(tl.dot(qubit, op2))), count+1
        return expvals1, expvals2


    def forward_partial_trace(self, state, kept_inds):
        """Partial trace for specified qubits in the output state of TTCircuit.

        Parameters
        ----------
        state : tt-tensor, input state to be evolved by unitary
        kept_inds : list of ints, indices of the qubits to be kept in the partial trace

        Returns
        -------
        tensor in matrix form, partial trace of the circuit's output state
        """
        circuit = self._build_circuit(state)
        if self.contractions['partial_trace_contraction'] is None:
            if self.equations['partial_trace_equation'] is None:
                self.equations['partial_trace_equation'] = contraction_eq(self.nqsystems, 2*self.nlsystems, kept_inds=kept_inds)
            self.contractions['partial_trace_contraction'] = contract_expression(self.equations['partial_trace_equation'], *[core.shape for core in circuit])
        return self.contractions['partial_trace_contraction'](*circuit)


    def state_inner_product(self, state, compare_state):
        """Inner product of input state evolved in unitary with a comparison state.

        Parameters
        ----------
        state : tt-tensor, input state to be evolved by unitary
        compare_state : tt-tensor, input state to be compared with evolved state

        Returns
        -------
        float, inner product of evolved state with compared state
        """
        eq = contraction_eq(self.nqsystems, self.nlsystems)
        built_layer = self._build_layer()
        circuit = compare_state + built_layer + state
        return contract(eq, *circuit)


    def overlap(self, state, compare_state):
        """Inner product of input state with a comparison state.

        Parameters
        ----------
        state : tt-tensor, input state
        compare_state : tt-tensor, state to be compared against

        Returns
        -------
        float, inner product of state with compared state
        """
        eq = overlap_eq(self.nqsystems)
        circuit = compare_state + state
        return contract(eq, *circuit)


    def apply_circuit_SCF(self, psi):
        """Applies circuit to input state, minimizes through SCF and returns evolved state.
               Performs || |psi'> - U|psi> || = 0 optimization
           NOTE: only works for a single layer of the circuit (for now)

        Parameters
        ----------
        psi : tt-tensor, input state |psi>

        Returns
        -------
        tt-tensor, evolved state |psi'> minimized with SCF procedure
        """
        #TODO: replace contract calls with tl.einsum
        max_iter = 10
        eps = 1.e-2
        layer = self._build_layer()
        Upsi = []
        for i in range(max_iter):
            # Apply operator to nodes in train and update operator, sweeping left then right
            Upsi, layer = self._apply_circuit_toleft(psi, layer)
            Upsi, layer = self._apply_circuit_toright(Upsi, layer)
            # Check exit condition <psi\tilde|U|psi> ~= 1
            overlap = self.overlap(Upsi, psi)
            print("OVERLAP: ", overlap)
            if abs(1-overlap) < eps: #TODO: complex values of overlap?
                return Upsi
            psi = Upsi
        print("SCF did not converge in {} iterations".format(max_iter))
        return Upsi


    def _apply_circuit_toright(self, state, layer):
        """Applies gates in layer to input state, sweeping through train left to right
           and updating the state and layer for future iterations

        Parameters
        ----------
        state : tt-tensor, input state |psi>
        layer: tt-tensor, gates in a layer to be applied to |psi>

        Returns
        -------
        tt-tensor, updated state U|psi>
        tt-tensor, updated circuit layer (U)
        """
        assert(len(state) == len(layer))
        Upsi = []
        # Swipe through train left to right
        for node in range(len(state)):
            right = True if node == len(state)-1 else False
            left = True if node == 0 else False

            # 1) apply operator (calculate Upsi ket)
            phi = state[node]
            U = layer[node]
            child = None if left else layer[node-1]
            Uphi = self._apply_local_circuit_toright(phi, U, child)

            # 2) orthonormalize ket (QR)
            UphiT, R = tl.qr(tl.transpose(Uphi[:,:,0]))
            Uphi[:,:,0] = tl.transpose(UphiT) # We can throw away the R degrees of freedom

            # 3) update basis U_ij = <psi\tilde_i|U|psi_j> (outer product of Upsi)
            if not right:
                # Update leaf U
                braket = [Uphi[0,:,0]] + [Uphi[0,:,0]]
                eq = 'i,j->ij' 
                U[0,:,:,0] = contract(eq, *braket)
                # Update child U
                if not left:
                    braket = [Uphi[1,:,0]] + [Uphi[1,:,0]]
                    child[0,:,:,0] = contract(eq, *braket)

            Upsi = Upsi + [Uphi]
        return Upsi, layer


    def _apply_circuit_toleft(self, state, layer):
        """Applies gates in layer to input state, sweeping through train right to left
           and updating the state and layer for future iterations

        Parameters
        ----------
        state : tt-tensor, input state |psi>
        layer: tt-tensor, gates in a layer to be applied to |psi>

        Returns
        -------
        tt-tensor, updated state U|psi>
        tt-tensor, updated circuit layer (U)
        """
        assert(len(state) == len(layer))
        Upsi = []
        # Swipe through train right to left
        for node in reversed(range(len(state))):
            right = True if node == len(state)-1 else False
            left = True if node == 0 else False

            # 1) apply operator (calculate Upsi ket)
            phi = state[node]
            U = layer[node]
            child = None if right else layer[node+1]
            Uphi = self._apply_local_circuit_toleft(phi, U, child)

            # 2) orthonormalize ket (QR)
            Uphi[0,:,:], R = tl.qr(Uphi[0,:,:]) # We can throw away the R degrees of freedom

            # 3) update basis U_ij = <psi\tilde_i|U|psi_j> (outer product of Upsi)
            if not left:
                # Update leaf U
                braket = [Uphi[0,:,0]] + [Uphi[0,:,0]]
                eq = 'i,j->ij' 
                U[0,:,:,0] = contract(eq, *braket)
                # Update child U
                if not right:
                    braket = [Uphi[0,:,1]] + [Uphi[0,:,1]]
                    child[0,:,:,0] = contract(eq, *braket)

            Upsi = [Uphi] + Upsi
        return Upsi, layer


    def _apply_local_circuit_toright(self, phi, gate, child=None):
        """Applies to a node phi its circuit gate (and its child's circuit gate, if applicable)

        Parameters
        ----------
        phi : tt-tensor, input state at one node (i.e. a basis function |phi>)
        gate: tt-tensor, gate operator belonging to node phi
        child: tt-tensor, gate operator belonging to child (left) of node phi

        Returns
        -------
        tt-tensor, updated basis function U|phi>
        """
        # First slice of phi is reserved for operator
        circuit = [gate] + [phi[0,:,:]]
        eq = 'aecf,eb->cb'
        Uphi = contract(eq, *circuit)
        # Second slice of phi is reserved for child node (to the left)
        if child is not None:
            circuit = [child] + [phi[1,:,:]] 
            Uphi = tl.stack((Uphi, contract(eq, *circuit)), axis=0)
        else:
            Uphi = Uphi.reshape(phi.shape)
        assert(Uphi.shape == phi.shape)
        return Uphi


    def _apply_local_circuit_toleft(self, phi, gate, child=None):
        """Applies to a node phi its circuit gate (and its child's circuit gate, if applicable)

        Parameters
        ----------
        phi : tt-tensor, input state at one node (i.e. a basis function |phi>)
        gate: tt-tensor, gate operator belonging to node phi
        child: tt-tensor, gate operator belonging to child (right) of node phi

        Returns
        -------
        tt-tensor, updated basis function U|phi>
        """
        # First slice of phi is reserved for operator
        circuit = [gate] + [phi[:,:,0]]
        eq = 'aecf,be->bc'
        Uphi = contract(eq, *circuit)
        # Second slice of phi is reserved for child node (to the right)
        if child is not None:
            circuit = [child] + [phi[:,:,1]] 
            Uphi = tl.stack((Uphi, contract(eq, *circuit)), axis=2)
        else:
            Uphi = Uphi.reshape(phi.shape)
        assert(Uphi.shape == phi.shape)
        return Uphi


    def _build_circuit(self, state, operator=[]):
        """Prepares the circuit gates and operators for forward pass of the tensor network.

        Parameters
        ----------
        state : tt-tensor, input state to be evolved by unitary
        operators : tt-tensor, operator for which to calculate the expectation value, used by the
                    forward_expectation_value method.

        Returns
        -------
        list of tt-tensors, unitaries and operators of the TTCircuit, ready for contraction
        """
        built_layer = self._build_layer()
        built_layer_dagger = [tt_dagger(built_layer[i]) for n in range(self.nlsystems, 0, -1) for i in range((n-1)*self.nqsystems, n*self.nqsystems)]
        return state + built_layer + operator + built_layer_dagger + state


    def _build_layer(self):
        """Prepares the ket unitary gates gates for forward pass of the tensor network.

        Returns
        -------
        list of tt-tensors, unitaries of the TTCircuit, ready for contraction
        """
        built_layer = [unitary.forward() for unitary in self.unitaries]
        if self.nlayers % self.layer_rep > 0:
            built_layer = built_layer[:self.nlayers]
        return layers_contract(built_layer, self.ncontral)


    def _generate_partial_trace_contraction_set(self, shapes):
        """Populates the partial trace equations and contractions attributes for each of the the single-qubit
        measurements, as required by Multi-Basis Encoding.

        Parameters
        ----------
        shapes : list of shape tuples, the shapes of the tt-tensors to be contracted over
        """
        partial_trace_contraction_set = []
        if self.equations['partial_trace_equation_set'] is None:
            self._generate_partial_trace_equation_set()
        for equation in self.equations['partial_trace_equation_set']:
            partial_trace_contraction_set.append(contract_expression(equation, *shapes))
        self.contractions['partial_trace_contraction_set'] = partial_trace_contraction_set


    def _generate_partial_trace_equation_set(self):
        """Generates the partial trace equations for each of the the single-qubit measurements,
        as required by Multi-Basis Encoding.

        """
        partial_trace_equation_set = []
        for segment in self.segments:
            equation = contraction_eq(self.nqsystems, 2*self.nlsystems, kept_inds=segment)
            partial_trace_equation_set.append(equation)
        self.equations['partial_trace_equation_set'] = partial_trace_equation_set


def tt_dagger(tt):
    """Transpose single-qubit matrices in tt-tensor format.

    Parameters
    ----------
    tt : tt-tensor

    Returns
    -------
    Transpose of tt
    """
    #return tl.conj(transpose(tt, 1, 2)) #pytorch
    return tl.conj(transpose(tt, (0,2,1,3)))  #numpy interface
