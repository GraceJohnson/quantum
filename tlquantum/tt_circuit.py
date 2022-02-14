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

import numpy #TODO: remove
import copy

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
        #circuit, expvals1, expvals2, count = self._build_circuit(state), tl.zeros((self.nqubits,), device=op1.device), tl.zeros((self.nqubits,), device=op1.device), 0
        circuit, expvals1, expvals2, count = self._build_circuit(state), tl.zeros((self.nqubits,), device=None), tl.zeros((self.nqubits,), device=None), 0
        if self.contractions['partial_trace_contraction_set'] is None:
            self._generate_partial_trace_contraction_set([core.shape for core in self._build_circuit(state)])
        print("SEGMENTS")
        print(self.segments)
        for ind in range(len(self.segments)):
            partial = self.contractions['partial_trace_contraction_set'][ind](*circuit)
            print(self.contractions['partial_trace_contraction_set'])
            partial_nqubits = int(tl.log2(tl.prod(tl.tensor(partial.shape)))/2)
            print("partial_nqubits : ", partial_nqubits)
            dims = [2 for i in range(partial_nqubits)]
            dims = [dims, dims]
            print("dims : ", dims)
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
        print(self.equations['partial_trace_equation'])
        print(kept_inds)
        return self.contractions['partial_trace_contraction'](*circuit)


    def partial_trace(self, state, kept_inds):
        """Partial trace for specified qubits in the output state of TTCircuit.
           NOTE: this isn't really a property of the circuit and more of the state..

        Parameters
        ----------
        state : tt-tensor, state to be traced
        kept_inds : list of ints, indices of the qubits to be kept in the partial trace

        Returns
        -------
        tensor in matrix form, partial trace of the state
        """
        circuit = state + state
        eq = "bHt,tJu,uLh,bid,dkf,fmh->ikmHJL" # for keeping all indices
        eq = "bHt,tJu,umh,bid,dkf,fmh->ikmHJ" # for keeping first 5 indices
        print("Projector?:")
        print(eq) 
        return contract(eq, *circuit)


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
        print(eq)
        built_layer = self._build_layer()
        circuit = compare_state + built_layer + state
        return contract(eq, *circuit)


    def overlap(self, state, compare_state):
        """Inner product of input state with a comparison state.
           NOTE: this should probably go in tt_state

        Parameters
        ----------
        state : tt-tensor, input state
        compare_state : tt-tensor, state to be compared against

        Returns
        -------
        float, inner product of state with compared state
        """
        eq = overlap_eq(self.nqsystems)
        print(eq)
        circuit = compare_state + state
        return contract(eq, *circuit)


    def apply_H(self, psi_orig):
        """ TODO
        """
        # NOTE: this is all you'd need for product operators, i.e. operators that don't cross bond dims
        # Assumes right node is top node
        psi = copy.deepcopy(psi_orig)
        layer = self._build_layer()

        for node in range(len(psi)-1):
            # apply operator and orthonormalize
            psi[node] = self._apply_local_circuit_toright(psi[node], layer[node])
            psi[node], psi[node+1] = self.local_orthonormalize_left(psi[node], psi[node+1])
        psi[-1] = self._apply_local_circuit_toright(psi[-1], layer[-1])
        psi[-1], _ = self.local_orthonormalize_left(psi[-1], numpy.array([[[1]]]))
        for node in psi:
            assert(self.check_left_orth(node))
        return psi


    def apply_H_rev(self, psi_orig):
        """ TODO
        """
        # Assumes left node is top node
        psi = copy.deepcopy(psi_orig)
        layer = self._build_layer()

        for node in reversed(range(1, len(psi)-1)):
            # apply operator and orthonormalize
            psi[node] = self._apply_local_circuit_toleft(psi[node], layer[node])
            psi[node], psi[node-1] = self.local_orthonormalize_right(psi[node], psi[node-1])
        psi[0] = self._apply_local_circuit_toleft(psi[0], layer[0])
        psi[0], _ = self.local_orthonormalize_right(psi[0], numpy.array([[[1]]]))
        for node in psi:
            assert(self.check_right_orth(node))
        return psi


    def apply_circuit_SCF(self, psi_orig):
        #TODO: replace contract calls with tl.einsum
        """Applies circuit layer to input state, minimizes through SCF and returns evolved state.
               Performs || |chi> - H|psi> || = 0 optimization

        Parameters
        ----------
        psi : tt-tensor, input state |psi>

        Returns
        -------
        tt-tensor, evolved state |chi> minimized with SCF procedure
        """
        return(self.apply_H(psi_orig))
        # NOTE: code below is still under construction. For now, this function
        # only works when two-qubit gates do not cross bond dimenstions

        max_iter = 10
        eps = 1.e-3
        psi = copy.deepcopy(psi_orig)
        layer = self._build_layer()
        mats = [tl.zeros((2,2)) for i in range(len(psi))] # Operator matrices at each node (chi x chi)

        # Initialize a random tensor network to apply operators to
        chi = []
        for phi in psi: 
            chi.append(numpy.random.randn(*phi.shape))
        # Orthogonalize random chi
        chi, _ = self.orthonorm_left(chi) # Right node is top, by convention

        Hpsi = self.apply_H(psi) # Save this to compare to
        assert(abs(self.overlap(Hpsi, Hpsi))-1.0 < 1e-4)

        print("--------------STARTING SCF--------------")
        for i in range(max_iter):

            for phi in chi:
                assert(self.check_left_orth(phi))
            assert(abs(self.overlap(chi, chi))-1.0 < 1e-4)

            # Apply operator to nodes in train and update operator, sweeping left then right
            chi, _ = self.orthonorm_right(chi) # make left node top
            assert(abs(self.overlap(chi, chi))-1.0 < 1e-4)

            chi = self._apply_circuit_toleft(chi, layer, mats)
            #chi[0], lnorm = self.local_orthonormalize_right(chi[0], numpy.array([[[1]]]))
            assert(abs(self.overlap(chi, chi))-1.0 < 1e-4)

            chi, _ = self.orthonorm_left(chi) # Make right node top
            assert(abs(self.overlap(chi, chi))-1.0 < 1e-4)

            chi = self._apply_circuit_toright(chi, layer, mats)
            #chi[-1], rnorm = self.local_orthonormalize_left(chi[-1], numpy.array([[[1]]]))
            assert(abs(self.overlap(chi, chi))-1.0 < 1e-4)

            #chi, _ = self.orthogonal_left(chi) # Right node is top, by convention

            # Check exit condition <psi\tilde|U|psi> ~= 1
            overlap = self.overlap(chi, Hpsi)
            overlap = tl.sqrt(overlap.real**2 + overlap.imag**2)
            print("SCF OVERLAP: ", overlap)
            if abs(1-overlap) < eps:
                return chi

        print("SCF did not converge in {} iterations".format(max_iter))
        return chi


    def _apply_circuit_toright(self, state_orig, layer_orig, mats):
        """Applies gates in layer to input state, sweeping through train left to right
           and updating the state and layer for future iterations

        Parameters
        ----------
        state : tt-tensor, input state |psi>
        layer: tt-tensor, gates in a layer to be applied to |psi>
        mats: tt-tensor, storage for operator matrices (one for each node)

        Returns
        -------
        tt-tensor, updated state U|psi>
        tt-tensor, updated circuit layer (U)
        """
        state = copy.deepcopy(state_orig)
        layer = copy.deepcopy(layer_orig)
        assert(len(state) == len(layer))

        # Swipe through train left to right
        for node in range(len(state)):
            right = True if node == len(state)-1 else False
            left = True if node == 0 else False

            # 1) apply operator (calculate Hpsi ket)
            child = None if left else mats[node-1]
            state[node] = self._apply_local_circuit_toright(state[node], layer[node], child)

            if not right:
                # 2) orthonormalize ket (QR)
                state[node] = self.node_orthonormalize_left(state[node])  # don't update next node, mats take care of this
                #state[node], state[node+1] = self.local_orthonormalize_left(state[node], state[node+1])  #NOTE

                # 3) Rebuild local operator: U_ij = <psi\tilde_i|U|psi_j> (outer product of Upsi)
                braket = [state[node]] + [state[node]]
                eq = 'kli,klj->ij'
                mats[node] = contract(eq, *braket)

        return state


    def _apply_circuit_toleft(self, state_orig, layer_orig, mats):
        """Applies gates in layer to input state, sweeping through train right to left
           and updating the state for future iterations

        Parameters
        ----------
        state : tt-tensor, input state |psi>
        layer: tt-tensor, gates in a layer to be applied to |psi>
        mats: tt-tensor, storage for operator matrices (one for each node)

        Returns
        -------
        tt-tensor, updated state U|psi>
        tt-tensor, updated circuit layer (U)
        """
        state = copy.deepcopy(state_orig)
        layer = copy.deepcopy(layer_orig)
        assert(len(state) == len(layer))

        # Swipe through train right to left
        for node in reversed(range(len(state))):
            right = True if node == len(state)-1 else False
            left = True if node == 0 else False

            # 1) apply operator (calculate Hpsi ket)
            child = None if right else mats[node+1]
            state[node] = self._apply_local_circuit_toleft(state[node], layer[node], child)

            if not left:
                # 2) orthonormalize ket (QR)
                state[node] = self.node_orthonormalize_right(state[node])  # Mats should handle updating the next node
                #state[node], state[node-1] = self.local_orthonormalize_right(state[node], state[node-1])  #NOTE

                # 3) Rebuild local operator: U_ij = <psi\tilde_i|U|psi_j> (outer product of Upsi)
                braket = [state[node]] + [state[node]]
                eq = 'ikl,jkl->ij'
                mats[node] = contract(eq, *braket)

        return state


    def _apply_local_circuit_toright(self, phi_orig, gate, child=None):
        """Applies to a node phi its circuit gate (and its child's operator matrix, if applicable)

        Parameters
        ----------
        phi : tt-tensor, input state at one node (i.e. a basis function |phi>)
        gate: tt-tensor, gate operator belonging to node phi
        child: tt-tensor, (2,2) operator matrix belonging to child (left) of node phi

        Returns
        -------
        tt-tensor, updated basis function U|phi>
        """
        phi = copy.deepcopy(phi_orig)
        # Middle dimension of phi is reserved for operator
        circuit = [gate] + [phi]
        eq = 'aecf,bed->bcd'
        Uphi = contract(eq, *circuit)
        # Left dim of phi is reserved for child node (to the left)
        if child is not None:
            circuit = [child] + [Uphi] 
            eq = 'bf,bcd->fcd'
            Uphi = contract(eq, *circuit)
        assert(Uphi.shape == phi.shape)
        return Uphi


    def _apply_local_circuit_toleft(self, phi_orig, gate, child=None):
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
        phi = copy.deepcopy(phi_orig)
        # Middle dimension of phi is reserved for operator
        circuit = [gate] + [phi]
        eq = 'aecf,bed->bcd'
        Uphi = contract(eq, *circuit)
        # Right dim of phi is reserved for child node (to the right)
        if child is not None:
            circuit = [child] + [Uphi] 
            eq = 'df,bcd->bcf'
        assert(Uphi.shape == phi.shape)
        return Uphi


    def local_orthonormalize_left(self, phi, phinext):
        """ TODO
        """
        phi = phi.transpose((1,0,2)) # switch convention
        s = phi.shape
        phi, R = tl.qr(phi.reshape(s[0]*s[1],s[2]))
        phi = phi.reshape((s[0],s[1],phi.shape[1]))
        phinext = phinext.transpose((1,0,2)) # switch convention
        phinext = numpy.tensordot(R, phinext, (1,1)).transpose((1,0,2))
        phi = phi.transpose((1,0,2)) # switch back 
        phinext = phinext.transpose((1,0,2)) # switch back
        return phi, phinext
   

    def local_orthonormalize_right(self, phi, phiprev):
        """ TODO
        """
        phi = phi.transpose((1,0,2)) # switch convention
        phi = phi.transpose((0,2,1))
        s = phi.shape
        phi, R = tl.qr(phi.reshape(s[0]*s[1],s[2]))
        phi = phi.reshape((s[0],s[1],phi.shape[1])).transpose((0,2,1))
        phiprev = phiprev.transpose((1,0,2)) # switch convention
        phiprev = numpy.tensordot(phiprev, R, (2,1))
        phi = phi.transpose((1,0,2)) # switch back
        phiprev = phiprev.transpose((1,0,2)) # switch back
        return phi, phiprev
   

    def node_orthonormalize_left(self, phi):
        """ TODO
        """
        phi = phi.transpose((1,0,2)) # switch convention
        s = phi.shape
        phi, R = tl.qr(phi.reshape(s[0]*s[1],s[2]))
        phi = phi.reshape((s[0],s[1],phi.shape[1]))
        phi = phi.transpose((1,0,2)) # switch back
        return phi
   

    def node_orthonormalize_right(self, phi):
        """ TODO
        """
        phi = phi.transpose((1,0,2)) # switch convention
        phi = phi.transpose((0,2,1))
        s = phi.shape
        phi, R = tl.qr(phi.reshape(s[0]*s[1],s[2]))
        phi = phi.reshape((s[0],s[1],phi.shape[1])).transpose((0,2,1))
        phi = phi.transpose((1,0,2)) # switch back
        return phi
   

    def orthogonal_right(self, psi): # This orthogonalizes but not normalizes, use on random tensors once
        """ TODO
        """
        psi, nrm = self.orthonorm_right(psi)
        #if nrm < 0:
        #    psi[0] = -psi[0]
        psi[0] = psi[0]/nrm
        return psi, nrm


    def orthogonal_left(self, psi): # This orthogonalizes but not normalizes, use on random tensors once
        """ TODO
        """
        psi, nrm = self.orthonorm_left(psi)
        #if nrm < 0:
        #    psi[-1] = -psi[-1]
        psi[-1] = psi[-1]/nrm
        return psi, nrm


    def orthonorm_right(self, psi):
        """ TODO
        """
        for i in reversed(range(1, len(psi))):
            psi[i], psi[i-1] = self.local_orthonormalize_right(psi[i], psi[i-1])
        # first tensor
        psi[0], Aprev = self.local_orthonormalize_right(psi[0], numpy.array([[[1]]]))
        nrm = Aprev[0,0,0].real
        return psi, nrm


    def orthonorm_left(self, psi):
        """ TODO
        """
        for i in range(len(psi)-1):
            psi[i], psi[i+1] = self.local_orthonormalize_left(psi[i], psi[i+1])
        # last tensor
        psi[-1], Anext = self.local_orthonormalize_left(psi[-1], numpy.array([[[1]]]))
        nrm = Anext[0,0,0].real
        return psi, nrm


    def check_left_orth(self, A): # To be used after a left orthonormalize
        """ TODO
        """
        eq = 'ijk,ijl->kl'
        matA = contract(eq, A, A)
        I = numpy.identity(matA.shape[0])
        return abs(numpy.linalg.norm(I-matA)) < 1e-6

        
    def check_right_orth(self, A): # To be used after a right orthonormalize
        """ TODO
        """
        eq = 'ijk,ljk->il'
        matA = contract(eq, A, A)
        I = numpy.identity(matA.shape[0])
        return abs(numpy.linalg.norm(I-matA)) < 1e-6
        
       
    def test_orthonorm(self, psi):
        """ TODO
        """
        print("\nTesting orthonormalization with random tensors")
        rand = []
        for phi in psi: 
            print(phi.shape)
            rand.append(numpy.random.randn(*phi.shape))
        for r in rand:
            print(r)
        randr, _ = self.orthogonal_right(rand)
        overlap = self.overlap(randr, rand)
        overlap = tl.sqrt(overlap.real**2 + overlap.imag**2)
        print("OVERLAP: ", overlap)
        randr_nrm = self.orthonorm_right(randr)
        overlap = self.overlap(randr_nrm, randr_nrm)
        overlap = tl.sqrt(overlap.real**2 + overlap.imag**2)
        print("SELF OVERLAP: ", overlap)
        randl, _ = self.orthogonal_left(rand)
        overlap = self.overlap(randl, rand)
        overlap = tl.sqrt(overlap.real**2 + overlap.imag**2)
        print("OVERLAP: ", overlap)


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
            print("partial trace equation set")
            print(equation)
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
