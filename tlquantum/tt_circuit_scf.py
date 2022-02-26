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



def apply_U(psi_orig, circuit):
    """ TODO
    """
    # NOTE: this is all you'd need for product operators, i.e. operators that don't cross bond dims
    # Assumes right node is top node
    psi = copy.deepcopy(psi_orig)
    layer = circuit._build_layer()

    for node in range(len(psi)-1):
        # apply operator and orthonormalize
        psi[node] = _apply_local_circuit_toright(psi[node], layer[node])
        psi[node], psi[node+1] = _local_orthonormalize_left(psi[node], psi[node+1])
    psi[-1] = _apply_local_circuit_toright(psi[-1], layer[-1])
    psi[-1], _ = _local_orthonormalize_left(psi[-1], numpy.array([[[1]]]))
    for node in psi:
        assert(check_left_orth(node))

    """
    densepsi = tl.tt_to_tensor(psi_orig)
    denseop = tl.tt_matrix_to_tensor(layer)
    n = len(densepsi.shape)
    check = tl.tensordot(denseop, densepsi, axes=n)
    densepsi = tl.tt_to_tensor(psi)
    """
  
    return psi


def apply_U_rev(psi_orig, circuit):
    """ TODO
    """
    # Assumes left node is top node
    psi = copy.deepcopy(psi_orig)
    layer = circuit._build_layer()

    for node in reversed(range(1, len(psi)-1)):
        # apply operator and orthonormalize
        psi[node] = _apply_local_circuit_toleft(psi[node], layer[node])
        psi[node], psi[node-1] = _local_orthonormalize_right(psi[node], psi[node-1])
    psi[0] = _apply_local_circuit_toleft(psi[0], layer[0])
    psi[0], _ = _local_orthonormalize_right(psi[0], numpy.array([[[1]]]))
    for node in psi:
        assert(check_right_orth(node))
    return psi


def apply_circuit_SCF(psi_orig, layers):
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
    nqsystems = layers[0].nqsystems

    # NOTE: code below is still under construction. For now, this function
    # only works for a single two-qubit operation

    #numpy.random.seed(4) # REMOVE after debugging

    #test_orthonorm(psi_orig, nqsystems)

    max_iter = 1
    eps = 1.e-3
    psi = copy.deepcopy(psi_orig)
    ops = []
    for op in layers:
        print(op)
        ops.append(op._build_layer())

    # Initialize a random tensor network to apply operators to
    chi = []
    for phi in psi: 
        chi.append(numpy.random.randn(*phi.shape))
        # TODO: only do random matrices on node that will actually get updated... otherwise just make a copy
        #        i.e. if it's an identity operator at that node, copy
        #chi.append(copy.deepcopy(phi))

    # Orthogonalize random chi
    chi, _ = _orthonorm_left(chi) # Right node is top, by convention

    print("--------------STARTING SCF--------------")
    for i in range(max_iter):

        """
        dense_psi = tl.tt_to_tensor(psi)
        denseops = []
        for op in ops:
            denseops.append(tl.tt_matrix_to_tensor(op))
        denseop = sum(denseops)
        n = len(dense_psi.shape)
        check = tl.tensordot(denseop, dense_psi, axes=n)
        """
  
        # Apply operator to nodes in train and update operator, sweeping left then right
        # -------------------------TO LEFT ---------------------------#
        psi, _ = _orthonorm_right(psi) # make left node top
        chi, _ = _orthonorm_right(chi) # make left node top
        mats = _represent_mats_left(chi, psi, ops)

        assert(abs(overlap(chi, chi, nqsystems))-1.0 < 1e-4)
        assert(abs(overlap(psi, psi, nqsystems))-1.0 < 1e-4)
        for phi in psi:
            assert(check_right_orth(phi))

        chi = _apply_circuit_toleft(psi, ops, mats)

        assert(abs(overlap(chi, chi, nqsystems))-1.0 < 1e-4)
        for phi in chi:
            assert(check_right_orth(phi))

        # ------------CHECK CONVERGENCE-------------#
        # Check exit condition <psi\tilde|U|psi> ~= 1
        umats = _represent_mats_left(chi, psi, ops)
        # Apply top
        hphi = tl.zeros((psi[0].shape), dtype=complex64)
        for l in range(len(ops)):
            hphi += _apply_local_circuit_toleft(psi[0], ops[l][0], umats[l][1])
        hphi = _node_orthonormalize_right(hphi)

        braket = [hphi] + [chi[0]]
        eq = 'ikl,jkl->ij'  # TODO: check this
        dot = contract(eq, *braket)
        fidelity = abs(dot[0][0])**2
        print("SCF FIDELITY (LEFT):", fidelity)

        psi, _ = _orthonorm_left(psi) # Make right node top
        chi, _ = _orthonorm_left(chi) # Make right node top

        if abs(1-fidelity) < eps:
            return chi

        # -------------------------TO RIGHT ---------------------------#
        psi, _ = _orthonorm_left(psi) # Make right node top
        chi, _ = _orthonorm_left(chi) # Make right node top
        mats = _represent_mats_right(chi, psi, ops)

        assert(abs(overlap(chi, chi, nqsystems))-1.0 < 1e-4)
        assert(abs(overlap(psi, psi, nqsystems))-1.0 < 1e-4)
        for phi in psi:
            assert(check_left_orth(phi))
        for phi in chi:
            assert(check_left_orth(phi))

        chi = _apply_circuit_toright(psi, ops, mats)

        assert(abs(overlap(chi, chi, nqsystems))-1.0 < 1e-4)
        for phi in chi:
            assert(check_left_orth(phi))

        # ------------CHECK CONVERGENCE-------------#
        # Check exit condition <psi\tilde|U|psi> ~= 1
        umats = _represent_mats_right(chi, psi, ops)
        # Apply top
        hphi = tl.zeros((psi[-1].shape), dtype=complex64)
        for l in range(len(ops)):
            hphi += _apply_local_circuit_toright(psi[-1], ops[l][-1], umats[l][-2])
        hphi = _node_orthonormalize_left(hphi)

        braket = [hphi] + [chi[-1]]
        eq = 'kli,klj->ij'  # TODO: check this
        dot = contract(eq, *braket)
        fidelity = abs(dot[0][0])**2
        print("SCF FIDELITY (RIGHT):", fidelity)

        if abs(1-fidelity) < eps:
            return chi

    print("SCF did not converge in {} iterations".format(max_iter))
    return chi


# ----------------------------------------------------------------------------#
#                    Helper functions for main routines                       #
# ----------------------------------------------------------------------------#

def _apply_circuit_toright(psi, layer_orig, mats):
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
    hpsi = copy.deepcopy(psi) # can actually just be empty, or better passed in as input
    layer = copy.deepcopy(layer_orig)
    assert(len(psi) == len(layer[0]))

    print("\n ------------- SWEEP RIGHT ---------------\n")
    # Swipe through train left to right
    for node in range(len(psi)):
        print("NODE: ", node)
        right = True if node == len(psi)-1 else False
        left = True if node == 0 else False

        # 1) apply operator (calculate Hpsi ket)
        hphi = tl.zeros((psi[node].shape), dtype=complex64)
        for l in range(len(layer)):
            child = None if left else mats[l][node-1]
            #hphi += _apply_local_circuit_toright(psi[node], layer[l][node], child)
            mphi = _apply_local_circuit_toright(psi[node], layer[l][node], child)
            print("layer op:")
            print(layer[l][node])
            print("child:")
            print(child)
            hphi += mphi

        print("hphi")
        print(hphi)
        hphi = _node_orthonormalize_left(hphi)  # don't update next node, mats take care of this
        print("orthonormalized hphi")
        print(hphi)
        if not right:
            print("REBUILDING OP")
            # 2) orthonormalize ket (QR)
            #hphi = _node_orthonormalize_left(hphi)  # don't update next node, mats take care of this

            # 3) Rebuild local operator: U_ij = <psi\tilde_i|U|psi_j> (outer product of Upsi)
            for l in range(len(layer)):
                child = None if left else mats[l][node-1]
                mats[l][node] = _represent_mat_right(hphi, psi[node], layer[l][node], child)
                print(mats[l][node])

        hpsi[node] = hphi

    return hpsi


def _apply_circuit_toleft(psi, layer_orig, mats):
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
    hpsi = copy.deepcopy(psi) # Just needs to be same shape
    layer = copy.deepcopy(layer_orig)
    assert(len(psi) == len(layer[0]))

    print("\n ------------- SWEEP LEFT ---------------\n")
    # Swipe through train right to left
    for node in reversed(range(len(psi))):
        print("NODE: ", node)
        right = True if node == len(psi)-1 else False
        left = True if node == 0 else False

        # 1) apply operator (calculate Hpsi ket)
        hphi = tl.zeros((psi[node].shape), dtype=complex64)
        for l in range(len(layer)):
            child = None if right else mats[l][node+1]
            hphi += _apply_local_circuit_toleft(psi[node], layer[l][node], child)
            print("layer op:")
            print(layer[l][node])
            print("child:")
            print(child)

        print("hphi")
        print(hphi)
        hphi = _node_orthonormalize_right(hphi)  # Mats should handle updating the next node
        print("orthonormalized hphi")
        print(hphi)
        if not left:
            print("REBUILDING OP, node ", node)
            # 2) orthonormalize ket (QR)
            #hphi = _node_orthonormalize_right(hphi)  # Mats should handle updating the next node

            # 3) Rebuild local operator: U_ij = <psi\tilde_i|U|psi_j> (outer product of Upsi)
            for l in range(len(layer)):
                child = None if right else mats[l][node+1]
                mats[l][node] = _represent_mat_left(hphi, psi[node], layer[l][node], child)
                print(mats[l][node])
  
        hpsi[node] = hphi

    return hpsi


def _apply_local_circuit_toright(phi_orig, gate, child=None):
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
        #eq = 'fb,bcd->fcd'
        Uphi = contract(eq, *circuit)
    assert(Uphi.shape == phi.shape)
    return Uphi


def _apply_local_circuit_toleft(phi_orig, gate, child=None):
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
        #eq = 'fb,bcd->fcd'
        Uphi = contract(eq, *circuit)
    assert(Uphi.shape == phi.shape)
    return Uphi


def _reduced_density_matrix_right(psi_orig):
    """ TODO
    """
    psi = copy.deepcopy(psi_orig)
    rho = [tl.zeros((2,2)) for i in range(len(psi))] # Rho at each node (TODO: chi x chi)
    # Loop top-down (right is top), top node is I
    rho[-1] = tl.tensor([[1,0],[0,1]], dtype=complex64) # (TODO: chi x chi)
    for node in reversed(range(len(psi)-1)):
        circuit = [rho[node+1]] + [psi[node+1]]
        eq = 'ab,acd->bcd' # matrix-tensor product
        ket = contract(eq, *circuit)
        circuit = [psi[node+1]] + [ket]
        eq = 'acd,bcd->ab' # tensor contraction
        rho[node] = contract(eq, *circuit)
    return rho
    

def _reduced_density_matrix_left(psi_orig):
    """ TODO
    """
    psi = copy.deepcopy(psi_orig)
    rho = [tl.zeros((2,2)) for i in range(len(psi))] # Rho at each node (TODO: chi x chi)
    # Loop top-down (left is top), top node is I
    rho[0] = tl.tensor([[1,0],[0,1]], dtype=complex64) # (TODO: chi x chi)
    for node in range(1, len(psi)):
        circuit = [rho[node-1]] + [psi[node-1]]
        eq = 'ab,cda->cdb' # matrix-tensor product
        ket = contract(eq, *circuit)
        circuit = [psi[node-1]] + [ket]
        eq = 'cda,cdb->ab' # tensor contraction
        rho[node] = contract(eq, *circuit)
    return rho
    

def _represent_mats_left(bra, ket, layer):
    """ TODO
    """
    # NOTE: don't do this if node is not active or if it's top node
    # Top node is a matrix of zeros(it never gets applied)
    assert(len(bra) == len(ket))
    mats = []
    # Loop through sum of products
    for l in range(len(layer)):
        mat = [tl.zeros((2,2)) for i in range(len(ket))] # Operator matrices at each node (chi x chi)
        # Loop through nodes (that aren't top node, left)
        for node in reversed(range(1, len(ket))):
            child = None if node==len(ket)-1 else mat[node+1]
            mat[node] = _represent_mat_left(bra[node], ket[node], layer[l][node], child)
        mats.append(mat)
    return mats


def _represent_mat_left(bra, ket, layer, child=None):
    """ TODO
    """
    # Apply child operators to ket (form Hket) (but don't change ket)
    hphi = _apply_local_circuit_toleft(ket, layer, child)
    assert(ket.shape == hphi.shape)
    assert(bra.shape == hphi.shape)
    # Represent local mat in new basis
    braket = [bra] + [hphi]
    #eq = 'ikl,jkl->ij'
    eq = 'jkl,ikl->ij'
    return contract(eq, *braket)


def _represent_mats_right(bra, ket, layer):
    """ TODO
    """
    # NOTE: don't do this if node is not active or if it's top node
    assert(len(bra) == len(ket))
    mats = []
    # Loop through sum of products
    for l in range(len(layer)):
        mat = [tl.zeros((2,2)) for i in range(len(ket))] # Operator matrices at each node (chi x chi)
        # Loop through nodes (that aren't top node, right)
        for node in range(len(ket)-1):
            child = None if node==0 else mat[node-1]
            mat[node] = _represent_mat_right(bra[node], ket[node], layer[l][node], child)
        mats.append(mat)
    return mats


def _represent_mat_right(bra, ket, layer, child=None):
    """ TODO
    """
    # Apply child operators to ket (form Hket) (but don't change ket)
    hphi = _apply_local_circuit_toright(ket, layer, child)
    assert(ket.shape == hphi.shape)
    assert(bra.shape == hphi.shape)
    # Represent local mat in new basis
    braket = [bra] + [hphi]
    #eq = 'kli,klj->ij'
    eq = 'klj,kli->ij'
    return contract(eq, *braket)


def _node_orthonormalize_left(phi):
    """ TODO
    """
    phi = phi.transpose((1,0,2)) # switch convention
    s = phi.shape
    phi, R = tl.qr(phi.reshape(s[0]*s[1],s[2]))
    # EXPERIMENTAL ############################
    if R[0][0] < -1.e-14:
        R = -R
        phi = -phi
    ############################################
    phi = phi.reshape((s[0],s[1],phi.shape[1]))
    phi = phi.transpose((1,0,2)) # switch back
    return phi


def _node_orthonormalize_right(phi):
    """ TODO
    """
    phi = phi.transpose((1,0,2)) # switch convention
    phi = phi.transpose((0,2,1))
    s = phi.shape
    phi, R = tl.qr(phi.reshape(s[0]*s[1],s[2]))
    # EXPERIMENTAL ############################
    if R[0][0] < -1.e-14:
        R = -R
        phi = -phi
    ############################################
    phi = phi.reshape((s[0],s[1],phi.shape[1])).transpose((0,2,1))
    phi = phi.transpose((1,0,2)) # switch back
    return phi


def _orthogonal_right(psi): # This orthogonalizes but not normalizes, use on random tensors once
    """ TODO
    """
    psi, nrm = _orthonorm_right(psi)
    #if nrm < 0:
    #    psi[0] = -psi[0]
    psi[0] = psi[0]/nrm
    return psi, nrm


def _orthogonal_left(psi): # This orthogonalizes but not normalizes, use on random tensors once
    """ TODO
    """
    psi, nrm = _orthonorm_left(psi)
    #if nrm < 0:
    #    psi[-1] = -psi[-1]
    psi[-1] = psi[-1]/nrm
    return psi, nrm


def _orthonorm_right(psi):
    """ TODO
    """
    for i in reversed(range(1, len(psi))):
        psi[i], psi[i-1] = _local_orthonormalize_right(psi[i], psi[i-1])
    # first tensor
    psi[0], Aprev = _local_orthonormalize_right(psi[0], numpy.array([[[1]]]))
    nrm = Aprev[0,0,0].real
    return psi, nrm


def _orthonorm_left(psi):
    """ TODO
    """
    for i in range(len(psi)-1):
        psi[i], psi[i+1] = _local_orthonormalize_left(psi[i], psi[i+1])
    # last tensor
    psi[-1], Anext = _local_orthonormalize_left(psi[-1], numpy.array([[[1]]]))
    nrm = Anext[0,0,0].real
    return psi, nrm


def _local_orthonormalize_left(phi, phinext):
    """ TODO
    """
    phi = phi.transpose((1,0,2)) # switch convention
    s = phi.shape
    phi, R = tl.qr(phi.reshape(s[0]*s[1],s[2]))
    # THIS BIT IS EXPERIMENTAL
    # TODO: complex numbers?
    # Check if first elem of R was neg
    if R[0][0] < -1.e-14:
        R = -R
        phi = -phi
    ############################################
    phi = phi.reshape((s[0],s[1],phi.shape[1]))
    phinext = phinext.transpose((1,0,2)) # switch convention
    phinext = numpy.tensordot(R, phinext, (1,1)).transpose((1,0,2))
    phi = phi.transpose((1,0,2)) # switch back 
    phinext = phinext.transpose((1,0,2)) # switch back
    return phi, phinext


def _local_orthonormalize_right(phi, phiprev):
    """ TODO
    """
    phi = phi.transpose((1,0,2)) # switch convention
    phi = phi.transpose((0,2,1))
    s = phi.shape
    phi, R = tl.qr(phi.reshape(s[0]*s[1],s[2]))
    # EXPERIMENTAL ############################
    if R[0][0] < -1.e-14:
        R = -R
        phi = -phi
    ############################################
    phi = phi.reshape((s[0],s[1],phi.shape[1])).transpose((0,2,1))
    phiprev = phiprev.transpose((1,0,2)) # switch convention
    phiprev = numpy.tensordot(phiprev, R, (2,1))
    phi = phi.transpose((1,0,2)) # switch back
    phiprev = phiprev.transpose((1,0,2)) # switch back
    return phi, phiprev


# ----------------------------------------------------------------------------#
#                      Utility functions for asserts                          #
# ----------------------------------------------------------------------------#


def overlap(state, compare_state, nqsystems):
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
    eq = overlap_eq(nqsystems)
    print(eq)
    circuit = compare_state + state
    return contract(eq, *circuit)


def check_left_orth(A): # To be used after a left orthonormalize
    """ TODO
    """
    eq = 'ijk,ijl->kl'
    matA = contract(eq, A, A)
    I = numpy.identity(matA.shape[0])
    return abs(numpy.linalg.norm(I-matA)) < 1e-6

    
def check_right_orth(A): # To be used after a right orthonormalize
    """ TODO
    """
    eq = 'ijk,ljk->il'
    matA = contract(eq, A, A)
    I = numpy.identity(matA.shape[0])
    return abs(numpy.linalg.norm(I-matA)) < 1e-6
    
   
def test_orthonorm(psi, nqsystems):
    """ TODO
    """
    print("\nTesting orthonormalization with random tensors")
    rand = []
    for phi in psi: 
        print(phi.shape)
        rand.append(numpy.random.randn(*phi.shape))
    for r in rand:
        print(r)
    randr, _ = _orthogonal_right(rand)
    overlap = overlap(randr, rand, nqsystmes)
    overlap = tl.sqrt(overlap.real**2 + overlap.imag**2)
    print("OVERLAP: ", overlap)
    randr_nrm, _ = _orthonorm_right(randr)
    overlap = overlap(randr_nrm, randr_nrm, nqsystems)
    overlap = tl.sqrt(overlap.real**2 + overlap.imag**2)
    print("SELF OVERLAP: ", overlap)
    randl, _ = _orthogonal_left(rand)
    overlap = overlap(randl, rand, nqsystems)
    overlap = tl.sqrt(overlap.real**2 + overlap.imag**2)
    print("OVERLAP: ", overlap)
