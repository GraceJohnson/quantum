import tensorly as tl
#tl.set_backend('pytorch')
tl.set_backend('numpy')
from tensorly import complex64, float64
from opt_einsum import contract
from numpy import ceil

from .tt_contraction import overlap_eq

import numpy #TODO: remove
import copy

DTYPE = complex64
#DTYPE = float64


def apply_U(psi_orig, circuit):
    """Applies circuit layer to input state, assuming layer is only product operators 
       (i.e. gates do not cross bond dimensions)

    Parameters
    ----------
    psi : tt-tensor, input state |psi>

    Returns
    -------
    tt-tensor, evolved state U|psi>
    """
    # Assumes right node is top node
    psi = copy.deepcopy(psi_orig)
    nqsystems = circuit.nqsystems
    layer = circuit._build_layer()

    for node in range(len(psi)-1):
        # apply operator and orthonormalize
        psi[node] = _apply_local_circuit_toright(psi[node], layer[node])
        psi[node], psi[node+1] = _local_orthonormalize_right(psi[node], psi[node+1])
    psi[-1] = _apply_local_circuit_toright(psi[-1], layer[-1])
    psi[-1], _ = _local_orthonormalize_right(psi[-1], numpy.array([[[1]]]))
    #for node in psi:
        #assert(check_right_orth(node)) # Top node is not necessarily normalized
    assert(abs(abs(overlap(psi, psi, nqsystems))-1.0) < 1e-4)

    return psi


def build_layer(circuit):
    ops = [[] for i in range(circuit[0].nqsystems)]
    for layer in circuit:
        print(layer)
        built = layer._build_layer()
        for i in range(len(built)):
            # Check for contracted dimensions (multi-qubit gates), form sum-of-products op out of these
            # TODO: handle cases where two-qubit gates on some nodes but not others (pad with I)
            shape = built[i].shape
            print("BUILT SHAPE: ", shape)
            assert(not(shape[0] > 1 and shape[3] > 1)) # TODO: handle cases where left and right dims > 1 (?)
            # Right has extra dims
            if shape[3] > 1:
                for j in range(shape[3]):
                    op = built[i][:,:,:,j] # grab jth slice of right dimension
                    op = op[:,:,:,None] # reshape to (1, p, p, 1)
                    ops[i].append(op)
            # Right has extra dims
            else:
                for j in range(shape[0]):
                    op = built[i][j,:,:,:] # grab jth slice of left dimension
                    op = op[None,:,:,:] # reshape to (1, p, p, 1)
                    ops[i].append(op)
    ops = [list(x) for x in zip(*ops)] # switch to outer loop over layers instead of nodes
    return ops
            

def apply_circuit_SCF(psi_orig, circuit):
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
    ops = build_layer(circuit)
    print("LEN OPS:", len(ops))

    # TODO: logic to apply_U if no two-qubit gates in layer (single layer already done, but can also be identities)
    if len(ops) == 1:
        return apply_U(psi_orig, *circuit) 

    # NOTE: code below is still under construction.
    # SCF convergence with small bond dimension has yet to be tested rigorously
    # Using a full bond dimension will converge in one sweep

    max_iter = 10
    eps = 1.e-3
    eps_diff = 1.e-4
    nqsystems = circuit[0].nqsystems

    # Initialize a random tensor network to apply operators to
    psi = copy.deepcopy(psi_orig)
    chi = []
    for phi in psi: 
        chi.append(numpy.random.randn(*phi.shape))
        # TODO: only do random matrices on node that will actually get updated... otherwise just make a copy
        #        i.e. if it's an identity operator at that node, copy
        #chi.append(copy.deepcopy(phi))

    f_left = 0.
    f_right = 0.
    f_left_prev = 1.
    f_right_prev = 1.

    print("--------------STARTING SCF--------------")
    for i in range(max_iter):

        # Apply operator to nodes in train and update operator, sweeping right then left
        # -------------------------TO RIGHT ---------------------------#
        psi, _ = orthonorm_left(psi) # Make left node top
        chi, _ = orthonorm_left(chi) # Make left node top
        mats_left = _represent_mats_left(chi, psi, ops)

        psi, _ = orthonorm_right(psi) # Make right node top
        chi, _ = orthonorm_right(chi) # Make right node top
        mats = _represent_mats_right(chi, psi, ops)

        assert(abs(abs(overlap(chi, chi, nqsystems))-1.0) < 1e-4)
        assert(abs(abs(overlap(psi, psi, nqsystems))-1.0) < 1e-4)

        chi = _apply_circuit_toright(psi, ops, mats, mats_left)

        assert(abs(abs(overlap(chi, chi, nqsystems))-1.0) < 1e-4)
        #for phi in chi: 
            #assert(check_right_orth(phi)) #top node orthonormal only upon convergence

        # ------------CHECK CONVERGENCE-------------#
        # Check exit condition <psi\tilde|U|psi> ~= 1
        umats = _represent_mats_right(chi, psi, ops)
        # Apply top
        hphi = tl.zeros((psi[-1].shape), dtype=DTYPE)
        for l in range(len(ops)):
            hphi += _apply_local_circuit_toright(psi[-1], ops[l][-1], umats[l][-2])
        hphi = _node_orthonormalize_right(hphi)

        braket = [hphi] + [chi[-1]]
        eq = 'kli,klj->ij'
        norm = contract(eq, hphi, hphi)
        norm = abs(norm[0][0]**2)
        #print("Norm hphi: ", norm)
        dot = contract(eq, *braket)
        f_right = abs(dot[0][0])**2
        f_right = f_right/norm # TESTING, seems you need this for states with complex numbers
        print("SCF FIDELITY (RIGHT):", f_right)
        
        if abs(1-f_right) < eps or abs(f_right - f_right_prev) < eps_diff:
            return chi
        f_right_prev = f_right


        # -------------------------TO LEFT ---------------------------#
        psi, _ = orthonorm_right(psi) # make right node top
        chi, _ = orthonorm_right(chi) # make right node top
        mats_right = _represent_mats_right(chi, psi, ops)

        psi, _ = orthonorm_left(psi) # make left node top
        chi, _ = orthonorm_left(chi) # make left node top
        mats = _represent_mats_left(chi, psi, ops)

        assert(abs(abs(overlap(chi, chi, nqsystems))-1.0) < 1e-4)
        assert(abs(abs(overlap(psi, psi, nqsystems))-1.0) < 1e-4)

        chi = _apply_circuit_toleft(psi, ops, mats, mats_right)

        assert(abs(abs(overlap(chi, chi, nqsystems))-1.0) < 1e-4)
        #for phi in chi: 
        #    assert(check_left_orth(phi)) #top node orthonormal only upon convergence

        # ------------CHECK CONVERGENCE-------------#
        # Check exit condition <psi\tilde|U|psi> ~= 1
        umats = _represent_mats_left(chi, psi, ops)
        # Apply top
        hphi = tl.zeros((psi[0].shape), dtype=DTYPE)
        for l in range(len(ops)):
            hphi += _apply_local_circuit_toleft(psi[0], ops[l][0], umats[l][1])
        hphi = _node_orthonormalize_left(hphi)

        braket = [hphi] + [chi[0]]
        eq = 'ikl,jkl->ij'
        dot = contract(eq, *braket)
        norm = contract(eq, hphi, hphi)
        norm = abs(norm[0][0]**2)
        #print("Norm hphi: ", norm)
        f_left = abs(dot[0][0])**2
        f_left = f_left/norm # TESTING
        print("SCF FIDELITY (LEFT): ", f_left)

        if abs(1.-f_left) < eps or abs(f_left - f_left_prev) < eps_diff:
            chi, _ = orthonorm_right(chi) # Make right node top
            return chi
        f_left_prev = f_left


    print("SCF did not converge in {} iterations".format(max_iter))
    return chi


# ----------------------------------------------------------------------------#
#                    Helper functions for main routines                       #
# ----------------------------------------------------------------------------#

def _apply_circuit_toright(psi, layer_orig, mats, mats_above):
    """Applies gates in layer to input state, sweeping through train left to right
       and updating the state and layer for future iterations

    Parameters
    ----------
    state : tt-tensor, input state |psi>
    layer: tt-tensor, gates in a layer to be applied to |psi>
    mats: tt-tensor, operator matrices, updated throughout sweep (one for each node)
    mats_above: tt-tensor, operator matrices from above, not updated (one for each node)

    Returns
    -------
    tt-tensor, updated state U|psi>
    """
    hpsi = copy.deepcopy(psi) # can actually just be empty, or better passed in as input
    layer = copy.deepcopy(layer_orig)
    assert(len(psi) == len(layer[0]))

    # Swipe through train left to right
    for node in range(len(psi)):
        right = True if node == len(psi)-1 else False
        left = True if node == 0 else False

        # 1) apply operator (calculate Hpsi ket)
        hphi = tl.zeros((psi[node].shape), dtype=DTYPE)
        for l in range(len(layer)):
            child = None if left else mats[l][node-1]
            parent = None if right else mats_above[l][node+1]
            hphi += _apply_local_circuit_toright(psi[node], layer[l][node], child, parent)

        #hphi = _node_orthonormalize_right(hphi)  # don't update next node, mats take care of this
        if not right:
            # 2) orthonormalize ket (QR) # TESTING
            hphi = _node_orthonormalize_right(hphi)  # don't update next node, mats take care of this

            # 3) Rebuild local operator: U_ij = <psi\tilde_i|U|psi_j> (outer product of Upsi)
            for l in range(len(layer)):
                child = None if left else mats[l][node-1]
                mats[l][node] = _represent_mat_right(hphi, psi[node], layer[l][node], child)

        hpsi[node] = hphi

    return hpsi


def _apply_circuit_toleft(psi, layer_orig, mats, mats_above):
    """Applies gates in layer to input state, sweeping through train right to left
       and updating the state for future iterations

    Parameters
    ----------
    state : tt-tensor, input state |psi>
    layer: tt-tensor, gates in a layer to be applied to |psi>
    mats: tt-tensor, operator matrices, updated throughout sweep (one for each node)
    mats_above: tt-tensor, operator matrices from above, not updated (one for each node)

    Returns
    -------
    tt-tensor, updated state U|psi>
    """
    hpsi = copy.deepcopy(psi) # Just needs to be same shape
    layer = copy.deepcopy(layer_orig)
    assert(len(psi) == len(layer[0]))

    # Swipe through train right to left
    for node in reversed(range(len(psi))):
        right = True if node == len(psi)-1 else False
        left = True if node == 0 else False

        # 1) apply operator (calculate Hpsi ket)
        hphi = tl.zeros((psi[node].shape), dtype=DTYPE)
        for l in range(len(layer)):
            child = None if right else mats[l][node+1]
            parent = None if left else mats_above[l][node-1]
            hphi += _apply_local_circuit_toleft(psi[node], layer[l][node], child, parent)

        #hphi = _node_orthonormalize_left(hphi)  # Mats should handle updating the next node
        if not left:
            # 2) orthonormalize ket (QR) # TESTING
            hphi = _node_orthonormalize_left(hphi)  # Mats should handle updating the next node

            # 3) Rebuild local operator: U_ij = <psi\tilde_i|U|psi_j> (outer product of Upsi)
            for l in range(len(layer)):
                child = None if right else mats[l][node+1]
                mats[l][node] = _represent_mat_left(hphi, psi[node], layer[l][node], child)
  
        hpsi[node] = hphi

    return hpsi


def _apply_local_circuit_toright(phi_orig, gate, child=None, parent=None):
    """Applies incoming operators to a node phi

    Parameters
    ----------
    phi : tt-tensor, input state at one node (i.e. a basis function |phi>)
    gate: tt-tensor, gate operator belonging to node phi
    child: tt-tensor, (bond_dim,bond_dim) operator matrix belonging to child (left) of node phi
    parent: tt-tensor, (bond_dim,bond_dim) operator matrix belonging to parent (right) of node phi

    Returns
    -------
    tt-tensor, updated basis function U|phi>
    """
    phi = copy.deepcopy(phi_orig)
    # Middle dimension of phi is reserved for gate operator
    circuit = [gate] + [phi]
    #eq = 'aecf,bed->bcd'
    eq = 'acef,bed->bcd' # transpose of matrix to fix non-symmetric bug
    Uphi = contract(eq, *circuit)
    # Left dimension of phi is reserved for child node (to the left)
    if child is not None:
        circuit = [child] + [Uphi] 
        #eq = 'bf,bcd->fcd'
        eq = 'fb,bcd->fcd'
        Uphi = contract(eq, *circuit)
    # Right dimension of phi is reserved for parent node (to the right)
    if parent is not None:
        circuit = [parent] + [Uphi] 
        #eq = 'df,bcd->bcf'
        eq = 'fd,bcd->bcf'
        Uphi = contract(eq, *circuit)
    assert(Uphi.shape == phi.shape)
    return Uphi


def _apply_local_circuit_toleft(phi_orig, gate, child=None, parent=None):
    """Applies incoming operators to a node phi

    Parameters
    ----------
    phi : tt-tensor, input state at one node (i.e. a basis function |phi>)
    gate: tt-tensor, gate operator belonging to node phi
    child: tt-tensor, (bond_dim,bond_dim) operator matrix belonging to child (right) of node phi
    parent: tt-tensor, (bond_dim,bond_dim) operator matrix belonging to parent (left) of node phi

    Returns
    -------
    tt-tensor, updated basis function U|phi>
    """
    phi = copy.deepcopy(phi_orig)
    # Middle dimension of phi is reserved for gate operator
    circuit = [gate] + [phi]
    #eq = 'aecf,bed->bcd'
    eq = 'acef,bed->bcd'
    Uphi = contract(eq, *circuit)
    # Right dimension of phi is reserved for child node (to the right)
    if child is not None:
        circuit = [child] + [Uphi] 
        #eq = 'df,bcd->bcf'
        eq = 'fd,bcd->bcf'
        Uphi = contract(eq, *circuit)
    # Left dimension of phi is reserved for parent node (to the left)
    if parent is not None:
        circuit = [parent] + [Uphi] 
        #eq = 'bf,bcd->fcd'
        eq = 'fb,bcd->fcd'
        Uphi = contract(eq, *circuit)
    assert(Uphi.shape == phi.shape)
    return Uphi


def _reduced_density_matrix_right(psi_orig):
    """ TODO
    """
    psi = copy.deepcopy(psi_orig)
    rho = [tl.zeros((2,2)) for i in range(len(psi))] # Rho at each node (TODO: chi x chi)
    # Loop top-down (right is top), top node is I
    rho[-1] = tl.tensor([[1,0],[0,1]], dtype=DTYPE) # (TODO: chi x chi)
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
    rho[0] = tl.tensor([[1,0],[0,1]], dtype=DTYPE) # (TODO: chi x chi)
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
        mat = [tl.zeros((2,2), dtype=DTYPE) for i in range(len(ket))] # Operator matrices at each node (chi x chi)
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
    #braket = [bra] + [hphi]
    braket = [tl.conj(bra)] + [hphi]
    eq = 'ikl,jkl->ij'
    #eq = 'jkl,ikl->ij' # This fixed original bug, but now transpose on apply
    return contract(eq, *braket)


def _represent_mats_right(bra, ket, layer):
    """ TODO
    """
    # NOTE: don't do this if node is not active or if it's top node
    assert(len(bra) == len(ket))
    mats = []
    # Loop through sum of products
    for l in range(len(layer)):
        mat = [tl.zeros((2,2), dtype=DTYPE) for i in range(len(ket))] # Operator matrices at each node (chi x chi)
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
    #braket = [bra] + [hphi]
    braket = [tl.conj(bra)] + [hphi]
    eq = 'kli,klj->ij'
    #eq = 'klj,kli->ij'  # This is what fixed original bug, but now transpose on apply
    return contract(eq, *braket)


def _node_orthonormalize_right(phi):
    """ TODO
    """
    # Left and center are children, already on left
    s = phi.shape
    phi, R = tl.qr(phi.reshape(s[0]*s[1],s[2]))
    if R[0][0] < -1.e-14:
        R = -R
        phi = -phi
    phi = phi.reshape((s[0],s[1],s[2]))
    return phi


def _node_orthonormalize_left(phi):
    """ TODO
    """
    # Right and center are children, put them on the left
    phi = phi.transpose((2,1,0))
    s = phi.shape
    phi, R = tl.qr(phi.reshape(s[0]*s[1],s[2]))
    if R[0][0] < -1.e-14:
        R = -R
        phi = -phi
    phi = phi.reshape((s[0],s[1],s[2])).transpose((2,1,0))
    return phi


def _orthogonal_right(psi): # This orthogonalizes but not normalizes, use on random tensors once
    """ TODO
    """
    psi, nrm = orthonorm_right(psi)
    #if nrm < 0:
    #    psi[-1] = -psi[-1]
    psi[-1] = psi[-1]/nrm
    return psi, nrm


def _orthogonal_left(psi): # This orthogonalizes but not normalizes, use on random tensors once
    """ TODO
    """
    psi, nrm = orthonorm_left(psi)
    #if nrm < 0:
    #    psi[0] = -psi[0]
    psi[0] = psi[0]/nrm
    return psi, nrm


def orthonorm_right(psi_orig):
    """ TODO
    """
    psi = copy.deepcopy(psi_orig)
    for i in range(len(psi)-1):
        psi[i], psi[i+1] = _local_orthonormalize_right(psi[i], psi[i+1])
    # last tensor
    psi[-1], Anext = _local_orthonormalize_right(psi[-1], numpy.array([[[1]]]))
    nrm = Anext[0,0,0].real
    return psi, nrm


def orthonorm_left(psi_orig):
    """ TODO
    """
    psi = copy.deepcopy(psi_orig)
    for i in reversed(range(1, len(psi))):
        psi[i], psi[i-1] = _local_orthonormalize_left(psi[i], psi[i-1])
    # first tensor
    psi[0], Aprev = _local_orthonormalize_left(psi[0], numpy.array([[[1]]]))
    nrm = Aprev[0,0,0].real
    return psi, nrm


def _local_orthonormalize_right(phi, phinext):
    """ TODO
    """
    # Left and center are children, already on left
    s = phi.shape
    phi, R = tl.qr(phi.reshape(s[0]*s[1],s[2]))
    # TODO: complex numbers? Resolved, I think
    # Check if first elem of R was neg
    if R[0][0] < -1.e-14:
        R = -R
        phi = -phi
    phi = phi.reshape((s[0],s[1],s[2]))
    # update next tensor: multiply with R from left
    phinext = numpy.tensordot(R, phinext, (1,0))
    return phi, phinext


def _local_orthonormalize_left(phi, phiprev):
    """ TODO
    """
    # Right and center are children, put them on the left
    phi = phi.transpose((2,1,0))
    s = phi.shape
    phi, R = tl.qr(phi.reshape(s[0]*s[1],s[2]))
    if R[0][0] < -1.e-14:
        R = -R
        phi = -phi
    phi = phi.reshape((s[0],s[1],s[2])).transpose((2,1,0))
    phiprev = numpy.tensordot(phiprev, R, (2,1))
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
    state_conj = [numpy.conj(node) for node in state] # TESTING
    # Send to vector, then do transpose too TODO
    braket = compare_state + state_conj
    return contract(eq, *braket)


def check_right_orth(A): # To be used after a right orthonormalize
    """ TODO
    """
    eq = 'ijk,ijl->kl'
    matA = contract(eq, A, A)
    matA = abs(matA)**2 # complex numbers
    I = numpy.identity(matA.shape[0])
    return abs(numpy.linalg.norm(I-matA)) < 1e-6

    
def check_left_orth(A): # To be used after a left orthonormalize
    """ TODO
    """
    eq = 'ijk,ljk->il'
    matA = contract(eq, A, A)
    matA = abs(matA)**2 # complex numbers
    I = numpy.identity(matA.shape[0])
    return abs(numpy.linalg.norm(I-matA)) < 1e-6
    
   
def test_orthonorm(psi, nqsystems):
    """ TODO
    """
    print("\nTesting orthonormalization with random tensors")
    print("nqsystems: ", nqsystems)
    rand = []
    for phi in psi: 
        print(phi.shape)
        #rand.append(numpy.random.randn(*phi.shape) + numpy.random.randn(*phi.shape)*1j)
        #rand.append(numpy.random.randn(*phi.shape) + numpy.zeros(phi.shape)*1j)
        rand.append(numpy.random.randn(*phi.shape))
    for r in rand:
        print(r)

    rand, _ = orthonorm_right(rand) # make right node top
    print("PSI RIGHT TOP")
    for phi in rand:
        print(phi)
        assert(check_right_orth(phi))
    randr = copy.deepcopy(rand)
    rand, _ = orthonorm_left(rand) # make left node top
    print("PSI LEFT TOP")
    for phi in rand:
        print(phi)
        assert(check_left_orth(phi))
    rand, _ = orthonorm_right(rand) # make right node top
    print("PSI RIGHT TOP")
    for phi in rand:
        print(phi)
        assert(check_right_orth(phi))
    print("PSI RIGHT TOP BEFORE")
    for phi in randr:
        print(phi)
        assert(check_right_orth(phi))

    randr, _ = _orthogonal_left(rand)
    ol = overlap(randr, rand, nqsystems)
    print("OVERLAP: ", ol)
    ol = tl.sqrt(ol.real**2 + ol.imag**2)
    print("OVERLAP: ", ol)
    randr_nrm, _ = orthonorm_left(randr)
    ol = overlap(randr_nrm, randr_nrm, nqsystems)
    print("SELF OVERLAP: ", ol)
    ol = tl.sqrt(ol.real**2 + ol.imag**2)
    print("SELF OVERLAP: ", ol)
    randl, _ = _orthogonal_right(rand)
    ol = overlap(randl, rand, nqsystems)
    print("OVERLAP: ", ol)
    ol = tl.sqrt(ol.real**2 + ol.imag**2)
    print("OVERLAP: ", ol)
