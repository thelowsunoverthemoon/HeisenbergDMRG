import numpy as np
from scipy.sparse.linalg import eigsh

from operator import itemgetter
from dmrg_classes import *
from dmrg_operators import *

single_site = Block(
    length = 1,
    basis = single_site_basis,
    hamiltonian = single_site_h,
    spin_z_operator = spin_z,
    spin_raise_operator = spin_raise
)

# merges block with a single site
def enlarge(block):
    return merge_blocks(block, single_site)

# merges the provided system and environment blocks
def create_super_block(system, env):
    return merge_blocks(system, env)

# Common code for creating both enlarged blocks and super-blocks
def merge_blocks(block1, block2):
    L1 = block1.length
    m_L1 = block1.basis
    L2 = block2.length
    m_L2 = block2.basis
    return Block(
        length = L1 + L2,
        basis = m_L1 * m_L2,
        hamiltonian = np.kron(block1.hamiltonian, np.identity(m_L2))
                    + np.kron(np.identity(m_L1), block2.hamiltonian)
                    + get_two_site_interaction(block1.spin_z_operator, block1.spin_raise_operator, block2.spin_z_operator, block2.spin_raise_operator),
        # Unclear if the two following operators have any meaning in the super-block, but this works correctly for enlarged blocks
        spin_z_operator = np.kron(np.identity(m_L1 * m_L2 // 2), spin_z),
        spin_raise_operator = np.kron(np.identity(m_L1 * m_L2 // 2), spin_raise)
    )

# computes the eigenvalues and eigenvectors of rho, and returns them as pairs
def diagonalize_matrix(rho):
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenstates = []
    for eval, evec in zip(eigenvalues, eigenvectors.transpose()):
        eigenstates.append((eval, evec))
    return (eigenvalues, eigenvectors, eigenstates)

def step(system, env, keep, debug):
    system_enlarge = enlarge(system)
    print_block(system_enlarge)
    env_enlarge = enlarge(env)
    print_block(env_enlarge)
    superblock = create_super_block(system_enlarge, env_enlarge)
    print_block(superblock)

    # get lowest energy groundstate (smallest eigenvalue)
    # wave_function represnts the ground state of the super-block, with its corresponding energy
    (energy,), wave_function = eigsh(superblock.hamiltonian, k=1, which="SA")
    
    # create rho_L, the reduced density matrix of the left enlarged block
    wave_function = wave_function.reshape([system_enlarge.basis, -1], order="C")
    rho_L = np.dot(wave_function, wave_function.conjugate().transpose())

    eigenvalues, eigenvectors, eigenstates = diagonalize_matrix(rho_L)
    eigenstates.sort(key=itemgetter(0), reverse=True)

    if (debug):
        print("Superblock Hamiltonian:\n", superblock.hamiltonian)
        print("Ground state energy: ", energy)
        print("Wave Function:\n", wave_function)
        print("Rho:\n", rho_L)
        print("Eigenvalues: ", eigenvalues)
        print("Eigenvectors:\n", eigenvectors)
        print("Eigenstates:\n", eigenstates)

    return system, 0

def infinite_dmrg(sites, keep, start, debug=False):
    block = start
    block, energy = step(block, block, keep, debug)
    # Commented avoid infinite loop
    # while 2 * block.length < sites:
    #    block, energy = step(block, block, keep)

if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True)
    sites = 100
    start = single_site

    print("Infinite DMRG with {} sites".format(sites))
    infinite_dmrg(sites = sites, keep = 20, start = start, debug=True)
    print("Finished")