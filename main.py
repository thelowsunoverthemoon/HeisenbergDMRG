import numpy as np
from dmrg_classes import *
from dmrg_operators import *

single_site = Block(
        length = 1,
        basis = single_site_basis,
        hamiltonian = single_site_h,
        spin_z_operator = spin_z,
        spin_raise_operator = spin_raise
    )

def step(system, env, keep):
    system_enlarge = enlarge(system)
    env_enlarge = enlarge(env)

    # TODO
    return system, 0

def infinite_dmrg(sites, keep, start):
    block = start
    print("Enlarged block: ")
    print_block(enlarge(block))
    # Commented avoid infinite loop
    # while 2 * block.length < sites:
    #    block, energy = step(block, block, keep)

def enlarge(block):
    return merge_blocks(block, single_site)

def create_symmetric_super_block(block):
    return merge_blocks(block, block)

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
        spin_z_operator = np.kron(np.identity(m_L1 * m_L2 // 2), spin_z),
        spin_raise_operator = np.kron(np.identity(m_L1 * m_L2 // 2), spin_raise)
    )

if __name__ == "__main__":

    sites = 100
    start = single_site

    print("Infinite DMRG with {} sites".format(sites))
    infinite_dmrg(sites = sites, keep = 20, start = start)
    print("Finished")