import numpy as np
from dmrg_classes import Block, EnlargedBlock
from dmrg_operators import *

def step(system, env, keep):
    system_enlarge = enlarge(system)
    env_enlarge = enlarge(env)

    # TODO
    return system, 0

def infinite_dmrg(sites, keep, start):
    block = start
    # Commented avoid infinite loop
    # while 2 * block.length < sites:
    #    block, energy = step(block, block, keep)

def enlarge(block):
    # TODO
    return block

if __name__ == "__main__":

    sites = 100
    start = Block(
        length = 1, basis = basis,
        hamiltonian = single_site_h,
        spin_z_operator = spin_z,
        spin_raise_operator = spin_raise
    ) 

    print("Infinite DMRG with {} sites".format(sites))
    infinite_dmrg(sites = 100, keep = 20, start = start)
    print("Finished")