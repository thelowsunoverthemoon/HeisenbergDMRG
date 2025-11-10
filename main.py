import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
def enlarge(block, Jx, Jy, Jz):
    return merge_blocks(block, single_site, Jx, Jy, Jz)

# merges the provided system and environment blocks
def create_super_block(system, env, Jx, Jy, Jz):
    return merge_blocks(system, env, Jx, Jy, Jz)

# Common code for creating both enlarged blocks and super-blocks
def merge_blocks(block1, block2, Jx, Jy, Jz):
    L1 = block1.length
    m_L1 = block1.basis
    L2 = block2.length
    m_L2 = block2.basis
    return Block(
        length = L1 + L2,
        basis = m_L1 * m_L2,
        hamiltonian = np.kron(block1.hamiltonian, np.identity(m_L2))
                    + np.kron(np.identity(m_L1), block2.hamiltonian)
                    + get_two_site_interaction(
                        block1.spin_z_operator,
                        block1.spin_raise_operator,
                        block2.spin_z_operator,
                        block2.spin_raise_operator,
                        Jx, Jy, Jz
                    ),
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

# changes the basis of the operator matrix using the transformation matrix
def transform_basis(operator, transformation):
    return transformation.conjugate().transpose().dot(operator.dot(transformation))

def step(system, env, keep, Jx, Jy, Jz, debug):
    system_enlarge = enlarge(system, Jx, Jy, Jz)
    env_enlarge = enlarge(env, Jx, Jy, Jz)
    superblock = create_super_block(system_enlarge, env_enlarge, Jx, Jy, Jz)
    
    if (debug):
        print("Skip")
        # print_block(system_enlarge)
        # print_block(env_enlarge)
        # print_block(superblock)

    # get lowest energy groundstate (smallest eigenvalue)
    # wave_function represnts the ground state of the super-block, with its corresponding energy
    (energy,), wave_function = eigsh(superblock.hamiltonian, k=1, which="SA")
    
    # create rho_L, the reduced density matrix of the left enlarged block
    wave_function = wave_function.reshape([system_enlarge.basis, -1], order="C")
    rho_L = np.dot(wave_function, wave_function.conjugate().transpose())

    eigenvalues, eigenvectors, eigenstates = diagonalize_matrix(rho_L)
    eigenstates.sort(key=itemgetter(0), reverse=True)
    
    new_basis = min(len(eigenstates), keep)
    used_eigenstates = eigenstates[:new_basis]
    transformation_matrix = np.zeros((system_enlarge.basis, new_basis), dtype='d', order='F')
    
    # fill the transformation matrix with the first new_basis eigenstates
    for i, (eval, evec) in enumerate(used_eigenstates):
        transformation_matrix[:, i] = evec
    
    new_length = system_enlarge.length
    new_hamiltonian = transform_basis(system_enlarge.hamiltonian, transformation_matrix)
    new_spin_z_operator = transform_basis(system_enlarge.spin_z_operator, transformation_matrix)
    new_spin_raise_operator = transform_basis(system_enlarge.spin_raise_operator, transformation_matrix)
    
    new_block = Block(
        length = new_length,
        basis = new_basis,
        hamiltonian = new_hamiltonian,
        spin_z_operator = new_spin_z_operator,
        spin_raise_operator = new_spin_raise_operator
    )

    if (debug):
        # print("Superblock Hamiltonian:\n", superblock.hamiltonian)
        # print("Ground state energy: ", energy)
        # print("Wave Function:\n", wave_function)
        # print("Rho:\n", rho_L)
        # print("Eigenvalues: ", eigenvalues)
        # print("Eigenvectors:\n", eigenvectors)
        # print("Eigenstates:\n", eigenstates)
        # print("Used eigenstates:\n", used_eigenstates)
        print("new block length: ", new_block.length)
        print("new block basis: ", new_block.basis)
        print("new block hamiltonian:\n", new_block.hamiltonian)
        print("new block spin z:\n", new_block.spin_z_operator)
        print("new block spin raise:\n", new_block.spin_raise_operator)
        print("energy: ", energy)

    return new_block, energy

def infinite_dmrg(sites, keep, start, Jx, Jy, Jz, debug=False):
    block = start
    # block, energy = step(block, block, keep, debug)
    
    while 2 * block.length < sites:
       block, energy = step(block, block, keep, Jx, Jy, Jz, debug=False)

    if (debug):
        # print("\n\nfinal block length: ", block.length)
        # print("\n\nfinal block basis: ", block.basis)
        # print("\n\nfinal block hamiltonian:\n", block.hamiltonian)
        # print("\n\nfinal block spin z:\n", block.spin_z_operator)
        # print("\n\nfinal block spin raise:\n", block.spin_raise_operator)
        print("\n\nfinal energy: ", energy)
        print("\n\nE/L: ", energy / (block.length * 2))

    return energy / (block.length * 2)


def gen_XXX_graph(name, title):
    j_list = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

    results = []
    start = single_site

    print(f"Creating graph {name}")
    for j in j_list:
        energy = infinite_dmrg(sites=30, keep=20, start=start, debug=False, Jx=j, Jy=j, Jz=j)
        results.append({'J': j, 'Energy': energy})

    df = pd.DataFrame(results)

    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='J', y='Energy')
    plt.title(title)
    plt.savefig(os.path.join('visual', name), dpi=300, bbox_inches='tight')
    plt.close()

def gen_XXZ_graph(name, title):
    j_list = [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0]
    start = single_site
    print(f"Creating graph {name}")

    data = []
    for jx in j_list:
        for jz in j_list:
            energy = infinite_dmrg(sites=10, keep=20, start=start, debug=False, Jx=jx, Jy=jx, Jz=jz)
            data.append({'Jx': jx, 'Jz': jz, 'Energy': energy})

    df = pd.DataFrame(data)
    heatmap_data = df.pivot(index='Jx', columns='Jz', values='Energy')

    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_data, cmap='coolwarm', cbar_kws={'label': 'Energy'})
    plt.xlabel('Jz')
    plt.ylabel('Jx')
    plt.title(title)
    plt.savefig(os.path.join('visual', name), dpi=300, bbox_inches='tight')
    plt.close()

def gen_keep_graph(name, title, zoom):
    keep_list = list(range(10, 51, 10))

    results = []
    start = single_site

    print(f"Creating graph {name}")
    for k in keep_list:
        
        energy = infinite_dmrg(sites=20, keep=k, start=start, debug=False, Jx=1.0, Jy=1.0, Jz=1.0)
        results.append({'Keep': k, 'Model': 'XXX with J = 1.0', 'Energy': energy})

        energy = infinite_dmrg(sites=20, keep=k, start=start, debug=False, Jx=1.0, Jy=1.0, Jz=0.5)
        results.append({'Keep': k, 'Model': 'XXZ with Jx = Jy = 1.0, Jz = 0.5', 'Energy': energy})
        
        energy = infinite_dmrg(sites=20, keep=k, start=start, debug=False, Jx=0.8, Jy=1.2, Jz=1.0)
        results.append({'Keep': k, 'Model': 'XYZ with Jx = 0.8, Jy = 1.2, Jz = 1.0', 'Energy': energy})

    df = pd.DataFrame(results)

    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='Keep', y='Energy', hue='Model')
    plt.title(title)

    if zoom:
        ymin = df['Energy'].min()
        ymax = df['Energy'].max()
        padding = 0.05 * (ymax - ymin)
        plt.ylim(ymin - padding, ymax + padding)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(os.path.join('visual', name), dpi=300, bbox_inches='tight')
    plt.close()

def gen_energy_graph(name, title, zoom):
    site_list = list(range(10, 101, 10))

    results = []
    start = single_site

    print(f"Creating graph {name}")
    for sites in site_list:
        
        energy = infinite_dmrg(sites=sites, keep=20, start=start, debug=False, Jx=1.0, Jy=1.0, Jz=1.0)
        results.append({'Sites': sites, 'Model': 'XXX with J = 1.0', 'Energy': energy})

        energy = infinite_dmrg(sites=sites, keep=20, start=start, debug=False, Jx=1.0, Jy=1.0, Jz=0.5)
        results.append({'Sites': sites, 'Model': 'XXZ with Jx = Jy = 1.0, Jz = 0.5', 'Energy': energy})
        
        energy = infinite_dmrg(sites=sites, keep=20, start=start, debug=False, Jx=0.8, Jy=1.2, Jz=1.0)
        results.append({'Sites': sites, 'Model': 'XYZ with Jx = 0.8, Jy = 1.2, Jz = 1.0', 'Energy': energy})

    df = pd.DataFrame(results)

    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='Sites', y='Energy', hue='Model')
    plt.title(title)

    if zoom:
        ymin = df['Energy'].min()
        ymax = df['Energy'].max()
        padding = 0.05 * (ymax - ymin)
        plt.ylim(ymin - padding, ymax + padding)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(os.path.join('visual', name), dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")

    np.set_printoptions(precision=10, suppress=True)

    os.makedirs('visual', exist_ok=True)
    gen_keep_graph('energy_vs_keep.png', 'Energy Per Site vs Keep (20 sites)', False)
    gen_keep_graph('energy_vs_keep_zoomed.png', 'Energy Per Site vs Keep (20 sites, Zoomed)', True)
    gen_XXZ_graph('energy_vs_j_xxz.png', 'Energy Per Site vs Jx, Jz (Jx and Jz from -2.0 to 2.0, 10 sites)')
    gen_XXX_graph('energy_vs_j_xxx.png', 'Energy Per Site vs J for XXX Heisenberg Models (J from -2 to 2, 30 sites)')
    gen_energy_graph('infinite_energy_vs_sites_zoomed.png', 'Energy Per Site vs Number of Sites for Different Heisenberg Models (Zoomed)', zoom=True)
    gen_energy_graph('infinite_energy_vs_sites.png', 'Energy Per Site vs Number of Sites for Different Heisenberg Models', zoom=False)