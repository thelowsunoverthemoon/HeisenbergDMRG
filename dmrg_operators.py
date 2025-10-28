import numpy as np

single_site_basis = 2 # 2 means states can be spin up or spin down
spin_z = np.array(
    [
        [0.5, 0],
        [0, -0.5]
    ],
    dtype='d'
) 

spin_raise = np.array(
    [
        [0, 1],
        [0, 0]
    ],
    dtype='d'
)

single_site_h = np.array(
    [
        [0, 0],
        [0, 0]
    ],
    dtype='d'
)

def get_two_site_interaction(spin_z_a, spin_raise_a, spin_z_b, spin_raise_b):
    J = Jz = 1
    xy = (J / 2) * (np.kron(spin_raise_a, spin_raise_b.conjugate().transpose()) + np.kron(spin_raise_a.conjugate().transpose(), spin_raise_b))
    z = Jz * np.kron(spin_z_a, spin_z_b)
    return xy + z