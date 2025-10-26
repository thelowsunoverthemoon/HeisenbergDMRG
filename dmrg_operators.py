import numpy as np

basis = 2 # 2 means states can be spin up or spin down
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

def get_two_site_h(spin_z_a, spin_raise_a, spin_z_b, spin_raise_b):
    xy = (1 / 2) * (kron(spin_z_a, spin_raise_a.conjugate().transpose()) + kron(spin_z_b.conjugate().transpose(), spin_raise_b))
    z = kron(Sz1, Sz2)
    return xy + z