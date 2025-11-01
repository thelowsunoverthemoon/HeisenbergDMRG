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

def get_two_site_interaction(spin_z_a, spin_raise_a, spin_z_b, spin_raise_b,
                             Jx=1.0, Jy=1.0, Jz=1.0):

    Sx_a = 0.5 * (spin_raise_a + spin_raise_a.T)
    Sx_b = 0.5 * (spin_raise_b + spin_raise_b.T)

    Sy_a = -0.5j * (spin_raise_a - spin_raise_a.T)
    Sy_b = -0.5j * (spin_raise_b - spin_raise_b.T)


    H = (Jx * np.kron(Sx_a, Sx_b) +
         Jy * np.kron(Sy_a, Sy_b) +
         Jz * np.kron(spin_z_a, spin_z_b))

    H = np.real_if_close(H, tol=1e-12)
    return H