class Block:
    def __init__(self, length, basis, hamiltonian, spin_z_operator, spin_raise_operator):
        self.length = length
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.spin_z_operator = spin_z_operator
        self.spin_raise_operator = spin_raise_operator

class EnlargedBlock:
    def __init__(self, length, basis, hamiltonian, spin_z_operator, spin_raise_operator):
        self.length = length
        self.basis = basis
        self.hamiltonian = hamiltonian
        self.spin_z_operator = spin_z_operator
        self.spin_raise_operator = spin_raise_operator

def print_block(b):
    print("L: {}, m_L: {} \nH: \n{} \nSz: \n{} \nSp: \n{}".format(b.length, b.basis, b.hamiltonian, b.spin_z_operator, b.spin_raise_operator))