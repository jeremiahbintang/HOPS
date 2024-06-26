import numpy as np
import scipy.sparse as sparse

def generate_physical_operators():
    """
    Computes some basic operators in the physical Hilbert space as np.ndarray
        
    Returns
    -------
    np.ndarray
        sigma_x pauli matrix, shape (2, 2): 
    np.ndarray
        sigma_z pauli matrix, shape (2, 2)
    np.ndarray
        identity matrix, shape (2, 2)
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    return sigma_x, sigma_z, eye

def generate_spin_boson_hamiltonian(delta=1, epsilon=0):
    """
    Computes the system Hamiltonian of the spin-boson model as np.ndarray
    
    Parameters
    ----------
    delta : float 
        parameter for the spin-boson model
    epsilon : float
        parameter for the spin-boson model
        
    Returns
    -------
    np.ndarray
        Hamiltonian of the spin-boson model, shape (2, 2) 
        $H = -1/2\Delta\sigma_x+1/2\epsilon\sigma_z$
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = -0.5*delta*sigma_x + 0.5*epsilon*sigma_z
    return H

def generate_quantum_battery_hamiltonian(w0, wc, Omega, f, a, a_dagger):
    """
    Returns
    -------
    np.ndarray
        Hamiltonian of the quantum battery model https://arxiv.org/pdf/2312.06389.pdf
        $H = w0 * sigma_plus @ sigma_min + wc * a^\dag @ a + f(t) * Omega * (sigma_plus @ a + sigma_min @ a^\dag)
    """
    sigma_plus = np.array([[0, 0], [1, 0]], dtype=complex)
    sigma_min = np.array([[0, 1j], [0, 0]], dtype=complex)
    H = w0 * np.kron(sigma_plus @ sigma_min, np.eye(a.shape[0])) + wc * np.kron(np.eye(2), a_dagger @ a) + f * Omega * (np.kron(sigma_plus, a) + np.kron(sigma_min, a_dagger))
    return H

def generate_auxiallary_operators(N_trunc, rescale_aux=True):
    """
    Computes the auxillary operators that are used in some of the
    HOPS/HOMPS implementations as np.ndarray
    
    Parameters
    ----------
    N_trunc : int
        dimension of the auxillary Hilbert space
    rescale_aux : int
        If this is set to true (default), the raising and lowering
        operators will include the factors sqrt(n + 1) and sqrt(n)
        respectively (this is necessary for HOMPS with rescaled 
        auxillary vectors)

    Returns
    -------
    np.ndarray 
        number operator N. Is of shape (N_trunc, N_trunc)
    np.ndarray 
        raising operator b_dagger. is of shape (N_trunc, N_trunc)
    np.ndarray 
        lowering operator b. is shape (N_trunc, N_trunc)
    np.ndarray 
        identity operator. is of shape (N_trunc, N_trunc)
    """
    # number operator
    N = np.diag(np.arange(0, N_trunc, dtype=complex))
    # lowering operator
    if rescale_aux:
        b = np.diag(np.sqrt(np.arange(1, N_trunc, dtype=complex)), 1)
    else:
        b = np.diag(np.sqrt(np.ones(N_trunc-1, dtype=complex)), 1)
    # raising operator
    if rescale_aux:
        b_dagger = np.diag(np.sqrt(np.arange(1, N_trunc, dtype=complex)), -1)
    else:
        b_dagger = np.diag(np.sqrt(np.ones(N_trunc-1, dtype=complex)), -1)
    # identity operator
    eye = np.eye(N_trunc, dtype=complex)
    # return the three operators
    return N, b_dagger, b, eye

def generate_physical_operators_sparse():
    """
    Computes some basic operators in the physical Hilbert space as scipy.sparse matrices
        
    Returns
    -------
    scipy.sparse.csr_matrix
        sigma_x pauli matrix, shape (2, 2): 
    scipy.sparse.csr_matrix
        sigma_z pauli matrix, shape (2, 2)
    scipy.sparse.csr_matrix
        identity matrix, shape (2, 2)
    """
    sigma_x = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    sigma_z = sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
    eye = sparse.csr_matrix(np.eye(2, dtype=complex))
    return sigma_x, sigma_z, eye

def generate_spin_boson_hamiltonian_sparse(delta=1, epsilon=0):
    """
    Computes the system Hamiltonian of the spin-boson model as scipy.sparse matrices
    
    Parameters
    ----------
    delta : float 
        parameter for the spin-boson model
    epsilon : float
        parameter for the spin-boson model
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Hamiltonian of the spin-boson model, shape (2, 2) 
        $H = -1/2\Delta\sigma_x+1/2\epsilon\sigma_z$
    """
    sigma_x = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    sigma_z = sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
    H = -0.5*delta*sigma_x + 0.5*epsilon*sigma_z
    return H

def generate_auxiallary_operators_sparse(N_trunc, rescale_aux=True):
    """
    Computes the auxillary operators that are used in some of the
    HOPS/HOMPS implementations as scipy sparse matrices
    
    Parameters
    ----------
    N_trunc : int
        dimension of the auxillary Hilbert space
    rescale_aux : int
        If this is set to true (default), the raising and lowering
        operators will include the factors sqrt(n + 1) and sqrt(n)
        respectively (this is necessary for HOMPS with rescaled 
        auxillary vectors)

    Returns
    -------
    scipy.sparse.csr_matrix
        number operator N. Is of shape (N_trunc, N_trunc)
    scipy.sparse.csr_matrix
        raising operator b_dagger. is of shape (N_trunc, N_trunc)
    scipy.sparse.csr_matrix
        lowering operator b. is shape (N_trunc, N_trunc)
    scipy.sparse.csr_matrix
        identity operator. is of shape (N_trunc, N_trunc)
    """
    # number operator
    N = sparse.csr_matrix(np.diag(np.arange(0, N_trunc, dtype=complex)))
    # lowering operator
    b = sparse.csr_matrix(np.diag(np.sqrt(np.arange(1, N_trunc, dtype=complex)), 1))
    # raising operator
    b_dagger = sparse.csr_matrix(np.diag(np.sqrt(np.arange(1, N_trunc, dtype=complex)), -1))
    
    # lowering operator
    if rescale_aux:
        b = sparse.csr_matrix(np.diag(np.sqrt(np.arange(1, N_trunc, dtype=complex)), 1))
    else:
        b = sparse.csr_matrix(np.diag(np.sqrt(np.ones(N_trunc-1, dtype=complex)), 1))
    # raising operator
    if rescale_aux:
        b_dagger = sparse.csr_matrix(np.diag(np.sqrt(np.arange(1, N_trunc, dtype=complex)), -1))
    else:
        b_dagger = sparse.csr_matrix(np.diag(np.sqrt(np.ones(N_trunc-1, dtype=complex)), -1))
    # identity operator
    eye = sparse.csr_matrix(np.eye(N_trunc, dtype=complex))
    # return the three operators
    return N, b_dagger, b, eye
