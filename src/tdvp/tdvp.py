"""
This file implements the Time Dependant Variational Principle (TDVP) algorithm.
Both the one-site and the two-site version are implemented. 
See "https://tensornetwork.org/mps/algorithms/timeevo/tdvp.html".
"""

import numpy as np
from ..mps import mps
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from numpy.linalg import qr, svd
from ..util.rq import rq
import time

class TDVP_Engine:
    """
    Base class for TDVP1_Engine and TDVP2_Engine that implements basic functions used in
    both algorithms
    """
    
    def __init__(self, psi, model, dt, chi_max, eps):
        """
        Initializes the TDVP_Engine
        """
        self.psi = psi
        self.model = model
        self.dt = dt
        self.chi_max = chi_max
        self.eps = eps
        # initialize left and right environment
        self.LPs = [None] * psi.L
        self.RPs = [None] * psi.L
        D = self.model.H_mpo[0].shape[0]
        chi = self.psi.Bs[0].shape[0]
        LP = np.zeros([chi, D, chi], dtype="float")  # vL wL* vL*
        RP = np.zeros([chi, D, chi], dtype="float")  # vR* wR* vR
        LP[:, 0, :] = np.eye(chi)
        RP[:, D - 1, :] = np.eye(chi)
        self.LPs[0] = LP
        self.RPs[-1] = RP
        # initialize necessary RPs
        for i in range(psi.L - 1, 0, -1):
            self.update_RP(i) 

    def update_RP(self, i):
        """
        Calculate RP right of site `i-1` from RP right of site `i`.
        """
        j = i - 1
        RP = self.RPs[i]  # vR* wR* vR
        B = self.psi.Bs[i]  # vL vR i
        Bc = B.conj()  # vL* vR* i*
        W = self.model.H_mpo[i]  # wL wR i i*
        RP = np.tensordot(B, RP, ([1, 0]))  # vL [vR] i; [vR*] wR* vR -> vL i wR* vR
        RP = np.tensordot(RP, W, ([1, 2], [3, 1]))  # vL [i] [wR*] vR; wL [wR] i [i*] -> vL vR wL i
        RP = np.tensordot(RP, Bc, ([1, 3], [1, 2]))  # vL [vR] wL [i]; vL* [vR*] [i*] -> vL wL vL*
        self.RPs[j] = RP  # vL wL vL* (== vR* wR* vR on site i-1)
    
    def update_LP(self, i):
        """
        Calculate LP left of site `i+1` from LP left of site `i`.
        """
        j = i + 1
        LP = self.LPs[i]  # vL wL vL*
        B = self.psi.Bs[i]  # vL vR i
        Bc = B.conj() # vL* vR* i*
        W = self.model.H_mpo[i] # wL wR i i*
        LP = np.tensordot(LP, B, ([2], [0])) # vL wL* [vL*]; [vL] vR i -> vL wL* vR i
        LP = np.tensordot(W, LP, ([0, 3], [1, 3]))  # [wL] wR i [i*]; vL [wL*] vR [i] -> wR i vL vR
        LP = np.tensordot(Bc, LP, ([0, 2], [2, 1]))  # [vL*] vR* [i*], wR [i] [vL] vR -> vR* wR vR
        self.LPs[j] = LP  # vR* wR vR (== vL wL* vL* on site i+1)

    def sweep(self):
        """
        Performs one sweep left -> right -> left.
        """                  
        raise NotImplementedError

class TDVP2_Engine(TDVP_Engine):
    """
    Class that implements the 2-site Time Dependent Variational Principle (TDVP) algorithm
    """
    
    def __init__(self, psi, model, dt, chi_max, eps):
        super().__init__(psi, model, dt, chi_max, eps)
        
    def sweep(self):
        """
        Performs one sweep left -> right -> left.
        """    
        # sweep from left to right
        for i in range(self.psi.L - 1):
            self.update_bond(i, sweep_right=True)
        # sweep from right to left
        for i in range(self.psi.L - 2, -1, -1):
            self.update_bond(i, sweep_right=False)
                      
    def update_bond(self, i, sweep_right):
        """
        Performs a single bond update at the given bond
        
        Parameters
        ----------
        i : int
            the bond index
        sweep_right : bool
            wether we are currently in a right or left sweep
        """
        j = i + 1
        # get two-site wavefunction
        theta = self.psi.get_theta_2(i) # vL, i, j, vR
        chi_vL, chi_i, chi_j, chi_vR = theta.shape
        # get effective two-site Hamiltonian
        Heff = compute_Heff_Twosite(self.LPs[i], self.RPs[j], self.model.H_mpo[i], self.model.H_mpo[j])
        theta = np.reshape(theta, [Heff.shape[0]])
        # evolve 2-site wave function forward in time
        theta = evolve(theta, Heff, self.dt/2)
        # split and truncate
        theta = np.reshape(theta, (chi_vL*chi_i, chi_j*chi_vR))
        U, S, V, norm_factor, _ = mps.split_and_truncate(theta, self.chi_max, self.eps)
        self.psi.norm *= norm_factor
        # put back into MPS
        if sweep_right:
            U = np.reshape(U, (chi_vL, chi_i, U.shape[1])) # (vL i) vR -> vL i vR 
            self.psi.Bs[i] = np.transpose(U, (0, 2, 1)) # vL i vR -> vL vR i 
            V = np.tensordot(np.diag(S), V, ([1], [0])) # vC [vC*]; [vC] (j vR) -> vC j vR = vL (j vR)
            V = np.reshape(V, (V.shape[0], chi_j, chi_vR)) # vL (j vR) -> vL j vR
            self.psi.Bs[j] = np.transpose(V, (0, 2, 1)) # vL j vR -> vL vR j
        else:
            U = np.tensordot(U, np.diag(S), ([1], [0])) # (vL i) [vC*]; [vC*] vC -> (vL i) vC
            U = np.reshape(U, (chi_vL, chi_i, U.shape[1])) # (vL i) vC -> vL i vC = vL i vR
            self.psi.Bs[i] = np.transpose(U, (0, 2, 1)) # vL i vR -> vL vR i
            V = np.reshape(V, (V.shape[0], chi_j, chi_vR)) # vL (j vR) -> vL j vR
            self.psi.Bs[j] = np.transpose(V, (0, 2, 1)) # vL j vR -> vL vR j
        if sweep_right == True:
            self.update_LP(i)
            if i < self.psi.L - 2:
                # extract single-site wavefunction
                psi = self.psi.Bs[j]
                psi_shape = psi.shape
                psi = psi.flatten()
                # compute effective one-site Hamiltonian
                Heff = compute_Heff_Onesite(self.LPs[j], self.RPs[j], self.model.H_mpo[j])
                # evolve 1-site wave function backwards in time
                psi = evolve(psi, Heff, -self.dt/2)
                psi = np.reshape(psi, psi_shape)
                # put back into MPS
                self.psi.Bs[j] = psi
        else:
            self.update_RP(j)
            if i > 0:
                # extract single-site wavefunction
                psi = self.psi.Bs[i]
                psi_shape = psi.shape
                psi = psi.flatten()
                # compute effective one-site Hamiltonian
                Heff = compute_Heff_Onesite(self.LPs[i], self.RPs[i], self.model.H_mpo[i])
                # evolve 1-site wave function backwards in time
                psi = evolve(psi, Heff, -self.dt/2)
                psi = np.reshape(psi, psi_shape)
                # put back into MPS
                self.psi.Bs[i] = psi

class TDVP1_Engine(TDVP_Engine):
    
    def __init__(self, psi, model, dt, chi_max=0, eps=0, mode='qr'):
        super().__init__(psi, model, dt, chi_max, eps)
        self.mode = mode
        
    def sweep(self):
        """
        Performs one sweep left -> right -> left.
        """
        # sweep from left to right
        for i in range(self.psi.L - 1):
            self.update_site(i)
            self.update_bond(i, sweep_right=True)
        # update last site
        self.update_site(self.psi.L - 1)
        # sweep from right to left
        for i in range(self.psi.L - 1, 0, -1):
            self.update_site(i)
            self.update_bond(i, sweep_right=False)
        # update first site
        self.update_site(0)
       
    def update_site(self, i):
        """
        Performs a single site update at the given bond
        
        Parameters
        ----------
        i : int
            the bond index
        """
        # extract single-site wavefunction
        psi = self.psi.Bs[i]
        psi_shape = psi.shape
        psi = psi.flatten()
        # compute effective one-site Hamiltonian
        Heff = compute_Heff_Onesite(self.LPs[i], self.RPs[i], self.model.H_mpo[i])
        # evolve 1-site wave function forwards in time
        psi = evolve(psi, Heff, self.dt/2)
        psi = np.reshape(psi, psi_shape)
        # put back into MPS
        self.psi.Bs[i] = psi
            
    def update_bond(self, i, sweep_right):
        """
        Performs a single bond update at the given bond
        
        Parameters
        ----------
        i : int
            the bond index
        sweep_right : bool
            wether we are currently in a right or left sweep
        """
        # First we factorize the current site to construct the zero-site tensor
        # and update the environment
        C = None
        B = np.transpose(self.psi.Bs[i], (0, 2, 1)) # vL vR i -> vL i vR
        chi_vL, chi_i, chi_vR = B.shape
        if sweep_right:
            B = np.reshape(B, (chi_vL*chi_i, chi_vR)) # vL i vR -> (vL i) vR
            if self.mode == 'qr':
                Q, R = qr(B)
                B = np.reshape(Q, (chi_vL, chi_i, Q.shape[1])) # (vL i) vC -> vL i vC
                self.psi.Bs[i] = np.transpose(B, (0, 2, 1)) # vL i vC -> vL vC i = vL vR i
                C = R
            else:
                U, S, V, norm, _ = mps.split_and_truncate(B, self.chi_max, self.eps)
                self.psi.norm *= norm
                B = np.reshape(U, (chi_vL, chi_i, U.shape[1])) # (vL i) vC -> vL i vC
                self.psi.Bs[i] = np.transpose(B, (0, 2, 1)) # vL i vC -> vL vC i = vL vR i
                C = np.tensordot(np.diag(S), V, ([1], [0])) # vC [vC*]; [vC] vR -> vC vR
            self.update_LP(i)
            # compute effective zero-site Hamiltonian
            Heff = compute_Heff_zero_site(self.LPs[i+1], self.RPs[i])
        else:
            B = np.reshape(B, (chi_vL, chi_i*chi_vR)) # vL i vR -> vL (i vR)
            if self.mode == 'qr':
                R, Q = rq(B)
                B = np.reshape(Q, (Q.shape[0], chi_i, chi_vR)) # vC (i vR) -> vC i vR
                self.psi.Bs[i] = np.transpose(B, (0, 2, 1)) # vC i vR -> vC vR i = vL vR i
                C = R
            else:
                U, S, V, norm, _ = mps.split_and_truncate(B, self.chi_max, self.eps)
                self.psi.norm *= norm
                B = np.reshape(V, (V.shape[0], chi_i, chi_vR)) # vC (i vR) -> vC i vR
                self.psi.Bs[i] = np.transpose(B, (0, 2, 1)) # vC i vR -> vC vR i = vL vR i
                C = np.tensordot(U, np.diag(S), ([1], [0])) # vL [vC]; [vC*] vC -> vL vC
            self.update_RP(i)
            # compute effective zero-site Hamiltonian
            Heff = compute_Heff_zero_site(self.LPs[i], self.RPs[i-1])
        C_shape = C.shape
        # evolve zero-site wave function backwards in time
        C = evolve(C.flatten(), Heff, -self.dt/2)
        C = np.reshape(C, C_shape)
        # put back into MPS
        if sweep_right:
            self.psi.Bs[i+1] = np.tensordot(C, self.psi.Bs[i+1], ([1], [0])) # vC [vR]; [vL] vR i -> vC vR i
        else:
            self.psi.Bs[i-1] = np.tensordot(self.psi.Bs[i-1], C, ([1], [0])) # vL [vR] i; [vL] vC -> vL i vC
            self.psi.Bs[i-1] = np.transpose(self.psi.Bs[i-1], (0, 2, 1)) # vL i vC -> vL vC i

class CBETDVP_Engine(TDVP_Engine):
    """
    
    """
    def __init__(self, psi, model, dt, chi_max=0, eps=0, mode='qr', D_max=20, D_tilde=2, truncation_threshold=1e-10, health_check_threshold=1e-10, enable_orthogonalization=False):
        super().__init__(psi, model, dt, chi_max, eps)
        self.mode = mode  
        self.D_max = D_max
        self.D_tilde = D_tilde
        self.truncation_threshold = truncation_threshold
        self.health_check_threshold = health_check_threshold
        self.enable_orthogonalization = enable_orthogonalization
        
        print(self.truncation_threshold)


    def get_right_orthogonal(self, i):
        # R_{l+2}
        RP = self.RPs[i]  # vR* wR* vR
        # W_{l+1}
        W = self.model.H_mpo[i]  # wL wR i i*
        # B_{l+1}
        B = self.psi.Bs[i]  # vL vR i
        B_dag = np.transpose(B.conj())  # i vR vL 
        R_temp = np.tensordot(B, W, ([2], [3])) # vC vR [i]; wL wR i [i*] -> vC vR wL wR i
        R_temp = np.tensordot(R_temp, RP, ([1, 3], [0, 1])) # vC [vR] wL [wR] i; [vR*] [wR*] vR -> vC wL i vR
        B_id = np.tensordot(B_dag, B, ([2], [0])) #i vR [vL];  [vL] vR i -> i vR vR i #TODO: How to compute this? CONTRACT 1 LEG
        prod = np.tensordot(R_temp, B_id, ([2, 3], [0, 1])) # vC wL [i] [vR]; [i] [vR] vR i -> vC wL vR i #TODO:  CONTRACT 2 LEGS
        prod = np.transpose(prod, (0, 1, 3, 2))
        R_orth = R_temp - prod 
        return R_orth 
    
    def get_left_orthogonal(self, i, U, S):
        # L_{l-1}
        LP = self.LPs[i]  # vL wL* vL*
        # W_{l}
        W = self.model.H_mpo[i]  # wL wR i i*
        # A_{l}
        B_left = self.psi.Bs[i]  # vL vR i
        B_left_dag = np.transpose(B_left.conj())  # i vR vL 
        # print("LP: ", LP.shape, "W: ", W.shape, "B_left: ", B_left.shape, "U: ", U.shape, "S: ", S.shape)
        L_temp = np.tensordot(LP, W, ([1], [0])) # vL [wL*] vL*; [wL] wR i i* -> vL vL* wR i i*
        L_temp = np.tensordot(L_temp, B_left, ([1, 4], [0, 2])) # vL [vL*] wR i [i*]; [vL] vR [i] -> vL wR i vR
        L_temp = np.tensordot(L_temp, U, ([3], [0])) # vL wR i [vR]; [vL] vR -> vL wR i vR
        L_temp = np.tensordot(L_temp, S, ([3], [0])) # vL wR i [vR]; [vL] vR -> vL wR i vR
        B_id = np.tensordot(B_left, B_left_dag, ([1], [1])) # vL [vR] i; i [vR] vL ->  vL i i vL
        prod = np.tensordot(B_id, L_temp, ([2, 3], [2, 0])) #  vL i [i] [vL]; [vL] wR [i] vR -> vL i wR vR
        prod = np.transpose(prod, (0, 2, 1, 3))
        L_orth = L_temp - prod   
        return L_orth
    
    def get_central_orthogonal(self, i, A_pr):
        # A_pr_dag_{l-1}
        A_pr_h = A_pr.conj().T # vR_right i vR_left
        # L_{l-2}
        LP = self.LPs[i-1]  # vL wL* vL*
        # L_{l-2}
        RP = self.RPs[i]  # vR* wR* vR
        # W_{l-1}
        W_left = self.model.H_mpo[i-1]  # wL wR i i*
        # W_{l+1}
        W_right = self.model.H_mpo[i]  # wL wR i i*
        # A_{l}
        B_left = self.psi.Bs[i-1]  # vL vR i
        # B_{l}
        B = self.psi.Bs[i]  # vL vR i
        B_dag = B.conj().T  # i vR vL 

        #vL bot, vL* top
        L_pr = np.tensordot(A_pr_h, LP, ([2], [0])) # vR_right i [vR_left]; [vL] wL* vL* -> vR_right i wL* vL*
        L_pr = np.tensordot(L_pr, W_left, ([1, 2], [2, 0])) # vR_right [i] [wL*] vL*; [wL] wR [i] i* -> vR_right vL* wR i*
        L_pr = np.tensordot(L_pr, B_left, ([3, 1], [2, 0])) # vR_right [vL*] wR [i*]; [vL] vR [i] -> vR_right wR vR

        C_temp = np.tensordot(L_pr, B, ([2], [0])) # vR_right wR [vR]; [vL] vR i -> vR_right wR vR i
        C_temp = np.tensordot(C_temp, W_right, ([1, 3], [0, 3])) # vR_right [wR] vR [i]; [wL] wR i [i*] -> vR_right vR wR i
        C_temp = np.tensordot(C_temp, RP, ([1, 2], [0, 1])) # vR_right [vR] [wR] i; [vR*] [wR*] vR -> vR_right i vR

        B_id = np.tensordot(B_dag, B, ([2], [0])) # i vR [vL];  [vL] vR i -> i vR vR i
        prod = np.tensordot(C_temp, B_id, ([1, 2], [0, 1])) # vR_top [i] [vR]; [i] [vR] vR_bot i -> vR_top vR_bot i
        prod = np.transpose(prod, (0, 2, 1))
        C_orth = C_temp - prod
        return C_orth

    def shrewd_selection(self, i, D_tilde):
        R_orth = self.get_right_orthogonal(i) # vC wL i vR
        R_orth = np.reshape(R_orth, (R_orth.shape[0], -1)) # vC x (wL i vL)
        U, S, Vh = svd(R_orth, full_matrices=False)
        S = np.diag(S)
        L_orth = self.get_left_orthogonal(i-1, U, S) # vR_down wR i vR_up
        L_orth_grouped = np.reshape(L_orth, (-1, L_orth.shape[3])) # (vR_down wR i) vR_up
        
        D_prime = int(np.floor(L_orth.shape[0] / L_orth.shape[1]))
        if D_prime == 0:
            D_prime = 1
        
        U_prime, S_prime, Vh_prime = svd(L_orth_grouped, full_matrices=False)

        S_prime = np.diag(S_prime)
        u_prime = U_prime[:, :D_prime] 
        s_prime = S_prime[:D_prime, :D_prime]
        
        u_prime = np.reshape(u_prime, (L_orth.shape[0], L_orth.shape[1], L_orth.shape[2], D_prime)) # vR_down wR i vR_up
        us_prime = np.tensordot(u_prime, s_prime, ([3], [0]))
        us_prime_reshape = np.transpose(us_prime, (0, 2, 1, 3)) # vR_down i wR vR_up
        us_prime_reshape_split = np.reshape(us_prime_reshape, (us_prime_reshape.shape[0] * us_prime_reshape.shape[1], -1)) # (vR_down i) (wR vR_up)

        U_hat, S_hat, Vh_hat = svd(us_prime_reshape_split, full_matrices=False)
        S_hat = np.diag(S_hat)
        nonzero_indices = np.where(S_hat > self.truncation_threshold)[0]
        if len(nonzero_indices) == 0:
            print("Skip shrewd selection as all indices are zero")
            return None
        u_hat = U_hat[:, nonzero_indices] # (vR_down i) wD'; wD' wD'

        u_hat_split = np.reshape(u_hat, (us_prime_reshape.shape[0], us_prime_reshape.shape[1], -1)) # vR_left i vR_right
        A = self.psi.Bs[i-1] # vL vR i
        Ah = A.conj().T # i vR vL
        Ah_Atr = np.tensordot(Ah, u_hat_split, ([0, 2], [1, 0])) # [i] vR [vL]; [vR_left] [i] vR_right -> vL vR_right
        A_pr = None
        health_check = np.max(np.abs(Ah_Atr))
        print("Health check of preselected A:", health_check)
        if health_check > self.health_check_threshold and self.enable_orthogonalization:
            print("Doing additional orthogonolization...")
            A_Ah = np.tensordot(A, Ah, ([1], [1])) # vL [vR] i; i [vR] vL ->  vL i i vL
            safe_u_hat = u_hat_split - np.tensordot(A_Ah, u_hat_split, ([2, 3], [1, 0])) # vL i [i] [vL]; [vR_left] [i] vR_right -> vL i vR
            # Ah_Atr = np.tensordot(Ah, safe_u_hat, ([0, 2], [1, 0])) # [i] vR [vL]; [vR_left] [i] vR_right -> vL vR_right
            # print("New health check of preselected A:", np.max(np.abs(Ah_Atr)))
            safe_u_hat_split = np.reshape(safe_u_hat, (safe_u_hat.shape[0] * safe_u_hat.shape[1], -1)) # (vL i) (vR)

            U_hat, S_hat, Vh_hat = svd(safe_u_hat_split, full_matrices=False)
            S_hat = np.diag(S_hat)
            nonzero_indices = np.where(S_hat > self.truncation_threshold)[0]
            if len(nonzero_indices) == 0:
                print("Skip shrewd selection as all indices are zero")
                return None
            u_hat = U_hat[:, nonzero_indices]
            print("Not skipped")
            u_hat_split = np.reshape(u_hat, (safe_u_hat.shape[0], safe_u_hat.shape[1], -1)) # vR_left i vR_right
            Ah_Atr = np.tensordot(Ah, u_hat_split, ([0, 2], [1, 0])) # [i] vR [vL]; [vR_left] [i] vR_right -> vL vR_right
            print("New health check of preselected A:", np.max(np.abs(Ah_Atr)))
            A_pr = u_hat_split
        else:
            A_pr = u_hat_split # vR_left i vR_right
        C_orth = self.get_central_orthogonal(i, A_pr) # vR_left wR_right i_right vR_right
        C_orth_grouped = np.reshape(C_orth, (C_orth.shape[0], -1)) # vR_left (wR_right i_right vR_right)
        U_tilde, S_tilde, Vh_tilde = svd(C_orth_grouped, full_matrices=False)
        S_tilde = np.diag(S_tilde)
        u_tilde = U_tilde[:, :D_tilde] 
        s_tilde = S_tilde[:D_tilde, :D_tilde]
        vh_tilde = Vh_tilde[:D_tilde, :]

        return np.tensordot(A_pr, u_tilde, ([2], [0])) # vR_left i [vR_right]; [vL] vR -> vR_left i vR_right 

    def update_dimension_with_shrewd_selection(self, i, D_tilde):
        A_old = self.psi.Bs[i-1] # vL vR i
        # shrewd selection -> truncated A
        A_tr = self.shrewd_selection(i, D_tilde) # vR_left i vR_right 
        if type(A_tr) != np.ndarray:
            return None
        A_tr = np.transpose(A_tr, (0, 2, 1)) # vR_left vR_right i 
        # Bs[i-1] = Bs[i-1] (+) truncated A
        A_ex = np.concatenate((A_old, A_tr), axis=1) 

        # update Bs[i] 
        A_ex_c = A_ex.conj().T  # i vR vL
        A_Adag = np.tensordot(A_old, A_ex_c, ([0, 2], [2, 0])) # [vL] vR [i]; [i] vR [vL] -> vR_right vR_left = vR vL
        C_old = self.psi.Bs[i] # vL vR i
        C_ex = np.tensordot(A_Adag, C_old, ([0], [0])) # [vR] vL; [vL] vR i -> vL vR i
        self.psi.Bs[i-1] = A_ex
        self.psi.Bs[i] = C_ex

        # update LP[i] using LP[i-1] and extended A
        LP = self.LPs[i-1]  # vL wL vL*
        W = self.model.H_mpo[i-1] # wL wR i i*
        LP = np.tensordot(LP, A_ex, ([2], [0])) # vL wL* [vL*]; [vR_left] vR_right i  -> vL wL* vR i
        LP = np.tensordot(W, LP, ([0, 3], [1, 3]))  # [wL] wR i [i*]; vL [wL*] vR [i] -> wR i vL vR
        LP = np.tensordot(A_ex_c, LP, ([0, 2], [1, 2]))  # [i*] vR* [vL*]; wR [i] [vL] vR -> vR* wR vR
        self.LPs[i] = LP  # vR* wR vR (== vL wL* vL* on site i+1)



    def sweep(self):
        """
        Performs one sweep left -> right -> left.
        """
        # sweep from left to right
        end_pos = self.psi.L - 1
        for i in range(end_pos):
            self.update_site(i)
            self.update_bond(i, sweep_right=True)
        self.update_site(end_pos)

        for i in range(end_pos, 0, -1):
            A_old = self.psi.Bs[i-1]
            if A_old.shape[1] != self.D_max:
                if A_old.shape[1] + self.D_tilde > self.D_max:
                   D = self.D_max - A_old.shape[1]
                else:
                   D = self.D_tilde
                self.update_dimension_with_shrewd_selection(i, D)

            self.update_site(i)
            self.update_bond(i, sweep_right=False)
        self.update_site(0)
       
    def update_site(self, i):
        """
        Performs a single site update at the given bond
        
        Parameters
        ----------
        i : int
            the bond index
        """
        # extract single-site wavefunction
        psi = self.psi.Bs[i]
        psi_shape = psi.shape
        psi = psi.flatten()
        # compute effective one-site Hamiltonian
        Heff = compute_Heff_Onesite(self.LPs[i], self.RPs[i], self.model.H_mpo[i])
        # evolve 1-site wave function forwards in time
        psi = evolve(psi, Heff, self.dt/2)
        psi = np.reshape(psi, psi_shape)
        # put back into MPS
        self.psi.Bs[i] = psi
            
    def update_bond(self, i, sweep_right):
        """
        Performs a single bond update at the given bond
        
        Parameters
        ----------
        i : int
            the bond index
        sweep_right : bool
            wether we are currently in a right or left sweep
        """
        # First we factorize the current site to construct the zero-site tensor
        # and update the environment
        C = None
        B = np.transpose(self.psi.Bs[i], (0, 2, 1)) # vL vR i -> vL i vR
        chi_vL, chi_i, chi_vR = B.shape
        if sweep_right:
            B = np.reshape(B, (chi_vL*chi_i, chi_vR)) # vL i vR -> (vL i) vR
            if self.mode == 'qr':
                Q, R = qr(B)
                B = np.reshape(Q, (chi_vL, chi_i, Q.shape[1])) # (vL i) vC -> vL i vC
                self.psi.Bs[i] = np.transpose(B, (0, 2, 1)) # vL i vC -> vL vC i = vL vR i
                C = R
            else:
                U, S, V, norm, _ = mps.split_and_truncate(B, self.chi_max, self.eps)
                self.psi.norm *= norm
                B = np.reshape(U, (chi_vL, chi_i, U.shape[1])) # (vL i) vC -> vL i vC
                self.psi.Bs[i] = np.transpose(B, (0, 2, 1)) # vL i vC -> vL vC i = vL vR i
                C = np.tensordot(np.diag(S), V, ([1], [0])) # vC [vC*]; [vC] vR -> vC vR
            self.update_LP(i)
            # compute effective zero-site Hamiltonian
            Heff = compute_Heff_zero_site(self.LPs[i+1], self.RPs[i])
        else:
            B = np.reshape(B, (chi_vL, chi_i*chi_vR)) # vL i vR -> vL (i vR)
            if self.mode == 'qr':
                R, Q = rq(B)
                B = np.reshape(Q, (Q.shape[0], chi_i, chi_vR)) # vC (i vR) -> vC i vR
                self.psi.Bs[i] = np.transpose(B, (0, 2, 1)) # vC i vR -> vC vR i = vL vR i
                C = R
            else:
                U, S, V, norm, _ = mps.split_and_truncate(B, self.chi_max, self.eps)
                self.psi.norm *= norm
                B = np.reshape(V, (V.shape[0], chi_i, chi_vR)) # vC (i vR) -> vC i vR
                self.psi.Bs[i] = np.transpose(B, (0, 2, 1)) # vC i vR -> vC vR i = vL vR i
                C = np.tensordot(U, np.diag(S), ([1], [0])) # vL [vC]; [vC*] vC -> vL vC
            self.update_RP(i)
            # compute effective zero-site Hamiltonian
            Heff = compute_Heff_zero_site(self.LPs[i], self.RPs[i-1])
        C_shape = C.shape
        # evolve zero-site wave function backwards in time
        C = evolve(C.flatten(), Heff, -self.dt/2)
        C = np.reshape(C, C_shape)
        # put back into MPS
        if sweep_right:
            self.psi.Bs[i+1] = np.tensordot(C, self.psi.Bs[i+1], ([1], [0])) # vC [vR]; [vL] vR i -> vC vR i
        else:
            self.psi.Bs[i-1] = np.tensordot(self.psi.Bs[i-1], C, ([1], [0])) # vL [vR] i; [vL] vC -> vL i vC
            self.psi.Bs[i-1] = np.transpose(self.psi.Bs[i-1], (0, 2, 1)) # vL i vC -> vL vC i

def compute_Heff_Twosite(LP, RP, W1, W2):
    """
    Computes the two-site effective hamiltonian
    |theta'> = H_eff |theta>
    
    Parameters
    ----------
    LP : np.nadarray, shape vL wL* vL*
        the left environment tensor
    RP : np.ndarray, shape vR* wR* vR
        the right environment tensor
    Wi : np.ndarray, shape wL wC i i*
        the MPO tensor acting on site i
    Wj : np.ndarray, shape wC wR j j*
        the MPO tensor acting on sitr j = i + 1
        
    Returns
    -------
    H_eff : np.ndarray
        the effective Hamiltonian as a matrix of dimensions
        '(vL i j vR) (vL* i* j* vR*)'
    """
    chi1, chi2 = LP.shape[0], RP.shape[2]
    d1, d2 = W1.shape[2], W2.shape[2]
    result = np.tensordot(LP, W1, ([1], [0])) # vL [wL*] vL*; [wL] wC i i* -> vL vL* wC i i*
    result = np.tensordot(result, W2, ([2], [0])) # vL vL* [wC] i i*; [wC] wR j j* -> vL vL* i i* wR j j*
    result = np.tensordot(result, RP, ([4], [1])) # vL vL* i i* [wR] j j*; vR* [wR*] vR -> vL vL* i i* j j* vR* vR
    result = np.transpose(result, (0, 2, 4, 7, 1, 3, 5, 6)) # vL vL* i i* j j* vR* vR -> vL i j vR vL* i* j* vR*
    mat_shape = chi1*chi2*d1*d2
    result = np.reshape(result, (mat_shape, mat_shape))
    return result

def compute_Heff_Onesite(LP, RP, W):
    """
    Computes the one-site effective hamiltonian
    |psi'> = H_eff |psi>
    
    Parameters
    ----------
    LP : np.nadarray, shape vL wL* vL*
        the left environment tensor
    RP : np.ndarray, shape vR* wR* vR
        the right environment tensor
    W : np.ndarray, shape wL wR i i*
        the MPO tensor acting on site i
        
    Returns
    -------
    H_eff : np.ndarray
        the effective Hamiltonian as a matrix of dimensions
        '(vL vR i) (vL* vR* i*)'
    """
    result = np.tensordot(LP, W, ([1], [0])) # vL [wL*] vL*; [wL] wR i i* -> vL vL* wR i i*
    result = np.tensordot(result, RP, ([2], [1])) # vL vL* [wR] i i*; vR* [wR*] vR -> vL vL* i i* vR* vR
    result = np.transpose(result, (0, 5, 2, 1, 4, 3)) # vL vL* i i* vR* vR -> vL vR i vL* vR* i*
    result = np.reshape(result, (result.shape[0]*result.shape[1]*result.shape[2], result.shape[3]*result.shape[4]*result.shape[5]))
    return result

def compute_Heff_zero_site(LP, RP):
    """
    Computes the one-site effective hamiltonian
    |C'> = H_eff |C>
    
    Parameters
    ----------
    LP : np.nadarray, shape vL wL* vL*
        the left environment tensor
    RP : np.ndarray, shape vR* wR* vR
        the right environment tensor
        
    Returns
    -------
    H_eff : np.ndarray
        the effective Hamiltonian as a matrix of dimensions
        '(vL vR) (vL* vR*)'
    """
    result = np.tensordot(LP, RP, ([1], [1])) # vL [wL*] vL*; vR* [wR*] vR -> vL vL* vR* vR
    result = np.transpose(result, (0, 3, 1, 2)) # vL vL* vR* vR -> vL vR vL* vR*
    result = np.reshape(result, (result.shape[0]*result.shape[1], result.shape[2]*result.shape[3]))
    return result

def evolve(psi, H, dt, debug=False):
    """
    Evolves the given vector by time dt using
    psi(t+dt) = exp(-i*H*dt) @ psi(t)

    Parameters
    ----------
    psi : np.ndarray
        vector of length N, state vector at time t
    H : np.ndarray
        matrix of shape (N, N), (effective) Hamiltonian
    dt : float
        time step
    """
    return expm_multiply(-1.j * H * dt, psi)
