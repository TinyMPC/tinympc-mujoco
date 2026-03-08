# src/rho_adapter.py
import autograd.numpy as np
from scipy.linalg import block_diag
from utils.hover_simulation import uhover, xg
from autograd import jacobian

class RhoAdapter:
    def __init__(self, rho_base=85.0, rho_min=60.0, rho_max=100.0, tolerance=1.1, method="analytical", clip = False, mode = 'hover'):
        self.rho_base = rho_base
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.tolerance = tolerance
        self.method = method  # "analytical" or "heuristic"
        self.rho_history = [rho_base]
        self.residual_history = []
        self.derivatives = None
        self.clip = clip
        self.mode = mode
    def initialize_derivatives(self, cache, eps=1e-4):
        """Initialize derivatives using autodiff"""
        print("Computing LQR sensitivity")
        
        def lqr_direct(rho):
            R_rho = cache['R'] + rho * np.eye(cache['R'].shape[0])
            A, B = cache['A'], cache['B']
            Q = cache['Q']
            
            # Compute Pgit ad
            P = Q
            for _ in range(10):
                K = np.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
                P = Q + A.T @ P @ (A - B @ K)
            
            K = np.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
            C1 = np.linalg.inv(R_rho + B.T @ P @ B)
            C2 = A - B @ K
            
            return np.concatenate([K.flatten(), P.flatten(), C1.flatten(), C2.flatten()])
        
        # Get derivatives using autodiff
        m, n = cache['Kinf'].shape
        derivs = jacobian(lqr_direct)(cache['rho'])
        
        # Reshape derivatives into matrices and store in cache
        k_size = m * n
        p_size = n * n
        c1_size = m * m
        c2_size = n * n
        
        # Store derivatives in the cache directly
        cache['dKinf_drho'] = derivs[:k_size].reshape(m, n)
        cache['dPinf_drho'] = derivs[k_size:k_size+p_size].reshape(n, n)
        cache['dC1_drho'] = derivs[k_size+p_size:k_size+p_size+c1_size].reshape(m, m)
        cache['dC2_drho'] = derivs[k_size+p_size+c1_size:].reshape(n, n)

    def format_matrices(self, x_prev, u_prev, v_prev, z_prev, g_prev, y_prev, cache, N):
        """Format matrices into the form needed for residual computation"""
        nx = x_prev.shape[0]  # Should be 12
        nu = u_prev.shape[0]  # Should be 4
        
        # Reshape inputs to ensure correct dimensions
        x_prev = x_prev.reshape(nx, -1)  
        u_prev = u_prev.reshape(nu, -1)
        v_prev = v_prev.reshape(nx, -1)
        z_prev = z_prev.reshape(nu, -1)

       
        # # Also let's see the actual values in cache
        # print("\nCache contents:")
        # for key, value in cache.items():
        #     if isinstance(value, np.ndarray):
        #         print(f"{key} shape:", value.shape)

        # 1. Form decision variable x (should be Nx*N + Nu*(N-1))
        x_decision = []
        for i in range(N):
            x_decision.append(x_prev[:, i].reshape(-1, 1))
            if i < N-1:
                x_decision.append(u_prev[:, i].reshape(-1, 1))
        x = np.vstack(x_decision)

        # 2. Form constraint matrices
        A_dynamics = []
        A_inputs = []
        A_base = cache['A']
        B_base = cache['B']

        #print(f"A_base: {A_base.shape}, B_base: {B_base.shape}")

        for i in range(N-1):
            # Dynamics constraints
            dyn_block = np.zeros((nx, (nx+nu)*(N-1) + nx))
            col_idx = i*(nx+nu)
            dyn_block[:, col_idx:col_idx+nx] = A_base
            dyn_block[:, col_idx+nx:col_idx+nx+nu] = B_base
            dyn_block[:, col_idx+nx+nu:col_idx+2*nx+nu] = -np.eye(nx)
            A_dynamics.append(dyn_block)
            
            # Input constraints
            input_block = np.zeros((nu, (nx+nu)*(N-1) + nx))
            input_block[:, col_idx+nx:col_idx+nx+nu] = np.eye(nu)
            A_inputs.append(input_block)

        A = np.vstack([np.vstack(A_inputs), np.vstack(A_dynamics)])

        # 3. Form z vector
        z_inputs = []
        z_dynamics = []
        for i in range(N-1):
            z_inputs.append(z_prev[:, i].reshape(-1, 1))
            z_dynamics.append(v_prev[:, i].reshape(-1, 1))
        z = np.vstack([np.vstack(z_inputs), np.vstack(z_dynamics)])

        # 4. Form y vector
        y_inputs = []
        y_dynamics = []
        for i in range(N-1):
            y_inputs.append(y_prev[:, i].reshape(-1, 1))
            y_dynamics.append(g_prev[:, i].reshape(-1, 1))
        y = np.vstack([np.vstack(y_inputs), np.vstack(y_dynamics)])

        # 5. Form cost matrix P
        Q = cache['Q']
        R = cache['R']

        P_blocks = []
        for i in range(N):
            if i < N-1:
                P_block = block_diag(Q, R)
            else:
                P_block = Q
            P_blocks.append(P_block)
        P = block_diag(*P_blocks)

        # # 6. Form cost vector q (zero for now)
        # print(f"xg: {xg.shape}")
        # print("xg[:12]:", xg[:12])
        # print("uhover:", uhover)

        q_blocks = []
        for i in range(N):
            # For hover, reference is just xg
            delta_x = x_prev[:, i] - xg[:12]
            q_x = Q @ delta_x.reshape(-1, 1)
            if i < N-1:
                # For hover, reference input is uhover
                delta_u = u_prev[:, i] - uhover
                q_u = R @ delta_u.reshape(-1, 1)
                q_blocks.extend([q_x, q_u])
            else:
                q_blocks.append(q_x)
        q = np.vstack(q_blocks)

        return x, A, z, y, P, q

    def compute_residuals(self, x, A, z, y, P, q):
        """Compute ADMM residuals"""
        # Primal residual        
        Ax = A @ x
        r_prim = Ax - z
        pri_res = np.linalg.norm(r_prim, ord=np.inf)
        pri_norm = max(np.linalg.norm(Ax, ord=np.inf), 
                      np.linalg.norm(z, ord=np.inf))

        # Dual residual
        r_dual = P @ x + q + A.T @ y
        dual_res = np.linalg.norm(r_dual, ord=np.inf)

        # Normalization terms
        Px = P @ x
        ATy = A.T @ y
        dual_norm = max(np.linalg.norm(Px, ord=np.inf),
                       np.linalg.norm(ATy, ord=np.inf),
                       np.linalg.norm(q, ord=np.inf))

        return pri_res, dual_res, pri_norm, dual_norm

    def predict_rho(self, pri_res, dual_res, pri_norm, dual_norm, current_rho):
        """Predict new rho value based on residuals"""

        normalized_pri = pri_res / (pri_norm + 1e-10)
        normalized_dual = dual_res / (dual_norm + 1e-10)

        ratio = normalized_pri / (normalized_dual + 1e-10)
            
        rho_new = current_rho * np.sqrt(ratio)
            

        rho_new = np.clip(rho_new, self.rho_min, self.rho_max)


        # if rho_new >= 1.2*current_rho or rho_new <= current_rho/1.2:
        #     return rho_new


        self.rho_history.append(rho_new)
        return rho_new



    def update_matrices(self, cache, new_rho):
        """Update matrices using derivatives stored in cache"""
        old_rho = cache['rho']
        delta_rho = new_rho - old_rho


        if self.mode == 'hover':
        
            updates = {
                'rho': new_rho,
                'Kinf': cache['Kinf'] + delta_rho * cache['dKinf_drho'],
                'Pinf': cache['Pinf'] + delta_rho * cache['dPinf_drho'],
                'C1': cache['C1'] + delta_rho * cache['dC1_drho'],
                'C2': cache['C2'] + delta_rho * cache['dC2_drho']
            }
        else:

            # NO C2 updates here for now for python numerical stability issues

             updates = {
                'rho': new_rho,
                'Kinf': cache['Kinf'] + delta_rho * cache['dKinf_drho'],
                'Pinf': cache['Pinf'] + delta_rho * cache['dPinf_drho'],
                'C1': cache['C1'] + delta_rho * cache['dC1_drho']
            }

 
        return updates

