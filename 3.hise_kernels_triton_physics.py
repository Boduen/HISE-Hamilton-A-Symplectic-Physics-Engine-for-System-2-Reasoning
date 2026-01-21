import torch
import triton
import triton.language as tl

# ==========================================
# 1. Triton Kernels (Low-Level GPU Ops)
# ==========================================

@triton.jit
def _variable_mass_fwd_kernel(
    m_ptr,              # Input Momentum: [Batch, Seq, d_inertial]
    f_proj_ptr,         # Input Force:    [Batch, Seq, d_inertial]
    mass_ptr,           # Scalar: [Batch, Seq, 1]
    epsilon_ptr,        # Scalar: [Batch, Seq, 1]
    gamma_ptr,          # Scalar: [Batch, Seq, 1]
    out_ptr,            # Output: [Batch, Seq, d_inertial] (New Memory)
    stride_batch, stride_seq, stride_dim,
    stride_scalar_batch, stride_scalar_seq, stride_scalar_dim,
    BLOCK_SIZE: tl.constexpr
):
    """
    Forward Pass: Symplectic Euler Update
    m_{t+1} = (1 - eps * gamma) * m_t + (eps / mass) * F_proj
    """
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    # Offsets for Scalars [B, S, 1]
    scalar_offset = pid_batch * stride_scalar_batch + pid_seq * stride_scalar_seq
    
    # Load Scalars
    mass_val = tl.load(mass_ptr + scalar_offset)
    eps_val = tl.load(epsilon_ptr + scalar_offset)
    gamma_val = tl.load(gamma_ptr + scalar_offset)
    
    # Compute Coefficients
    # Clamp mass to avoid division by zero (Gradient Explosion Protection)
    safe_mass = mass_val + 1e-6
    decay = 1.0 - (eps_val * gamma_val)
    force_scale = eps_val / safe_mass
    
    # Offsets for Vectors [B, S, D]
    base_offset = pid_batch * stride_batch + pid_seq * stride_seq
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < stride_dim
    
    # Load Vectors
    m_val = tl.load(m_ptr + base_offset + offs, mask=mask)
    f_val = tl.load(f_proj_ptr + base_offset + offs, mask=mask)
    
    # Physics Update
    m_new = decay * m_val + force_scale * f_val
    
    # Store Result (Not In-place!)
    tl.store(out_ptr + base_offset + offs, m_new, mask=mask)


@triton.jit
def _variable_mass_bwd_vector_kernel(
    grad_out_ptr,       # dL/dm_new (Incoming Gradient)
    mass_ptr, epsilon_ptr, gamma_ptr,
    grad_m_ptr,         # Output: dL/dm
    grad_f_ptr,         # Output: dL/dF
    stride_batch, stride_seq, stride_dim,
    stride_scalar_batch, stride_scalar_seq, stride_scalar_dim,
    BLOCK_SIZE: tl.constexpr
):
    """
    Backward Pass for Vector Inputs (Momentum & Force)
    dL/dm = dL/dm_new * (1 - eps * gamma)
    dL/dF = dL/dm_new * (eps / mass)
    """
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    # Load Scalars
    scalar_offset = pid_batch * stride_scalar_batch + pid_seq * stride_scalar_seq
    mass_val = tl.load(mass_ptr + scalar_offset)
    eps_val = tl.load(epsilon_ptr + scalar_offset)
    gamma_val = tl.load(gamma_ptr + scalar_offset)
    
    # Precompute gradients coefficients
    decay = 1.0 - (eps_val * gamma_val)
    force_scale = eps_val / (mass_val + 1e-6)
    
    # Load Incoming Gradient
    base_offset = pid_batch * stride_batch + pid_seq * stride_seq
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < stride_dim
    
    grad_out = tl.load(grad_out_ptr + base_offset + offs, mask=mask)
    
    # Compute Input Gradients
    grad_m = grad_out * decay
    grad_f = grad_out * force_scale
    
    # Store
    tl.store(grad_m_ptr + base_offset + offs, grad_m, mask=mask)
    tl.store(grad_f_ptr + base_offset + offs, grad_f, mask=mask)


# ==========================================
# 2. PyTorch Autograd Function (The Bridge)
# ==========================================

class SymplecticPhysicsOps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, f_proj, mass, epsilon, gamma):
        # 1. Prepare Output Buffer
        m_new = torch.empty_like(m)
        
        # 2. Shape & Stride Info
        batch, seq, d_inertial = m.shape
        BLOCK_SIZE = triton.next_power_of_2(d_inertial)
        grid = (batch, seq)
        
        # 3. Launch Forward Kernel
        _variable_mass_fwd_kernel[grid](
            m, f_proj, mass, epsilon, gamma, m_new,
            m.stride(0), m.stride(1), m.stride(2),
            mass.stride(0), mass.stride(1), mass.stride(2),
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # 4. Save Tensors for Backward (Critical for Training)
        ctx.save_for_backward(m, f_proj, mass, epsilon, gamma)
        return m_new

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        m, f_proj, mass, epsilon, gamma = ctx.saved_tensors
        
        # Prepare Gradients for Vectors
        grad_m = torch.empty_like(m)
        grad_f = torch.empty_like(f_proj)
        
        # Launch Backward Kernel (Vector Gradients)
        batch, seq, d_inertial = m.shape
        BLOCK_SIZE = triton.next_power_of_2(d_inertial)
        grid = (batch, seq)
        
        _variable_mass_bwd_vector_kernel[grid](
            grad_output, mass, epsilon, gamma,
            grad_m, grad_f,
            m.stride(0), m.stride(1), m.stride(2),
            mass.stride(0), mass.stride(1), mass.stride(2),
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # --- Compute Scalar Gradients (Mass, Epsilon, Gamma) in PyTorch ---
        # While we could do this in Triton, PyTorch's reduction is optimized enough 
        # for [Batch, Seq] size and easier to maintain for numerical stability.
        
        # Coefficients derived from chain rule:
        # m_new = (1 - eps*gam)*m + (eps/M)*f
        
        # dL/d(eps) = grad_out * (-gamma * m + f / mass)
        term_eps = (-gamma * m) + (f_proj / (mass + 1e-6))
        grad_epsilon = (grad_output * term_eps).sum(dim=-1, keepdim=True)
        
        # dL/d(gamma) = grad_out * (-eps * m)
        term_gamma = -epsilon * m
        grad_gamma = (grad_output * term_gamma).sum(dim=-1, keepdim=True)
        
        # dL/d(mass) = grad_out * (-eps * f / mass^2)
        term_mass = -epsilon * f_proj / ((mass + 1e-6)**2)
        grad_mass = (grad_output * term_mass).sum(dim=-1, keepdim=True)
        
        return grad_m, grad_f, grad_mass, grad_epsilon, grad_gamma


# ==========================================
# 3. Public API
# ==========================================

def fused_agi_update(m, f_proj, mass, epsilon, gamma):
    """
    Fused Symplectic Integrator with Autograd Support.
    
    Args:
        m: Momentum [Batch, Seq, D]
        f_proj: Projected Force [Batch, Seq, D]
        mass: Semantic Mass [Batch, Seq, 1]
        epsilon: Step size [Batch, Seq, 1]
        gamma: Friction [Batch, Seq, 1]
        
    Returns:
        m_new: Updated Momentum [Batch, Seq, D] (Preserves Gradient Chain)
    """
    # Safety: Ensure inputs are contiguous for Triton
    if not m.is_contiguous(): m = m.contiguous()
    if not f_proj.is_contiguous(): f_proj = f_proj.contiguous()
    
    # Broadcast Check
    if mass.dim() == 2: mass = mass.unsqueeze(-1)
    if epsilon.dim() == 2: epsilon = epsilon.unsqueeze(-1)
    if gamma.dim() == 2: gamma = gamma.unsqueeze(-1)
    
    # Execute Function
    return SymplecticPhysicsOps.apply(m, f_proj, mass, epsilon, gamma)
