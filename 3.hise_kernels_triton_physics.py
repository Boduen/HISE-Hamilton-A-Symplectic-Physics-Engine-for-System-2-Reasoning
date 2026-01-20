import torch
import triton
import triton.language as tl


@triton.jit
def variable_mass_symplectic_kernel(
    m_ptr,              # [Batch, Seq, d_inertial]
    f_proj_ptr,         # [Batch, Seq, d_inertial]
    mass_ptr,           # [Batch, Seq, 1]
    epsilon_ptr,        # [Batch, Seq, 1]
    gamma_ptr,          # [Batch, Seq, 1]
    stride_m_batch, stride_m_seq, stride_m_dim,
    stride_scalar_batch, stride_scalar_seq, stride_scalar_dim,
    BLOCK_SIZE: tl.constexpr
):
    """
    Symplectic Euler Update with per-token Mass and Step-size.
    Physics: m_{t+1} = (1 - eps * gamma) * m_t + (eps / mass) * F_proj
    """
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    # Calculate offsets
    base_offset = pid_batch * stride_m_batch + pid_seq * stride_m_seq
    scalar_offset = pid_batch * stride_scalar_batch + pid_seq * stride_scalar_seq
    
    # Load Scalars (Shared across d_inertial dimension)
    mass_val = tl.load(mass_ptr + scalar_offset)
    eps_val = tl.load(epsilon_ptr + scalar_offset)
    gamma_val = tl.load(gamma_ptr + scalar_offset)
    
    # Calculate Decay
    decay = 1.0 - (eps_val * gamma_val)
    
    # Vectorized Load of Momentum and Force
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < stride_m_dim
    
    m_ptrs = m_ptr + base_offset + offs
    f_ptrs = f_proj_ptr + base_offset + offs
    
    m_val = tl.load(m_ptrs, mask=mask)
    f_val = tl.load(f_ptrs, mask=mask)
    
    # Update Momentum: Force is dampened by Mass (Inertia)
    # F_effective = (epsilon / Mass) * F
    force_term = (eps_val / (mass_val + 1e-6)) * f_val
    m_new = decay * m_val + force_term
    
    # Store result
    tl.store(m_ptrs, m_new, mask=mask)


def fused_agi_update(m, f_proj, mass, epsilon, gamma):
    """
    Python wrapper for the Variable Mass kernel.
    Handles shape broadcasting and grid dispatch.
    """
    batch, seq, d_inertial = m.shape
    grid = (batch, seq)
    BLOCK_SIZE = triton.next_power_of_2(d_inertial)
    
    # Ensure scalars are [B, S, 1]
    if mass.dim() == 2: mass = mass.unsqueeze(-1)
    if epsilon.dim() == 2: epsilon = epsilon.unsqueeze(-1)
    if gamma.dim() == 2: gamma = gamma.unsqueeze(-1)


    variable_mass_symplectic_kernel[grid](
        m, f_proj, mass, epsilon, gamma,
        m.stride(0), m.stride(1), m.stride(2),
        mass.stride(0), mass.stride(1), mass.stride(2),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return m