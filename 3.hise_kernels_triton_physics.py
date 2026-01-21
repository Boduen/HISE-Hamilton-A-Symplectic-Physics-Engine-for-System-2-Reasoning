import torch
import triton
import triton.language as tl

# ==========================================
# 1. Forward Kernel: Fused Linear Recurrence
# ==========================================
# 數學形式: h_t = alpha_t * h_{t-1} + beta_t * u_t
# 在 HISE 中:
# alpha_t = (1 - epsilon * gamma)
# beta_t  = (epsilon / mass)
# u_t     = f_proj

@triton.jit
def _fused_recurrence_fwd_kernel(
    # Inputs
    f_proj_ptr,     # [Batch, Seq, D]
    mass_ptr,       # [Batch, Seq, 1]
    eps_ptr,        # [Batch, Seq, 1]
    gamma_ptr,      # [Batch, Seq, 1]
    h0_ptr,         # [Batch, 1, D] (Initial State)
    # Outputs
    m_out_ptr,      # [Batch, Seq, D] (Momentum History)
    # Strides
    stride_b, stride_s, stride_d,       # f_proj, m_out
    stride_sb, stride_ss, stride_sd,    # scalars
    # Constants
    SEQ_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    # 1. Parallelize over Batch and Dimension (D_inertial)
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    # 2. Initialize State (Momentum)
    # Load h0 if provided, else 0
    h_curr = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    # 指針偏移量計算
    batch_offset = pid_b * stride_b
    scalar_batch_offset = pid_b * stride_sb
    dim_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    
    # Mask for dimension boundary
    dim_mask = dim_offset < stride_d # Assuming D is stride_d in last dim
    
    # 如果有初始狀態 h0，這裡加載 (略，為簡化預設為0，可自行擴充)
    
    # 3. The Loop (Moved from Python to Triton SRAM)
    # Iterate over time sequence
    for t in range(SEQ_LEN):
        # --- A. Load Scalars for time t ---
        # Scalars are [Batch, Seq, 1], shared across Dimension
        scalar_ptr_offset = scalar_batch_offset + t * stride_ss
        
        mass_val = tl.load(mass_ptr + scalar_ptr_offset)
        eps_val = tl.load(eps_ptr + scalar_ptr_offset)
        gamma_val = tl.load(gamma_ptr + scalar_ptr_offset)
        
        # --- B. Compute Coefficients (Alpha, Beta) ---
        # Alpha (Decay): (1 - eps * gamma)
        alpha = 1.0 - (eps_val * gamma_val)
        
        # Beta (Input Scale): (eps / mass)
        # Add epsilon to mass to avoid division by zero
        beta = eps_val / (mass_val + 1e-6)
        
        # --- C. Load Input Force Vector ---
        f_ptr = f_proj_ptr + batch_offset + (t * stride_s) + dim_offset
        f_val = tl.load(f_ptr, mask=dim_mask, other=0.0)
        
        # --- D. Update State (Recurrence) ---
        # h_t = alpha * h_{t-1} + beta * f_t
        h_curr = alpha * h_curr + beta * f_val
        
        # --- E. Store Result ---
        out_ptr = m_out_ptr + batch_offset + (t * stride_s) + dim_offset
        tl.store(out_ptr, h_curr, mask=dim_mask)


# ==========================================
# 2. Backward Kernel: Fused Adjoint Recurrence
# ==========================================
# 反向傳播本質上是時間反向的遞歸 (Running backwards in time)
# dL/dh_{t-1} = dL/dh_t * alpha_t + dL/dLoss_term

@triton.jit
def _fused_recurrence_bwd_kernel(
    # Gradients from top (dL/dm_out)
    grad_out_ptr,   # [Batch, Seq, D]
    
    # Original Inputs for re-computing coeffs
    f_proj_ptr, mass_ptr, eps_ptr, gamma_ptr,
    m_out_ptr,      # Need history for some derivatives
    
    # Output Gradients
    grad_f_ptr,     # [Batch, Seq, D]
    grad_mass_ptr,  # [Batch, Seq, 1]
    grad_eps_ptr,   # [Batch, Seq, 1]
    grad_gamma_ptr, # [Batch, Seq, 1]
    
    # Strides
    stride_b, stride_s, stride_d,
    stride_sb, stride_ss, stride_sd,
    
    SEQ_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    # Gradient Accumulator for the Hidden State (running backwards)
    d_h_next = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    # Offsets
    batch_offset = pid_b * stride_b
    scalar_batch_offset = pid_b * stride_sb
    dim_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offset < stride_d
    
    # Iterate BACKWARDS from T-1 to 0
    for t in range(SEQ_LEN - 1, -1, -1):
        # --- 1. Load Data at time t ---
        # Scalars
        scalar_ptr_offset = scalar_batch_offset + t * stride_ss
        mass = tl.load(mass_ptr + scalar_ptr_offset)
        eps = tl.load(eps_ptr + scalar_ptr_offset)
        gamma = tl.load(gamma_ptr + scalar_ptr_offset)
        
        # Inputs & State History
        # Note: To compute dL/d(alpha), we need h_{t-1}. 
        # But for simplicity in this fast kernel, we can approximate or reload.
        # Strict correctness requires loading m_{t-1}.
        
        # Load m_{t-1} (Previous state). If t=0, it's 0.
        prev_h_val = tl.zeros([BLOCK_D], dtype=tl.float32)
        if t > 0:
            m_prev_ptr = m_out_ptr + batch_offset + ((t-1) * stride_s) + dim_offset
            prev_h_val = tl.load(m_prev_ptr, mask=dim_mask)
            
        # Load Force input
        f_ptr = f_proj_ptr + batch_offset + (t * stride_s) + dim_offset
        f_val = tl.load(f_ptr, mask=dim_mask)
        
        # Load Incoming Gradient dL/dm_t
        grad_out_ptr_t = grad_out_ptr + batch_offset + (t * stride_s) + dim_offset
        d_m_curr = tl.load(grad_out_ptr_t, mask=dim_mask)
        
        # Total gradient at this step = Incoming from above + Recurrent from future
        d_h_total = d_m_curr + d_h_next
        
        # --- 2. Compute Derivatives ---
        # Model: h_t = (1 - eps*gam) * h_{t-1} + (eps/mass) * f_t
        
        alpha = 1.0 - (eps * gamma)
        beta = eps / (mass + 1e-6)
        
        # A. Gradient w.r.t Input Force f_t
        # dL/df = dL/dh * beta
        d_f = d_h_total * beta
        tl.store(grad_f_ptr + batch_offset + (t*stride_s) + dim_offset, d_f, mask=dim_mask)
        
        # B. Gradient w.r.t Hidden State h_{t-1} (Pass to next iteration)
        # dL/dh_{t-1} = dL/dh_t * alpha
        d_h_next = d_h_total * alpha
        
        # C. Gradients w.r.t Scalars (Mass, Eps, Gamma)
        # These are reductions over dimension D.
        # dL/d(alpha) = dL/dh * h_{t-1}
        d_alpha = tl.sum(d_h_total * prev_h_val, axis=0)
        
        # dL/d(beta) = dL/dh * f_t
        d_beta = tl.sum(d_h_total * f_val, axis=0)
        
        # Chain Rule for Scalars
        # alpha = 1 - eps*gam  => d_alpha/d_eps = -gam, d_alpha/d_gam = -eps
        # beta = eps/mass      => d_beta/d_eps = 1/mass, d_beta/d_mass = -eps/mass^2
        
        d_eps_val = d_alpha * (-gamma) + d_beta * (1.0 / (mass + 1e-6))
        d_gamma_val = d_alpha * (-eps)
        d_mass_val = d_beta * (-eps / ((mass + 1e-6) * (mass + 1e-6)))
        
        # Store Scalar Gradients (Atomic Add is safer if multiple blocks map to same scalar, 
        # but here we parallelize over D, so we need to be careful.
        # SIMPLIFICATION: We assume BLOCK_D covers full dimension D, or we perform reduction outside.
        # For this example, we assume d_inertial <= 1024 so 1 block handles it.
        tl.store(grad_eps_ptr + scalar_ptr_offset, d_eps_val)
        tl.store(grad_gamma_ptr + scalar_ptr_offset, d_gamma_val)
        tl.store(grad_mass_ptr + scalar_ptr_offset, d_mass_val)


# ==========================================
# 3. PyTorch Autograd Interface
# ==========================================

class FusedRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f_proj, mass, epsilon, gamma):
        # shapes
        B, S, D = f_proj.shape
        
        # Output buffer
        m_out = torch.empty_like(f_proj)
        
        # Grid
        # Parallelize over Batch and Dimension Blocks
        BLOCK_D = triton.next_power_of_2(D)
        num_warps = 4
        if BLOCK_D > 1024: 
            BLOCK_D = 1024 # Hardware limit
            # Note: If D > 1024, need loop over D blocks, simplified here.
        
        grid = (B, 1) # Simplified: Assumes D fits in one block or we stride inside kernel (omitted)
        
        # Launch
        _fused_recurrence_fwd_kernel[grid](
            f_proj, mass, epsilon, gamma, None,
            m_out,
            f_proj.stride(0), f_proj.stride(1), f_proj.stride(2),
            mass.stride(0), mass.stride(1), mass.stride(2),
            SEQ_LEN=S, BLOCK_D=BLOCK_D,
            num_warps=num_warps
        )
        
        ctx.save_for_backward(f_proj, mass, epsilon, gamma, m_out)
        return m_out

    @staticmethod
    def backward(ctx, grad_out):
        f_proj, mass, epsilon, gamma, m_out = ctx.saved_tensors
        B, S, D = f_proj.shape
        
        # Gradients
        grad_f = torch.empty_like(f_proj)
        grad_mass = torch.empty_like(mass)
        grad_eps = torch.empty_like(epsilon)
        grad_gamma = torch.empty_like(gamma)
        
        BLOCK_D = triton.next_power_of_2(D)
        if BLOCK_D > 1024: BLOCK_D = 1024
        grid = (B, 1)
        
        _fused_recurrence_bwd_kernel[grid](
            grad_out,
            f_proj, mass, epsilon, gamma, m_out,
            grad_f, grad_mass, grad_eps, grad_gamma,
            f_proj.stride(0), f_proj.stride(1), f_proj.stride(2),
            mass.stride(0), mass.stride(1), mass.stride(2),
            SEQ_LEN=S, BLOCK_D=BLOCK_D
        )
        
        return grad_f, grad_mass, grad_eps, grad_gamma


def fused_agi_update(past_momentum, f_proj, mass, epsilon, gamma):
    """
    Standard Interface:
    If past_momentum is provided (Inference), use single step (Logic handled in layer).
    If past_momentum is None (Training), use Fused Scan.
    
    NOTE: The SoftTCMLayer logic needs to pass the FULL SEQUENCE 'f_proj' here for training.
    """
    # Use the Fused Scan for Training Speedup
    # Input shapes: [Batch, Seq, Dim]
    
    # Broadcast check
    if mass.dim() == 2: mass = mass.unsqueeze(-1)
    if epsilon.dim() == 2: epsilon = epsilon.unsqueeze(-1)
    if gamma.dim() == 2: gamma = gamma.unsqueeze(-1)
    
    return FusedRecurrenceFunction.apply(f_proj, mass, epsilon, gamma)
