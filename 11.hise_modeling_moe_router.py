import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PhysicsRouter(nn.Module):
    """
    Implements 'Thermodynamic Routing' with Load Balancing.
    [FIXED]: 
    1. Removed hard-coded logic (+5.0).
    2. Added learnable physics bias.
    3. Implemented Load Balancing Loss to prevent expert collapse.
    """
    def __init__(self, config, num_experts: int = 4):
        super().__init__()
        self.d_model = config.d_model
        self.num_experts = num_experts
        
        # Router Gate: Projects input to expert logits
        self.gate = nn.Linear(config.d_model, num_experts, bias=False)
        
        # [NEW] Learnable Physics Bias
        # Instead of hard-coding "Mass > Threshold -> Expert 3", we let the model learn
        # how Mass impacts expert selection.
        # Shape: [Num_Experts] - Each expert has a sensitivity to Semantic Mass
        self.mass_bias = nn.Parameter(torch.zeros(num_experts))


    def forward(self, hidden_states: torch.Tensor, mass: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [Batch, Seq, Dim]
            mass: [Batch, Seq, 1] - The Semantic Mass derived from PSD.
        
        Returns:
            router_logits: [Batch*Seq, Num_Experts]
            expert_indices: [Batch*Seq, TopK]
            aux_loss: Scalar tensor for load balancing
        """
        B, T, C = hidden_states.shape
        flat_hidden = hidden_states.view(-1, C)
        flat_mass = mass.view(-1, 1)


        # 1. Base Routing Logits
        logits = self.gate(flat_hidden) # [N, Num_Experts]
        
        # 2. Apply Physics Bias (Learnable)
        # We scale the mass influence. If mass is high, it boosts experts with high mass_bias.
        # This preserves the "System 1/2" intent but makes it differentiable.
        physics_impact = flat_mass * self.mass_bias.unsqueeze(0)
        logits = logits + physics_impact
        
        # 3. Calculate Load Balancing Loss (Auxiliary Loss)
        # Reference: Switch Transformer / Mixtral paper
        probs = F.softmax(logits, dim=-1)
        
        # limit top_k selection
        top_k = 2
        top_k_weights, top_k_indices = torch.topk(probs, top_k, dim=-1)
        
        # Calculate Aux Loss (Coefficient usually 0.01 - 0.1 in main loop)
        # Importance: Sum of probabilities assigned to each expert
        expert_importance = probs.sum(0)
        # Load: Count of tokens assigned to each expert (approximate via gradients)
        # We use a soft proxy for load to keep it differentiable or just use importance variance
        # Here we implement the standard "Mean Squared importance" to encourage uniformity
        target_load = probs.size(0) / self.num_experts
        aux_loss = ((expert_importance - target_load) ** 2).mean()
        
        return logits, top_k_indices, aux_loss, top_k_weights


class MoPEBlock(nn.Module):
    """
    A Mixture-of-Experts block where experts are specialized Physics Engines.
    """
    def __init__(self, config, num_experts=4):
        super().__init__()
        self.router = PhysicsRouter(config, num_experts)
        
        # Experts: Can be specialized MLPs or SoftTCMLayers
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, 4 * config.d_model),
                nn.GELU(),
                nn.Linear(4 * config.d_model, config.d_model)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, hidden_states, mass):
        """
        Returns:
            output: [Batch, Seq, Dim]
            aux_loss: Scalar
        """
        B, T, C = hidden_states.shape
        flat_hidden = hidden_states.view(-1, C)
        
        # 1. Route
        logits, indices, aux_loss, weights = self.router(flat_hidden, mass)
        
        # 2. Dispatch & Execute
        final_output = torch.zeros_like(flat_hidden)
        
        # Naive Loop (Compatible with standard PyTorch)
        # For production speedup, use 'torch.scatter_add' or Triton Permutation
        for i, expert in enumerate(self.experts):
            # Identification: Which tokens chose this expert?
            # indices: [N, TopK]
            
            # Mask for 1st choice
            mask1 = (indices[:, 0] == i)
            # Mask for 2nd choice
            mask2 = (indices[:, 1] == i)
            
            combined_mask = mask1 | mask2
            
            if combined_mask.any():
                # Extract tokens for this expert
                # Note: This slicing is slow but functionally correct for training
                selected_tokens = flat_hidden[combined_mask]
                
                # Expert Forward
                expert_out = expert(selected_tokens)
                
                # Scatter back results
                # We need to multiply by the routing weight
                
                # Handle 1st choice weights
                if mask1.any():
                    # Get weights for tokens where this expert was 1st choice
                    # We need to map back carefully.
                    # For simplicity in this naive impl, we iterate full batch logic:
                    w1 = weights[mask1, 0].unsqueeze(-1)
                    # We need to subset expert_out corresponding to mask1
                    # This naive loop logic is complex to get perfectly right in 20 lines without scatter
                    # So we use the "Zero Masking" approach for readability:
                    
                    # Full batch forward (Wasteful but correct graph) * Mask
                    # Ideally, use Megablocks or scatter/gather.
                    pass 

        # --- Efficient Dispatch Implementation (Replacing Naive Loop) ---
        # Reshape to [N, TopK, C] to handle weights easily
        # But since experts are standard NN modules, we process expert-by-expert
        
        results = torch.zeros_like(flat_hidden)
        
        for k in range(2): # For Top-1 and Top-2
            expert_idx = indices[:, k] # [N]
            w = weights[:, k].unsqueeze(-1) # [N, 1]
            
            for e_idx, expert in enumerate(self.experts):
                # Boolean mask for tokens that selected expert e_idx at rank k
                token_mask = (expert_idx == e_idx)
                
                if token_mask.any():
                    inp = flat_hidden[token_mask]
                    out = expert(inp)
                    # Accumulate: results[masked] += weight * output
                    # In-place add with masking requires index_put or scatter
                    # We use a masked add for simplicity
                    results[token_mask] += w[token_mask] * out
                    
        return results.view(B, T, C), aux_loss
