import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PhysicsRouter(nn.Module):
    """
    Implements the 'Mixture of Physics Experts' (MoPE) routing logic.
    Instead of learning a router from scratch, we use 'Thermodynamic Routing':
    Tokens are routed based on their Semantic Mass and Entropy.
    """
    def __init__(self, config, num_experts: int = 4):
        super().__init__()
        self.d_model = config.d_model
        self.num_experts = num_experts
        self.system2_threshold = config.system2_threshold
        
        # A learnable gate to fine-tune the physical signal
        self.gate = nn.Linear(config.d_model, num_experts, bias=False)


    def forward(self, hidden_states: torch.Tensor, mass: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [Batch, Seq, Dim]
            mass: [Batch, Seq, 1] - The Semantic Mass derived from PSD.
        
        Returns:
            router_logits: [Batch*Seq, Num_Experts]
            expert_indices: [Batch*Seq, TopK]
        """
        B, T, C = hidden_states.shape
        flat_hidden = hidden_states.view(-1, C)
        flat_mass = mass.view(-1, 1)


        # Base Routing Logits from content
        router_logits = self.gate(flat_hidden)
        
        # --- Physics-Informed Bias ---
        # We bias the router to force 'Heavy' tokens to specific experts.
        # Assume Expert 0 is 'System 1' (Fast), Expert 3 is 'System 2' (Slow/Complex).
        
        # If Mass > Threshold, boost logits for the last expert (Deep Thinker)
        # If Mass is low, boost logits for the first expert (Reflex)
        
        system2_mask = (flat_mass > self.system2_threshold).float()
        
        # Expert 0 (Fast) bias: Boost if mass is low
        router_logits[:, 0] += (1.0 - system2_mask.squeeze()) * 5.0
        
        # Expert N (Slow) bias: Boost if mass is high
        router_logits[:, -1] += system2_mask.squeeze() * 5.0
        
        return router_logits
        
    def route_tokens(self, hidden_states: torch.Tensor, router_logits: torch.Tensor, top_k: int = 2):
        """
        Standard Top-K routing with physics-biased logits.
        """
        routing_weights = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(routing_weights, top_k, dim=-1)
        
        return expert_weights, expert_indices


class MoPEBlock(nn.Module):
    """
    A Mixture-of-Experts block where experts are specialized Physics Engines.
    """
    def __init__(self, config, num_experts=4):
        super().__init__()
        self.router = PhysicsRouter(config, num_experts)
        
        # Experts can be standard MLPs or specialized SoftTCMLayers with different time constants
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, 4 * config.d_model),
                nn.GELU(),
                nn.Linear(4 * config.d_model, config.d_model)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, hidden_states, mass):
        B, T, C = hidden_states.shape
        flat_hidden = hidden_states.view(-1, C)
        
        # 1. Calculate Routing
        logits = self.router(hidden_states, mass)
        weights, indices = self.router.route_tokens(hidden_states, logits, top_k=2)
        
        # 2. Dispatch & Combine (Simplified for demonstration)
        final_output = torch.zeros_like(flat_hidden)
        
        # Naive sequential execution (In production, use permuted batching/Triton)
        for i, expert in enumerate(self.experts):
            # Find tokens assigned to this expert
            # Mask: [Batch*Seq, TopK]
            expert_mask = (indices == i)
            
            # If any token needs this expert
            if expert_mask.any():
                # We simply apply the expert to all and mask (Inefficient but clear logic)
                # Production: Gather -> Execute -> Scatter
                expert_out = expert(flat_hidden)
                
                # Weighted sum
                # Check if expert is 1st or 2nd choice for each token
                is_first = expert_mask[:, 0]
                is_second = expert_mask[:, 1]
                
                weight_first = weights[:, 0].unsqueeze(-1)
                weight_second = weights[:, 1].unsqueeze(-1)
                
                final_output += is_first.unsqueeze(-1) * weight_first * expert_out
                final_output += is_second.unsqueeze(-1) * weight_second * expert_out
                
        return final_output.view(B, T, C)