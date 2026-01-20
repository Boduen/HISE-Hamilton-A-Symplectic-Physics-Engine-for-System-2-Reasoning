import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import HISEConfig


class CognitiveGearbox(nn.Module):
    """
    Implements the 'Mass-Entropy' equivalence from Projective Spectral Dynamics (PSD).
    Dynamically calculates semantic mass M(t) and adjusts time-step epsilon(t).
    """
    def __init__(self, config: HISEConfig):
        super().__init__()
        self.base_epsilon = config.epsilon
        self.min_epsilon = config.epsilon * config.min_epsilon_scale
        self.threshold = config.system2_threshold
        
        # Learnable Boltzmann Constant equivalent for information
        # Corresponds to k_b in PSD theory
        self.k_b = nn.Parameter(torch.tensor(1.0)) 


    def compute_local_entropy(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq, Dim] -> Entropy: [Batch, Seq, 1]
        # Calculates Shannon entropy of the local semantic distribution
        prob = F.softmax(x, dim=-1)
        log_prob = F.log_softmax(x, dim=-1)
        entropy = -(prob * log_prob).sum(dim=-1, keepdim=True)
        return entropy


    def derive_mass(self, entropy: torch.Tensor, fsi_score: torch.Tensor) -> torch.Tensor:
        """
        Derives Mass M(t) based on Complexity Cost.
        Formula: M ~ sqrt(k_b * H * ln(2))
        """
        # Complexity Cost
        complexity_cost = self.k_b * entropy * 0.693 # ln(2) approx
        
        # Base Mass from Complexity
        mass_base = torch.sqrt(torch.clamp(complexity_cost, min=1e-6))
        
        # FSI Modulation: High risk (low FSI) drastically increases Mass (Inertia)
        # to prevent hallucination (Axiom Smuggling).
        fsi_safe = fsi_score.unsqueeze(-1) + 1e-6
        safety_factor = torch.clamp(1.0 / fsi_safe, max=10.0)
        
        mass_dynamic = mass_base * safety_factor
        return mass_dynamic


    def forward(self, h: torch.Tensor, fsi_score: torch.Tensor):
        entropy = self.compute_local_entropy(h)
        mass_t = self.derive_mass(entropy, fsi_score)
        
        # Determine System Mode
        # Mass > Threshold implies System 2 (Heavy/Slow)
        is_system_2 = mass_t > self.threshold
        
        # Adaptive Epsilon (Step Size)
        # System 1 -> Base Epsilon (Fast)
        # System 2 -> Min Epsilon (Slow, fine-grained)
        epsilon_t = torch.where(
            is_system_2,
            torch.tensor(self.min_epsilon, device=h.device),
            torch.tensor(self.base_epsilon, device=h.device)
        )
        
        return mass_t, epsilon_t, is_system_2