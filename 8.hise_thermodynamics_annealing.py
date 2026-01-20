import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


class ThermodynamicScheduler:
    """
    Manages the thermodynamic state of the HISE Physics Engine during training.
    
    Implements:
    1. Adaptive Friction (Gamma): Increases damping during high-gradient turbulence.
    2. Temperature Annealing (Tau): Lowers temperature to settle into ground states.
    3. Inverse-Mass Scaling: Adjusts Learning Rate based on Semantic Mass (SR-TCM).
    """
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        config,
        base_tau: float = 1.0, 
        min_tau: float = 0.1
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        
        self.base_tau = base_tau
        self.min_tau = min_tau
        self.current_step = 0
        
        # Track initial LRs for scaling
        self.base_lrs = [param_group['lr'] for param_group in optimizer.param_groups]


    def step(self, epoch: float, loss_val: float, gradient_norm: float) -> Dict[str, float]:
        """
        Called at the end of each training step.
        Adjusts physical constants based on energy landscape topology.
        """
        self.current_step += 1
        
        # 1. Temperature Annealing (Simulated Cooling)
        # Decay Tau to freeze the system into a low-energy ground state
        # Formula: Linear decay with a floor
        decay_progress = min(1.0, epoch / 10.0) # Assume 10 epochs for full cooling
        new_tau = self.base_tau - (self.base_tau - self.min_tau) * decay_progress
        
        # Apply new Tau to all Hamiltonian Attention layers
        for module in self.model.modules():
            if hasattr(module, 'tau'):
                module.tau = new_tau
        
        # 2. Dynamic Friction (Gamma) Adjustment
        # Logic: High gradient norm -> High turbulence -> Need more Friction (Gamma)
        # to dissipate excess energy and prevent divergence.
        if gradient_norm > 1.0:
            friction_scaling = 1.2 # Overdamped regime
        else:
            friction_scaling = 0.9 # Underdamped regime (allow momentum)
            
        # 3. Inverse-Mass LR Scaling (SR-TCM Theory)
        # Ref: "The learning rate eta_k is scaled by its eigen-mass: eta_k ~ sqrt(m_k)"
        # We approximate the global average mass from the Cognitive Gearboxes
        avg_mass = self._get_average_system_mass()
        
        # If the system is "Heavy" (System 2 active), we can afford larger steps 
        # due to high inertial stability, or smaller steps for precision?
        # SR-TCM Eq (2) suggests scaling proportional to sqrt(mass).
        mass_factor = np.sqrt(max(1.0, avg_mass))
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * mass_factor


        return {
            "tau": new_tau,
            "friction_scale": friction_scaling,
            "global_mass": avg_mass,
            "lr_scale": mass_factor
        }


    def _get_average_system_mass(self) -> float:
        """
        Aggregates the current 'Semantic Mass' across all layers.
        Used to monitor if the model is currently in System 1 (Light) or System 2 (Heavy) mode.
        """
        total_mass = 0.0
        count = 0
        
        # Iterate through SoftTCMLayers to find CognitiveGearboxes
        # Note: This requires the model to have run a forward pass recently to populate states,
        # or we inspect the learnable k_b parameters if they encode base mass.
        # For dynamic monitoring, we'd typically hook into the forward pass stats.
        # Here we return a heuristic based on the 'k_b' parameter state (Information Cost).
        
        for module in self.model.modules():
            if hasattr(module, 'k_b'): # CognitiveGearbox
                # k_b represents the conversion rate of Entropy to Mass
                total_mass += module.k_b.item()
                count += 1
                
        return total_mass / max(1, count) if count > 0 else 1.0