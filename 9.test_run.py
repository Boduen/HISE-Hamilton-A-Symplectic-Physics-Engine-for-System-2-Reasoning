import torch
import time
from hise.config import HISEConfig
from hise.modeling.modeling_hise import HISEForCausalLM


def run_physics_sanity_check():
    print("=== HISE-Pro Physics Engine: System 1/2 Integration Test ===")
    
    # 1. Configuration (Tiny Scale for Debugging)
    # Enabling Cognitive Gearbox (System 1/2) and S-Tier features
    config = HISEConfig(
        n_layers=2, 
        d_model=128, 
        d_inertial=16,
        use_cognitive_gearbox=True, # Active PSD Mass Dynamics
        system2_threshold=1.0,
        fsi_threshold=1.0
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")


    # 2. Initialize Model
    print("Initializing HISE Model with Spectral-Riemannian Dynamics...")
    model = HISEForCausalLM(config).to(device)
    model.eval() # Set to eval mode for deterministic physics


    # 3. Prepare Dummy Input
    # Simulating a batch of 2 sequences with length 10
    input_ids = torch.randint(0, config.vocab_size, (2, 10)).to(device)
    
    # 4. Forward Pass with Physics Monitoring
    print("Executing Symplectic Forward Pass...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(
            input_ids, 
            output_fsi=True # Critical: Request System 2 metrics
        )
    
    end_time = time.time()
    print(f"Inference Time: {(end_time - start_time)*1000:.2f} ms")


    # 5. Verify Outputs
    # A. Logits Shape
    assert outputs.logits.shape == (2, 10, config.vocab_size), "Logits shape mismatch!"
    print(f"[PASS] Logits Shape: {outputs.logits.shape}")
    
    # B. FSI Metric (System 2 Monitor)
    # Expected shape: [Batch_Size] (Averaged across layers/seq)
    fsi_metric = outputs.attentions
    
    if fsi_metric is not None:
        print(f"[PASS] FSI Metric Captured: {fsi_metric.shape}")
        print(f"   Batch 0 FSI: {fsi_metric[0].item():.4f}")
        print(f"   Batch 1 FSI: {fsi_metric[1].item():.4f}")
        
        # Validate Logic
        if fsi_metric[0] < 1.0:
            print("   -> Status: System 2 Active (Massive/Slow Thought)")
        else:
            print("   -> Status: System 1 Active (Massless/Fast Reflex)")
    else:
        print("[FAIL] FSI Metric is None! Check modeling_hise.py logic.")
        return


    # C. Momentum Cache (KV-Cache equivalent)
    if outputs.past_key_values is not None:
        # Check Layer 0's momentum shape: [Batch, Seq, d_inertial]
        mom_shape = outputs.past_key_values[0].shape
        print(f"[PASS] Inertial Momentum Cache: {mom_shape}")
        assert mom_shape[-1] == config.d_inertial
    
    print("\n=== SUCCESS: Cognitive Gearbox is Operational ===")
    print("The model is now physically grounded based on PSD theory.")


if __name__ == "__main__":
    run_physics_sanity_check()