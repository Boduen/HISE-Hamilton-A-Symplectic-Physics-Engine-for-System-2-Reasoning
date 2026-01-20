import torch
from typing import List, Dict, Optional, Tuple


class FSISafetyValve:
    """
    Implements the Semantic Nyquist Limit check via Fisher Semantic Information (FSI) monitoring.
    Acts as the external supervisor for the System 1/2 Cognitive Gearbox.
    """
    def __init__(self, threshold: float = 1.0, verbose: bool = True):
        self.threshold = threshold # FSI < 1.0 implies Semantic Aliasing (Hallucination)
        self.verbose = verbose


    def check_safety(self, fsi_score: float, current_token_id: int, tokenizer) -> Dict:
        """
        Evaluates the Fisher Semantic Information of the current generation step.
        """
        status = "safe"
        action = "continue"
        
        # FSI < 1.0: The model's internal mass is insufficient to stabilize the semantic trajectory.
        # This indicates 'Axiom Smuggling': inventing entropy to satisfy the prompt.
        if fsi_score < self.threshold:
            status = "hallucination_risk"
            action = "trigger_rag"
            
            if self.verbose:
                token_str = tokenizer.decode([current_token_id])
                print(f"[HISE-Guard] ALERT: FSI {fsi_score:.4f} < {self.threshold}. "
                      f"Token '{token_str}' triggered Axiom Smuggling warning.")
        
        return {
            "fsi": fsi_score,
            "status": status,
            "action": action
        }


class RAGController:
    """
    Mock RAG Controller. 
    In the PSD framework, RAG acts as an 'Entropy Sink', injecting low-entropy axioms 
    to cool down the system and restore geodesic stability.
    """
    def retrieve_context(self, query_embedding: Optional[torch.Tensor] = None) -> str:
        # Placeholder for Milvus/Pinecone retrieval
        # In a real implementation, this would query a vector DB
        print("[RAG] Retrieving external axioms to restore thermodynamic balance...")
        return " [System Note: External Axiom Retrieved: The gravitational constant is G = 6.674e-11.] "


def generate_with_safety(
    model, 
    tokenizer, 
    input_ids: torch.Tensor, 
    max_new_tokens: int = 50, 
    rag_controller: Optional[RAGController] = None
):
    """
    Generation loop with Physics-Informed Safety checks.
    """
    if rag_controller is None:
        rag_controller = RAGController()
        
    safety_valve = FSISafetyValve(threshold=model.config.fsi_threshold)
    
    generated = input_ids
    past_momentums = None
    
    for _ in range(max_new_tokens):
        # Prepare inputs: use only the last token if caching is active
        model_inputs = generated if past_momentums is None else generated[:, -1:]
        
        # Forward Pass with Physics Monitoring
        outputs = model(
            model_inputs, 
            past_momentums=past_momentums, 
            use_cache=True,
            output_fsi=True # Crucial: Request FSI metrics from the physics engine
        )
        
        next_token_logits = outputs.logits[:, -1, :]
        past_momentums = outputs.past_key_values
        
        # Decode FSI from the overloaded 'attentions' field (S-Tier Hack)
        # Shape: [Batch] (Averaged across layers)
        fsi_metric_batch = outputs.attentions 
        
        # Greedy decoding for demo
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Safety Check (Analyze Batch Index 0)
        if fsi_metric_batch is not None:
            current_fsi = fsi_metric_batch[0].item()
            safety_res = safety_valve.check_safety(current_fsi, next_token.item(), tokenizer)
            
            if safety_res["action"] == "trigger_rag":
                # System 2 Failure detected: External intervention required.
                
                # 1. Retrieve Context
                context_str = rag_controller.retrieve_context()
                context_ids = tokenizer.encode(context_str, return_tensors="pt").to(input_ids.device)
                
                # 2. Inject Axioms (Append context to generation)
                # Note: In a full implementation, we might rewind and re-generate.
                # Here we append and let the physics engine settle on the new low-entropy data.
                generated = torch.cat([generated, context_ids], dim=1)
                
                # 3. Reset Momentums (Optional but recommended to clear "bad" inertia)
                # past_momentums = None 
                
                # For this demo, we just print and continue to show flow
                print(f"[HISE-Guard] Context injected. Resuming generation...")
                
        
        generated = torch.cat([generated, next_token], dim=1)
        
        # Simple stopping criteria
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    return generated