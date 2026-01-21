from transformers import PretrainedConfig

class HISEConfig(PretrainedConfig):
    model_type = "hise"

    def __init__(
        self,
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        max_position_embeddings=8192,
        
        # --- Physics Parameters ---
        d_inertial=64,          
        epsilon=0.1,            
        tau=1.0,                
        lambda_conf=0.01,       
        
        # --- Cognitive Gearbox (System 1/2) ---
        use_cognitive_gearbox=True,
        min_epsilon_scale=0.1,  
        system2_threshold=1.0,  
        
        # --- S-Tier Engineering (MOE & Memory) ---
        use_paged_momentum=False,
        fsi_threshold=1.0,
        
        # [NEW] MoE Configuration
        use_moe=False,          # Master switch for MoPE
        num_experts=4,          # Number of physics experts
        num_experts_per_tok=2,  # Top-k routing
        moe_loss_weight=0.01,   # Aux loss weight
        
        initializer_range=0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_position_embeddings = max_position_embeddings
        
        self.d_inertial = d_inertial
        self.epsilon = epsilon
        self.tau = tau
        self.lambda_conf = lambda_conf
        
        self.use_cognitive_gearbox = use_cognitive_gearbox
        self.min_epsilon_scale = min_epsilon_scale
        self.system2_threshold = system2_threshold
        
        self.use_paged_momentum = use_paged_momentum
        self.fsi_threshold = fsi_threshold
        
        # [NEW] Initialize MoE params
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_loss_weight = moe_loss_weight
        
        self.initializer_range = initializer_range
        
        super().__init__(**kwargs)
