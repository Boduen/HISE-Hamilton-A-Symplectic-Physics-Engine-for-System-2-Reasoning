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
        d_inertial=64,          # Rank of the inertial manifold (r)
        epsilon=0.1,            # Base symplectic step size (System 1)
        tau=1.0,                # Base temperature
        lambda_conf=0.01,       # Harmonic confinement strength
        
        # --- Cognitive Gearbox (AGI Parameters) ---
        use_cognitive_gearbox=True,
        min_epsilon_scale=0.1,  # Factor to reduce step size in System 2
        system2_threshold=1.0,  # Mass threshold to trigger System 2
        
        # --- S-Tier Engineering ---
        use_paged_momentum=False,
        fsi_threshold=1.0,
        
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
        self.initializer_range = initializer_range
        
        super().__init__(**kwargs)