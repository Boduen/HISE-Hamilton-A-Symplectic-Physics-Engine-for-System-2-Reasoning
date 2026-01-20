import os
import json
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import wandb


from hise.config import HISEConfig
from hise.modeling.modeling_hise import HISEForCausalLM
from hise.thermodynamics.annealing import ThermodynamicScheduler


# --- 1. 分布式環境設置 ---
def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        # 單機調試模式
        return 0, 1, 0


# --- 2. 物理感知數據集 (Mock) ---
class CausalPhysicsDataset(Dataset):
    def __init__(self, size=10000, seq_len=512):
        self.size = size
        self.seq_len = seq_len
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 實戰中請替換為真實的 Tokenized Data (OpenWebText/RedPajama)
        return {
            "input_ids": torch.randint(0, 50257, (self.seq_len,), dtype=torch.long),
            "labels": torch.randint(0, 50257, (self.seq_len,), dtype=torch.long)
        }


# --- 3. 辛-費雪損失函數 (Physics-Informed Loss) ---
class SymplecticFisherLoss(nn.Module):
    def __init__(self, lambda_phy=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_phy = lambda_phy
        
    def forward(self, logits, labels, fsi_scores):
        # A. 標準預測誤差 (Prediction Error)
        # Shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # B. 物理約束誤差 (Physical Constraint)
        # 懲罰 FSI < 1.0 的行為 (即幻覺/公理走私)
        # Loss_phy = ReLU(1.0 - FSI) -> 只有當 FSI 低於閾值時才懲罰
        if fsi_scores is not None:
            # fsi_scores: [Layers, Batch] -> mean over layers -> [Batch]
            avg_fsi = fsi_scores.mean(dim=0) 
            violation = torch.relu(1.0 - avg_fsi) # 只懲罰 FSI < 1.0
            loss_phy = violation.mean()
        else:
            loss_phy = 0.0
            
        return loss_ce + self.lambda_phy * loss_phy, loss_ce, loss_phy


# --- 4. 訓練主迴圈 ---
def train():
    rank, world_size, local_rank = setup_distributed()
    is_master = rank == 0
    
    # 初始化 WandB (僅主節點)
    if is_master:
        wandb.init(project="HISE-Pro-Evolution", name="Run-01-System1-to-2")


    # A. 載入課表 (Curriculum)
    with open("train/curriculum_config.json", "r") as f:
        curriculum = json.load(f)
    
    # B. 模型配置
    config = HISEConfig(
        n_layers=12, d_model=768, d_inertial=64, # 7B 規模可設為 n_layers=32, d_model=4096
        use_cognitive_gearbox=True,
        vocab_size=50257
    )
    
    model = HISEForCausalLM(config).cuda()
    
    # 啟用 DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    # 熱力學調度器 (取代傳統 Cosine Scheduler)
    thermo_scheduler = ThermodynamicScheduler(model, optimizer, config)
    
    # 物理損失函數
    criterion = SymplecticFisherLoss(lambda_phy=curriculum['loss_function']['lambda_physics'])
    
    # 數據加載
    dataset = CausalPhysicsDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler) # micro_batch=8
    
    # C. 演化迴圈
    global_step = 0
    model.train()
    
    for epoch in range(3): # 演化 Epochs
        if sampler: sampler.set_epoch(epoch)
        
        for batch in dataloader:
            inputs = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            
            # --- 1. 課表階段檢測 (Curriculum Check) ---
            current_stage = None
            for stage in curriculum['stages']:
                if stage['step_range'][0] <= global_step <= stage['step_range'][1]:
                    current_stage = stage
                    break
            
            # 動態注入物理參數 (Mass, Epsilon)
            # 這是 "S-Tier" 的操作：直接修改運行中模型的 Config
            if current_stage and hasattr(model, "module"): # Handle DDP wrapping
                raw_model = model.module
            else:
                raw_model = model
                
            # 根據階段調整物理常數 (模擬宇宙冷卻)
            if current_stage:
                params = current_stage['physics_params']
                raw_model.config.epsilon = params.get('epsilon', 0.1)
                # base_mass 等其他參數可透過 CogGearbox 內部調整
                
            # --- 2. 前向傳播 (Symplectic Forward) ---
            # output_fsi=True 是關鍵：我們需要物理指標來計算 Loss
            outputs = model(inputs, labels=labels, output_fsi=True)
            
            # --- 3. 計算物理損失 ---
            # outputs.attentions 這裡被 Hack 用來傳遞 FSI
            fsi_scores = outputs.attentions 
            loss, loss_ce, loss_phy = criterion(outputs.logits, labels, fsi_scores)
            
            # --- 4. 反向傳播與演化 ---
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪 (防止哈密頓量發散)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # 熱力學退火 (調整溫度與學習率)
            phy_stats = thermo_scheduler.step(epoch, loss.item(), grad_norm)
            
            global_step += 1
            
            # --- 5. 遙測日誌 (Telemetry) ---
            if is_master and global_step % 10 == 0:
                # 計算當前系統狀態
                avg_mass = phy_stats['global_mass']
                
                log_data = {
                    "train/loss_total": loss.item(),
                    "train/loss_ce": loss_ce.item(),
                    "train/loss_physics": loss_phy.item(),
                    "physics/temperature_tau": phy_stats['tau'],
                    "physics/global_mass": avg_mass,
                    "physics/epsilon": raw_model.config.epsilon,
                    "physics/grad_norm": grad_norm.item(),
                    "curriculum/stage": current_stage['name'] if current_stage else "Unknown"
                }
                
                wandb.log(log_data)
                
                print(f"[Step {global_step}] Stage: {log_data['curriculum/stage']} | "
                      f"Loss: {loss.item():.4f} (Phy: {loss_phy.item():.4f}) | "
                      f"Mass: {avg_mass:.2f} | Tau: {log_data['physics/temperature_tau']:.2f}")


    if is_master:
        print(">>> Training Complete. Physical Model Converged.")
        # 保存演化完成的模型
        raw_model.save_pretrained("checkpoints/hise-pro-evolved")
        
    dist.destroy_process_group()


if __name__ == "__main__":
    train()