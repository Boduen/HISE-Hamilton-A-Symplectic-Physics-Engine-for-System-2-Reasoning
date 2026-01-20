import torch
from typing import Optional, List, Dict


class PagedMomentumManager:
    """
    Manages physical memory for Hamiltonian Momentum states (m).
    Implements a PagedAttention-style memory layout to eliminate fragmentation
    during long-context System 2 reasoning.
    """
    def __init__(self, 
                 num_blocks: int, 
                 block_size: int, 
                 d_inertial: int, 
                 dtype=torch.float16,
                 device="cuda"):
        self.block_size = block_size
        self.d_inertial = d_inertial
        
        # Pre-allocate contiguous memory block
        # Shape: [Num_Blocks, Block_Size, D_Inertial]
        self.momentum_block_tables = torch.zeros(
            (num_blocks, block_size, d_inertial), 
            dtype=dtype, 
            device=device
        )
        self.free_blocks = list(range(num_blocks))
        self.seq_to_block_table: Dict[int, List[int]] = {} 


    def allocate(self, seq_id: int, seq_len: int) -> List[int]:
        """Allocates physical blocks for a new sequence."""
        needed_blocks = (seq_len + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < needed_blocks:
            raise RuntimeError(f"OOM: Not enough PagedMomentum blocks! Needed {needed_blocks}, Free {len(self.free_blocks)}")
            
        blocks = [self.free_blocks.pop() for _ in range(needed_blocks)]
        self.seq_to_block_table[seq_id] = blocks
        return blocks


    def get_physical_pointer(self, seq_id: int, token_pos: int):
        """
        Resolves logical position to physical memory address.
        Used by Triton kernels to write m_new.
        """
        blocks = self.seq_to_block_table[seq_id]
        block_idx = token_pos // self.block_size
        block_offset = token_pos % self.block_size
        
        physical_block = blocks[block_idx]
        return self.momentum_block_tables[physical_block, block_offset]


    def free(self, seq_id: int):
        """Reclaims blocks when a sequence finishes."""
        if seq_id in self.seq_to_block_table:
            blocks = self.seq_to_block_table.pop(seq_id)
            self.free_blocks.extend(blocks)