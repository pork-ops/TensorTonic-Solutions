def gpu_occupancy(
    threads_per_block: int, registers_per_thread: int, shared_mem_per_block: int,
    max_threads_per_sm: int, max_warps_per_sm: int, max_blocks_per_sm: int,
    max_registers_per_sm: int, max_shared_mem_per_sm: int, warp_size: int = 32,
) -> dict:
    """
    Returns a dict: blocks_per_sm (int), resident_warps (int), occupancy (float).
    """
    warps_per_block, need_extra = divmod(threads_per_block, warp_size)
    if need_extra: 
        warps_per_block = warps_per_block + 1 
    total_threads = warps_per_block * warp_size
    total_registers = total_threads * registers_per_thread

    
    limit_A = max_threads_per_sm // total_threads
    limit_B = max_warps_per_sm // warps_per_block

    if total_registers:
        limit_C = max_registers_per_sm // total_registers
    else:
        limit_C = max_blocks_per_sm # no limit case
    if shared_mem_per_block:
        limit_D = max_shared_mem_per_sm // shared_mem_per_block
    else:
        limit_D = max_blocks_per_sm # no limit case
    
    b = min([limit_A, limit_B, limit_C, limit_D, max_blocks_per_sm])
    
    return {
        "blocks_per_sm": b,
        "resident_warps": b*warps_per_block,
        "occupancy": float((b*warps_per_block)/max_warps_per_sm)
    }
