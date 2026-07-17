import os
import time
import torch

def occupy_gpu_memory(gpu_id: int, target_gb: float = 12.0):
    # 1. 指定仅使用目标GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无GPU设备")
    
    device = torch.device("cuda:0")
    # 单精度float32：每个元素4字节
    byte_per_elem = 4
    target_bytes = target_gb * 1024 ** 3
    total_elem = int(target_bytes / byte_per_elem)

    print(f"=== 准备占用 GPU{gpu_id} 显存 {target_gb}GB ===")
    print(f"计算张量总元素数量: {total_elem:,}")
    
    # 分配大张量锁定显存
    try:
        # 分两段张量防止一次性分配失败
        tensor1 = torch.randn((total_elem // 2), device=device, dtype=torch.float32)
        tensor2 = torch.randn((total_elem - total_elem // 2), device=device, dtype=torch.float32)
        print("显存占用成功，保持占用中...")
        print("按 Ctrl+C 即可立即终止程序并释放显存")
        
        # 持续循环维持占用
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n检测到终止信号，释放显存，程序退出")
        # 删除张量释放显存
        del tensor1, tensor2
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"分配显存失败: {e}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # ========== 配置区 ==========
    TARGET_GPU = 0       # 要占用的GPU编号，修改这里切换GPU
    TARGET_MEM_GB = 12.0 # 需要占用的显存大小GB
    # ===========================
    occupy_gpu_memory(gpu_id=TARGET_GPU, target_gb=TARGET_MEM_GB)

    # python /data/zjy_work/GPU.py