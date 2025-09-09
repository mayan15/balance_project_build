import psutil
import GPUtil
import time

def get_cpu_usage():
    """获取 CPU 使用率"""
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    """获取内存使用率"""
    memory = psutil.virtual_memory()
    return memory.percent

def get_disk_io():
    """获取磁盘 IO 信息"""
    disk_io = psutil.disk_io_counters()
    return {
        "read_bytes": disk_io.read_bytes,
        "write_bytes": disk_io.write_bytes
    }

def get_gpu_usage():
    """获取 GPU 使用率"""
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            "id": gpu.id,
            "name": gpu.name,
            "load": gpu.load * 100,  # GPU 使用率
            "memory_used": gpu.memoryUsed,  # 已用显存
            "memory_total": gpu.memoryTotal  # 总显存
        })
    return gpu_info

def monitor_resources(interval=5):
    """定时监控服务器资源使用率"""
    while True:
        print("\n===== 服务器资源使用率 =====")
        print(f"CPU 使用率: {get_cpu_usage()}%")
        print(f"内存使用率: {get_memory_usage()}%")
        
        disk_io = get_disk_io()
        print(f"磁盘读取字节数: {disk_io['read_bytes']}")
        print(f"磁盘写入字节数: {disk_io['write_bytes']}")
        
        gpu_info = get_gpu_usage()
        if gpu_info:
            for gpu in gpu_info:
                print(f"GPU {gpu['id']} ({gpu['name']}):")
                print(f"  GPU 使用率: {gpu['load']:.2f}%")
                print(f"  显存使用: {gpu['memory_used']}MB / {gpu['memory_total']}MB")
        else:
            print("未检测到 GPU")
        
        print("===========================")
        time.sleep(interval)

if __name__ == "__main__":
    monitor_resources(interval=5)  # 每 5 秒监控一次