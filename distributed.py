from torch import distributed as dist

def setup(rank):
    dist.init_process_group(backend='nccl', init_method='env://')
    return dist.get_world_size()

def cleanup():
    dist.destroy_process_group()

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt