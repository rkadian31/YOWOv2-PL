import torch
import torch.distributed as dist
import os
import subprocess
import pickle
import math

def all_gather(data):
    """Enhanced all_gather with high resolution support"""
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # Check if data contains high resolution tensors
    is_high_res = False
    if isinstance(data, dict) and 'tensor' in data:
        is_high_res = any(dim >= 1080 for dim in data['tensor'].shape)

    # Optimize serialization for high resolution data
    if is_high_res:
        # Chunk large tensors before serialization
        chunk_size = 1024 * 1024  # 1MB chunks
        buffer = optimize_high_res_serialization(data, chunk_size)
    else:
        buffer = pickle.dumps(data)

    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # Enhanced size gathering for high resolution
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # Optimize memory usage for high resolution
    if is_high_res:
        # Use chunked gathering for large tensors
        tensor_list = gather_large_tensors(tensor, max_size, size_list, world_size)
    else:
        # Original gathering for standard resolution
        tensor_list = []
        for _ in size_list:
            tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
        if local_size != max_size:
            padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
            tensor = torch.cat((tensor, padding), dim=0)
        dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def optimize_high_res_serialization(data, chunk_size):
    """Optimize serialization for high resolution data"""
    if isinstance(data, dict) and 'tensor' in data:
        tensor = data['tensor']
        if tensor.numel() * tensor.element_size() > chunk_size:
            chunks = tensor.chunk(math.ceil(tensor.numel() * tensor.element_size() / chunk_size))
            data['tensor_chunks'] = chunks
            data['is_chunked'] = True
            del data['tensor']
    return pickle.dumps(data)

def gather_large_tensors(tensor, max_size, size_list, world_size):
    """Gather large tensors efficiently"""
    chunk_size = 256 * 1024 * 1024  # 256MB chunks
    num_chunks = math.ceil(max_size / chunk_size)
    tensor_list = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, max_size)
        chunk_tensor_list = []
        
        for _ in range(world_size):
            chunk_tensor_list.append(
                torch.empty((end_idx - start_idx,), 
                          dtype=torch.uint8, 
                          device="cuda")
            )
        
        if i == num_chunks - 1 and tensor.size(0) < end_idx:
            padding = torch.empty(
                size=(end_idx - tensor.size(0),),
                dtype=torch.uint8,
                device="cuda"
            )
            chunk_tensor = torch.cat((tensor[start_idx:], padding), dim=0)
        else:
            chunk_tensor = tensor[start_idx:end_idx]
            
        dist.all_gather(chunk_tensor_list, chunk_tensor)
        tensor_list.extend(chunk_tensor_list)

    return tensor_list

def reduce_dict(input_dict, average=True):
    """Enhanced reduce_dict with high resolution support"""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        is_high_res = False

        # Check for high resolution tensors
        for k in sorted(input_dict.keys()):
            if isinstance(input_dict[k], torch.Tensor):
                is_high_res = any(dim >= 1080 for dim in input_dict[k].shape)
            names.append(k)
            values.append(input_dict[k])

        if is_high_res:
            # Use chunked reduction for high resolution
            values = chunk_and_reduce_tensors(values, world_size)
        else:
            # Original reduction
            values = torch.stack(values, dim=0)
            dist.all_reduce(values)

        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def chunk_and_reduce_tensors(tensors, world_size):
    """Chunk and reduce large tensors"""
    chunk_size = 64 * 1024 * 1024  # 64MB chunks
    reduced_tensors = []
    
    for tensor in tensors:
        if tensor.numel() * tensor.element_size() > chunk_size:
            chunks = tensor.chunk(
                math.ceil(tensor.numel() * tensor.element_size() / chunk_size)
            )
            reduced_chunks = []
            for chunk in chunks:
                dist.all_reduce(chunk)
                reduced_chunks.append(chunk)
            reduced_tensors.append(torch.cat(reduced_chunks))
        else:
            dist.all_reduce(tensor)
            reduced_tensors.append(tensor)
    
    return torch.stack(reduced_tensors, dim=0)

def init_distributed_mode(args):
    """Enhanced distributed initialization with high resolution support"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    # Configure for high resolution
    if hasattr(args, 'high_resolution') and args.high_resolution:
        # Set optimal NCCL parameters for high resolution
        os.environ['NCCL_IB_TIMEOUT'] = '23'
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
        
        # Adjust CUDA memory allocation
        torch.cuda.set_device(args.gpu)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        torch.cuda.set_device(args.gpu)

    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
