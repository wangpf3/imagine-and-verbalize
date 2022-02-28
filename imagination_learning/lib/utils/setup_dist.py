import os
import torch
import torch.distributed as dist
import socket
import subprocess

def setup(params):
    
    params.is_slurm_job = 'SLURM_JOB_ID' in os.environ
    print("SLURM job: %s" % str(params.is_slurm_job))

    # SLURM job
    if params.is_slurm_job:

        # assert params.local_rank == -1   # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            'SLURM_JOB_ID',
            'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
            'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
            'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
        ]

        PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            print(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        # params.job_id = os.environ['SLURM_JOB_ID']

        # number of nodes / node ID
        params.n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        params.node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        params.local_rank = int(os.environ['SLURM_LOCALID'])
        params.global_rank = int(os.environ['SLURM_PROCID'])

        # number of processes / GPUs per node
        params.world_size = int(os.environ['SLURM_NTASKS'])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        params.master_addr = hostnames.split()[0].decode('utf-8')
        os.environ['MASTER_PORT'] = str(int(os.environ['SLURM_JOB_ID'])) 
        # assert 10001 <= params.master_port <= 20000 or params.world_size == 1
        print(PREFIX + "Master address: %s" % params.master_addr)
        print(PREFIX + "Master port   : %i" % params.master_port)

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = params.master_addr
        # os.environ['MASTER_PORT'] = str(params.master_port)
        os.environ['WORLD_SIZE'] = str(params.world_size)
        os.environ['RANK'] = str(params.global_rank)

    elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
        print('using multi node multi gpu')
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        local_size = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        global_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        #node_rank = int(os.environ['OMPI_COMM_WORLD_NODE_RANK'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']

        params.master_addr = master_addr
        params.master_port = master_port

        params.global_rank = global_rank
        params.world_size = world_size
        params.local_rank = local_rank
        params.n_gpu_per_node = local_size
        params.n_nodes = world_size // local_size
        params.node_id = global_rank // local_size

        # set environment variables for 'env://'
        os.environ['WORLD_SIZE'] = str(params.world_size)
        os.environ['RANK'] = str(params.global_rank)

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif params.local_rank != -1:
        print('using single node multi gpu')

        # assert params.master_port == -1

        # read environment variables
        params.global_rank = int(os.environ['RANK'])
        params.world_size = int(os.environ['WORLD_SIZE'])
        params.n_gpu_per_node = int(os.environ['NGPU'])
        os.environ['MASTER_PORT'] = str(8090 + int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])) 

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node

    # local job (single GPU)
    else:
        print('using single gpu')
        assert params.local_rank == -1
        assert params.master_port == -1
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = -1
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1

    # define whether this is the master process / if we are in distributed mode
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1
    if params.multi_gpu:
        params.is_master = params.node_id == 0 and params.local_rank == 0
    else:
        params.is_master = True

    # summary
    PREFIX = "%i - " % params.global_rank
    print(PREFIX + "Number of nodes: %i" % params.n_nodes)
    print(PREFIX + "Node ID        : %i" % params.node_id)
    print(PREFIX + "Local rank     : %i" % params.local_rank)
    print(PREFIX + "Global rank    : %i" % params.global_rank)
    print(PREFIX + "World size     : %i" % params.world_size)
    print(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    print(PREFIX + "Master         : %s" % str(params.is_master))
    print(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    print(PREFIX + "Hostname       : %s" % socket.gethostname())

    if params.multi_gpu:
        torch.cuda.set_device(params.local_rank)

    if params.multi_gpu:
        # initialize the process group
        dist.init_process_group("nccl", init_method='env://')
        # master_uri = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        # dist.init_process_group(
        #         backend='nccl',
        #         init_method=master_uri,
        #         world_size=params.world_size,
        #         rank=params.global_rank
        # )

def cleanup():
    dist.destroy_process_group()
