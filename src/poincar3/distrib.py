import os
import torch
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


__all__ = [
    "init_distributed",
    "is_main_process",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "setup_multinode_nccl_env",
]

_is_distributed: bool = False
_is_initialized: bool = False


def init_distributed():
    global _is_distributed
    global _is_initialized
    if _is_initialized:
        raise RuntimeError("Distributed training already initialized.")
    local_rank = int(os.environ.get("LOCAL_RANK", default=-1))
    if local_rank != -1:
        logger.info(f"Initializing distributed training on local rank {local_rank}")

        # Default NCCL watchdog timeout (10 min) is too tight for this dataloader:
        # tail-latency `__getitem__` calls on the shared Lustre mount have been
        # measured up to ~370s even under moderate contention (see
        # experiments/stress_test_teacher_len.py), and that's on a single node --
        # multi-node runs share the same filesystem and have more concurrent
        # readers. 30 min gives real stragglers room to finish without masking a
        # true hang for too long.
        torch.distributed.init_process_group(  # ty: ignore[possibly-unbound-attribute]
            backend="nccl", timeout=timedelta(minutes=30)
        )
        torch.cuda.set_device(local_rank)
    _set_is_distributed(local_rank != -1)
    _is_initialized = True


def get_rank() -> int:
    if not is_distributed():
        return 0
    return torch.distributed.get_rank()  # type: ignore[possibly-unbound-attribute]


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return torch.distributed.get_world_size()  # type: ignore[possibly-unbound-attribute]


def is_main_process():
    return get_rank() == 0
    # if not is_distributed():
    #     return True
    # return torch.distributed.get_rank() == 0  # type: ignore[possibly-unbound-attribute]


def is_distributed():
    global _is_distributed
    return _is_distributed


def _set_is_distributed(value: bool):
    global _is_distributed
    _is_distributed = value


def setup_multinode_nccl_env() -> None:
    """Must be called before `init_distributed()` creates the NCCL
    communicator -- sets process-wide env vars, not something `init_process_
    group` args can express.

    Originally lived inline in `experiments/train.py`'s `if __name__ ==
    "__main__":` block; factored out here so every multi-node-capable
    entrypoint (SSL pretraining and the `experiments/ffrecon/train_*.py`
    finetune scripts, via `poincar3.finetune.cli_main`) gets the same fabric
    setup instead of only the one script that happened to have it first.
    """
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["HDF5_USE_FILE_LOCKING"] = "0"
    # Enable the NCCL flight recorder so a collective timeout dumps a stack
    # trace / trace buffer instead of "FlightRecorder is disabled" -- must be
    # set before init_process_group() creates the NCCL communicator.
    # TORCH_NCCL_TRACE_BUFFER_SIZE was renamed to TORCH_FR_BUFFER_SIZE in newer
    # torch releases; set both since we can't be sure which one this torch
    # version actually reads.
    os.environ.setdefault("TORCH_FR_BUFFER_SIZE", "2000")
    os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "2000")
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    logger.info(
        f"[nccl flight recorder] RANK={os.environ.get('RANK', '<unset>')} "
        f"TORCH_FR_BUFFER_SIZE={os.environ['TORCH_FR_BUFFER_SIZE']} "
        f"TORCH_NCCL_TRACE_BUFFER_SIZE={os.environ['TORCH_NCCL_TRACE_BUFFER_SIZE']} "
        f"TORCH_NCCL_DUMP_ON_TIMEOUT={os.environ['TORCH_NCCL_DUMP_ON_TIMEOUT']}"
    )
    # Everything below only matters once traffic crosses a node boundary --
    # forcing NCCL_NET to the AWS OFI/libfabric plugin on a single-node job
    # makes NCCL try to init that plugin even though intra-node collectives
    # never need a network transport (NVLink/PCIe P2P handles those), and on
    # this cluster that init can hard-fail ("Failed to initialize any NET
    # plugin") outside a real multi-node `srun` allocation. slurm_util's
    # `submit --dist` exports SLURM_NNODES (see
    # .venv-aarch64/.../slurm_util/submit.py wrap_command), so use that to
    # gate on node count; default to 1 (single-node behavior) when unset,
    # e.g. for local/non-slurm runs.
    num_nodes = int(os.environ.get("SLURM_NNODES", "1"))
    if num_nodes > 1:
        # Arrhenius's fast fabric is HPE Slingshot over libfabric (cxi provider),
        # not plain Mellanox IB verbs. The pip `nvidia-nccl-cu13` wheel this repo
        # depends on has no libfabric support built in, so without these it
        # silently falls back to NCCL's plain socket transport for anything that
        # crosses a node boundary -- measured at ~6.5 GB/s busbw multi-node vs.
        # ~330 GB/s intra-node with scripts/nccl_bench.py. These three vars,
        # taken from Arrhenius's own `GPU/buildenv-gcccuda/2026.03-cu13.0`
        # module (which bundles the same NCCL 2.29.7 plus the aws-ofi-nccl
        # plugin), point the pip NCCL at that plugin instead and recovered
        # ~90 GB/s busbw in the same benchmark -- an ~14x fix.
        os.environ.setdefault(
            "NCCL_NET_PLUGIN",
            "/software/sse2/el9_gh200/manual/AWS_OFI_NCCL/1.19.2/g14/cu13.0/mp43/hpc1/lib/libnccl-net-ofi.so",
        )
        os.environ.setdefault("NCCL_NET", "AWS Libfabric")
        os.environ.setdefault("FI_PROVIDER", "cxi")
        # The three vars above are the only ones Arrhenius's own NCCL module sets
        # -- they're enough to get a plain nccl-tests-style all_reduce benchmark
        # to 90 GB/s busbw, but NOT enough for real training: we hit an immediate
        # deadlock at step 1 with just those three. Two other GH200+Slingshot HPC
        # sites (CSCS/Todi, Sigma2/Olivia -- same libfabric cxi provider, same
        # aws-ofi-nccl plugin) document this exact failure class and converge on
        # the same fix: the CXI provider's default eager-message/rendezvous path
        # can deadlock real NCCL traffic (many concurrent, variably-sized
        # collectives) even though a single fixed-size all_reduce benchmark is
        # fine. Forcing the rendezvous protocol for everything (by zeroing the
        # eager threshold) avoids that path entirely. NCCL_PROTO=^LL128 is a
        # separate, unrelated correctness fix both sites also call out: LL128 has
        # a known data-corruption bug on Slingshot GH200 nodes.
        os.environ.setdefault("FI_CXI_RDZV_THRESHOLD", "0")
        os.environ.setdefault("FI_CXI_RDZV_EAGER_SIZE", "0")
        os.environ.setdefault("FI_CXI_RDZV_GET_MIN", "0")
        os.environ.setdefault("FI_CXI_DISABLE_HOST_REGISTER", "1")
        # NOT "userfaultfd": the DataLoader (src/poincar3/run.py setup_sequence_data)
        # uses num_workers>0 with the default fork start method and
        # persistent_workers=False, so it forks a fresh batch of worker processes
        # every epoch. userfaultfd-based monitoring services page-fault
        # notifications on a background thread in this process; fork() only
        # clones the calling thread, so a forked worker that touches a COW page
        # from a region the parent registered can raise a userfault that nothing
        # is listening for -- the worker hangs forever, which surfaces as
        # "DataLoader timed out after 120 seconds" on some rank. Hit this after
        # ~2000 steps in practice. memhooks intercepts malloc/mmap/munmap
        # per-process instead of an async fd handoff, so it doesn't have this
        # cross-fork gap.
        os.environ.setdefault("FI_MR_CACHE_MONITOR", "memhooks")
        os.environ.setdefault("FI_CXI_DEFAULT_CQ_SIZE", "131072")
        os.environ.setdefault("NCCL_NET_GDR_LEVEL", "PHB")
        os.environ.setdefault("NCCL_CROSS_NIC", "1")
        os.environ.setdefault("NCCL_PROTO", "^LL128")
        # Log unconditionally (every rank) rather than gating on RANK==0 -- a
        # single one-off line is too easy to lose in a screen/byobu-captured
        # slurm log (cursor-repositioning escape codes can overwrite it), and we
        # can't be 100% sure RANK is populated this early on this launcher.
        logger.info(
            f"[nccl net plugin] NCCL_NET={os.environ['NCCL_NET']} "
            f"NCCL_NET_PLUGIN={os.environ['NCCL_NET_PLUGIN']} "
            f"FI_PROVIDER={os.environ['FI_PROVIDER']}"
        )
        logger.info(
            f"[nccl cxi tuning] FI_CXI_RDZV_THRESHOLD={os.environ['FI_CXI_RDZV_THRESHOLD']} "
            f"FI_CXI_RDZV_EAGER_SIZE={os.environ['FI_CXI_RDZV_EAGER_SIZE']} "
            f"FI_CXI_RDZV_GET_MIN={os.environ['FI_CXI_RDZV_GET_MIN']} "
            f"FI_CXI_DISABLE_HOST_REGISTER={os.environ['FI_CXI_DISABLE_HOST_REGISTER']} "
            f"FI_MR_CACHE_MONITOR={os.environ['FI_MR_CACHE_MONITOR']} "
            f"FI_CXI_DEFAULT_CQ_SIZE={os.environ['FI_CXI_DEFAULT_CQ_SIZE']} "
            f"NCCL_NET_GDR_LEVEL={os.environ['NCCL_NET_GDR_LEVEL']} "
            f"NCCL_CROSS_NIC={os.environ['NCCL_CROSS_NIC']} "
            f"NCCL_PROTO={os.environ['NCCL_PROTO']}"
        )
    else:
        logger.info("[nccl net plugin] single-node job (SLURM_NNODES=1) -- skipping libfabric/CXI overrides")
