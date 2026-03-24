"""torch.distributed — distributed stubs."""

import torch.distributed.distributed_c10d as distributed_c10d  # noqa: F401


def is_nccl_available():
    return False


def is_mpi_available():
    return False


def is_gloo_available():
    return False


def is_initialized():
    return False
