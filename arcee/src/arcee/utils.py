from __future__ import annotations


def prepare_wandb_name(wandb_job_name: str) -> str:
    """
    Ensures that a wandb job name for a run is valid.

    Args:
        wandb_job_name: the proposed wandb name
    """
    wandb_job_name = wandb_job_name.replace("/", "-")
    if len(wandb_job_name) > 64:
        wandb_job_name = wandb_job_name[0:64]
    return wandb_job_name
