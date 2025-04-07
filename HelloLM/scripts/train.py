import os
import gc
import json
import math
import torch
import plotly.graph_objects as go
from time import time
from tqdm import tqdm
from collections import deque
from argparse import ArgumentParser
from plotly.subplots import make_subplots
from torch.multiprocessing import spawn
from torch.distributed import (
    init_process_group,
    destroy_process_group,
    barrier,
    all_reduce,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from HelloLM.config import MODEL_CONFIG, TRAIN_CONFIG
from HelloLM.config import ModelConfig, TrainConfig
from HelloLM.utils.logger import logger, setup_logger
from HelloLM.utils.tools import log_env_metadata, to_abs_path, ensure_directory
from HelloLM.model.model import HelloModel
from HelloLM.data.loader import create_dataloader


# configure train worker GPU device
def setup_distributed(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "17676"
    init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


# configure train subprocess (reproduction, optimization)
def setup_environment(seed):
    # set seeds for all library
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # envs
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["NCCL_DEBUG"] = "INFO"

    # optimization settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: torch.nn.Module,
    device,
    use_amp=True,
    scalar=None,
):
    # make sure the batch is on the same device with model
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    if torch.cuda.is_available() and use_amp and scalar is not None:
        with autocast("cuda", dtype=torch.float16):
            logits: torch.Tensor = model(input_batch)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), target_batch.flatten()
            )
    else:
        logits: torch.Tensor = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
        )

    return loss


def calc_loss_loader(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device,
    num_batches=None,
    use_amp=True,
    scalar=None,
):
    total_loss = 0.0

    # resolve value
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for index, (input_batch, target_batch) in enumerate(dataloader):
        if index < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device, use_amp, scalar
            )
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    device,
    evaluation_iter: int,
    use_amp=True,
    scalar=None,
    distributed=False,
):
    # turn model into evaluation mode
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_dataloader,
            model,
            device,
            num_batches=evaluation_iter,
            use_amp=use_amp,
            scalar=scalar,
        )
        validation_loss = calc_loss_loader(
            validation_dataloader,
            model,
            device,
            num_batches=evaluation_iter,
            use_amp=use_amp,
            scalar=scalar,
        )

    # change back to train model
    model.train()

    # synchronize losses across process
    if distributed:
        train_loss_tensor = torch.tensor([train_loss], device=device)
        validation_loss_tensor = torch.tensor([validation_loss], device=device)

        all_reduce(train_loss_tensor)
        all_reduce(validation_loss_tensor)

        # get current world size
        world_size = torch.distributed.get_world_size()
        train_loss = train_loss_tensor.item() / world_size
        validation_loss = validation_loss_tensor.item() / world_size

    return train_loss, validation_loss


def plot_losses(steps_seen, tokens_seen, train_losses, val_losses, name):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=steps_seen, y=train_losses, name="Training Loss (Epochs)"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=steps_seen,
            y=val_losses,
            name="Validation Loss (Epochs)",
            line=dict(dash="dot"),
        ),
        secondary_y=False,
    )

    # Update axes
    fig.update_xaxes(title_text="Steps", showgrid=True)
    fig.update_yaxes(title_text="Loss", secondary_y=False)

    fig.update_layout(
        title="Training and Validation Loss",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=[
            dict(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text=f"Total Tokens Seen: {max(tokens_seen):,}",
                showarrow=False,
            )
        ],
    )

    ensure_directory("plots")
    fig.write_html(f"plots/{name}.html")


@logger.catch
def save_model_backup(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    epoch: int,
    train_losses,
    validation_losses,
    tokens_seen,
    total_tokens,
    step,
    update_count,
    scaler,
    lr_scheduler,
    use_amp,
    ckpt_path,
    ckpt_queue: deque,
    is_end_of_epoch,
):
    ckpt_filepath = os.path.join(
        to_abs_path(ckpt_path),
        f"backup-{'ep' if is_end_of_epoch else 'step'}-{'epoch' if is_end_of_epoch else step}.pth",
    )
    tmp_path = f"{ckpt_filepath}.tmp"

    try:
        model_state = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "tokens_seen": tokens_seen,
            "total_tokens": total_tokens,
            "update_count": update_count,
        }

        if use_amp and scaler is not None:
            checkpoint["scalar_state_dict"] = scaler.state_dict()

        if lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = lr_scheduler.state_dict()

        torch.save(checkpoint, tmp_path)

        if os.path.exists(tmp_path):
            if os.path.exists(ckpt_filepath):
                os.remove(ckpt_filepath)
            os.rename(tmp_path, ckpt_filepath)
            logger.info(f"Saved checkpoint: {ckpt_filepath}")
            ckpt_queue.append(ckpt_filepath)

        for f in os.listdir(ckpt_path):
            file_path = os.path.join(ckpt_path, f)
            if (
                f.startswith("backup-")
                and f.endswith(".pth")
                and not f.endswith(".tmp")
                and file_path not in ckpt_queue
            ):
                os.remove(file_path)
                logger.info(f"Deleted checkpoint not in queue: {file_path}")

    except Exception as e:
        logger.error("Error saving checkpoint backup")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Error saving checkpoint: {e}")


# main training function
def _train(
    train_config: TrainConfig,
    # model and dataset
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimizer: torch.optim.AdamW,
    # train metadata
    device,
    start_epoch=0,
    start_step=0,
    backup_path="ckpts",
    # misc
    memory_efficient=True,
    checkpoint=None,
    rank=0,
    lr_scheduler=None,
):
    # trace data
    trace_train_loss = checkpoint.get("train_losses", []) if checkpoint else []
    trace_validation_loss = (
        checkpoint.get("validation_losses", []) if checkpoint else []
    )
    trace_tokens_seen = checkpoint.get("tokens_seen", []) if checkpoint else []

    # insight
    tokens_seen = checkpoint.get("total_tokens", 0) if checkpoint else 0
    step = start_step
    update_count = 0

    # mixed precision scaler
    scaler = (
        GradScaler("cuda")
        if (torch.cuda.is_available() and train_config["use_mixed_precision"])
        else None
    )
    if (
        checkpoint
        and "scaler_state_dict" in checkpoint
        and (torch.cuda.is_available() and train_config["use_mixed_precision"])
    ):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # initialize checkpoint queue
    ckpt_queue = deque(maxlen=train_config["max_backup_nums"])

    # add existing checkpoints to the queue if resuming
    backup_path = to_abs_path(backup_path)
    if os.path.exists(backup_path):
        if rank == 0 or train_config["distributed"]:
            ckpt_files = []
            for f in os.listdir(backup_path):
                if (
                    f.startswith("backup-")
                    and f.endswith(".pth")
                    and not f.endswith(".tmp")
                ):
                    ckpt_files.append(os.path.join(backup_path, f))

            # sort by creation time
            ckpt_files.sort(key=lambda v: os.path.getctime(v))
            ckpt_queue.extend(ckpt_files)

            # delete files not in the queue
            for f in os.listdir(backup_path):
                file_path = os.path.join(backup_path, f)
                if (
                    f.startswith("backup-")
                    and f.endswith(".pth")
                    and not f.endswith(".tmp")
                    and file_path not in ckpt_queue
                ):
                    os.remove(file_path)
                    logger.info(f"Deleted checkpoint not in queue: {file_path}")

    # wait for all process setup
    if train_config["distributed"]:
        barrier()

    # training statistics
    running_loss = 0.0
    batch_times = deque(maxlen=100)
    total_batch_time = 0.0

    epoch_progress = None
    epoch_iter = range(start_epoch, train_config["target_epochs"])
    if rank == 0 or not train_config["distributed"]:
        epoch_progress = tqdm(epoch_iter, desc="Training Epochs")

    for epoch in (
        epoch_iter if rank == 0 or not train_config["distributed"] else epoch_progress
    ):
        # setup batch progress bar
        batch_progress = None
        if rank == 0 or train_config["distributed"]:
            batch_progress = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")

        # reset sampler
        if train_config["distributed"] and hasattr(
            train_dataloader.sampler, "set_epoch"
        ):
            train_dataloader.sampler.set_epoch(epoch)

        # turn model into train mode
        model.train()
        epoch_start_time = time()

        for index, (input_batch, target_batch) in enumerate(train_dataloader):
            input_batch: torch.Tensor
            target_batch: torch.Tensor
            batch_start_time = time()
            batch_tokens = input_batch.shape[0] * input_batch.shape[1]
            tokens_seen += batch_tokens

            if (
                torch.cuda.is_available()
                and train_config["use_mixed_precision"]
                and scaler is not None
            ):
                with autocast("cuda", dtype=torch.float16):
                    loss = calc_loss_batch(
                        input_batch,
                        target_batch,
                        model,
                        device,
                        use_amp=train_config["use_mixed_precision"],
                        scalar=scaler,
                    )
                    # normalize loss for gradient accumulation
                    loss = loss / train_config["gradient_accumulation_steps"]

                # scales loss and backward pass
                scaler.scale(loss).backward()

                # gradient accumulation
                if (index + 1) % train_config["gradient_accumulation_steps"] == 0 or (
                    index + 1
                ) == len(train_dataloader):
                    # clip gradients to avoid explosion
                    torch.nn.utils.clip_grads_with_norm_(
                        model.parameters(), max_norm=1.0
                    )

                    # unscale before optimizer
                    scaler.unscale_(optimizer)

                    # update weights
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    # step LR scheduler
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    update_count += 1
            else:
                # normal training process
                loss = calc_loss_batch(
                    input_batch, target_batch, model, device, use_amp=False
                )
                loss = loss / train_config["gradient_accumulation_steps"]

                # backward called
                loss.backward()

                if (index + 1) % train_config["gradient_accumulation_steps"] == 0 or (
                    index + 1
                ) == len(train_dataloader):
                    torch.nn.utils.clip_grads_with_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    update_count += 1

            step += 1
            running_loss += loss.item() * train_config["gradient_accumulation_steps"]

            # timing calculation
            batch_time = time() - batch_start_time
            batch_times.append(batch_time)
            total_batch_time += batch_time

            # update progress bar
            if batch_progress is not None:
                lr_value = optimizer.param_groups[0]["lr"]
                batch_progress.set_postfix(
                    {
                        "loss": running_loss / (index + 1),
                        "tokens/sec": batch_tokens / batch_time,
                        "learning rate": lr_value,
                    }
                )
                batch_progress.update(1)

            # evaluation
            if step % train_config["evaluation_step"] == 0 and (
                update_count % train_config["gradient_accumulation_steps"] == 0
            ):
                train_loss, validation_loss = evaluate_model(
                    model,
                    train_dataloader,
                    validation_dataloader,
                    device,
                    evaluation_iter=train_config["evaluation_iter"],
                    use_amp=train_config["use_mixed_precision"],
                    scalar=scaler,
                    distributed=train_config["distributed"],
                )

                trace_train_loss.append(train_loss)
                trace_validation_loss.append(validation_loss)
                trace_tokens_seen.append(tokens_seen)

                if rank == 0 or not train_config["distributed"]:
                    logger.info(
                        f"Step {step}, Epoch {epoch}, Updates {update_count}, tokens/sec={sum(batch_times) / len(batch_times)}"
                    )
                    logger.info(
                        f"Train Loss {train_loss: 6f}, Val Loss {validation_loss: 6f}"
                    )

                if memory_efficient:
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if rank == 0 or not train_config["distributed"]:
                        if torch.cuda.is_available():
                            allocated_mem = torch.cuda.memory_allocated(device) / (
                                1024**3
                            )
                            reserved_mem = torch.cuda.memory_reserved(device) / (
                                1024**3
                            )
                            logger.info(
                                f"CUDA memory: {allocated_mem:.3f}GB allocated, {reserved_mem:.3f}GBN reserved"
                            )

            if step % train_config["backup_steps"] == 0 and (
                rank == 0 or train_config["distributed"]
            ):
                logger.info("Saving checkpoint backup...")

                save_model_backup(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_losses=trace_train_loss,
                    validation_losses=trace_validation_loss,
                    tokens_seen=trace_tokens_seen,
                    total_tokens=tokens_seen,
                    step=step,
                    update_count=update_count,
                    scaler=scaler,
                    lr_scheduler=lr_scheduler,
                    use_amp=train_config["use_mixed_precision"],
                    ckpt_path=train_config["backup_path"],
                    ckpt_queue=ckpt_queue,
                    is_end_of_epoch=False,
                )

                plot_losses(
                    steps_seen=list(range(len(trace_train_loss))),
                    tokens_seen=trace_tokens_seen,
                    train_losses=trace_train_loss,
                    val_losses=trace_validation_loss,
                    name=f"ep{epoch}-step{step}",
                )

        if rank == 0 or train_config["distributed"]:
            if batch_progress is not None:
                batch_progress.close()

            epoch_time = time() - epoch_start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")

            logger.info("Saving epoch checkpoint backup...")

            save_model_backup(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_losses=trace_train_loss,
                validation_losses=trace_validation_loss,
                tokens_seen=trace_tokens_seen,
                total_tokens=tokens_seen,
                step=step,
                update_count=update_count,
                scaler=scaler,
                lr_scheduler=lr_scheduler,
                use_amp=train_config["use_mixed_precision"],
                ckpt_path=train_config["backup_path"],
                ckpt_queue=ckpt_queue,
                is_end_of_epoch=True,
            )

            plot_losses(
                steps_seen=list(range(len(trace_train_loss))),
                tokens_seen=trace_tokens_seen,
                train_losses=trace_train_loss,
                val_losses=trace_validation_loss,
                name=f"ep{epoch}-step{step}",
            )

        if train_config["distributed"]:
            barrier()

    return trace_train_loss, trace_validation_loss, trace_tokens_seen


@logger.catch
def train_unit(
    rank: int, model_config: ModelConfig, train_config: TrainConfig, ckpt_path: str
):
    # setup environment to process
    setup_environment(seed=train_config["seed"])

    if train_config["distributed"]:
        device = torch.device(f"cuda:{rank}")
        logger.info(
            f"Process {rank}/{train_config['world_size']} using device: {device}"
        )
    else:
        logger.info("Setting up single process training...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("CUDA is ready to use")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("macOS MPS is ready to use")
        elif torch.cpu.is_available():
            device = torch.device("cpu")
            logger.info("CPU train only")
        else:
            logger.error("At least one device should be available for train")
            raise RuntimeError("No device available for PyTorch")

    # create model
    model = HelloModel(model_config=model_config)
    model.to(device)

    # wrap model inside DDP in distributed training
    if train_config["distributed"]:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        betas=(0.9, 0.95),  # common practice
    )

    # load dataset
    logger.info("Loading datasets...")
    train_dataloader = create_dataloader(
        payload=train_config["train_datasets_path"],
        column_name=train_config["train_datasets_column_name"],
        model_config=model_config,
        train_config=train_config,
        shuffle=True,
        drop_last=True,
        use_cache=True if train_config["cache_path"] else False,
        cache_path=train_config["cache_path"],
        rank=-1 if not train_config["distributed"] else rank,
    )
    validation_dataloader = create_dataloader(
        payload=train_config["validation_datasets_path"],
        column_name=train_config["validation_datasets_column_name"],
        model_config=model_config,
        train_config=train_config,
        shuffle=True,
        drop_last=True,
        use_cache=True if train_config["cache_path"] else False,
        cache_path=train_config["cache_path"],
        rank=-1 if not train_config["distributed"] else rank,
    )

    # calculate total steps roughly in current device
    total_steps = math.ceil(
        len(train_dataloader)
        / train_config["batch_size_per_device"]
        / (1 if not train_config["distributed"] else train_config["world_size"])
    )
    warmup_steps = train_config["warmup_steps"]

    # log insight for different training mode
    if rank == 0 or not train_config["distributed"]:
        logger.info("Dataloaders created successfully")
        if train_config["distributed"]:
            logger.info(f"Loaded totally {len(train_dataloader)} training samples")
            logger.info(
                f"Loaded {len(train_dataloader) / train_config['world_size']} training samples per device"
            )
            logger.info(
                f"Loaded totally {len(validation_dataloader)} validation samples"
            )
            logger.info(
                f"Loaded {len(validation_dataloader) / train_config['world_size']} validation samples per device"
            )
        else:
            logger.info(f"Loaded {len(train_dataloader)} training samples")
            logger.info(f"Loaded {len(validation_dataloader)} validation samples")

    # create lr scheduler
    lr_scheduler = None
    if isinstance(total_steps, int):
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # cosine decay phase
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    # initialize start steps
    start_epoch = 0
    start_step = 0

    # checkpoint data
    checkpoint = None

    # load checkpoint file
    resolved_ckpt_path = to_abs_path(ckpt_path) if ckpt_path else None
    if ckpt_path and os.path.exists(resolved_ckpt_path):
        logger.info(f"Loading checkpoint file: {ckpt_path}")
        try:
            checkpoint = torch.load(resolved_ckpt_path, map_location=device)

            # check if the checkpoint is a complete checkpoint or just weights
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # handle DDP model
                if train_config["distributed"]:
                    # handle non-DDP checkpoint
                    if not any(
                        k.startswith("module.")
                        for k in checkpoint["model_state_dict"].keys()
                    ):
                        # convert non-DDP checkpoint to DDP
                        model.module.load_state_dict(
                            {k: v for k, v in checkpoint["model_state_dict"].items()}
                        )
                    else:
                        # load DDP checkpoint into model directly
                        model.module.load_state_dict(checkpoint["model_state_dict"])
                else:
                    # handle DDP checkpoint
                    if any(
                        k.startswith("module.")
                        for k in checkpoint["model_state_dict"].keys()
                    ):
                        # convert DDP checkpoint into non-DDP
                        model.load_state_dict(
                            {
                                k.replace("module.", ""): v
                                for k, v in checkpoint["model_state_dict"].items()
                            }
                        )
                    else:
                        # load non-DDP checkpoint into model directly
                        model.load_state_dict(checkpoint["model_state_dict"])

                # cleanup cache before loading optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(
                        f"[GPU{rank}] memory before loading optimizer: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
                    )

                # load optimizer state with CPU tensor
                raw_optimizer_state = checkpoint["optimizer_state_dict"]

                # move to CPU to avoid GPU OOM
                optimizer_state = {}
                for k, v in raw_optimizer_state.items():
                    if isinstance(v, dict):
                        # create a empty dict first
                        optimizer_state[k] = {}
                        # move one by one
                        for state_k, state_v in v.items():
                            # if the state value is a Tensor, then move it
                            if isinstance(state_v, torch.Tensor):
                                optimizer_state[k][state_k] = state_v.cpu()
                            else:
                                # copy other data directly
                                optimizer_state[k][state_k] = state_v

                # load optimizer state
                optimizer.load_state_dict(optimizer_state)

                # load lr_scheduler
                if lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                # get back training state
                start_epoch = checkpoint.get("epoch", 1)
                start_step = checkpoint.get("step", 0) + 1

                # if rank is 0 (main process) or non-distributed
                if rank == 0 or not train_config["distributed"]:
                    logger.info(
                        f"Loaded checkpoint from Epoch {start_epoch}, Step {start_step}"
                    )
            else:
                # load weights only
                model.load_state_dict(checkpoint)
                # if rank is 0 (main process) or non-distributed
                if rank == 0 or not train_config["distributed"]:
                    logger.info(
                        "Loaded model weights checkpoint only (no training / optimizer state)"
                    )
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {ckpt_path}")
            raise RuntimeError(f"Checkpoint loading failed, {e}")

    # wait for all process to finish loading checkpoint (if there has one)
    if train_config["distributed"]:
        barrier()

    # set up gradient accumulation steps
    gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
    if gradient_accumulation_steps > 1 and (
        rank == 0 or not train_config["distributed"]
    ):
        logger.info(f"Using gradient accumulation with {gradient_accumulation_steps}")
        effective_batch_size = (
            train_config["batch_size_per_device"] * gradient_accumulation_steps
        )
        if train_config["distributed"]:
            effective_batch_size *= train_config["world_size"]
        logger.info(f"Effective batch size is {effective_batch_size}")

    # ensure checkpoint directory
    if train_config["backup_path"]:
        ensure_directory(train_config["backup_path"])

    # start training
    if rank == 0 or not train_config["distributed"]:
        logger.info("Starting training...")

    trace_train_loss, trace_validation_loss, trace_tokens_seen = _train(
        train_config=train_config,
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        device=device,
        start_epoch=start_epoch,
        start_step=start_step,
        backup_path=train_config["backup_path"],
        memory_efficient=True,
        checkpoint=checkpoint,
        rank=rank,
        lr_scheduler=lr_scheduler,
    )

    # cleanup
    if train_config["distributed"]:
        destroy_process_group()

    if rank == 0 or not train_config["distributed"]:
        logger.info("Training completed successfully")

    return trace_train_loss, trace_validation_loss, trace_tokens_seen, model


# worker to handle training result (model saving and plot)
def train_worker(
    rank: int, model_config: ModelConfig, train_config: TrainConfig, ckpt_path: str
):
    # setup distributed environment
    setup_distributed(rank=rank, world_size=train_config["world_size"])

    # call train unit
    trace_train_loss, trace_validation_loss, trace_tokens_seen, model = train_unit(
        model_config=model_config,
        train_config=train_config,
        rank=rank,
        ckpt_path=ckpt_path
    )

    if rank == 0 or not train_config["distributed"]:
        plot_losses(
            steps_seen=list(range(len(trace_train_loss))),
            tokens_seen=trace_tokens_seen,
            train_losses=trace_train_loss,
            val_losses=trace_validation_loss,
            name="final",
        )

        try:
            temp_path = "model.pth.tmp"
            model_state = (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            )
            checkpoint = {
                "model_state_dict": model_state,
                "train_losses": trace_train_loss,
                "validation_losses": trace_validation_loss,
                "tokens_seen": trace_tokens_seen,
            }
            torch.save(checkpoint, temp_path)
            if os.path.exists(temp_path):
                if os.path.exists("model.pth"):
                    os.remove("model.pth")
                os.rename(temp_path, "model.pth")
                logger.info("Saved final model to model.pth")
        except Exception as e:
            logger.error("Error saving final model")
            raise RuntimeError(f"Error saving model: {e}")

    # cleanup
    if train_config["distributed"]:
        destroy_process_group()

    return


# _main function is used to start one (or more) train worker(s)
@logger.catch
def _main(
    model_config: ModelConfig,
    train_config: TrainConfig,
    ckpt_path,
):
    # logging insight
    logger.info(f"Model configuration:\n{json.dumps(MODEL_CONFIG, indent=2)}")
    logger.info(f"Training configuration:\n{json.dumps(TRAIN_CONFIG, indent=2)}")
    if train_config["distributed"]:
        logger.info(
            f"Distributed training is enabled with {train_config['world_size']} processes"
        )
    else:
        logger.info("The training will perform on single device")

    if ckpt_path:
        if os.path.exists(to_abs_path(ckpt_path)):
            logger.info("Checkpoint file specified")
            logger.info(f"The training process will resume from: {ckpt_path}")
        else:
            logger.warning(
                f"The checkpoint file {ckpt_path} specified could not be found"
            )
            logger.warning("The training process will start in fresh")
            logger.info("Stopped training")
            return

    # distributed training is only work when enough cuda device available currently
    if (
        train_config["distributed"]
        and torch.cuda.device_count() < train_config["world_size"]
    ):
        logger.error("No enough CUDA device(s) for distributed training")
        logger.error(
            f"Need {train_config['world_size']} CUDA device(s), only found {torch.cuda.device_count()}"
        )

    # start one or more train worker(s)
    if train_config["distributed"] and train_config["world_size"] > 1:
        spawn(
            train_worker,
            args=(model_config, train_config, ckpt_path),
            nprocs=train_config["world_size"],
            join=True,
        )
    else:
        train_worker(0, model_config, train_config, ckpt_path)


# (module) is used to parse command line argument
if __name__ == "__main__":
    # resolve arguments
    parser = ArgumentParser(description="Just a simple CLI to train HelloLM (demo)")

    # log control
    logging_options = parser.add_argument_group("logging options")
    logging_options.add_argument(
        "--disable-log-to-file",
        help="whether to log to file",
        action="store_true",
        default=False,
    )
    logging_options.add_argument(
        "--log-file-path", help="target path to save logging files", default="logs"
    )
    logging_options.add_argument(
        "--log-file-split",
        help="whether to split warning/error to another log file",
        action="store_true",
        default=False,
    )

    # checkpoint and backup control
    ckpt_backup_options = parser.add_argument_group("checkpoint and backup options")
    ckpt_backup_options.add_argument(
        "-c",
        "--checkpoint",
        help="path to checkpoint file to resume training from",
        type=str,
        default=None,
    )
    ckpt_backup_options.add_argument(
        "--disable-backup",
        help="whether to disable auto backup for training",
        action="store_true",
        default=False,
    )
    ckpt_backup_options.add_argument(
        "--backup-steps",
        help="specify how many steps to perform an automatic backup",
        type=int,
        default=400,
    )
    ckpt_backup_options.add_argument(
        "--max-backup-num",
        help="specify the maximum number of simultaneous automatic backups",
    )

    # distributed training control
    distributed_control = parser.add_argument_group("distributed options")
    distributed_control.add_argument(
        "-d",
        "--distributed",
        help="enable distributed training on multiple GPUs",
        action="store_true",
    )
    distributed_control.add_argument(
        "-w", "--world-size", help="number of GPUs to be used in training", type=int
    )

    # parse
    args = parser.parse_args()

    # initialize logger
    setup_logger(
        disable_log_to_file=args.disable_log_to_file,
        log_file_path=args.log_file_path,
        log_file_split=args.log_file_split,
    )

    # log system, package version (and driver) info
    log_env_metadata()

    # checkpoint resume
    ckpt_path = args.checkpoint

    # resolve distributed config state (default to False)
    TRAIN_CONFIG["distributed"] = args.distributed or TRAIN_CONFIG.get(
        "distributed", False
    )
    # world_size will be 2 if distributed==True, otherwise it will be 1
    TRAIN_CONFIG["world_size"] = (
        args.world_size
        if args.world_size is not None
        else (2 if TRAIN_CONFIG["distributed"] else 1)
    )

    _main(
        model_config=MODEL_CONFIG,
        train_config=TRAIN_CONFIG,
        ckpt_path=ckpt_path,
    )
