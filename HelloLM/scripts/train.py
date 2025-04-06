import os
import torch
import plotly.graph_objects as go
from collections import deque
from torch.utils.data import DataLoader
from importlib.metadata import version
from HelloLM.data.loader import create_dataloader
from HelloLM.data.tokenizer import create_tokenizer
from HelloLM.model.model import HelloModel, simple_generate
from HelloLM.config import MODEL_CONFIG, TRAIN_CONFIG
from plotly.subplots import make_subplots


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(
    input_batch: torch.Tensor, target_batch: torch.Tensor, model, device
):
    # make sure to move batch to same device with model
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits: torch.Tensor = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(dataloader: DataLoader, model, device, num_batches=None):
    total_loss = 0.0
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model: torch.nn.Module,
    train_dataloader,
    validation_dataloader,
    device,
    evaluation_iter,
):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_dataloader, model, device, num_batches=evaluation_iter
        )
        val_loss = calc_loss_loader(
            validation_dataloader, model, device, num_batches=evaluation_iter
        )
    # change back to train mode
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, test_output_context):
    model.eval()
    context_size = model.positional_embedding.weight.shape[0]
    encoded = text_to_token_ids(test_output_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = simple_generate(
            model=model, index=encoded, max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimizer: torch.optim.AdamW,
    device,
    target_epochs,
    evaluation_step,
    evaluation_iter,
    tokenizer,
    test_output_context,
    start_epoch=0,
    start_step=0,
    checkpoint_data=None,
):
    # trace - restore from checkpoint if available
    trace_train_loss = checkpoint_data.get('train_losses', []) if checkpoint_data else []
    trace_validation_loss = checkpoint_data.get('val_losses', []) if checkpoint_data else []
    trace_tokens_seen = checkpoint_data.get('tokens_seen', []) if checkpoint_data else []

    # insight variable - restore from checkpoint or initialize
    tokens_seen = checkpoint_data.get('total_tokens', 0) if checkpoint_data else 0
    step = start_step

    checkpoint_queue = deque(maxlen=5)
    
    # Add existing checkpoints to the queue if resuming
    if os.path.exists("ckpts"):
        # Find existing checkpoints and add them to the queue
        checkpoint_files = []
        for f in os.listdir("ckpts"):
            if f.startswith("backup_ep-") and f.endswith(".pth") and not f.endswith(".tmp"):
                checkpoint_files.append(os.path.join("ckpts", f))
        
        # Sort by creation time (newest first) to keep the most recent backups
        checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        
        # Add the most recent ones to the queue
        for f in checkpoint_files[:checkpoint_queue.maxlen]:
            checkpoint_queue.append(f)
            
        # Remove any old backups not in the queue
        for f in checkpoint_files[checkpoint_queue.maxlen:]:
            print(f"Removing old checkpoint: {f}")
            try:
                os.remove(f)
            except Exception as e:
                print(f"Failed to remove old checkpoint {f}: {e}")
    
    # state
    print(f'The checkpoint queue has {len(checkpoint_queue)} checkpoints initialized')

    # create ckpts dir
    os.makedirs("ckpts", exist_ok=True)

    for epoch_num in range(start_epoch, start_epoch + target_epochs):
        model.train()

        for input_batch, target_batch in train_dataloader:
            if os.path.exists("stop.txt"):
                print("Stop signal detected. Saving checkpoint and stopping training.")
                try:
                    checkpoint_path = f"ckpts/ep-{epoch_num}_step-{step}.pth"
                    temp_path = f"{checkpoint_path}.tmp"
                    # Save complete checkpoint with training state
                    checkpoint = {
                        'epoch': epoch_num,
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_losses': trace_train_loss,
                        'val_losses': trace_validation_loss,
                        'tokens_seen': trace_tokens_seen,
                        'total_tokens': tokens_seen
                    }
                    torch.save(checkpoint, temp_path)
                    if os.path.exists(temp_path):
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        os.rename(temp_path, checkpoint_path)
                        print(f"Saved stop checkpoint to {checkpoint_path}")
                except Exception as e:
                    print(f"Error saving stop checkpoint: {e}")
                os.remove("stop.txt")
                return trace_train_loss, trace_validation_loss, trace_tokens_seen

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            step += 1

            # Backup checkpoint every 200 steps
            if step % 200 == 0:
                checkpoint_path = f"ckpts/backup_ep-{epoch_num}_step-{step}.pth"
                try:
                    temp_path = f"{checkpoint_path}.tmp"
                    checkpoint = {
                        'epoch': epoch_num,
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_losses': trace_train_loss,
                        'val_losses': trace_validation_loss,
                        'tokens_seen': trace_tokens_seen,
                        'total_tokens': tokens_seen
                    }
                    torch.save(checkpoint, temp_path)
                    if os.path.exists(temp_path):
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        os.rename(temp_path, checkpoint_path)
                        print(f"Saved checkpoint to {checkpoint_path}")
                        checkpoint_queue.append(checkpoint_path)
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
                    if os.path.exists(temp_path):
                        print(f"Temporary checkpoint file remains at {temp_path}")

                if len(checkpoint_queue) > checkpoint_queue.maxlen:
                    oldest_checkpoint = checkpoint_queue.popleft()
                    print(f'Checkpoint queue is full, removing {oldest_checkpoint}')
                    if os.path.exists(oldest_checkpoint):
                        print(f"Removing oldest checkpoint: {oldest_checkpoint}")
                        try:
                            os.remove(oldest_checkpoint)
                        except Exception as e:
                            print(f"Failed to remove checkpoint {oldest_checkpoint}: {e}")

            if step % evaluation_step == 0:
                train_loss, validation_loss = evaluate_model(
                    model,
                    train_dataloader,
                    validation_dataloader,
                    device,
                    evaluation_iter,
                )
                trace_train_loss.append(train_loss)
                trace_validation_loss.append(validation_loss)
                trace_tokens_seen.append(tokens_seen)
                # trace information
                print(
                    f"Epoch {epoch_num + 1} [Step {step:08d}]: "
                    f"Train loss {train_loss:.5f}, Validation loss {validation_loss:.5f}"
                )

        generate_and_print_sample(model, tokenizer, device, test_output_context)
        try:
            # Save epoch checkpoint with the same robust method
            checkpoint_path = f"ckpts/ep-{epoch_num}.pth"
            temp_path = f"{checkpoint_path}.tmp"
            # Save complete checkpoint with training state
            checkpoint = {
                'epoch': epoch_num,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': trace_train_loss,
                'val_losses': trace_validation_loss,
                'tokens_seen': trace_tokens_seen,
                'total_tokens': tokens_seen
            }
            torch.save(checkpoint, temp_path)
            if os.path.exists(temp_path):
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                os.rename(temp_path, checkpoint_path)
                print(f"Saved epoch checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"Error saving epoch checkpoint: {e}")

    return trace_train_loss, trace_validation_loss, trace_tokens_seen


def _main(model_config, train_config, checkpoint_path=None):
    # check environment
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA Ready")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS Ready")
    else:
        device = torch.device("cpu")
        print("CPU Ready")

    # load module into selected device
    model = HelloModel(model_config=model_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    
    # Initialize checkpoint variables
    start_epoch = 0
    start_step = 0
    checkpoint_data = None
    
    # Load checkpoint if specified
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Check if it's a complete checkpoint or just model state
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Load model state
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Free memory before loading optimizer state
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"GPU memory before loading optimizer: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                
                # Load optimizer state with CPU tensors first
                optimizer_state = checkpoint['optimizer_state_dict']
                
                # Move optimizer state to CPU first to avoid OOM
                cpu_optimizer_state = {}
                for k, v in optimizer_state.items():
                    if isinstance(v, dict):
                        cpu_optimizer_state[k] = {}
                        for state_k, state_v in v.items():
                            if isinstance(state_v, torch.Tensor):
                                cpu_optimizer_state[k][state_k] = state_v.cpu()
                            else:
                                cpu_optimizer_state[k][state_k] = state_v
                    else:
                        cpu_optimizer_state[k] = v
                optimizer.load_state_dict(cpu_optimizer_state)
                
                # Extract training progress data
                start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
                start_step = checkpoint.get('step', 0) + 1    # Start from next step
                checkpoint_data = {
                    'train_losses': checkpoint.get('train_losses', []),
                    'val_losses': checkpoint.get('val_losses', []),
                    'tokens_seen': checkpoint.get('tokens_seen', []),
                    'total_tokens': checkpoint.get('total_tokens', 0)
                }

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"GPU memory after loading optimizer: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"Resuming from epoch {start_epoch}, step {start_step}")
            else:
                # Old format, just model state
                model.load_state_dict(checkpoint)
                print("Loaded model weights only (no training state)")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")

    print("creating dataloader")

    # load dataset
    train_dataloader = create_dataloader(
        [
            "data/wikitext/train-00000-of-00002.parquet",
            "data/wikitext/train-00001-of-00002.parquet",
        ],
        column_name="text",
        batch_size=train_config["batch_size"],
        max_length=model_config["context_length"],
        stride=model_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=2,
        use_cache=True,
        cache_path=train_config["dataset_cache_path"],
    )

    validation_dataloader = create_dataloader(
        "data/wikitext/validation-00000-of-00001.parquet",
        column_name="text",
        batch_size=train_config["batch_size"],
        max_length=model_config["context_length"],
        stride=model_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
        use_cache=True,
        cache_path=train_config["dataset_cache_path"],
    )

    print("created dataloader")

    print(
        f"About {train_dataloader.dataset.total_samples / train_dataloader.batch_size} steps a epoch"
    )

    # tokenizer for test output
    tokenizer = create_tokenizer()

    print("start training")

    trace_train_loss, trace_validation_loss, trace_tokens_seen = train(
        model,
        train_dataloader,
        validation_dataloader,
        optimizer,
        device,
        target_epochs=train_config["target_epochs"],
        evaluation_step=5,
        evaluation_iter=1,
        test_output_context="Watching the stars and",
        tokenizer=tokenizer,
        start_epoch=start_epoch,
        start_step=start_step,
        checkpoint_data=checkpoint_data,
    )

    print("finished training")

    return trace_train_loss, trace_validation_loss, trace_tokens_seen, model


def plot_losses(steps_seen, tokens_seen, train_losses, val_losses):
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

    fig.write_html("training_validation_loss.html")


if __name__ == "__main__":
    print(f"PyTorch version is {version('torch')}")
    
    # Check for checkpoint argument
    import sys
    checkpoint_path = None
    
    # Parse command line arguments for checkpoint path
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint file {checkpoint_path} not found")
            checkpoint_path = None
        else:
            print(f"Will resume training from: {checkpoint_path}")

    # training
    trace_train_loss, trace_validation_loss, trace_tokens_seen, model = _main(
        model_config=MODEL_CONFIG, train_config=TRAIN_CONFIG, checkpoint_path=checkpoint_path
    )

    # plot
    steps_seen = list(range(len(trace_train_loss))) * 5
    plot_losses(
        steps_seen=steps_seen,
        tokens_seen=trace_tokens_seen,
        train_losses=trace_train_loss,
        val_losses=trace_validation_loss,
    )

    # save model with robust method
    try:
        temp_path = "model.pth.tmp"
        # Save as complete checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'train_losses': trace_train_loss,
            'val_losses': trace_validation_loss,
            'tokens_seen': trace_tokens_seen
        }
        torch.save(checkpoint, temp_path)
        if os.path.exists(temp_path):
            if os.path.exists("model.pth"):
                os.remove("model.pth")
            os.rename(temp_path, "model.pth")
            print("Saved final model to model.pth")
    except Exception as e:
        print(f"Error saving final model: {e}")
