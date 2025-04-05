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
):
    # trace
    trace_train_loss = []
    trace_validation_loss = []
    trace_tokens_seen = []

    # insight variable
    tokens_seen = 0
    step = -1

    checkpoint_queue = deque(maxlen=5)

    for epoch_num in range(target_epochs):
        model.train()

        for input_batch, target_batch in train_dataloader:
            if os.path.exists("stop.txt"):
                print("Stop signal detected. Saving checkpoint and stopping training.")
                torch.save(model.state_dict(), f'ckpts/ep-{epoch_num}_step-{step}.pth')
                return trace_train_loss, trace_validation_loss, trace_tokens_seen

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            step += 1

            # Backup checkpoint every 200 steps
            if step % 200 == 0:
                checkpoint_path = f'ckpts/backup_ep-{epoch_num}_step-{step}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                checkpoint_queue.append(checkpoint_path)

                # Remove oldest checkpoint if max backups exceeded
                if len(checkpoint_queue) > checkpoint_queue.maxlen:
                    oldest_checkpoint = checkpoint_queue.popleft()
                    if os.path.exists(oldest_checkpoint):
                        os.remove(oldest_checkpoint)

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
        torch.save(model.state_dict(), f'ckpts/ep-{epoch_num}.pth')

    return trace_train_loss, trace_validation_loss, trace_tokens_seen


def _main(model_config, train_config):
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

    print('creating dataloader')

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

    print('created dataloader')

    # tokenizer for test output
    tokenizer = create_tokenizer()

    print('start training')

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
    )

    print('finished training')

    return trace_train_loss, trace_validation_loss, trace_tokens_seen, model


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=epochs_seen, y=train_losses, name="Training Loss (Epochs)"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs_seen,
            y=val_losses,
            name="Validation Loss (Epochs)",
            line=dict(dash="dot"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=tokens_seen, y=train_losses, name="Training Loss (Tokens)", opacity=0
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Epochs", showgrid=True)
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_xaxes(
        title_text="Tokens Seen", secondary_y=True, overlaying="x", side="top"
    )

    fig.update_layout(
        title="Training and Validation Loss",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.write_html("training_validation_loss.html")


if __name__ == "__main__":
    print(f"PyTorch version is {version('torch')}")

    # training
    trace_train_loss, trace_validation_loss, trace_tokens_seen, model = _main(
        model_config=MODEL_CONFIG, train_config=TRAIN_CONFIG
    )

    # plot
    epochs_seen = list(range(len(trace_train_loss)))
    plot_losses(
        epochs_seen=epochs_seen,
        tokens_seen=trace_tokens_seen,
        train_losses=trace_train_loss,
        val_losses=trace_validation_loss,
    )

    # save model
    torch.save(model.state_dict(), 'model.pth')
