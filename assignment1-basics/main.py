from tests.adapters import *
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import pickle


def sample_top_p(probs: torch.Tensor, p: float) -> int:
    """Top-p (nucleus) sampling: sample from the smallest set of tokens whose cumulative probability ≥ p."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff = (cumulative_probs > p).float().argmax().item() + 1
    top_p_probs = sorted_probs[:cutoff]
    top_p_probs = top_p_probs / top_p_probs.sum()  # normalize

    sampled_index = torch.multinomial(top_p_probs, 1).item()
    return sorted_indices[sampled_index].item()


@torch.no_grad()
def generate_text(
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eot_token_id: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    model.eval()
    model.to(device)

    # Encode prompt
    input_ids: List[int] = tokenizer.encode(prompt)
    input_ids = input_ids[-model.context_len:]  # truncate if needed
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    for _ in range(max_new_tokens):
        # Truncate to last `context_len` tokens
        input_crop = input_tensor[:, -model.context_len:]

        # Forward pass to get logits
        logits = model(input_crop)  # [1, T, vocab_size]
        next_token_logits = logits[0, -1, :]  # last token prediction

        # Apply temperature
        next_token_logits = next_token_logits / temperature
        probs = run_softmax(next_token_logits, dim=-1)

        # Sample using top-p
        next_token_id = sample_top_p(probs, top_p)

        # Append to input
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1)

        # Stop if EOT
        if next_token_id == eot_token_id:
            break

    # Decode
    output_ids = input_tensor[0].tolist()
    return tokenizer.decode(output_ids)


def train():
    dataset = torch.load("tokenized_text.pt")
    checkpoint_path = 'model.pt'
    vocab_size = 5000
    context_length = 256
    d_model = 512
    d_ff = 1344
    rope_theta = 10000
    num_layers = 4
    num_heads = 16
    learning_rate = 3e-4
    weight_decay = 0.01
    max_iters = 5000
    batch_size = 64
    gradient_clip = 1.0
    warmup_iters = 250
    cosine_cycle_iters = 2500
    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    model.train()
    AdamW = get_adamw_cls()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    iteration = 0
    writer = SummaryWriter(log_dir="logs")

    for it in trange(iteration, max_iters, desc="Training"):
        inputs, targets = run_get_batch(dataset, batch_size, context_length, device="cpu")
        logits = model(inputs)
        loss = run_cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(model.parameters(), gradient_clip)
        lr = run_get_lr_cosine_schedule(it, learning_rate, 1e-5, warmup_iters, cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if it % 10 == 0:
            writer.add_scalar("Loss", loss.item(), it)
            writer.add_scalar("LR", lr, it)

        if it % 100 == 0:
            print(f"[Iter {it}] Loss: {loss.item():.4f} | LR: {lr:.6f}")

    run_save_checkpoint(model, optimizer, 5000, checkpoint_path)


def test():
    with open('tokenizer_data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    merges_loaded = loaded_data['merges']
    vocab_loaded = loaded_data['vocab']
    tokenizer = get_tokenizer(
        vocab=vocab_loaded,
        merges=merges_loaded,
        special_tokens=["<|endoftext|>"]
    )
    print('loaded tokenzier')

    vocab_size = 5000
    context_length = 256
    d_model = 512
    d_ff = 1344
    rope_theta = 10000
    num_layers = 4
    num_heads = 16
    checkpoint_path = 'model.pt'
    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    run_load_checkpoint(checkpoint_path, model, None)

    eot_token_id = tokenizer.encode("<|endoftext|>")
    print(eot_token_id)

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="Tom and Lily were playing with",
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.95,
        eot_token_id=eot_token_id
    )
    print(text)


if __name__ == '__main__':
    test()
