import argparse
import time
import torch
import torch.nn as nn
from timeit import default_timer as timer
from cs336_basics.adapters import *


class SimpleTransformer(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model, nhead=8)
            for _ in range(n_layers)
        ])
        self.final = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.final(self.layers(x))


def benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    model = SimpleTransformer(args.d_model, args.n_layers).to(device)
    data = torch.randn(args.batch_size, args.context, args.d_model, device=device, requires_grad=True)

    optimizer = torch.optim.Adam(model.parameters())

    def run_step():
        optimizer.zero_grad()
        output = model(data)
        loss = output.sum()
        if not args.forward_only:
            loss.backward()
        return loss.item()

    # warm-up
    for _ in range(args.warmup):
        run_step()
        torch.cuda.synchronize()

    # measure
    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        start = timer()
        run_step()
        torch.cuda.synchronize()
        end = timer()
        times.append(end - start)

    import numpy as np
    print(f"Mean: {np.mean(times):.4f}s, Std: {np.std(times):.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--forward_only", action='store_true')
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    benchmark(args)
