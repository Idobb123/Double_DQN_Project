import torch
import torch.nn as nn
import torch.optim as optim

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def make_model(input_dim, output_dim):
    # simple MLP like MountainCar
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
    ).to(device)

def train_one_batch(model, batch_size, input_dim, output_dim):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # dummy data
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, output_dim, (batch_size,), device=device)
    criterion = nn.MSELoss()

    logits = model(x)
    # simple target
    target = torch.zeros_like(logits)
    target[torch.arange(batch_size), y] = 1.0
    loss = criterion(logits, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def find_max_batch(input_dim=2, output_dim=3, start=32, max_bs=1024):
    model = make_model(input_dim, output_dim)
    bs = start
    last_success = start
    while bs <= max_bs:
        try:
            print(f"Trying batch size = {bs} ...", end='')
            train_one_batch(model, bs, input_dim, output_dim)
            print(" OK")
            last_success = bs
            bs *= 2
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(" OOM")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                break
            else:
                raise e
    print(f"Max successful batch size: {last_success}")

if __name__ == "__main__":
    find_max_batch()
