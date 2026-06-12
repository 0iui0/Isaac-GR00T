#!/usr/bin/env python3
"""Train IQL Critic + Value for QGF guidance on CR5AF offline data.

Usage:
    .venv/bin/python examples/CR5AF/train_iql_critic.py \
        --data-path /datasets/cr5af_grasp_housing_qgf.npz \
        --output-dir /tmp/cr5af_iql_critic

Output:
    critic.pt, value.pt, training loss plot
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Networks ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, dims, activation=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """Q(s, a) — takes state + action, outputs scalar Q-value."""
    def __init__(self, obs_dim=16, act_dim=16, hidden_dim=256):
        super().__init__()
        self.net = MLP([obs_dim + act_dim, hidden_dim, hidden_dim, hidden_dim, 1])

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class Value(nn.Module):
    """V(s) — takes state, outputs scalar V-value for expectile regression."""
    def __init__(self, obs_dim=16, hidden_dim=256):
        super().__init__()
        self.net = MLP([obs_dim, hidden_dim, hidden_dim, hidden_dim, 1])

    def forward(self, obs):
        return self.net(obs)


# ─── IQL Loss ─────────────────────────────────────────────────────────────

def expectile_loss(pred, target, tau=0.7):
    """Asymmetric expectile regression used by IQL for V(s)."""
    diff = target - pred
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff ** 2).mean()


# ─── Data ─────────────────────────────────────────────────────────────────

class QGFDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        d = np.load(data_path)
        self.obs = torch.from_numpy(d["observations"]).float()        # (N, 16)
        self.act = torch.from_numpy(d["actions"]).float()              # (N, 16)
        self.rew = torch.from_numpy(d["rewards"]).float().unsqueeze(1)  # (N, 1)
        self.mask = torch.from_numpy(d["masks"]).float().unsqueeze(1)   # (N, 1)
        self.next_obs = torch.from_numpy(d["next_observations"]).float() # (N, 16)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx], self.rew[idx], \
               self.mask[idx], self.next_obs[idx]


# ─── Training ─────────────────────────────────────────────────────────────

def train(data_path, output_dir, batch_size=256, lr=3e-4,
          steps=100000, tau=0.7, gamma=0.99, eval_interval=1000):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dataset = QGFDataset(data_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=0, pin_memory=True)
    data_iter = iter(loader)

    # Networks
    critic = Critic().to(device)
    critic_target = Critic().to(device)
    critic_target.load_state_dict(critic.state_dict())
    value = Value().to(device)

    opt_c = torch.optim.Adam(critic.parameters(), lr=lr)
    opt_v = torch.optim.Adam(value.parameters(), lr=lr)

    step = 0
    losses_c, losses_v = [], []
    t0 = time.time()

    while step < steps:
        try:
            obs, act, rew, mask, next_obs = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            obs, act, rew, mask, next_obs = next(data_iter)

        obs = obs.to(device)
        act = act.to(device)
        rew = rew.to(device)
        mask = mask.to(device)
        next_obs = next_obs.to(device)

        # ── Value loss (expectile regression) ──
        q = critic(obs, act).detach()
        v = value(obs)
        loss_v = expectile_loss(v, q, tau)
        opt_v.zero_grad()
        loss_v.backward()
        opt_v.step()

        # ── Critic loss (TD) ──
        with torch.no_grad():
            next_v = value(next_obs)
            target_q = rew + gamma * mask * next_v
        q_pred = critic(obs, act)
        loss_c = F.mse_loss(q_pred, target_q)
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()

        # ── Soft target update ──
        for p, tp in zip(critic.parameters(), critic_target.parameters()):
            tp.data.copy_(0.005 * p.data + 0.995 * tp.data)

        losses_c.append(loss_c.item())
        losses_v.append(loss_v.item())

        if step % eval_interval == 0:
            q_mean = q_pred.mean().item()
            v_mean = v.mean().item()
            dt = time.time() - t0
            print(f"[{step:6d}/{steps}] "
                  f"Q={q_mean:.3f} V={v_mean:.3f} "
                  f"c_loss={loss_c.item():.4f} v_loss={loss_v.item():.4f} "
                  f"({dt*1000/eval_interval:.0f}ms/step)")
            t0 = time.time()

        step += 1

    # Save
    torch.save(critic.state_dict(), os.path.join(output_dir, "critic.pt"))
    torch.save(value.state_dict(), os.path.join(output_dir, "value.pt"))
    print(f"\nSaved critic.pt and value.pt to {output_dir}")

    # Quick evaluation
    print("\nQuick eval (last batch):")
    print(f"  Q(s,a) range: [{q_pred.min().item():.3f}, {q_pred.max().item():.3f}]")
    print(f"  V(s) range:   [{v.min().item():.3f}, {v.max().item():.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/datasets/cr5af_grasp_housing_qgf.npz")
    parser.add_argument("--output-dir", default="/tmp/cr5af_iql_critic")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--tau", type=float, default=0.7)
    args = parser.parse_args()

    train(args.data_path, args.output_dir, batch_size=args.batch_size,
          steps=args.steps, tau=args.tau)
