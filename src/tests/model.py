import argparse
import yaml
import torch
import numpy as np
from model import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = create_model(in_channels=cfg.get("in_channels",4), out_channels=cfg.get("out_channels",3), feature_size=cfg.get("feature_size",48)).to(device)
    model.train()

    # synthetic batch: (B, C, H, W, D)
    roi = tuple(cfg.get("roi", (128,128,128)))
    x = torch.randn((1, cfg.get("in_channels",4),) + roi, device=device)
    y = torch.randint(0, cfg.get("out_channels",3), (1,) + roi, device=device)

    out = model(x)
    print(f"Model output shape: {out.shape}")

    # compute a dummy loss to check backward
    loss = out.mean()
    loss.backward()
    print("Backward OK.\nTest passed.")


if __name__ == '__main__':
    main()