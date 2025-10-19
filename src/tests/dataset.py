import argparse
import yaml
from dataset import get_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    train_loader, val_loader = get_loaders(cfg["data_dir"], cfg["json_list"], roi=tuple(cfg.get("roi", (128,128,128))), batch_size=cfg.get("batch_size",2), num_workers=2)

    print("-- Iterating a few training batches --")
    for i, b in enumerate(train_loader):
        imgs = b["image"]
        labs = b["label"]
        print(f"Batch {i}: image shape {imgs.shape} label shape {labs.shape} dtype {imgs.dtype}")
        if i + 1 >= args.num_samples:
            break

    print("-- Sanity checks on a validation case --")
    for i, b in enumerate(val_loader):
        print(f"Val {i}: image shape {b['image'].shape}")
        break

    print("Dataset tests completed.")


if __name__ == '__main__':
    main()