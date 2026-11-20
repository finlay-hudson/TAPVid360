from pathlib import Path

import torch


class CheckpointSaver:
    def __init__(self, out_root):
        self.best_epoch = -1
        self.out_root = Path(out_root)

    def save(self, current_epoch: int, models: dict, dists: dict, losses: dict, save_epoch_number=True):
        val_epochs = list(dists["val"].keys())
        mean_dists = torch.stack(list(dists["val"].values())).mean(axis=[1, 2, 3])
        min_dist_idx = mean_dists.argmin()
        min_dist_val = mean_dists.min()
        last_mean_dist = mean_dists[-1]
        best_epoch = val_epochs[min_dist_idx] + 1
        model_out_dict = {k: v.state_dict() for k, v in models.items()}
        model_out_dict["extras"] = {"dists": dists, "losses": losses}
        model_out_dict["epoch"] = current_epoch
        if best_epoch != self.best_epoch:
            self.best_epoch = best_epoch
            torch.save(model_out_dict, self.out_root / "checkpoint_best.pth")
        torch.save(model_out_dict, self.out_root / "checkpoint_last.pth")
        if save_epoch_number:
            torch.save(model_out_dict, get_epoch_out_dir(self.out_root, current_epoch) / "checkpoint.pth")
        with open(self.out_root / "checkpoint_info.txt", 'w') as file:
            # Write two lines to the file
            file.write(f'Best Epoch: {self.best_epoch}\n')
            file.write(f'Best Epoch mean dist {min_dist_val:.5f}\n')
            file.write(f'Last Epoch {current_epoch}\n')
            file.write(f'Last Epoch mean dist {last_mean_dist:.5f}\n')

    def load(self, name: str):
        assert (self.out_root / name).exists(), f"Checkpoint {self.out_root / name} does not exist"
        return torch.load(self.out_root / name, weights_only=False)


def get_epoch_out_dir(ims_out_root, epoch_num):
    if ims_out_root is None:
        return None

    epoch_ims_out_root = ims_out_root / f"epoch_{epoch_num}"
    epoch_ims_out_root.mkdir(exist_ok=True)

    return epoch_ims_out_root
