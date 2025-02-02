import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import numpy as np
import tqdm
import hydra
from omegaconf import OmegaConf
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
import torch.optim as optim

from lib.networks.clip import load_and_freeze_clip, encoded_text
from lib.datasets.datasets import get_dataloader
from lib.utils.model_utils import build_seq_cvae
from lib.utils.metric import AverageMeter
from lib.utils.file import (
    make_save_folder, 
    wandb_login,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_object(config)
    config = edict(config)
    data_config = config.dataset
    dataset_name = data_config.name
    # wandb = wandb_login(training_config)

    model_name = config.seq_cvae.model_name
    model_save_path = config.seq_cvae.weight_path
    best_model_save_path = model_save_path.replace(".pth", "_best.pth")
    save_root = config.seq_cvae.save_root
    make_save_folder(save_root)
    
    dataloader = get_dataloader("Sequence"+dataset_name, config, data_config)
    seq_cvae = build_seq_cvae(config)
    optimizer = optim.AdamW(seq_cvae.parameters(), lr=config.seq_cvae.lr)
    
    clip_model = load_and_freeze_clip(config.clip.clip_version)
    clip_model = clip_model.cuda()
    
    cur_loss = 9999
    best_loss = 9999
    best_epoch = -1
    
    nepoch = config.seq_cvae.iteration / (data_config.data_num/data_config.text_num)
    nepoch = int(np.ceil(nepoch / 50.0) * 50)
    with tqdm.tqdm(range(nepoch)) as pbar:
        for epoch in pbar:
            seq_cvae.train()
            loss_meter = AverageMeter()
            loss_recon_meter = AverageMeter()
            loss_kld_meter = AverageMeter()
            for item in dataloader:
                seq_time = item["seq_time"].cuda()
                text = item["text"]
                enc_text = encoded_text(clip_model, text)

                bs = seq_time.shape[0]
                output, mu, logvar = seq_cvae(seq_time, enc_text)
                
                recon_loss = F.mse_loss(output, seq_time, reduction="sum")
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                losses = recon_loss + kl_div

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                loss_meter.update(losses.item(), bs)
                loss_recon_meter.update(recon_loss.item(), bs)
                loss_kld_meter.update(kl_div.item(), bs)
            
            cur_loss = loss_meter.avg
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_epoch = epoch+1
                torch.save(
                    {
                        "model": seq_cvae.state_dict(), 
                        "epoch": epoch, 
                        "loss": cur_loss, 
                    },
                    best_model_save_path, 
                )
            pbar.set_description(f"{model_name} | lr: {optimizer.param_groups[0]['lr']} | Best loss: {best_loss:.4f} ({best_epoch}), Cur loss: {cur_loss:.4f}")
            
    torch.save(
        {
            "model": seq_cvae.state_dict(), 
            "epoch": nepoch, 
            "loss": cur_loss, 
        },
        model_save_path, 
    )


if __name__ == "__main__":
    main()