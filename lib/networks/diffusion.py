import tqdm
import math

import torch
import torch.nn as nn

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)

class Diffusion(nn.Module):
    def __init__(
            self, lhand_layer, rhand_layer, 
            beta_1=1e-4, beta_T=0.02, T=1000, 
            schedule_name="cosine", 
        ):
        super().__init__()

        self.lhand_layer = lhand_layer
        self.rhand_layer = rhand_layer
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        if schedule_name == "linear":
            betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
        elif schedule_name == "cosine":
            betas = betas_for_alpha_bar(
                T,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )

        alphas = 1 - betas
        alpha_bars = torch.cumprod(
            alphas, 
            dim = 0
        )

        alpha_prev_bars = torch.cat([torch.Tensor([1]), alpha_bars[:-1]])
        sigmas = torch.sqrt((1 - alpha_prev_bars) / (1 - alpha_bars)) * torch.sqrt(1 - (alpha_bars / alpha_prev_bars))

        posterior_variance = (
            betas * (1.0 - alpha_prev_bars) / (1.0 - alpha_bars)
        )
        posterior_log_variance_clipped = torch.log(
            torch.hstack([posterior_variance[1], posterior_variance[1:]])
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_prev_bars", alpha_prev_bars)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)

    def forward(
        self, model, x_lhand, x_rhand, 
        x_obj, obj_feat, 
        timesteps=None, enc_text=None, 
        get_target=False, 
        valid_mask_lhand=None, 
        valid_mask_rhand=None, 
        valid_mask_obj=None, 
    ):
        assert enc_text is not None
        return_list = []

        if timesteps == None:
            timesteps = torch.randint(0, len(self.alpha_bars), (x_obj.size(0), )).to(x_obj.device)
            used_alpha_bars = self.alpha_bars[timesteps][:, None, None]
            epsilon_lhand = torch.randn_like(x_lhand)
            x_tilde_lhand = torch.sqrt(used_alpha_bars) * x_lhand + torch.sqrt(1 - used_alpha_bars) * epsilon_lhand
            epsilon_rhand = torch.randn_like(x_rhand)
            x_tilde_rhand = torch.sqrt(used_alpha_bars) * x_rhand + torch.sqrt(1 - used_alpha_bars) * epsilon_rhand
            epsilon_obj = torch.randn_like(x_obj)
            x_tilde_obj = torch.sqrt(used_alpha_bars) * x_obj + torch.sqrt(1 - used_alpha_bars) * epsilon_obj
            
        else:
            timesteps = torch.Tensor([timesteps for _ in range(x_obj.size(0))]).to(x_obj.device).long()
            x_tilde_lhand = x_lhand
            x_tilde_rhand = x_rhand
            x_tilde_obj = x_obj

        pred_X0_lhand, \
        pred_X0_rhand, \
        pred_X0_obj \
            = model(
                x_tilde_lhand, x_tilde_rhand, 
                x_tilde_obj, obj_feat, 
                timesteps, enc_text, 
                valid_mask_lhand,
                valid_mask_rhand, 
                valid_mask_obj
            )
        return_list.append(pred_X0_lhand)
        return_list.append(pred_X0_rhand)
        return_list.append(pred_X0_obj)

        if get_target:
            return_list.append(epsilon_lhand)
            return_list.append(epsilon_rhand)
            return_list.append(epsilon_obj)
            return_list.append(used_alpha_bars)
        return return_list
    
    @torch.no_grad()
    def sampling(
        self, 
        model, obj_feat, 
        enc_text, max_nframes, 
        hand_nfeats, obj_nfeats,
        valid_mask_lhand, 
        valid_mask_rhand, 
        valid_mask_obj, 
        device, 
        return_middle=False, 
    ):
        sampling_number = len(enc_text)
        sample_lhand = torch.randn([sampling_number, max_nframes, hand_nfeats]).to(device)
        sample_rhand = torch.randn([sampling_number, max_nframes, hand_nfeats]).to(device)
        sample_obj = torch.randn([sampling_number, max_nframes, obj_nfeats]).to(device)

        sample_lhand, sample_rhand, sample_obj = self.ddpm_loop(
            sample_lhand, sample_rhand, sample_obj, 
            model, obj_feat, enc_text, 
            valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
            return_middle, 
        )
        return sample_lhand, sample_rhand, sample_obj
    
    def ddpm_loop(
            self, sample_lhand, sample_rhand, sample_obj, 
            model, obj_feat, enc_text, 
            valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
            return_middle, 
        ):
        for t_idx in tqdm.tqdm(
            reversed(range(len(self.alpha_bars))), 
            desc="sampling",
            total=len(self.alpha_bars)
        ):
            noise_lhand = torch.zeros_like(sample_lhand) if t_idx == 0 else torch.randn_like(sample_lhand)
            noise_rhand = torch.zeros_like(sample_rhand) if t_idx == 0 else torch.randn_like(sample_rhand)
            noise_obj = torch.zeros_like(sample_obj) if t_idx == 0 else torch.randn_like(sample_obj)

            pred_X0_lhand, pred_X0_rhand, pred_X0_obj \
                = self.forward(
                    model, sample_lhand, sample_rhand, 
                    sample_obj, obj_feat, 
                    timesteps=t_idx, enc_text=enc_text, 
                    valid_mask_lhand=valid_mask_lhand, 
                    valid_mask_rhand=valid_mask_rhand, 
                    valid_mask_obj=valid_mask_obj
                )
            beta = self.betas[t_idx]
            alpha = self.alphas[t_idx]
            alpha_prev_bar = self.alpha_prev_bars[t_idx]
            alpha_bar = self.alpha_bars[t_idx]
            log_variance = self.posterior_log_variance_clipped[t_idx]

            coefficient_X0 = (beta*torch.sqrt(alpha_prev_bar)/(1-alpha_bar))
            coefficient_noise = ((1-alpha_prev_bar)*torch.sqrt(alpha)/(1-alpha_bar))

            mu_xt_lhand = pred_X0_lhand*coefficient_X0+sample_lhand*coefficient_noise
            mu_xt_rhand = pred_X0_rhand*coefficient_X0+sample_rhand*coefficient_noise
            mu_xt_obj = pred_X0_obj*coefficient_X0+sample_obj*coefficient_noise
            
            sample_lhand = mu_xt_lhand + torch.exp(0.5*log_variance) * noise_lhand
            sample_rhand = mu_xt_rhand + torch.exp(0.5*log_variance) * noise_rhand
            sample_obj = mu_xt_obj + torch.exp(0.5*log_variance) * noise_obj
            if return_middle and t_idx == 500:
                return sample_lhand, sample_rhand, sample_obj
        return sample_lhand, sample_rhand, sample_obj