import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import autocast
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import k_diffusion as K

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)

        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

class KDiffusionSampler:
    def __init__(self, m, sampler):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)
        self.schedule = sampler

    def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T):
        sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)

        samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}, disable=False)

        return samples_ddim, None


def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].half().to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/inpaiting-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        help="diffusion sampler to be used",
        choices=["plms", "ddim", "k_dpm_2_a", "k_dpm_2", "k_euler_a", "k_euler", "k_heun", "k_lms"],
        default="k_euler"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=4,
        help="downsampling factor",
    )
    opt = parser.parse_args()

    masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.half().to(device)


    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(model)
    elif opt.sampler == 'k_dpm_2_a':
        sampler = KDiffusionSampler(model,'dpm_2_ancestral')
    elif opt.sampler == 'k_dpm_2':
        sampler = KDiffusionSampler(model,'dpm_2')
    elif opt.sampler == 'k_euler_a':
        sampler = KDiffusionSampler(model,'euler_ancestral')
    elif opt.sampler == 'k_euler':
        sampler = KDiffusionSampler(model,'euler')
    elif opt.sampler == 'k_heun':
        sampler = KDiffusionSampler(model,'heun')
    elif opt.sampler == 'k_lms':
        sampler = KDiffusionSampler(model,'lms')
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                for image, mask in tqdm(zip(images, masks)):
                    outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                    batch = make_batch(image, mask, device=device)

                    # encode masked image and concat downsampled mask
                    c = model.cond_stage_model.encode(batch["masked_image"])
                    cc = torch.nn.functional.interpolate(batch["mask"],
                                                        size=c.shape[-2:])
                    c = torch.cat((c, cc), dim=1)

                    uc = model.cond_stage_model.encode(batch["masked_image"])
                    ucc = torch.nn.functional.interpolate(batch["mask"],
                                                        size=c.shape[-2:])
                    uc = torch.cat((uc, ucc), dim=1)

                    start_code = torch.randn([opt.n_samples, 3, opt.H // opt.f, opt.W // opt.f], device=device)

                    shape = (c.shape[1]-1,)+c.shape[2:]
                    samples_ddim, _ = sampler.sample(S=opt.steps,
                                                    conditioning=c,
                                                    batch_size=c.shape[0],
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta,
                                                    x_T=start_code)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)

                    image = torch.clamp((batch["image"]+1.0)/2.0,
                                        min=0.0, max=1.0)
                    mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                    predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                                min=0.0, max=1.0)

                    inpainted = (1-mask)*image+mask*predicted_image
                    inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                    Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
