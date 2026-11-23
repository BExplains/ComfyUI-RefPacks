"""Experimental testing adapter: FLUX.2 native reference_latents conditioning and 128-channel image latents."""
import torch

TARGET = "flux2"
MULTIPLE = 16
MEDIA = "image"
MODEL_CLASSES = ("Flux2",)
REFERENCE_KINDS = ("image",)


def validate_vae(vae):
    channels = getattr(vae, "latent_channels", None)
    dimensions = getattr(vae, "latent_dim", None)
    if channels is not None and channels != 128 or dimensions is not None and dimensions != 2:
        raise ValueError("The connected VAE is incompatible with flux2. Use that model family's VAE.")


def validate(z, kind):
    if kind != "image" or z.ndim != 4 or z.shape[0] != 1 or z.shape[1] != 128:
        raise ValueError("Flux 2 references must be image latents [1,128,H,W], encoded using the Flux 2 VAE.")
    return z


def receive(positive, negative, entries, width, height, count, guidance, device):
    refs = [validate(e["latent"], e["kind"]) for e in entries]
    positive = [[t, {**m, "reference_latents": refs, "guidance": guidance}] for t, m in positive]
    negative = [[t, {**m, "reference_latents": refs, "guidance": guidance}] for t, m in negative]
    latent = {"samples": torch.zeros([count, 128, height // 16, width // 16], device=device)}
    return positive, negative, latent, f"Flux 2: {count} image(s), {len(refs)} reference image(s)."
