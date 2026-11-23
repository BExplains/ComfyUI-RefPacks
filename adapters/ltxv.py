"""Experimental testing adapter: LTXV 0.9.8 first-image / leading-video conditioning, matching LTXVImgToVideo."""
import torch

TARGET = "ltxv"
MULTIPLE = 32
MEDIA = "video"
MODEL_CLASSES = ("LTXV",)
REFERENCE_KINDS = ("image", "video")
EXCLUDE_MODEL_CLASSES = ("LTXAV",)


def validate_vae(vae):
    channels = getattr(vae, "latent_channels", None)
    dimensions = getattr(vae, "latent_dim", None)
    if channels is not None and channels != 128 or dimensions is not None and dimensions != 3:
        raise ValueError("The connected VAE is incompatible with ltxv. Use that model family's VAE.")


def validate(z, kind):
    if kind not in ("image", "video") or z.ndim != 5 or z.shape[0] != 1 or z.shape[1] != 128:
        raise ValueError("LTXV references require [1,128,T,H,W] latents encoded with the LTXV VAE.")
    if kind == "image" and z.shape[2] != 1:
        raise ValueError("An LTXV image reference must have one latent frame.")
    return z


def reference_frame_count(length):
    return 1 + 8 * max(0, (length - 1 + 7) // 8)


def align_frames(length):
    return 1 + 8 * max(1, (length - 1 + 7) // 8)


def check_entries(entries, width, height):
    # This model conditions on a first image / video prefix, not arbitrary identities or outfits.
    if len(entries) != 1:
        raise ValueError("LTXV 0.9.8 accepts one first-image or leading-video reference here. Combine references into an image before encoding; arbitrary multi-reference identity conditioning is not supported.")
    e = entries[0]
    ref = validate(e["latent"], e["kind"])
    if tuple(ref.shape[-2:]) != (height // 32, width // 32):
        raise ValueError(f"LTXV reference resolution must match the output {width}x{height}. Prepare the pack at that resolution.")


def receive(positive, negative, entries, width, height, length, frame_rate, strength, device):
    check_entries(entries, width, height)
    ref = entries[0]["latent"]
    frames = align_frames(length)
    latent = torch.zeros([1, 128, (frames - 1) // 8 + 1, height // 32, width // 32], device=device)
    used = min(ref.shape[2], latent.shape[2])
    latent[:, :, :used] = ref[:, :, :used].to(device)
    mask = torch.ones([1, 1, latent.shape[2], 1, 1], device=device)
    mask[:, :, :used] = 1.0 - strength
    positive = [[t, {**m, "frame_rate": frame_rate}] for t, m in positive]
    negative = [[t, {**m, "frame_rate": frame_rate}] for t, m in negative]
    return positive, negative, {"samples": latent, "noise_mask": mask}, f"LTXV 0.9.8: {frames} frames at {frame_rate:g} fps; {used} reference latent frame(s) used as a leading guide."
