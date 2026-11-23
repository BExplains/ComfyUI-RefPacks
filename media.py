"""Expose cached reference pixels and decode stored audio for downstream nodes."""
import torch


def extract_media(references, audio_vae=None, vae=None):
    images, audio, notes = [], [], []
    missing_audio = 0
    for group in references.get("groups", []):
        for entry in group["entries"]:
            kind = entry["kind"]
            if kind in ("image", "video"):
                if vae is not None:
                    with torch.inference_mode():
                        pixels = vae.decode(entry["latent"].clone())
                    if pixels.ndim == 5:
                        pixels = pixels.reshape(-1, *pixels.shape[-3:])
                    if pixels.ndim != 4 or pixels.shape[-1] != 3:
                        raise ValueError("The visual VAE must return RGB images [frames,height,width,3]. Use the VAE matching this pack's encoding.")
                else:
                    pixels = entry.get("vision")
                if pixels is None:
                    notes.append(f"{group['name']}: no cached pixels available.")
                    continue
                # Separate list items preserve differing dimensions without resizing.
                images.append(pixels.detach().cpu().clone())
                if kind == "video" and vae is None:
                    notes.append(f"{group['name']}: {len(pixels)} cached video preview frames; these may be sampled, not the complete clip.")
            elif kind == "audio":
                if audio_vae is None:
                    missing_audio += 1
                    continue
                if not hasattr(audio_vae, "audio_sample_rate"):
                    raise ValueError("Connect the matching audio VAE, not an image or video VAE.")
                from comfy_extras.nodes_audio import vae_decode_audio
                with torch.inference_mode():
                    decoded = vae_decode_audio(audio_vae, {"samples": entry["audio_latent"].clone()})
                audio.append({"waveform": decoded["waveform"].detach().cpu().clone(),
                              "sample_rate": decoded["sample_rate"]})
    if missing_audio:
        notes.append(f"{missing_audio} audio reference(s) contain latents only. Connect the matching audio_vae to extract playable Audio.")
    if audio:
        notes.append("Audio is reconstructed from VAE latents, not the original source waveform.")
    source = "VAE reconstructions" if vae is not None else "cached reference pixels"
    notes.insert(0, f"Media outputs: {len(images)} visual entry batch(es), {len(audio)} audio clip(s). Images are {source}, not original source files.")
    return images, audio, "\n".join(notes)
