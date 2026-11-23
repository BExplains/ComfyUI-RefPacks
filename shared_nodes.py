"""Common node interfaces, backed by explicitly compatible model adapters."""
import math
import torch
from .core import FORMAT, VERSION, validate_pack
from .adapters import ADAPTERS, for_model
from .media import extract_media

CATEGORY = "BExplains/RefPacks"
PACK_TYPE = "BEXPLAINS_REFPACK"
SET_TYPE = "BEXPLAINS_REFERENCES"



def _check_vae(vae, adapter):
    validator = getattr(adapter, "validate_vae", None)
    if validator is not None:
        validator(vae)


def _canvas(width, height, multiple):
    if width < multiple or height < multiple or width % multiple or height % multiple:
        raise ValueError(f"Width and height must be positive multiples of {multiple} for this adapter.")


def _pixels(image, width, height):
    import comfy.utils
    return comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)


def _entry(vae, adapter, pixels, kind, **metadata):
    z = vae.encode(pixels)
    adapter.validate(z, kind)
    return {"kind": kind, "latent": z.detach().cpu().contiguous(),
            "vision": pixels.detach().cpu().float().contiguous(), **metadata}


def _pack(vae, adapter, name, entries):
    encoder = vae.first_stage_model
    return validate_pack({"format": FORMAT, "version": VERSION, "name": name,
        "target": adapter.TARGET, "source_target": adapter.TARGET,
        "encoder_class": type(encoder).__module__ + "." + type(encoder).__name__, "entries": entries})


def _entries(references, target):
    if not references or not references.get("groups"):
        raise ValueError("Add at least one reference pack.")
    entries = []
    report = []
    for group in references["groups"]:
        family = group.get("target", group.get("source_target"))
        if family != target:
            raise ValueError(f"Pack '{group['name']}' was encoded for {family}, not {target}. Re-encode its source with the target model's VAE; cached latents are not interchangeable.")
        for e in group["entries"]:
            if e["kind"] == "audio":
                raise ValueError(f"The {target} adapter does not support audio references.")
            entries.append(e)
            report.append(f"Reference {len(entries)}: {group['role']} | {group['name']} | {group.get('source_file') or '(unsaved)'}")
    return entries, "\n".join(report)


class BExplainsEncodeImageReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "images": ("IMAGE",),
            "model_family": ([name for name, adapter in ADAPTERS.items() if "image" in getattr(adapter, "REFERENCE_KINDS", ("image",))],), "name": ("STRING", {"default": "reference"}),
            "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16})}}
    RETURN_TYPES = (PACK_TYPE,)
    RETURN_NAMES = ("refpack",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Prepare one cached entry per image with the selected model family's VAE. Flux 2 and LTXV use different latent formats. For LTXV, match the final video resolution. H3 preparation is in its separate collection."

    def encode(self, vae, images, model_family, name, width, height):
        adapter = ADAPTERS[model_family]
        _canvas(width, height, adapter.MULTIPLE)
        _check_vae(vae, adapter)
        entries = [_entry(vae, adapter, _pixels(image.unsqueeze(0), width, height), "image") for image in images]
        return (_pack(vae, adapter, name, entries),)


class BExplainsEncodeVideoReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "frames": ("IMAGE",),
            "model_family": ([name for name, adapter in ADAPTERS.items() if "video" in getattr(adapter, "REFERENCE_KINDS", ())],),
            "name": ("STRING", {"default": "video_reference"}),
            "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0}),
            "width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 32})}}
    RETURN_TYPES = (PACK_TYPE,)
    RETURN_NAMES = ("refpack",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Prepare an LTXV leading-video guide once. Pad the final frame to the 8n+1 grid. This is prefix conditioning, not arbitrary motion transfer."

    def encode(self, vae, frames, name, fps, width, height, model_family="LTXV 0.9.8"):
        adapter = ADAPTERS[model_family]
        if "video" not in getattr(adapter, "REFERENCE_KINDS", ()):
            raise ValueError("This adapter cannot encode video references.")
        _canvas(width, height, adapter.MULTIPLE)
        _check_vae(vae, adapter)
        if not math.isfinite(fps) or fps <= 0 or frames.shape[0] < 1:
            raise ValueError("Supply frames and a positive finite source frame rate.")
        pixels = _pixels(frames, width, height)
        count = adapter.reference_frame_count(len(pixels))
        if count > len(pixels):
            pixels = torch.cat([pixels, pixels[-1:].repeat(count - len(pixels), 1, 1, 1)])
        entry = _entry(vae, adapter, pixels, "video", timestamps=[i / fps for i in range(count)],
                       fps=fps, frame_count=count, source_frame_count=len(frames))
        return (_pack(vae, adapter, name, [entry]),)


class _Receiver:
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "vae", "reference_report")
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    @classmethod
    def _inputs(cls):
        return {"model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",), "references": (SET_TYPE,),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16})}

    def _prepare(self, model, clip, vae, references, prompt, negative_prompt, width, height, media):
        adapter = for_model(model, media)
        _canvas(width, height, adapter.MULTIPLE)
        _check_vae(vae, adapter)
        entries, report = _entries(references, adapter.TARGET)
        # Validate before running expensive text encoders.
        for e in entries:
            adapter.validate(e["latent"], e["kind"])
        if hasattr(adapter, "check_entries"):
            adapter.check_entries(entries, width, height)
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        return adapter, entries, report, positive, negative


class BExplainsRefPackToImage(_Receiver):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = cls._inputs()
        inputs.update(batch_size=("INT", {"default": 1, "min": 1, "max": 64}),
                      guidance=("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0}))
        return {"required": inputs}
    DESCRIPTION = "Experimental shared image receiver with a Flux 2 testing adapter; not officially supported yet. Reuse cached Flux 2 references and native reference_latents conditioning. Keeps the user prompt unchanged. Connect vae output to VAEDecode."

    def encode(self, model, clip, vae, references, prompt, negative_prompt, width, height, batch_size, guidance):
        import comfy.model_management
        adapter, entries, report, positive, negative = self._prepare(model, clip, vae, references, prompt, negative_prompt, width, height, "image")
        positive, negative, latent, details = adapter.receive(positive, negative, entries, width, height, batch_size, guidance, comfy.model_management.intermediate_device())
        return model, positive, negative, latent, vae, report + "\n" + details


class BExplainsRefPackToVideo(_Receiver):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = cls._inputs()
        inputs["width"] = ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32})
        inputs["height"] = ("INT", {"default": 512, "min": 64, "max": 8192, "step": 32})
        inputs.update(length=("INT", {"default": 97, "min": 9, "max": 4096,
                      "tooltip": "Output video frames. The LTXV adapter rounds up to 8n+1."}),
                      frame_rate=("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0}),
                      strength=("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}))
        return {"required": inputs}
    DESCRIPTION = "Experimental shared video receiver with an LTXV 0.9.8 testing adapter; not officially supported yet. One cached first image or leading-video guide, native noise mask and frame-rate conditioning. Keep the model's normal sampler/scheduler downstream."

    def encode(self, model, clip, vae, references, prompt, negative_prompt, width, height, length, frame_rate, strength):
        import comfy.model_management
        adapter, entries, report, positive, negative = self._prepare(model, clip, vae, references, prompt, negative_prompt, width, height, "video")
        positive, negative, latent, details = adapter.receive(positive, negative, entries, width, height, length, frame_rate, strength, comfy.model_management.intermediate_device())
        return model, positive, negative, latent, vae, report + "\n" + details


class BExplainsInspectReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"references": (SET_TYPE,)},
                "optional": {"vae": ("VAE", {"tooltip": "Matching image/video VAE for reconstructing full visual latents. Leave disconnected to return cached pixels."}),
                             "audio_vae": ("VAE", {"tooltip": "Matching audio VAE for reconstructing stored audio latents."})}}
    RETURN_TYPES = (SET_TYPE, "STRING", "IMAGE", "AUDIO")
    RETURN_NAMES = ("references", "report", "Images", "Audio")
    OUTPUT_IS_LIST = (False, False, True, True)
    FUNCTION = "inspect"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def inspect(self, references, audio_vae=None, vae=None):
        lines = []
        for group in references.get("groups", []):
            lines.append(f"{group['role']}: {group['name']} | target: {group.get('target', group.get('source_target'))} | encoder: {group.get('encoder_class', 'unspecified')} | file: {group.get('source_file') or '(unsaved)'}")
            for e in group["entries"]:
                z = e.get("latent", e.get("audio_latent"))
                lines.append(f"  {e['kind']}: {list(z.shape)}")
        images, audio, media_report = extract_media(references, audio_vae, vae)
        report = "\n".join(lines + ["", media_report])
        return {"ui": {"text": [report]}, "result": (references, report, images, audio)}
