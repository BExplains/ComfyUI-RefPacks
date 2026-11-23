"""BExplains RefPacks: direct saved-latent references for native H3 Image."""


class BExplainsSetH3ReferenceLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "latent": ("LATENT",),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "BExplains/RefPacks"
    DESCRIPTION = (
        "Append a saved H3 Image VAE latent to native H3 reference conditioning. "
        "Chain nodes to add references in order. No image decoding or VAE encoding. "
        "Does not provide reference pixels to the text encoder."
    )

    def apply(self, conditioning, latent):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("Connect a LATENT containing samples, such as Load Latent.")
        z = latent["samples"]
        if getattr(z, "is_nested", False):
            raise ValueError("Use a reference encoded with the H3 Image VAE, not a joint video/audio latent.")
        if getattr(z, "ndim", None) == 4:
            z = z.unsqueeze(2)
        if getattr(z, "ndim", None) != 5 or z.shape[1] != 24:
            raise ValueError(
                "Expected an H3 Image VAE latent [1, 24, 1, H, W]. "
                "Flux and Qwen Image latents are incompatible."
            )
        if z.shape[0] != 1 or z.shape[2] != 1:
            raise ValueError("Use one still image per latent. Chain nodes for multiple references; video references are not supported by this node.")
        if z.shape[-2] <= 0 or z.shape[-1] <= 0:
            raise ValueError("Reference latent spatial dimensions must be positive.")
        if not isinstance(conditioning, (list, tuple)) or not conditioning:
            raise ValueError("Connect native H3 CONDITIONING, such as MiniMax H3 Image Conditioning's positive output.")

        reference = {
            "kind": "image",
            "latent_h": z.shape[-2],
            "latent_w": z.shape[-1],
            "latent": z,
        }
        result = []
        for entry in conditioning:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[1], dict):
                raise ValueError("Expected native ComfyUI CONDITIONING entries [embedding, metadata].")
            metadata = entry[1].copy()
            existing = metadata.get("minimax_refs", [])
            if not isinstance(existing, (list, tuple)):
                raise ValueError("Existing minimax_refs must be a list of H3 reference blocks.")
            metadata["minimax_refs"] = list(existing) + [reference.copy()]
            result.append([entry[0], metadata])
        return (result,)


NODE_CLASS_MAPPINGS = {
    "BExplainsSetH3ReferenceLatent": BExplainsSetH3ReferenceLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BExplainsSetH3ReferenceLatent": "BExplains Set H3 Reference Latent",
}
