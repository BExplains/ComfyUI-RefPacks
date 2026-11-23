"""Receiver integrations; additional collections can register adapters here."""
from . import flux2, ltxv

ADAPTERS = {}


def register_adapter(display_name, adapter):
    """An adapter owns family/shape validation and the native conditioning protocol."""
    for field in ("TARGET", "MULTIPLE", "MEDIA", "MODEL_CLASSES", "validate", "receive"):
        if not hasattr(adapter, field):
            raise ValueError(f"Adapter {display_name} is missing {field}.")
    if display_name in ADAPTERS and ADAPTERS[display_name] is not adapter:
        raise ValueError(f"Adapter {display_name} is already registered.")
    ADAPTERS[display_name] = adapter


def for_model(model, media):
    names = {c.__name__ for c in type(getattr(model, "model", None)).__mro__}
    candidates = [adapter for adapter in ADAPTERS.values()
                  if adapter.MEDIA == media and names.intersection(adapter.MODEL_CLASSES)
                  and not names.intersection(getattr(adapter, "EXCLUDE_MODEL_CLASSES", ()))]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError("Multiple RefPack adapters match this model; remove the conflicting adapter registrations.")
    raise ValueError(f"No {media} RefPack adapter is installed for this model. Use Flux 2 for images, LTXV 0.9.8 for video, or the separate H3 collection.")


register_adapter("Flux 2", flux2)
register_adapter("LTXV 0.9.8", ltxv)
