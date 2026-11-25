"""Common RefPacks nodes; model collections depend on this stable package."""
import sys

canonical = sys.modules.get("bexplains_refpacks")
if canonical is not None and canonical is not sys.modules[__name__]:
    NODE_CLASS_MAPPINGS = canonical.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = canonical.NODE_DISPLAY_NAME_MAPPINGS
else:
    sys.modules["bexplains_refpacks"] = sys.modules[__name__]
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
__version__ = "1.2.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Keep imports stable when ComfyUI loads this folder under a generated module name.
for name, module in list(sys.modules.items()):
    if name.startswith(__name__ + "."):
        sys.modules.setdefault("bexplains_refpacks" + name[len(__name__):], module)
