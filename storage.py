"""Non-pickle, atomic, versioned reference containers."""
import json
import os
from pathlib import Path
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file

from .core import FORMAT, VERSION, validate_pack, validate_references

META_KEY = "bexplains_refpack"


def save_pack(pack, destination, overwrite=False):
    validate_pack(pack)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise ValueError(f"{destination.name} already exists. Enable overwrite or choose a different name.")
    tensors, manifest = {}, {k: v for k, v in pack.items() if k != "entries"}
    manifest["entries"] = []
    for i, e in enumerate(pack["entries"]):
        record = {k: v for k, v in e.items() if k not in ("latent", "vision", "audio_latent")}
        for field in ("latent", "vision", "audio_latent"):
            if field not in e:
                continue
            key = f"references.{i}.{field}"
            tensors[key] = e[field].detach().cpu().contiguous().clone()
            record[field + "_key"] = key
        manifest["entries"].append(record)
    return _write_container(tensors, {META_KEY: json.dumps(manifest, ensure_ascii=False)}, destination, overwrite)


def load_pack(source):
    with safe_open(str(source), framework="pt", device="cpu") as f:
        raw = (f.metadata() or {}).get(META_KEY)
        if raw is None:
            raise ValueError("This is not a BExplains RefPack. For an older RefMod, use Import H3 RefMod and save the converted pack once.")
        manifest = json.loads(raw)
        if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
            raise ValueError("Unsupported RefPack format or version.")
        entries = []
        for record in manifest.get("entries", []):
            e = dict(record)
            for field in ("latent", "vision", "audio_latent"):
                if field + "_key" not in e:
                    continue
                key = e.pop(field + "_key")
                e[field] = f.get_tensor(key).clone()
            entries.append(e)
    manifest["entries"] = entries
    return validate_pack(manifest)


REFERENCES_META_KEY = "bexplains_references"
REFERENCES_FORMAT = "bexplains.references"
REFERENCES_VERSION = 1
REFERENCES_SUFFIX = ".references.safetensors"
TENSOR_FIELDS = ("latent", "vision", "audio_latent")


def _write_container(tensors, metadata, destination, overwrite):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise ValueError(f"{destination.name} already exists. Enable overwrite or choose a different name.")
    fd, temporary = tempfile.mkstemp(prefix=".refpack-", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        save_file(tensors, temporary, metadata=metadata)
        # Explicit overwrite is required; avoid silently replacing a concurrently saved file.
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(destination)


def save_references(references, destination, overwrite=False):
    """Embed the entire ordered reference set and tensors in one independent file."""
    validate_references(references)
    tensors = {}
    stored = {k: v for k, v in references.items() if k != "groups"}
    stored["groups"] = []
    for gi, group in enumerate(references["groups"]):
        saved_group = {k: v for k, v in group.items() if k != "entries"}
        saved_group["entries"] = []
        for ei, entry in enumerate(group["entries"]):
            record = {k: v for k, v in entry.items() if k not in TENSOR_FIELDS}
            for field in TENSOR_FIELDS:
                if field not in entry:
                    continue
                key = f"groups.{gi}.entries.{ei}.{field}"
                tensors[key] = entry[field].detach().cpu().contiguous().clone()
                record[field + "_key"] = key
            saved_group["entries"].append(record)
        stored["groups"].append(saved_group)
    manifest = {"format": REFERENCES_FORMAT, "version": REFERENCES_VERSION,
                "name": Path(destination).name.removesuffix(REFERENCES_SUFFIX), "references": stored}
    return _write_container(tensors, {REFERENCES_META_KEY: json.dumps(manifest, ensure_ascii=False)}, destination, overwrite)


def load_references(source):
    with safe_open(str(source), framework="pt", device="cpu") as f:
        raw = (f.metadata() or {}).get(REFERENCES_META_KEY)
        if raw is None:
            raise ValueError("This is not a saved RefPacks reference set. Use Load RefPack for individual packs.")
        manifest = json.loads(raw)
        if manifest.get("format") != REFERENCES_FORMAT or manifest.get("version") != REFERENCES_VERSION:
            raise ValueError("Unsupported RefPacks reference-set format or version.")
        stored = manifest.get("references")
        if not isinstance(stored, dict) or not isinstance(stored.get("groups"), list):
            raise ValueError("Invalid reference-set manifest.")
        for group in stored["groups"]:
            for entry in group.get("entries", []):
                for field in TENSOR_FIELDS:
                    key = entry.pop(field + "_key", None)
                    if key is not None:
                        entry[field] = f.get_tensor(key).clone()
    return validate_references(stored)
