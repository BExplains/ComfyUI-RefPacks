# Changelog

## 1.1.2 — 2026-11-24

- Remove hardcoded example filenames from RefPack and RefPack Bundle load selectors. Only discovered files are listed; empty folders produce empty lists.
- Preserve storage paths, file formats and internal node IDs.

## 1.1.1 — 2026-11-24

- Rename the shared load/save node display names to Load RefPack Bundles and Save RefPack Bundles; update usage documentation.
- Preserve internal node IDs, existing workflows, the models/refpacks/references folder and .references.safetensors files.

## 1.1.0 — 2026-11-23

- Add Auto/Image/Video generation mode to H3 encoders. Auto uses length 1-4 for images and 5+ for video; explicit Video rounds up to at least five frames. VAE selection no longer chooses generation mode.

- Add image-only style groups and H3 `<Style n>` aliases resolving to actual Picture slots; preserve style roles through scene save/load.

- Add Images and Audio list outputs to both inspectors, with optional visual/audio VAEs for reconstruction and cached-pixel extraction without a VAE.
- Preserve existing output indices and entry dimensions; keep video frames batched and audio clips separate.
- Rename the H3 inspector to BExplains Inspect Refpacks without changing its internal ID.

- Stop tracking Python bytecode caches; keep them excluded from source and release archives.

## 1.0.1

- Rename encoder display names to Encode Image Refpack and Encode Video Refpack.
- Update README, workflow instructions and error messages to use the new names.
- Preserve internal node IDs, saved workflow compatibility and encoding behavior.

## 1.0.0

- Prepare Comfy Registry metadata, MIT licensing and clean release archives.
- Document registry installation and explicit companion-package installation.
- Correct stale test paths after splitting the H3 collection; 87 CPU tests pass.

- Prepare both packages for the first public release at version 1.0.0.
- Rewrite each README with installation, quick start, node reference, prompting,
  compatibility, performance, troubleshooting, FAQ, issue reporting and credits.
- Document the actual repositories and explicit sibling installation folder names.
- Include structured text and JSON examples, exact raw-prompt detection behavior,
  typed roles, Auto Outfit, H3 image/video intent and reusable scene collections.
- Retain Flux 2 and LTXV 0.9.8 as experimental testing targets, without promoting
  them to officially supported integrations.
- Keep existing node IDs, version-1 asset/set formats and generation behavior.

## 0.9.2 — unreleased development build

- Add Character, Item and Motion reference roles to the shared Add RefPack node.
- Compile Character/Item aliases into distinct native H3 Subject IDs, with
  person-specific and object-specific definitions and retention guidance.
- Compile Motion aliases into their actual native Video slots.
- Make generic Subject definitions neutral rather than assuming a person.
- Preserve legacy Subject IDs and store stable shared subject_id metadata for new
  groups so appending packs or saving/loading reference sets does not renumber them.
- Show typed aliases and native Subject IDs in H3 inspection and prompt reports.
- Auto Outfit excludes items and explicitly targets the sole wearer when items
  are also present. Raw prompt passthrough continues to bypass all alias expansion.

## 0.9.1 — unreleased development build

- Add Save RefPacks References and Load RefPacks References to the shared package.
- Store the full assembled set in one .references.safetensors file, including
  cached tensors, group/entry order, names, role IDs, provenance and metadata.
- Preserve references exactly on loading; original individual files are not needed.
- Use atomic writes and the existing explicit overwrite protection.
- Keep collection files in models/refpacks/references with a separate loader list.

## 0.9.0 — unreleased development build

- Keep development at 0.9.0; a 1.0 release requires the project owner's decision.
- Mark Flux 2 and LTXV 0.9.8 as experimental testing targets, not officially supported integrations.
- Add opt-in Allow Raw Prompt on both H3 receiver nodes and the H3 inspector.
  Recognized structured text/JSON passes through exactly, with reference tensors retained.
- Split shared RefPack persistence, grouping, preparation and receiver interfaces
  from the separate H3 node collection; preserve legacy node IDs and version-1 packs.
- Add RefPack to Image Encoder with a Flux 2 adapter, and RefPack to Video Encoder
  with an LTXV 0.9.8 adapter, plus preparation nodes and two demo workflows.
- Add RefPack to H3 Encoder with VAE-based output intent and VAE passthrough.
- Image lengths 1–4 allocate independent batch items, each with one temporal frame.
  Values above 4 fall back to one image with an explicit report.
- Add a composable diffusion wrapper that processes image batch items separately
  per denoising step to satisfy native H3's explicit batch-one restriction.
- Keep architecture-specific conditioning when using an H3 video model for stills;
  support normal image VAEs and still-frame-tuned H3 video decoders.
- Update four H3 render examples to connect decoding VAE intent explicitly.
- Keep H3 Duration and all existing H3 preparation, inspection and audio support.

## 0.6.0 — H3 Duration calculator (minor)

- Add seconds-to-frame conversion at 24 fps, rounding up to the 17n+5 grid.
- This feature was delivered before version labels were synchronized; the prior
  archive still displayed 0.5.0. Version labels are synchronized from 0.9.0 onward.

## 0.5.0

- Add reference tag inspection and compiled prompt previews.
- Save RefPack filename defaults subsequently changed to plain names with automatic
  .safetensors suffix handling; extension-appending behavior was already present.
