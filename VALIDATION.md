# Validation — working tree (inspector media outputs)

101 CPU tests passed with real PyTorch tensors and safetensors serialization.
ComfyUI encoders/model interfaces use recording doubles; no pretrained weights
were loaded and no live ComfyUI frontend or GPU generation was performed.

Checked:

- Character/Item aliases resolve to distinct shared Subject IDs in both H3 prompt
  formats, with correct person/object definitions and multiple views per group.
- Motion aliases resolve to actual native video slots, including interleaved
  video/motion groups and multiple video entries.
- Typed counters remain independent; native Subject IDs stay stable on append
  and reference-set save/load, including legacy groups without subject_id fields.
- Inspector aliases match compiled prompt addressing; missing aliases and invalid
  role/media combinations fail explicitly. Auto Outfit excludes items, and raw
  passthrough leaves new aliases untouched.

- Complete reference-set round trips preserve mixed image/video/audio tensors,
  metadata, group/entry order, role IDs and compiled H3 addressing. Original source
  files can be removed, and additional packs can be appended after loading.
- Shared Flux/LTX sets and older H3 groups round-trip without metadata rewriting.
- Reference-set writes retain the existing explicit overwrite protection, refuse
  concurrent collisions atomically, and clean up failed temporary writes. Separate
  loader choices, automatic suffix handling and path confinement are checked.

- Raw-prompt opt-in preserves structured text and JSON exactly, including whitespace;
  skips aliases, auto-outfit expansion and prompt tag checks; retains reference
  payload identity, ordinary shot compilation and model/data validation. Both H3
  receiver nodes and inspector previews share the same behavior.
- Both package versions are 1.0.0; Flux 2 and LTXV are labeled experimental testing
  integrations rather than officially supported models. The 1.0.0 version does not
  promote those experimental adapters to official model support.

- All previous H3 container, audio, reference ordering, prompt, inspection,
  profiling and native-conditioning payload checks remain passing.
- Flux 2 image preparation, multi-reference save/load fidelity, native conditioning
  metadata, batch allocation and unchanged prompts. Empty latent allocation was
  compared with the actual native EmptyFlux2LatentImage implementation.
- LTXV image conditioning matches actual native LTXVImgToVideo latent/noise-mask
  output for the same encoded input. Checks also cover frame-rate metadata,
  video-reference padding, prefix truncation and resolution/multi-reference errors.
- Receiving nodes make no VAE encode calls; incompatible families fail explicitly.
- H3 VAE intent detection covers image VAEs, video VAEs and single-frame-tuned video
  decoders. Lengths 1–4 allocate separate batch items with T=1 for both architectures.
  The video architecture retains a matching audio batch; lengths over 4 fall back to
  one image with a report. An image diffusion model cannot be used for video output.
- The actual native H3 image-VAE temporal guard accepts batches 1–4 with T=1 and
  rejects multi-frame temporal inputs.
- Both native H3 diffusion models explicitly reject batch size greater than one.
  The collection's wrapper splits each denoising forward into batch-one items and
  reassembles predictions. Checks cover image and joint audio/video stream layouts,
  timesteps, context, masks, CFG labels, unchanged reference payloads, registration
  without mutating the original model, and composition with native WrapperExecutor.
- Shared-only imports need no H3 collection. Both shared-first and H3-first load
  orders pass without duplicate node registrations.
- All ten workflow graphs pass link/type/registration checks. Four H3 render examples
  connect the intended decoding VAE through the new receiver.
- Python sources compile and both frontend scripts pass JavaScript syntax checks.

The new image-batch wrapper and Flux/LTX adapters still need validation with actual
model weights on the user's GPU. The LTX example is a minimal single-pass demo,
not validation of the official multi-scale distilled upscaling pipeline.

## Run CPU tests

Install torch and safetensors in the test environment, then run from the directory
containing both collection folders:

```bash
python -m unittest discover -s ComfyUI-RefPacks/tests -v
```

Native-source comparisons are optional. To repeat those checks, set paths to the
corresponding ComfyUI source snapshots before running:

- H3_NATIVE_SOURCE: comfy_extras/nodes_minimax_h3.py
- REFPACKS_NATIVE_FLUX_SOURCE: comfy_extras/nodes_flux.py
- REFPACKS_NATIVE_LTX_SOURCE: comfy_extras/nodes_lt.py
- REFPACKS_NATIVE_H3_VAE_SOURCE: comfy/ldm/minimax/vae.py
- REFPACKS_NATIVE_NESTED_SOURCE: comfy/nested_tensor.py
- REFPACKS_NATIVE_PATCHER_SOURCE: comfy/patcher_extension.py

## Native source snapshot SHA-256

- `nodes_minimax_h3.py`: `32da0ea654f884b6c9dac74caaf81f1354dfd27686109b1b81952a24689f6492`
- `nodes_flux.py`: `e79791b1ac1d302271fb062bed5e3cebdb7d3cfc25bba4cbcadf81a7bcb5a95d`
- `nodes_lt.py`: `5d921cda52072e25c2c415d50b149998ab002049371af2a8ba33271ae3be9d71`
- `minimax/vae.py`: `fd32db0e1fd277fac561c6232323afc53957ed6eb48cc50eca18730a0a6bb0d1`
- `nested_tensor.py`: `d4d1f56e90bbf1b758de069a0949d6c26c0329cd3894188124a65235aa704aeb`
- `patcher_extension.py`: `1c7b3b855278d0b95d1c1a705fa0b3c91a265f320e2f97bdeb9d15469d6d4ac7`

## Inspector media outputs

Seven additional CPU tests cover cached pixel fidelity and isolation, mixed image sizes, sampled video frames, full video reconstruction, independent audio clips, native audio decoder layout/output sample rate, saved asset/scene extraction, and unchanged original inspector output indices. No pretrained-model reconstruction was performed. Set `REFPACKS_NATIVE_AUDIO_SOURCE` to `comfy_extras/nodes_audio.py` for the native audio decoder check. Both VAE inputs are optional.

Style-role tests cover both H3 prompt formats, actual Picture indices, multi-image styles, independent numbering without new Subject IDs, image-only validation, missing aliases, saved scenes, inspector mappings and unchanged raw passthrough.

Generation mode checks cover Auto boundaries (1–4 images, 5+ video), explicit Image/Video overrides, minimum video length and temporal rounding, decoder/model incompatibility, and dropdown defaults/tooltips. Earlier VAE-selected-intent checks now select explicit modes where required.
