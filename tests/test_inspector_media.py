"""Media extraction preserves saved pixels and keeps existing inspector sockets."""
import ast
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import test_refpacks as h
from bexplains_refpacks.media import extract_media


class InspectorMediaTests(unittest.TestCase):
    def test_mixed_size_images_remain_exact_and_independent(self):
        pack = h.pack(count=2)
        pack['entries'][1]['vision'] = torch.rand(1, 16, 48, 3)
        refs = h.core.add_pack(pack, role='reference')
        images, audio, report = extract_media(refs)
        self.assertEqual([tuple(v.shape) for v in images], [(1,32,32,3),(1,16,48,3)])
        self.assertEqual(audio, [])
        for output, entry in zip(images, pack['entries']):
            self.assertTrue(torch.equal(output, entry['vision']))
            output.zero_()
            self.assertGreater(entry['vision'].sum(), 0)

    def test_cached_video_frames_and_missing_audio_are_explicit(self):
        refs = h.core.add_pack(h.pack('video'), role='video')
        refs = h.core.add_pack(h.audio_pack(), refs, 'audio')
        images, audio, report = extract_media(refs)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape[0], 2)
        self.assertEqual(audio, [])
        self.assertIn('not the complete clip', report)
        self.assertIn('audio_vae', report)
        self.assertEqual(extract_media({'groups': []})[:2], ([], []))

    def test_audio_clips_stay_separate_and_use_native_decoder(self):
        refs = h.core.add_pack(h.audio_pack(), role='audio')
        refs = h.core.add_pack(h.audio_pack('second'), refs, 'audio')
        before = refs['groups'][0]['entries'][0]['audio_latent'].clone()
        calls = []
        module = types.ModuleType('comfy_extras.nodes_audio')
        def decode(vae, samples):
            calls.append(samples)
            samples['samples'].zero_()
            return {'waveform': torch.ones(1,2,len(calls)*80), 'sample_rate': 48000}
        module.vae_decode_audio = decode
        with patch.dict(sys.modules, {module.__name__: module}):
            images, audio, report = extract_media(refs, types.SimpleNamespace(audio_sample_rate=32000))
        self.assertEqual(images, [])
        self.assertEqual([a['waveform'].shape[-1] for a in audio], [80,160])
        self.assertEqual([a['sample_rate'] for a in audio], [48000,48000])
        self.assertTrue(torch.equal(before, refs['groups'][0]['entries'][0]['audio_latent']))
        self.assertIn('reconstructed', report)
        with self.assertRaisesRegex(ValueError, 'matching audio VAE'):
            extract_media(refs, object())

    def test_saved_pack_and_scene_pixels_can_be_extracted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            original = h.pack(count=2)
            h.storage.save_pack(original, path/'asset.safetensors')
            refs = h.core.add_pack(h.storage.load_pack(path/'asset.safetensors'), role='reference')
            h.storage.save_references(refs, path/'scene.safetensors')
            loaded = h.storage.load_references(path/'scene.safetensors')
            images, _, _ = extract_media(loaded)
            for image, entry in zip(images, original['entries']):
                self.assertTrue(torch.equal(image, entry['vision']))

    def test_both_inspectors_append_sockets_without_changing_existing_outputs(self):
        refs = h.core.add_pack(h.pack(), role='reference')
        shared = h.shared_package.nodes.BExplainsInspectReferences
        h3 = h.nodes.BExplainsInspectRefPacks
        for cls, old_types in ((shared, ('BEXPLAINS_REFERENCES','STRING')),
                               (h3, ('BEXPLAINS_REFERENCES','STRING','STRING'))):
            result = cls().inspect(refs)['result']
            self.assertIs(result[0], refs)
            self.assertEqual(cls.RETURN_TYPES, old_types + ('IMAGE','AUDIO'))
            self.assertEqual(cls.OUTPUT_IS_LIST, (False,)*len(old_types) + (True,True))
            self.assertEqual(len(result[-2]), 1)
            self.assertEqual(result[-1], [])
            self.assertIn('audio_vae', cls.INPUT_TYPES()['optional'])
            self.assertIn('vae', cls.INPUT_TYPES()['optional'])

    def test_visual_vae_returns_full_video_batch_without_mutating_latents(self):
        pack = h.pack('video')
        original = pack['entries'][0]['latent'].clone()
        frames = torch.rand(1,22,32,48,3)
        def decode(latent):
            latent.fill_(99)
            return frames
        refs = h.core.add_pack(pack, role='video')
        images, _, report = extract_media(refs, vae=types.SimpleNamespace(decode=decode))
        self.assertEqual(len(images),1)
        self.assertEqual(tuple(images[0].shape),(22,32,48,3))
        self.assertTrue(torch.equal(images[0],frames.reshape(22,32,48,3)))
        self.assertTrue(torch.equal(pack['entries'][0]['latent'],original))
        self.assertIn('VAE reconstructions',report)

    def test_native_audio_decode_layout_and_output_sample_rate(self):
        source = os.environ.get('REFPACKS_NATIVE_AUDIO_SOURCE')
        if not source:
            self.skipTest('Set REFPACKS_NATIVE_AUDIO_SOURCE for native audio decode parity.')
        tree = ast.parse(Path(source).read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'vae_decode_audio')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), source, 'exec'), namespace)
        module = types.ModuleType('comfy_extras.nodes_audio')
        module.vae_decode_audio = namespace['vae_decode_audio']
        waveform = torch.linspace(-0.1,0.1,160).reshape(1,80,2)
        vae = types.SimpleNamespace(audio_sample_rate=32000, audio_sample_rate_output=48000,
                                    decode=lambda latent: waveform.clone())
        refs = h.core.add_pack(h.audio_pack(), role='audio')
        with patch.dict(sys.modules, {module.__name__: module}):
            _, audio, _ = extract_media(refs, vae)
        self.assertEqual(audio[0]['sample_rate'],48000)
        self.assertTrue(torch.equal(audio[0]['waveform'],waveform.movedim(-1,1)))
