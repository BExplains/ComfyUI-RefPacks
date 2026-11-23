import types
import unittest
import test_refpacks as h
from test_receivers import MiniMaxH3ImageVAE, MiniMaxH3VideoVAE, setUpModule


class GenerationModeTests(unittest.TestCase):
    def run_mode(self, length, mode='Auto', image_model=False, image_vae=False):
        refs=h.core.add_pack(h.pack(),role='subject')
        vae=types.SimpleNamespace(first_stage_model=(MiniMaxH3ImageVAE() if image_vae else MiniMaxH3VideoVAE()))
        return h.nodes.BExplainsRefPackToH3().compile(
            h.FakeModel(image_model),h.RecordingClip(),refs,'<Subject 1> stands.','',64,64,
            length,False,vae,False,generation_mode=mode)

    def test_auto_boundary_is_length_driven_with_video_vae(self):
        for length in (1,2,3,4):
            result=self.run_mode(length)
            video=result[3]['samples'].unbind()[0]
            self.assertEqual(tuple(video.shape),(length,24,1,4,4))
        for length,frames in ((5,5),(6,22),(22,22),(23,39)):
            result=self.run_mode(length)
            self.assertIn('output intent: h3_video',result[4])
            self.assertIn(f'output frames: {frames}',result[4])

    def test_explicit_modes_override_length(self):
        for length in (1,4):
            self.assertIn('output frames: 5',self.run_mode(length,'Video')[4])
        result=self.run_mode(73,'Image')
        self.assertEqual(result[3]['samples'].unbind()[0].shape[2],1)
        self.assertIn('exceeds the 4-image limit',result[4])
        self.assertEqual(self.run_mode(3,'Image',True,True)[3]['samples'].shape[0],3)

    def test_invalid_mode_and_video_incompatibility(self):
        with self.assertRaisesRegex(ValueError,'must be Auto'):
            self.run_mode(1,'Other')
        with self.assertRaisesRegex(ValueError,'cannot generate video'):
            self.run_mode(5,image_model=True)
        with self.assertRaisesRegex(ValueError,'Video VAE'):
            self.run_mode(5,image_vae=True)
        with self.assertRaisesRegex(ValueError,'Video VAE'):
            self.run_mode(1,'Video',image_vae=True)

    def test_both_schemas_offer_documented_auto_default(self):
        for node in (h.nodes.BExplainsRefPackToH3,h.nodes.BExplainsH3RefPackConditioning):
            options,details=node.INPUT_TYPES()['optional']['generation_mode']
            self.assertEqual(options,['Auto','Image','Video'])
            self.assertEqual(details['default'],'Auto')
            self.assertIn('1-4',details['tooltip'])
            self.assertIn('5 frames minimum',details['tooltip'])
