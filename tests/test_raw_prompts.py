"""Raw H3 prompt opt-in, exact passthrough and inspector/encoder parity."""
import json
import unittest
import test_refpacks as h


class RawPromptTests(unittest.TestCase):
    def setUp(self):
        self.refs=h.core.add_pack(h.pack(),role='subject')
        self.refs=h.core.add_pack(h.pack(name='dress'),self.refs,role='outfit')
        self.raw='\n  subject_definitions:\n<Subject 99> is the person in <Picture 1>.\n\nsummary:\n<Subject 99> wears <Outfit 1>.\ndetailed_description:\nMy exact instruction.  \n'

    def test_opt_in_preserves_exact_prompt_and_reference_payloads(self):
        for task in ('h3_image','h3_video'):
            normal=h.core.assemble(self.refs,'<Subject 1> smiles',task)
            raw=h.core.assemble(self.refs,self.raw,task,auto_outfit=True,allow_raw_prompt=True)
            self.assertEqual(raw[0],self.raw)
            self.assertEqual(raw[3],normal[3])
            for actual,expected in zip(raw[1],normal[1]):
                self.assertIs(actual['latent'],expected['latent'])
            for actual,expected in zip(raw[2],normal[2]):
                self.assertIs(actual['data'],expected['data'])
            self.assertIn('<Outfit 1>',raw[0]) # No alias replacement.
            self.assertEqual(raw[0].count('subject_definitions:'),1)

    def test_structured_text_requires_opt_in(self):
        for raw in (self.raw,'summary:\nA camera move','description: A portrait',json.dumps({'description':'A portrait'})):
            self.assertTrue(h.core.is_raw_prompt(raw))
            with self.assertRaisesRegex(ValueError,'Enable Allow Raw Prompt'):
                h.core.assemble(self.refs,raw,'h3_image')

    def test_json_is_not_parsed_and_reserialized(self):
        raw=' \n{ "subject_definitions" : "<Subject 9> is shown in <Picture 1>",\n "description" : "Leave \\n and spaces untouched" }\n '
        self.assertTrue(h.core.is_raw_prompt(raw))
        self.assertEqual(h.core.assemble(self.refs,raw,'h3_image',allow_raw_prompt=True)[0],raw)

    def test_plain_instructions_compile_identically_with_toggle_on_or_off(self):
        for prompt in ('<Subject 1> smiles.', 'Use a summary of the scene as inspiration.', '{"ordinary":"not an H3 section"}'):
            self.assertFalse(h.core.is_raw_prompt(prompt))
            off=h.core.assemble(self.refs,prompt,'h3_video',allow_raw_prompt=False)
            on=h.core.assemble(self.refs,prompt,'h3_video',allow_raw_prompt=True)
            self.assertEqual(off[0],on[0])
            self.assertIn('wearing the outfit',on[0])

    def test_raw_mode_keeps_model_and_reference_validation(self):
        refs=h.core.add_pack(h.pack('video'),role='video')
        with self.assertRaisesRegex(ValueError,'still-image references only'):
            h.core.assemble(refs,self.raw,'h3_image',allow_raw_prompt=True)
        for prompt in ('',' \n '):
            with self.assertRaisesRegex(ValueError,'shot prompt'):
                h.core.assemble(self.refs,prompt,'h3_video',allow_raw_prompt=True)

    def test_legacy_and_new_encoders_tokenize_raw_exactly(self):
        for image in (True,False):
            for new in (False,True):
                clip=h.RecordingClip()
                if new:
                    vae=type('VAE',(),{'first_stage_model':type('MiniMaxH3ImageVAE' if image else 'MiniMaxH3VideoVAE',(),{})()})()
                    result=h.nodes.BExplainsRefPackToH3().compile(h.FakeModel(image),clip,self.refs,self.raw,'negative unchanged',64,64,1 if image else 5,True,vae,False,allow_raw_prompt=True)
                else:
                    result=h.nodes.BExplainsH3RefPackConditioning().compile(h.FakeModel(image),clip,self.refs,self.raw,'negative unchanged',64,64,1 if image else 5,True,False,allow_raw_prompt=True)
                self.assertEqual(clip.calls[0][0],self.raw)
                self.assertEqual(clip.calls[1][0],'negative unchanged')
                self.assertEqual(result[1][0][1]['bexplains_refpacks']['prompt'],self.raw)
                self.assertEqual(len(result[1][0][1]['minimax_refs']),2)

    def test_inspector_uses_same_raw_prompt_path(self):
        inspector=h.nodes.BExplainsInspectRefPacks()
        for fmt in ('H3 Image','H3 Video'):
            result=inspector.inspect(self.refs,self.raw,fmt,True,allow_raw_prompt=True)
            self.assertEqual(result['result'][2],self.raw)
            denied=inspector.inspect(self.refs,self.raw,fmt,True)
            self.assertIn('Enable Allow Raw Prompt',denied['ui']['text'][0])
            self.assertEqual(denied['result'][2],'')

    def test_toggle_defaults_to_false_on_all_h3_prompt_nodes(self):
        for cls in (h.nodes.BExplainsRefPackToH3,h.nodes.BExplainsH3RefPackConditioning,h.nodes.BExplainsInspectRefPacks):
            self.assertFalse(cls.INPUT_TYPES()['optional']['allow_raw_prompt'][1]['default'])
