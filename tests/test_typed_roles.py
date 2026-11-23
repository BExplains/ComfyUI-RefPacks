"""Character/Item subjects and Motion aliases share stable native addressing."""
import copy
from pathlib import Path
import tempfile
import unittest
import test_refpacks as h

common=__import__(h.shared_package.__name__+'.core',fromlist=['unused'])


class TypedRoleTests(unittest.TestCase):
    def refs(self):
        refs=common.add_pack(h.pack(name='woman',count=2),role='character')
        return common.add_pack(h.pack(name='microphone'),refs,'item')

    def test_character_and_item_compile_to_distinct_native_subjects(self):
        refs=self.refs()
        for task in ('h3_image','h3_video'):
            compiled,blocks,items,report=h.core.assemble(refs,'<Character 1> holds <Item 1> in her hand.',task)
            self.assertIn('<Subject 1> holds <Subject 2> in her hand.',compiled)
            self.assertIn('<Subject 1> is one person',compiled)
            self.assertIn('<Picture 1> and <Picture 2>',compiled)
            self.assertIn('<Subject 2> is one object shown in <Picture 3>',compiled)
            self.assertNotIn('<Character 1>',compiled)
            self.assertNotIn('<Item 1>',compiled)
            self.assertIn('<Character 1> -> <Subject 1>',report)
            self.assertIn('<Item 1> -> <Subject 2>',report)
            self.assertEqual(len(blocks),3)
            self.assertIs(blocks[2]['latent'],refs['groups'][1]['entries'][0]['latent'])

    def test_independent_alias_counters_and_shared_subject_counter(self):
        refs=None
        for role in ('character','item','character','item'):
            refs=common.add_pack(h.pack(),refs,role)
        self.assertEqual([(g['role'],g['role_id'],g['subject_id']) for g in refs['groups']],
                         [('character',1,1),('item',1,2),('character',2,3),('item',2,4)])
        text=h.core.assemble(refs,'<Character 2> hands <Item 2> to <Character 1>, who holds <Item 1>.','h3_image')[0]
        self.assertIn('<Subject 3> hands <Subject 4> to <Subject 1>, who holds <Subject 2>.',text)

    def test_motion_uses_actual_native_video_indices(self):
        refs=common.add_pack(h.pack('video',name='background'),role='video')
        refs=common.add_pack(h.pack(),refs,'character')
        refs=common.add_pack(h.pack('video',count=2,name='dance'),refs,'motion')
        text,blocks,items,report=h.core.assemble(refs,'<Character 1> performs <Motion 1>.','h3_video')
        self.assertIn('<Subject 1> performs <Video 2> and <Video 3>.',text)
        self.assertIn('<Motion 1>: dance -> <Video 2>, <Video 3>',report)
        self.assertIn('requested motion, poses',text)
        self.assertEqual([i['type'] for i in items],['image','video','video','video'])
        with self.assertRaisesRegex(ValueError,'still-image references only'):
            h.core.assemble(refs,'<Character 1> performs <Motion 1>.','h3_image')

    def test_legacy_subject_ids_and_append_stability(self):
        refs=common.add_pack(h.pack(),role='subject',role_id=4)
        refs['groups'][0].pop('subject_id') # Legacy saved set.
        refs=common.add_pack(h.pack(),refs,'character')
        refs=common.add_pack(h.pack(),refs,'item')
        self.assertEqual(common.subject_numbers(refs),{0:4,1:1,2:2})
        extended=common.add_pack(h.pack(),refs,'subject')
        self.assertEqual(extended['groups'][-1]['role_id'],3)
        self.assertEqual(common.subject_numbers(extended),{0:4,1:1,2:2,3:3})
        self.assertEqual(common.subject_numbers(refs),{0:4,1:1,2:2})
        text=h.core.assemble(extended,'<Subject 4> watches <Character 1> hold <Item 1>.','h3_image')[0]
        self.assertIn('<Subject 4> watches <Subject 1> hold <Subject 2>.',text)

    def test_generic_subject_definition_is_not_person_specific(self):
        refs=common.add_pack(h.pack(name='car'),role='subject')
        text=h.core.assemble(refs,'<Subject 1> is parked.','h3_video')[0]
        self.assertIn('<Subject 1> is one subject',text)
        self.assertNotIn('one person',text)
        self.assertNotIn('face, hair',text)

    def test_auto_outfit_targets_wearer_and_not_item(self):
        refs=common.add_pack(h.pack(name='microphone'),role='item')
        refs=common.add_pack(h.pack(name='woman'),refs,'character')
        refs=common.add_pack(h.pack(name='dress'),refs,'outfit')
        text=h.core.assemble(refs,'<Character 1> holds <Item 1>.','h3_video')[0]
        self.assertIn('<Subject 2> is wearing the outfit in <Picture 3>.',text)
        self.assertNotIn('<Subject 1> is wearing',text)
        item_only=common.add_pack(h.pack(),role='item')
        item_only=common.add_pack(h.pack(),item_only,'outfit')
        text=h.core.assemble(item_only,'<Item 1> rests on a table.','h3_image')[0]
        self.assertNotIn('wearing the outfit',text)
        multiple=common.add_pack(h.pack(),refs,'character')
        self.assertNotIn('wearing the outfit',h.core.assemble(multiple,'<Character 1> greets <Character 2>.','h3_video')[0])

    def test_unknown_typed_aliases_and_incompatible_media_fail(self):
        for alias in ('<Character 2>','<Item 2>','<Motion 1>'):
            with self.assertRaisesRegex(ValueError,'missing'):
                h.core.assemble(self.refs(),alias,'h3_video')
        for role in ('character','item'):
            with self.assertRaises(ValueError):common.add_pack(h.pack('video'),role=role)
        with self.assertRaises(ValueError):common.add_pack(h.pack(),role='motion')

    def test_inspector_and_saved_collection_keep_same_aliases(self):
        refs=self.refs();refs=common.add_pack(h.pack('video'),refs,'motion')
        prompt='<Character 1> holds <Item 1> while moving as shown in <Motion 1>.'
        expected=h.core.assemble(refs,prompt,'h3_video')[0]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scene.references.safetensors'
            h.storage.save_references(refs,path)
            loaded=h.storage.load_references(path)
        self.assertEqual(common.subject_numbers(loaded),common.subject_numbers(refs))
        result=h.nodes.BExplainsInspectRefPacks().inspect(loaded,prompt,'H3 Video')
        self.assertEqual(result['result'][2],expected)
        self.assertIn('<Character 1> -> <Subject 1> -> <Picture 1>, <Picture 2>',result['result'][1])
        self.assertIn('<Item 1> -> <Subject 2> -> <Picture 3>',result['result'][1])
        self.assertIn('<Motion 1> -> <Video 1>',result['result'][1])
        extended=common.add_pack(h.pack(),loaded,'character')
        self.assertEqual(extended['groups'][-1]['role_id'],2)
        self.assertEqual(extended['groups'][-1]['subject_id'],3)

    def test_raw_mode_still_bypasses_typed_alias_rewriting(self):
        raw='description:\n<Character 1> holds <Item 1>.'
        self.assertEqual(h.core.assemble(self.refs(),raw,'h3_image',allow_raw_prompt=True)[0],raw)

    def test_duplicate_subject_id_rejected_across_different_roles(self):
        refs=self.refs();refs['groups'][1]['subject_id']=refs['groups'][0]['subject_id']
        with self.assertRaisesRegex(ValueError,'unique across'):
            common.validate_references(refs)
