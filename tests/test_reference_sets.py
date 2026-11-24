"""Self-contained saved reference sets: fidelity, ordering, portability and protection."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import test_refpacks as h

common=__import__(h.shared_package.__name__+'.core',fromlist=['unused'])
nodes=__import__(h.shared_package.__name__+'.nodes',fromlist=['unused'])
storage=h.storage


def assert_tree(test,actual,expected):
    if isinstance(expected,torch.Tensor):
        test.assertEqual(actual.dtype,expected.dtype)
        test.assertTrue(torch.equal(actual,expected))
    elif isinstance(expected,dict):
        test.assertEqual(set(actual),set(expected))
        for key in expected:assert_tree(test,actual[key],expected[key])
    elif isinstance(expected,list):
        test.assertEqual(len(actual),len(expected))
        for a,e in zip(actual,expected):assert_tree(test,a,e)
    else:test.assertEqual(actual,expected)


class ReferenceSetTests(unittest.TestCase):
    def mixed_set(self):
        first=h.pack(count=2,name='Mai');first['source_file']='people/mai.safetensors'
        refs=common.add_pack(first,role='subject')
        refs=common.add_pack(h.pack(name='Stage'),refs,'location')
        refs=common.add_pack(h.pack(name='Dress'),refs,'outfit')
        refs=common.add_pack(h.pack('video',name='Walk'),refs,'video')
        refs=common.add_pack(h.audio_pack(),refs,'audio')
        refs['name']='Opening scene'
        refs['note']='Preserve Unicode: Bjørn / 舞台'
        return refs

    def test_full_roundtrip_preserves_order_ids_metadata_tensors_and_prompt(self):
        refs=self.mixed_set();before=copy.deepcopy(refs)
        prompt='<Subject 1> wears <Outfit 1> in <Location 1>, moving as shown in <Video 1>, with <Audio 1>.'
        expected=h.core.assemble(refs,prompt,'h3_video')[0]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scene.references.safetensors'
            storage.save_references(refs,path)
            loaded=storage.load_references(path)
            with safe_open(str(path),framework='pt',device='cpu') as f:
                manifest=json.loads(f.metadata()[storage.REFERENCES_META_KEY])
                self.assertEqual(manifest['name'],'scene')
                self.assertEqual(manifest['format'],'bexplains.references')
        assert_tree(self,loaded,refs)
        assert_tree(self,refs,before)
        self.assertEqual(h.core.assemble(loaded,prompt,'h3_video')[0],expected)
        self.assertEqual([(g['role'],g['role_id']) for g in loaded['groups']],[(g['role'],g['role_id']) for g in refs['groups']])

    def test_does_not_depend_on_original_files_and_can_append_after_load(self):
        with tempfile.TemporaryDirectory() as d:
            original=Path(d)/'original.safetensors';storage.save_pack(h.pack(),original)
            pack=storage.load_pack(original);pack['source_file']=str(original)
            refs=common.add_pack(pack,role='subject')
            combined=Path(d)/'bundle.references.safetensors'
            storage.save_references(refs,combined);original.unlink()
            loaded=storage.load_references(combined)
            extended=common.add_pack(h.pack(name='another'),loaded,'subject')
            self.assertEqual([g['role_id'] for g in extended['groups']],[1,2])
            self.assertEqual(len(loaded['groups']),1)

    def test_shared_formats_and_legacy_h3_groups_are_preserved(self):
        for target,z in [('flux2',torch.rand(1,128,2,2)),('ltxv',torch.rand(1,128,1,2,2))]:
            pack={'format':common.FORMAT,'version':common.VERSION,'target':target,'name':target,
                  'encoder_class':'test.Encoder','entries':[{'kind':'image','latent':z,'vision':torch.rand(1,64,64,3)}]}
            refs=common.add_pack(pack,role='reference')
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'scene.references.safetensors';storage.save_references(refs,path)
                assert_tree(self,storage.load_references(path),refs)
        refs=common.add_pack(h.pack(),role='subject')
        refs['groups'][0].pop('target') # Older H3 groups have only source_target.
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'legacy.references.safetensors';storage.save_references(refs,path)
            assert_tree(self,storage.load_references(path),refs)

    def test_overwrite_refusal_keeps_original_and_explicit_overwrite_replaces(self):
        refs=self.mixed_set()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scene.references.safetensors';storage.save_references(refs,path)
            original=path.read_bytes()
            changed=copy.deepcopy(refs);changed['groups'][0]['name']='New character'
            with self.assertRaisesRegex(ValueError,'overwrite'):
                storage.save_references(changed,path)
            self.assertEqual(path.read_bytes(),original)
            storage.save_references(changed,path,True)
            self.assertEqual(storage.load_references(path)['groups'][0]['name'],'New character')
            self.assertEqual(list(Path(d).glob('.refpack-*')),[])

    def test_atomic_collision_and_failed_write_cleanup(self):
        refs=self.mixed_set()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scene.references.safetensors'
            # Simulate another writer winning after the existence check.
            def racing_link(source,destination):
                Path(destination).write_bytes(b'other writer')
                raise FileExistsError('concurrent file')
            with patch.object(storage.os,'link',side_effect=racing_link),self.assertRaises(FileExistsError):
                storage.save_references(refs,path)
            self.assertEqual(path.read_bytes(),b'other writer')
            self.assertEqual(list(Path(d).glob('.refpack-*')),[])
            with patch.object(storage,'save_file',side_effect=RuntimeError('failed write')),self.assertRaises(RuntimeError):
                storage.save_references(refs,path,True)
            self.assertEqual(path.read_bytes(),b'other writer')
            self.assertEqual(list(Path(d).glob('.refpack-*')),[])

    def test_invalid_sets_and_wrong_file_types_are_rejected(self):
        invalid=[{}, {'target':None,'groups':[]},self.mixed_set()]
        invalid[-1]['groups'].append(copy.deepcopy(invalid[-1]['groups'][0]))
        for refs in invalid:
            with self.assertRaises(ValueError):common.validate_references(refs)
        with tempfile.TemporaryDirectory() as d:
            individual=Path(d)/'pack.safetensors';storage.save_pack(h.pack(),individual)
            with self.assertRaisesRegex(ValueError,'not a saved RefPacks reference set'):storage.load_references(individual)
            scene=Path(d)/'scene.references.safetensors';storage.save_references(self.mixed_set(),scene)
            with self.assertRaisesRegex(ValueError,'not a BExplains RefPack'):storage.load_pack(scene)
            save_file({},str(scene),metadata={storage.REFERENCES_META_KEY:json.dumps({'format':storage.REFERENCES_FORMAT,'version':999})})
            with self.assertRaisesRegex(ValueError,'Unsupported'):storage.load_references(scene)

    def test_save_load_nodes_suffix_paths_passthrough_cache_and_registration(self):
        refs=self.mixed_set()
        with tempfile.TemporaryDirectory() as d,patch.object(nodes,'REFERENCES_ROOT',Path(d)):
            saver=nodes.BExplainsSaveRefPacksReferences();loader=nodes.BExplainsLoadRefPacksReferences()
            for name in ('scene','second.safetensors','subdir/third.references.safetensors'):
                result=saver.save(refs,name,False)['result']
                self.assertIs(result[0],refs)
                self.assertTrue(result[1].endswith('.references.safetensors'))
                self.assertNotIn('.safetensors.references',result[1])
                assert_tree(self,loader.load(result[1])[0],refs)
                self.assertNotEqual(loader.IS_CHANGED(result[1]),'missing')
            self.assertEqual(loader.IS_CHANGED('missing.references.safetensors'),'missing')
            for name in ('../outside.references.safetensors','/tmp/outside.references.safetensors'):
                with self.assertRaises(ValueError):saver.save(refs,name,True)
                with self.assertRaises(ValueError):loader.load(name)
            with self.assertRaises(ValueError):saver.save(refs,' ',False)
        self.assertIn('BExplainsSaveRefPacksReferences',h.shared_package.NODE_CLASS_MAPPINGS)
        self.assertIn('BExplainsLoadRefPacksReferences',h.shared_package.NODE_CLASS_MAPPINGS)

    def test_reference_sets_have_separate_loader_choices(self):
        def files(category):
            return ['nested/segment.references.safetensors'] if category=='bexplains_refpacks_references' else ['actual.safetensors','references/scene.references.safetensors']
        with patch.object(nodes.folder_paths,'get_filename_list',side_effect=files):
            self.assertEqual(nodes._files(), ['actual.safetensors'])
            self.assertEqual(nodes._reference_files(), ['nested/segment.references.safetensors'])
        with patch.object(nodes.folder_paths,'get_filename_list',return_value=[]):
            self.assertEqual(nodes._files(), [])
            self.assertEqual(nodes._reference_files(), [])
