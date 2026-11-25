import copy
from pathlib import Path
import tempfile
import unittest

import test_refpacks as h
from test_reference_sets import assert_tree, common, nodes


class MergeBundleTests(unittest.TestCase):
    def test_collision_remapping_and_source_immutability(self):
        first = common.add_pack(h.pack(name='First'), role='character')
        first = common.add_pack(h.pack(name='First style'), first, 'style')
        second = common.add_pack(h.pack(name='Second'), role='subject')
        second = common.add_pack(h.pack(name='Third'), second, 'character')
        second = common.add_pack(h.pack(name='Second style'), second, 'style')
        before = copy.deepcopy([first, second])
        result = common.merge_bundles([first, second])
        self.assertEqual([g['role_id'] for g in result['groups']], [1, 1, 3, 2, 2])
        self.assertEqual(list(common.subject_numbers(result).values()), [1, 3, 2])
        self.assertEqual([g['name'] for g in result['groups']],
                         ['First', 'First style', 'Second', 'Third', 'Second style'])
        assert_tree(self, [first, second], before)
        self.assertIs(result['groups'][0]['entries'][0]['latent'], first['groups'][0]['entries'][0]['latent'])
        result['groups'][0]['entries'][0]['new_metadata'] = 'isolated'
        self.assertNotIn('new_metadata', first['groups'][0]['entries'][0])
        compiled = h.core.assemble(result, '<Character 2> in the style of <Style 2>', 'h3_video')[0]
        self.assertNotIn('<Style 2>', compiled)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'merged.references.safetensors'
            h.storage.save_references(result, path)
            assert_tree(self, h.storage.load_references(path), result)

    def test_slots_are_numeric_and_chains_accept_merged_output(self):
        bundles = [common.add_pack(h.pack(name=str(i)), role='subject') for i in range(4)]
        result = nodes.BExplainsMergeRefPackBundles().merge(
            bundles[0], bundles[1], bundle_10=bundles[3], bundle_3=bundles[2])[0]
        self.assertEqual([g['name'] for g in result['groups']], ['0', '1', '2', '3'])
        self.assertEqual([g['role_id'] for g in result['groups']], [1, 2, 3, 4])
        result = common.add_pack(h.pack(name='Append'), result, 'subject')
        self.assertEqual(result['groups'][-1]['role_id'], 5)
        self.assertIn('BExplainsMergeRefPackBundles', h.shared_package.NODE_CLASS_MAPPINGS)
        schema = nodes.BExplainsMergeRefPackBundles.INPUT_TYPES()
        self.assertEqual(len(schema['optional']), 62)
        self.assertEqual(schema['optional']['bundle_64'], ('BEXPLAINS_REFERENCES',))

    def test_media_roles_targets_and_duplicate_inputs(self):
        image = common.add_pack(h.pack(), role='subject')
        media = common.add_pack(h.pack('video'), role='motion')
        media = common.add_pack(h.audio_pack(), media, 'audio')
        result = common.merge_bundles([image, media, image])
        self.assertEqual(result['target'], 'h3_video')
        self.assertEqual(len(result['groups']), 4)
        self.assertEqual(result['groups'][-1]['role_id'], 2)
        foreign = copy.deepcopy(image)
        foreign['groups'][0]['target'] = 'flux2'
        self.assertEqual(common.merge_bundles([image, foreign])['target'], 'mixed')

    def test_invalid_and_empty_bundles_are_rejected(self):
        image = common.add_pack(h.pack(), role='subject')
        with self.assertRaisesRegex(ValueError, 'at least two'):
            common.merge_bundles([image])
        for invalid in (None, {}, common.empty_set()):
            with self.assertRaises(ValueError):
                common.merge_bundles([image, invalid])
