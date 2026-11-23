"""Routing checks only: tensor shape stand-ins, no model inference."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("bexplains_refpacks_nodes", Path(__file__).resolve().parents[1] / "nodes.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ShapeTensor:
    is_nested = False

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)

    def unsqueeze(self, axis):
        return ShapeTensor(self.shape[:axis] + (1,) + self.shape[axis:])


class ReferenceRoutingTests(unittest.TestCase):
    def setUp(self):
        self.node = module.BExplainsSetH3ReferenceLatent()
        self.embedding = object()
        self.z = ShapeTensor((1, 24, 1, 64, 96))

    def test_chain_preserves_existing_references_without_mutating_branches(self):
        original_ref = {"kind": "image", "name": "existing"}
        source = [[self.embedding, {"minimax_refs": [original_ref], "strength": 0.7}]]
        first = self.node.apply(source, {"samples": self.z})[0]
        second_z = ShapeTensor((1, 24, 1, 32, 48))
        second = self.node.apply(first, {"samples": second_z})[0]
        self.assertEqual(len(source[0][1]["minimax_refs"]), 1)
        self.assertEqual(len(first[0][1]["minimax_refs"]), 2)
        refs = second[0][1]["minimax_refs"]
        self.assertEqual(len(refs), 3)
        self.assertIs(refs[0], original_ref)
        self.assertIs(refs[1]["latent"], self.z)
        self.assertIs(refs[2]["latent"], second_z)
        self.assertIs(second[0][0], self.embedding)
        self.assertEqual(second[0][1]["strength"], 0.7)
        self.assertEqual((refs[1]["latent_h"], refs[1]["latent_w"]), (64, 96))

    def test_each_conditioning_entry_retains_its_own_metadata(self):
        source = [[self.embedding, {"start_percent": 0.2}], [object(), {"end_percent": 0.8}]]
        output = self.node.apply(source, {"samples": self.z})[0]
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0][1]["start_percent"], 0.2)
        self.assertEqual(output[1][1]["end_percent"], 0.8)
        self.assertNotIn("minimax_refs", source[0][1])
        self.assertIsNot(output[0][1]["minimax_refs"], output[1][1]["minimax_refs"])

    def test_four_dimensional_still_reference(self):
        output = self.node.apply([[self.embedding, {}]], {"samples": ShapeTensor((1, 24, 64, 96))})[0]
        self.assertEqual(output[0][1]["minimax_refs"][0]["latent"].shape, self.z.shape)

    def test_rejects_incompatible_or_ambiguous_latents(self):
        for shape in [(1, 128, 26, 46), (1, 16, 1, 64, 64), (2, 24, 1, 64, 64), (1, 24, 5, 64, 64), (1, 24, 1, 0, 64)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.node.apply([[self.embedding, {}]], {"samples": ShapeTensor(shape)})

    def test_registration_uses_bexplains_namespace(self):
        self.assertIs(module.NODE_CLASS_MAPPINGS["BExplainsSetH3ReferenceLatent"], type(self.node))
        self.assertEqual(self.node.CATEGORY, "BExplains/RefPacks")


if __name__ == "__main__":
    unittest.main()
