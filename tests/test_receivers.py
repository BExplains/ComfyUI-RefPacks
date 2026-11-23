"""Integration checks with real tensors; native interfaces use recording doubles."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
import torch
import test_refpacks as h

shared = h.shared_package
nodes = __import__(shared.__name__ + '.shared_nodes', fromlist=['unused'])
common = __import__(shared.__name__ + '.core', fromlist=['unused'])
ltxv = __import__(shared.__name__ + '.adapters.ltxv', fromlist=['unused'])
flux2 = __import__(shared.__name__ + '.adapters.flux2', fromlist=['unused'])


class Flux2: pass
class LTXV: pass
class LTXAV: pass
class MiniMaxH3VideoVAE:
    def __init__(self, still_frame=None): self.still_frame = still_frame
class MiniMaxH3ImageVAE: pass


class VAE:
    def __init__(self, family):
        self.family = family
        self.calls = []
        self.first_stage_model = object()
    def encode(self, pixels):
        self.calls.append(pixels.clone())
        n, height, width, _ = pixels.shape
        shape = (1,128,height//16,width//16) if self.family == 'Flux 2' else (1,128,(n-1)//8+1,height//32,width//32)
        return torch.full(shape, float(pixels.mean()))


def av_latent(width, height, latent_t, audio_t, batch_size=1):
    # Use native NestedTensor when supplied, otherwise a recording interface double.
    from comfy.nested_tensor import NestedTensor
    return {'samples': NestedTensor([torch.zeros(batch_size,24,latent_t,height//16,width//16),torch.zeros(batch_size,32,2,audio_t)])}


def setUpModule():
    import importlib.util, sys
    source = os.environ.get('REFPACKS_NATIVE_NESTED_SOURCE')
    if source:
        spec = importlib.util.spec_from_file_location('comfy.nested_tensor', source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        class TestNestedTensor:
            is_nested = True
            def __init__(self, tensors): self.tensors = list(tensors)
            def unbind(self): return self.tensors
        module = types.ModuleType('comfy.nested_tensor')
        module.NestedTensor = TestNestedTensor
        sys.modules[module.__name__] = module
    h.native._av_latent = av_latent


class ReceiverTests(unittest.TestCase):
    def prepare(self, family, count=1):
        vae=VAE(family)
        pack=nodes.BExplainsEncodeImageReferences().encode(vae,torch.rand(count,64,64,3),family,'reference',64,64)[0]
        return vae, pack, common.add_pack(pack,role='reference')

    def test_flux_prepare_roundtrip_and_receive_multiple_references(self):
        vae,pack,refs=self.prepare('Flux 2',2)
        with tempfile.TemporaryDirectory() as d:
            h.storage.save_pack(pack,Path(d)/'flux.safetensors')
            loaded=h.storage.load_pack(Path(d)/'flux.safetensors')
        self.assertEqual(loaded['target'],'flux2')
        self.assertEqual(loaded['encoder_class'],pack['encoder_class'])
        self.assertTrue(torch.equal(pack['entries'][1]['latent'],loaded['entries'][1]['latent']))
        refs=common.add_pack(loaded,role='reference')
        clip=h.RecordingClip()
        result=nodes.BExplainsRefPackToImage().encode(types.SimpleNamespace(model=Flux2()),clip,vae,refs,'Use image 1 and image 2','',64,64,4,4.0)
        self.assertEqual(tuple(result[3]['samples'].shape),(4,128,4,4))
        self.assertIs(result[4],vae)
        self.assertEqual(len(result[1][0][1]['reference_latents']),2)
        self.assertEqual(result[1][0][1]['guidance'],4)
        self.assertEqual(len(vae.calls),2) # No VAE calls in receiving stage.
        self.assertEqual(clip.calls[0][0],'Use image 1 and image 2')

    def test_flux_native_empty_latent_parity(self):
        source=os.environ.get('REFPACKS_NATIVE_FLUX_SOURCE')
        if not source:self.skipTest('Set REFPACKS_NATIVE_FLUX_SOURCE for upstream source parity.')
        native_path=Path(source)
        tree=ast.parse(native_path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EmptyFlux2LatentImage')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        method.decorator_list=[];method.returns=None
        ns={'torch':torch,'comfy':h.comfy,'io':types.SimpleNamespace(NodeOutput=lambda *x:x)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(native_path),'exec'),ns)
        expected=ns['execute'](None,64,96,3)[0]['samples']
        _,_,latent,_=flux2.receive([],[],[{'kind':'image','latent':torch.zeros(1,128,4,4)}],64,96,3,4,'cpu')
        self.assertTrue(torch.equal(expected,latent['samples']))

    def test_ltx_image_prefix_native_parity(self):
        vae,pack,refs=self.prepare('LTXV 0.9.8')
        clip=h.RecordingClip()
        result=nodes.BExplainsRefPackToVideo().encode(types.SimpleNamespace(model=LTXV()),clip,vae,refs,'Camera pans','',64,64,17,25,0.8)
        z=result[3]['samples']; mask=result[3]['noise_mask']
        self.assertEqual(tuple(z.shape),(1,128,3,2,2))
        self.assertTrue(torch.equal(z[:,:,:1],pack['entries'][0]['latent']))
        self.assertTrue(torch.equal(z[:,:,1:],torch.zeros_like(z[:,:,1:])))
        self.assertTrue(torch.allclose(mask[:,:,:1],torch.full_like(mask[:,:,:1],0.2)))
        self.assertTrue(torch.equal(mask[:,:,1:],torch.ones_like(mask[:,:,1:])))
        self.assertEqual(result[1][0][1]['frame_rate'],25)
        self.assertEqual(result[2][0][1]['frame_rate'],25)
        self.assertEqual(len(vae.calls),1)
        # Execute actual upstream LTXVImgToVideo's implementation for equivalent input.
        source=os.environ.get('REFPACKS_NATIVE_LTX_SOURCE')
        if not source:return
        path=Path(source)
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LTXVImgToVideo')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        method.decorator_list=[];method.returns=None
        ns={'torch':torch,'comfy':h.comfy,'io':types.SimpleNamespace(NodeOutput=lambda *x:x)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),ns)
        actual=ns['execute'](None,[],[],pack['entries'][0]['vision'],VAE('LTXV 0.9.8'),64,64,17,1,0.8)[2]
        self.assertTrue(torch.equal(z,actual['samples']))
        self.assertTrue(torch.equal(mask,actual['noise_mask']))

    def test_ltx_video_padding_and_prefix_truncation(self):
        vae=VAE('LTXV 0.9.8')
        frames=torch.rand(10,64,64,3)
        pack=nodes.BExplainsEncodeVideoReference().encode(vae,frames,'clip',25,64,64)[0]
        self.assertEqual(pack['entries'][0]['frame_count'],17)
        self.assertEqual(len(vae.calls[0]),17)
        self.assertTrue(torch.equal(vae.calls[0][-1],frames[-1]))
        refs=common.add_pack(pack,role='video')
        result=nodes.BExplainsRefPackToVideo().encode(types.SimpleNamespace(model=LTXV()),h.RecordingClip(),vae,refs,'Continue','',64,64,9,25,1)
        self.assertEqual(result[3]['samples'].shape[2],2)
        self.assertEqual(len(vae.calls),1)

    def test_foreign_packs_and_unsupported_models_fail_before_text_encoding(self):
        vae,pack,refs=self.prepare('Flux 2')
        clip=h.RecordingClip()
        with self.assertRaisesRegex(ValueError,'Re-encode'):
            nodes.BExplainsRefPackToVideo().encode(types.SimpleNamespace(model=LTXV()),clip,vae,refs,'','',64,64,17,25,1)
        self.assertEqual(clip.calls,[])
        with self.assertRaises(ValueError):
            nodes.BExplainsRefPackToVideo().encode(types.SimpleNamespace(model=LTXAV()),clip,vae,refs,'','',64,64,17,25,1)
        with self.assertRaises(ValueError):
            nodes.BExplainsEncodeImageReferences().encode(VAE('LTXV 0.9.8'),torch.rand(1,64,64,3),'Flux 2','wrong',64,64)

    def test_ltx_multiple_references_and_canvas_mismatch_fail(self):
        vae,pack,refs=self.prepare('LTXV 0.9.8',2)
        with self.assertRaisesRegex(ValueError,'one first-image'):
            nodes.BExplainsRefPackToVideo().encode(types.SimpleNamespace(model=LTXV()),h.RecordingClip(),vae,refs,'','',64,64,17,25,1)
        refs=common.add_pack({**pack,'entries':pack['entries'][:1]},role='reference')
        with self.assertRaisesRegex(ValueError,'resolution'):
            nodes.BExplainsRefPackToVideo().encode(types.SimpleNamespace(model=LTXV()),h.RecordingClip(),vae,refs,'','',96,64,17,25,1)

    def test_h3_vae_selects_intent_and_images_use_batch_not_time(self):
        refs=common.add_pack(h.pack(),role='subject')
        node=h.nodes.BExplainsRefPackToH3()
        for model_image in (False,True):
            vae=types.SimpleNamespace(first_stage_model=MiniMaxH3ImageVAE())
            for count in (1,2,3,4):
                clip=h.RecordingClip()
                result=node.compile(h.FakeModel(model_image),clip,refs,'<Subject 1> smiles','',64,64,count,True,vae,False)
                samples=result[3]['samples']
                video=samples.unbind()[0] if samples.is_nested else samples
                self.assertEqual(tuple(video.shape),(count,24,1,4,4))
                self.assertIs(result[5],vae)
                self.assertEqual('images' in clip.calls[0][1],model_image)
                self.assertEqual('minimax_ref_items' in clip.calls[0][1],not model_image)
                if not model_image:self.assertEqual(tuple(samples.unbind()[1].shape),(count,32,2,2))

    def test_h3_fallback_and_video_vae_minimum(self):
        refs=common.add_pack(h.pack(),role='subject')
        node=h.nodes.BExplainsRefPackToH3()
        for stage in (MiniMaxH3ImageVAE(),MiniMaxH3VideoVAE(still_frame=3)):
            with contextlib.redirect_stdout(io.StringIO()):
                result=node.compile(h.FakeModel(),h.RecordingClip(),refs,'<Subject 1> smiles','',64,64,175,True,types.SimpleNamespace(first_stage_model=stage),False)
            self.assertEqual(result[3]['samples'].unbind()[0].shape[0],1)
            self.assertEqual(result[3]['samples'].unbind()[0].shape[2],1)
            self.assertIn('exceeds the 4-image limit',result[4])
        vae=types.SimpleNamespace(first_stage_model=MiniMaxH3VideoVAE())
        result=node.compile(h.FakeModel(),h.RecordingClip(),refs,'<Subject 1> smiles','',64,64,4,True,vae,False)
        self.assertEqual(result[3]['samples'][0].shape[2],2) # 5 video frames, not 4 images.
        with self.assertRaisesRegex(ValueError,'cannot generate video'):
            node.compile(h.FakeModel(True),h.RecordingClip(),refs,'','',64,64,5,True,vae,False)

    def test_h3_rejects_foreign_packs_but_preserves_video_refs_for_video_architecture(self):
        vae=types.SimpleNamespace(first_stage_model=MiniMaxH3ImageVAE())
        _,_,refs=self.prepare('Flux 2')
        with self.assertRaisesRegex(ValueError,'Re-encode'):
            h.nodes.BExplainsRefPackToH3().compile(h.FakeModel(),h.RecordingClip(),refs,'','',64,64,1,True,vae,False)
        refs=common.add_pack(h.pack('video'),role='video')
        result=h.nodes.BExplainsRefPackToH3().compile(h.FakeModel(),h.RecordingClip(),refs,'Use <Video 1> as a reference','',64,64,1,True,vae,False)
        self.assertEqual(result[3]['samples'].unbind()[0].shape[2],1)
        self.assertIs(result[1][0][1]['minimax_refs'][0]['latent'],refs['groups'][0]['entries'][0]['latent'])
        with self.assertRaisesRegex(ValueError,'still-image references only'):
            h.nodes.BExplainsRefPackToH3().compile(h.FakeModel(True),h.RecordingClip(),refs,'Use <Video 1>','',64,64,1,True,vae,False)


class CollectionTests(unittest.TestCase):
    def test_both_install_orders_and_shared_package_independence(self):
        import subprocess, sys
        root=Path(__file__).resolve().parents[2]
        for order in ('shared_first','h3_first'):
            script='''
import importlib.util, pathlib, sys, tempfile, types
root=pathlib.Path(sys.argv[1]);order=sys.argv[2]
folders=types.ModuleType('folder_paths');folders.models_dir=tempfile.mkdtemp()
folders.add_model_folder_path=lambda *a:None;folders.get_filename_list=lambda *a:[]
sys.modules['folder_paths']=folders

def load(name, directory):
    path=root/directory/'__init__.py'
    spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(path.parent)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

if order=='shared_first':
    shared=load('comfy_generated_shared','BExplains_RefPacks')
    assert not any('MiniMaxH3' in k for k in shared.NODE_CLASS_MAPPINGS)
    assert not any(k.endswith('.legacy') for k in sys.modules)
    h3=load('comfy_generated_h3','BExplains_H3_RefPacks')
else:
    h3=load('comfy_generated_h3','BExplains_H3_RefPacks')
    shared=load('comfy_generated_shared','BExplains_RefPacks')
assert shared.NODE_CLASS_MAPPINGS is sys.modules['bexplains_refpacks'].NODE_CLASS_MAPPINGS
assert not set(shared.NODE_CLASS_MAPPINGS).intersection(h3.NODE_CLASS_MAPPINGS)
assert 'BExplainsRefPackToImage' in shared.NODE_CLASS_MAPPINGS
assert 'BExplainsRefPackToVideo' in shared.NODE_CLASS_MAPPINGS
assert 'BExplainsRefPackToH3' in h3.NODE_CLASS_MAPPINGS
'''
            result=subprocess.run([sys.executable,'-c',script,str(root),order],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_actual_h3_image_vae_guard_accepts_batch_but_rejects_time(self):
        source=os.environ.get('REFPACKS_NATIVE_H3_VAE_SOURCE')
        if not source:self.skipTest('Set REFPACKS_NATIVE_H3_VAE_SOURCE for the upstream image-VAE guard check.')
        tree=ast.parse(Path(source).read_text())
        original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='H3ImageVAE')
        decode=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='decode')
        class Parent:
            def decode(self,z,output_buffer=None):return z
        cls=ast.ClassDef(name='ImageGuard',bases=[ast.Name(id='Parent',ctx=ast.Load())],keywords=[],body=[decode],decorator_list=[])
        ns={'Parent':Parent}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),source,'exec'),ns)
        vae=ns['ImageGuard']()
        for batch in (1,2,3,4):self.assertEqual(vae.decode(torch.zeros(batch,24,1,4,4)).shape[0],batch)
        with self.assertRaisesRegex(ValueError,'single-frame'):
            vae.decode(torch.zeros(1,24,4,4,4))


class H3BatchForwardTests(unittest.TestCase):
    def setUp(self):
        self.batching=__import__(h.package.__name__+'.batching',fromlist=['unused'])

    def test_batch_predictions_equal_separate_forwards_for_both_architectures(self):
        for joint in (False,True):
            calls=[]
            x=torch.arange(4*24*1*2*2,dtype=torch.float32).reshape(4,24,1,2,2)
            audio=torch.ones(4,32,2,2)
            t=torch.arange(4,dtype=torch.float32)
            context=torch.arange(4,dtype=torch.float32).reshape(4,1,1)
            ref=torch.ones(1,24,1,2,2)
            payload={'cond_video_latents':[ref], 'seed':42}
            mask=torch.arange(4,dtype=torch.float32).reshape(4,1,1)
            options={'cond_or_uncond':[0,1]}
            def forward(item,step,ctx,opts,**kwargs):
                primary=item[0] if joint else item
                if primary.shape[0]!=1:raise ValueError('MiniMax H3 supports batch size 1')
                self.assertEqual(ctx.shape[0],1)
                self.assertEqual(kwargs['denoise_mask'].shape[0],1)
                self.assertIs(kwargs['minimax_payload']['cond_video_latents'][0],ref)
                calls.append((float(step.item()),opts['cond_or_uncond']))
                kwargs['minimax_payload']['private_cache']=1
                v=primary + step.reshape(1,1,1,1,1) + ctx.reshape(1,1,1,1,1)
                return [v,item[1]+step.reshape(1,1,1,1)] if joint else v
            actual=self.batching.split_image_batch(forward,[x,audio] if joint else x,t,context,options,minimax_payload=payload,denoise_mask=mask)
            expected=x+2*t.reshape(4,1,1,1,1)
            self.assertTrue(torch.equal(actual[0] if joint else actual,expected))
            if joint:self.assertTrue(torch.equal(actual[1],audio+t.reshape(4,1,1,1)))
            self.assertEqual(calls,[(0,[0]),(1,[0]),(2,[1]),(3,[1])])
            self.assertNotIn('private_cache',payload)
            self.assertEqual(options,{'cond_or_uncond':[0,1]})

    def test_wrapper_registration_is_local_composable_and_idempotent(self):
        original=h.FakeModel()
        wrapped=self.batching.enable_image_batches(original)
        twice=self.batching.enable_image_batches(wrapped)
        self.assertEqual(original.model_options,{})
        wrappers=twice.model_options['transformer_options']['wrappers']['diffusion_model']
        self.assertEqual(len(wrappers[self.batching.KEY]),1)
        source=os.environ.get('REFPACKS_NATIVE_PATCHER_SOURCE')
        if not source:return
        import importlib.util
        spec=importlib.util.spec_from_file_location('native_patcher_batch_test',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        calls=[]
        def extra(executor,*args,**kwargs):
            calls.append('extra');return executor(*args,**kwargs)
        def native_forward(x,timestep,context,transformer_options,**kwargs):
            self.assertEqual(x.shape[0],1);return x+1
        executor=module.WrapperExecutor.new_executor(native_forward,[self.batching.split_image_batch,extra])
        x=torch.zeros(4,24,1,2,2)
        result=executor.execute(x,torch.ones(4),torch.ones(1,2,3),{})
        self.assertTrue(torch.equal(result,torch.ones_like(x)))
        self.assertEqual(calls,['extra']*4)
