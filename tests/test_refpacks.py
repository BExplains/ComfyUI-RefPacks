"""CPU reference-container and routing checks. These do not test generated motion."""
import ast
import contextlib
import copy
import io
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = tempfile.TemporaryDirectory()
folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = FOLDERS.name
folder_paths.add_model_folder_path = lambda *args: None
folder_paths.get_filename_list = lambda *args: []
sys.modules["folder_paths"] = folder_paths

# Provide the Comfy interfaces needed for unit checks, without loading any model.
comfy = types.ModuleType("comfy")
comfy.__path__ = []
utils = types.ModuleType("comfy.utils")
def common_upscale(samples, width, height, method, crop):
    return torch.nn.functional.interpolate(samples,size=(height,width),mode="bilinear",align_corners=False)
utils.common_upscale = common_upscale
management = types.ModuleType("comfy.model_management")
management.intermediate_device = lambda: torch.device("cpu")
comfy.utils = utils
comfy.model_management = management
sys.modules.update({"comfy":comfy,"comfy.utils":utils,"comfy.model_management":management})
extension = types.ModuleType("comfy.patcher_extension")
extension.WrappersMP = types.SimpleNamespace(OUTER_SAMPLE="outer_sample", SAMPLER_SAMPLE="sampler_sample", DIFFUSION_MODEL="diffusion_model")
def add_wrapper_with_key(kind,key,wrapper,options,is_model_options=False):
    options = options.setdefault("transformer_options",{}) if is_model_options else options
    options.setdefault("wrappers",{}).setdefault(kind,{}).setdefault(key,[]).append(wrapper)
extension.add_wrapper_with_key = add_wrapper_with_key
sys.modules[extension.__name__] = extension
extras = types.ModuleType("comfy_extras")
extras.__path__ = []
sys.modules["comfy_extras"] = extras

native = types.ModuleType("comfy_extras.nodes_minimax_h3")
def resize(image,width,height,crop):
    return common_upscale(image.movedim(-1,1),width,height,"lanczos",crop).movedim(1,-1)
native._resize = resize
def empty_av(width,height,length):
    n=max(5,length)
    while n % 17 != 5:n+=1
    t=2 if n<=5 else ((n-5)//17)*5+2
    return {"samples":(torch.zeros(1,24,t,height//16,width//16),torch.zeros(1,32,2,round(n/24*40)))},n
native._empty_av_latent=empty_av
def short_edge(w,h,low,high):
    scale=min(max(min(w,h),low),high)/min(w,h)
    return tuple(max(32,round(side*scale/32)*32) for side in (w,h))
native._size_short_edge=short_edge
def image_refs(images,vae,vision_size,vae_size):
    originals=list(images.values());visions=[];refs=[]
    for image in originals:
        w,h=image.shape[2],image.shape[1]
        v=resize(image,*vision_size(w,h),"disabled")
        z=vae.encode(resize(image,*vae_size(w,h),"disabled"))
        visions.append(v);refs.append({"kind":"image","latent_h":z.shape[-2],"latent_w":z.shape[-1],"latent":z})
    return originals,visions,refs
native._image_refs=image_refs
def encode_ref_audio(vae,audio):
    z=vae.encode(audio['waveform'][:1].movedim(1,-1))
    return z,z.shape[-1]
native._encode_ref_audio=encode_ref_audio
sys.modules[native.__name__]=native

spec=importlib.util.spec_from_file_location("bexplains_refpacks_under_test",ROOT/"__init__.py",submodule_search_locations=[str(ROOT)])
package=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=package
spec.loader.exec_module(package)
shared_package=package
spec=importlib.util.spec_from_file_location("bexplains_h3_under_test",ROOT.parent/"BExplains_H3_RefPacks"/"__init__.py",submodule_search_locations=[str(ROOT.parent/"BExplains_H3_RefPacks")])
package=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=package
spec.loader.exec_module(package)
core=sys.modules[spec.name+".core"]
storage=sys.modules["bexplains_refpacks.storage"] if "bexplains_refpacks.storage" in sys.modules else sys.modules[shared_package.__name__+".storage"]
nodes=sys.modules[spec.name+".nodes"]
diagnostics=sys.modules[spec.name+".diagnostics"]
ALL_NODE_CLASSES={**shared_package.NODE_CLASS_MAPPINGS, **package.NODE_CLASS_MAPPINGS}



class MiniMaxH3:pass
class MiniMaxH3Image(MiniMaxH3):pass
class FakeModel:
    def __init__(self,image=False):
        self.model=MiniMaxH3Image() if image else MiniMaxH3()
        self.model_options={}
    def clone(self):
        result=copy.copy(self)
        result.model_options=copy.deepcopy(self.model_options)
        return result


class RecordingVAE:
    first_stage_model=object()
    def __init__(self):self.calls=[]
    def encode(self,pixels):
        self.calls.append(pixels.clone())
        n,h,w,_=pixels.shape
        t=1 if n==1 else 2 if n<=5 else ((n-5)//17)*5+2
        # Real tensors; deterministic content allows exact serialization comparisons.
        return torch.full((1,24,t,h//16,w//16),float(pixels.mean()))


class RecordingClip:
    def __init__(self):self.calls=[]
    def tokenize(self,prompt,**kwargs):
        self.calls.append((prompt,kwargs));return (prompt,kwargs)
    def encode_from_tokens_scheduled(self,tokens):
        return [[torch.ones(1,8,5120),{"minimax_token_tags":torch.ones(8,dtype=torch.int64)}]]


class RecordingAudioVAE:
    audio_sample_rate=32000
    first_stage_model=type('H3AudioVAE',(),{})()
    def __init__(self):self.calls=[]
    def encode(self,pixels):
        self.calls.append(pixels.clone())
        return torch.full((1,32,2,8),float(pixels.mean()))


def audio_pack(name='voice'):
    return {"format":core.FORMAT,"version":core.VERSION,"name":name,"target":"h3_video",
            "entries":[{"kind":"audio","audio_latent":torch.rand(1,32,2,8),"ref_audio_t":8,"duration_seconds":0.2}]}


def pack(kind="image",count=1,name="reference",target="h3_video"):
    entries=[]
    for i in range(count):
        e={"kind":kind,"latent":torch.full((1,24,7 if kind=="video" else 1,2,2),float(i)),"vision":torch.full((2 if kind=="video" else 1,32,32,3),0.25+i*0.1)}
        if kind=="video":e.update(timestamps=[0.0,0.5],frame_count=22,fps=24.0)
        entries.append(e)
    return {"format":core.FORMAT,"version":core.VERSION,"name":name,"target":target,"entries":entries}


class RefPacksTests(unittest.TestCase):
    def test_loader_reports_selected_filename_in_inspector(self):
        filename='inspect_test.safetensors'
        p=pack();p['source_file']='stale_name.safetensors'
        storage.save_pack(p,nodes.ROOT/filename)
        loaded=nodes.BExplainsLoadRefPack().load(filename,True)[0]
        refs=core.add_pack(loaded,role='reference')
        result=nodes.BExplainsInspectRefPacks().inspect(refs)['result'][1]
        self.assertIn('<Picture 1> -> inspect_test.safetensors',result)
        self.assertNotIn('stale_name',result)

    def test_inspector_locates_picture_eight_without_encoder_and_preserves_set(self):
        person=pack(count=6,name='Yoleida');person['source_file']='people/yoleida.safetensors'
        props=pack(count=3,name='Props');props['source_file']='props/studio.safetensors'
        refs=core.add_pack(props,core.add_pack(person,role='subject'),'reference')
        inspected=nodes.BExplainsInspectRefPacks().inspect(refs)
        self.assertIs(inspected['result'][0],refs)
        self.assertIn('<Picture 8> -> props/studio.safetensors | pack: Props | entry 2 of 3',inspected['result'][1])
        self.assertIn('people/yoleida.safetensors: 6 picture(s)',inspected['result'][1])
        self.assertEqual(inspected['result'][2],'')
        self.assertIn('No text encoder has run',inspected['ui']['text'][0])
        self.assertEqual(list(nodes.BExplainsInspectRefPacks.INPUT_TYPES()['required']),['references'])

    def test_inspector_matches_native_order_for_mixed_reordered_references(self):
        refs=core.add_pack(audio_pack(),role='audio')
        refs=core.add_pack(pack('video'),refs,'video')
        refs=core.add_pack(pack(count=2),refs,'reference')
        mapping=nodes.BExplainsInspectRefPacks().inspect(refs)['result'][1]
        layout=list(core.reference_layout(refs))
        self.assertEqual([x[-1] for x in layout],['<Picture 1>','<Picture 2>','<Video 1>','<Audio 1>'])
        positions=[mapping.index(x[-1]+' ->') for x in layout]
        self.assertEqual(positions,sorted(positions))
        self.assertIs(core.add_pack(None,refs,'reference'),refs)

    def test_inspector_prompt_preview_matches_conditioning_compiler(self):
        refs=core.add_pack(pack(),role='subject')
        refs=core.add_pack(pack(name='outfit'),refs,'outfit')
        for fmt,task in [('H3 Image','h3_image'),('H3 Video','h3_video')]:
            result=nodes.BExplainsInspectRefPacks().inspect(refs,'<Subject 1> wears <Outfit 1>.',fmt,False)
            self.assertEqual(result['result'][2],core.assemble(refs,'<Subject 1> wears <Outfit 1>.',task,False)[0])
            self.assertIn('<Outfit 1> -> <Picture 2>',result['result'][1])

    def test_inspector_keeps_mapping_when_optional_prompt_is_invalid(self):
        refs=core.add_pack(pack(),role='reference')
        result=nodes.BExplainsInspectRefPacks().inspect(refs,'<Picture 8>')
        self.assertIn('<Picture 1> ->',result['result'][1])
        self.assertIn('missing <Picture 8>',result['ui']['text'][0])
        self.assertEqual(result['result'][2],'')

    def test_framing_and_motion_are_not_injected_with_or_without_video(self):
        refs=core.add_pack(pack(),role='subject')
        for video in (False,True):
            if video:refs=core.add_pack(pack('video'),refs,'video')
            compiled=core.assemble(refs,'Close-up of <Subject 1>. Cut to a wide shot.','h3_video')[0]
            self.assertNotIn('Use the referenced movement within',compiled)
            self.assertNotIn('Use one continuous shot',compiled)
            self.assertNotIn("keep the performer's body visible",compiled)
            self.assertIn('Close-up of <Subject 1>. Cut to a wide shot.',compiled)

    def test_general_pictures_have_no_role_specific_definitions_in_either_model(self):
        p=pack(count=3,name='lighting_and_props')
        refs=core.add_pack(p,role='reference')
        prompt='Light a room as in <Picture 1>, with the object in <Picture 2> and palette from <Picture 3>.'
        for task in ('h3_image','h3_video'):
            compiled,blocks,items,report=core.assemble(refs,prompt,task)
            self.assertIn(prompt,compiled)
            for unwanted in ('one person whose identity','outfit reference','location, environment','source performance','continuous shot','body visible'):
                self.assertNotIn(unwanted,compiled)
            self.assertEqual(len(blocks),3)
            for i,b in enumerate(blocks):self.assertIs(b['latent'],p['entries'][i]['latent'])
            self.assertIn('<Picture 1>, <Picture 2>, <Picture 3>',report)

    def test_general_pictures_mix_with_subject_and_preserve_literal_slots(self):
        refs=core.add_pack(pack(count=2),role='reference')
        refs=core.add_pack(pack(),refs,'subject')
        compiled,blocks,_,report=core.assemble(refs,'<Subject 1> holds the item in <Picture 1>, lit as in <Picture 2>.','h3_video')
        self.assertIn('shown in <Picture 3>',compiled)
        self.assertEqual(len(blocks),3)
        with self.assertRaisesRegex(ValueError,'missing <Picture 4>'):core.assemble(refs,'<Picture 4>','h3_video')

    def test_audio_roundtrip_lossless_and_legacy_visual_files_still_load(self):
        p=audio_pack()
        p['encoder_class']='H3AudioVAE'
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'audio.safetensors';storage.save_pack(p,path)
            loaded=storage.load_pack(path)
        self.assertTrue(torch.equal(loaded['entries'][0]['audio_latent'],p['entries'][0]['audio_latent']))
        self.assertEqual(loaded['encoder_class'],'H3AudioVAE')
        self.assertNotIn('vision',loaded['entries'][0])
        self.assertNotIn('latent',loaded['entries'][0])

    def test_audio_encode_routes_native_payload_without_generation_reencoding(self):
        vae=RecordingAudioVAE()
        p=nodes.BExplainsH3EncodeAudio().encode(vae,{'waveform':torch.rand(1,2,6400),'sample_rate':32000},'voice')[0]
        self.assertEqual(vae.calls[0].shape,(1,6400,2))
        self.assertEqual(p['entries'][0]['duration_seconds'],0.2)
        refs=core.add_pack(p,core.add_pack(pack(),role='subject'),'audio')
        clip=RecordingClip()
        result=nodes.BExplainsH3RefPackConditioning().compile(FakeModel(),clip,refs,'<Subject 1> speaks using the voice in <Audio 1>.','',64,64,22,True,True)
        self.assertEqual(clip.calls[0][1]['minimax_ref_items'][-1],{'type':'audio'})
        block=result[1][0][1]['minimax_refs'][-1]
        self.assertEqual(set(block),{'kind','audio_latent','ref_audio_t'})
        self.assertIs(block['audio_latent'],p['entries'][0]['audio_latent'])
        self.assertEqual(len(vae.calls),1)
        self.assertNotIn('No dialogue.',clip.calls[0][0])
        self.assertNotIn('No music is requested.',clip.calls[0][0])
        self.assertIn('8 audio time steps',result[4])

    def test_audio_addressing_order_and_default_window_preserve_full_audio(self):
        a,b=audio_pack('voice'),audio_pack('music')
        refs=core.add_pack(a,role='audio')
        refs=core.add_pack(pack('video'),refs,'video')
        refs=core.add_pack(pack(),refs,'subject')
        refs=core.add_pack(b,refs,'audio')
        window=core.reference_window(refs,5)
        self.assertIs(window['groups'][0]['entries'][0],a['entries'][0])
        _,blocks,items,report=core.assemble(window,'<Subject 1> follows <Video 1> with <Audio 1> and <Audio 2>.','h3_video')
        self.assertEqual([i['type'] for i in items],['image','video','audio','audio'])
        self.assertIs(blocks[2]['audio_latent'],a['entries'][0]['audio_latent'])
        self.assertIn('<Audio 2>',report)
        with self.assertRaisesRegex(ValueError,'missing <Audio 3>'):core.assemble(refs,'<Audio 3>','h3_video')
        with self.assertRaisesRegex(ValueError,'still-image references only'):core.assemble(core.add_pack(a,role='audio'),'<Audio 1>','h3_image')

    def test_audio_validation_and_role_mismatch_fail(self):
        for z in (torch.zeros(1,24,2,8),torch.zeros(1,32,2,0),torch.full((1,32,2,8),float('nan'))):
            with self.assertRaises(ValueError):core.validate_audio_latent(z)
        p=audio_pack();p['entries'][0]['ref_audio_t']=9
        with self.assertRaises(ValueError):core.validate_pack(p)
        with self.assertRaises(ValueError):core.add_pack(audio_pack(),role='subject')
        with self.assertRaises(ValueError):core.add_pack(pack(),role='audio')
        with self.assertRaises(ValueError):nodes.BExplainsH3EncodeAudio().encode(RecordingAudioVAE(),{'waveform':torch.zeros(2,2,32),'sample_rate':32000},'bad')

    def test_native_audio_helper_resampling_and_payload_parity(self):
        source=os.environ.get('H3_NATIVE_SOURCE')
        if not source:self.skipTest('Set H3_NATIVE_SOURCE for native audio parity.')
        tree=ast.parse(Path(source).read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_encode_ref_audio')
        resamples=[]
        def resample(waveform,sr,target):
            resamples.append((sr,target))
            return torch.nn.functional.interpolate(waveform,size=round(waveform.shape[-1]*target/sr),mode='linear',align_corners=False)
        env={'comfy':types.SimpleNamespace(audio=types.SimpleNamespace(resample=resample))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),source,'exec'),env)
        helper=native._encode_ref_audio
        try:
            native._encode_ref_audio=env['_encode_ref_audio']
            audio={'waveform':torch.rand(1,2,4800),'sample_rate':48000}
            vae=RecordingAudioVAE()
            p=nodes.BExplainsH3EncodeAudio().encode(vae,audio,'voice')[0]
            expected,t=env['_encode_ref_audio'](RecordingAudioVAE(),audio)
            self.assertTrue(torch.equal(expected,p['entries'][0]['audio_latent']))
            self.assertEqual(t,p['entries'][0]['ref_audio_t'])
            self.assertEqual(resamples,[(48000,32000),(48000,32000)])
            self.assertEqual(vae.calls[0].shape,(1,3200,2))
        finally:native._encode_ref_audio=helper

    def test_roundtrip_is_bitwise_lossless_and_preserves_metadata(self):
        p=pack("video")
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"dance.safetensors"
            storage.save_pack(p,path)
            loaded=storage.load_pack(path)
            for field in ("latent","vision"):
                self.assertTrue(torch.equal(p["entries"][0][field],loaded["entries"][0][field]))
            self.assertEqual(loaded["entries"][0]["timestamps"],[0.0,0.5])
            self.assertEqual(loaded["target"],"h3_video")
            with self.assertRaises(ValueError):storage.save_pack(p,path)

    def test_reordering_video_before_or_after_subject_keeps_native_binding(self):
        character,dance=pack(count=2,name="Mai"),pack("video",name="Dance")
        a=core.add_pack(character,role="subject")
        a=core.add_pack(dance,a,"video")
        b=core.add_pack(dance,role="video")
        b=core.add_pack(character,b,"subject")
        outa=core.assemble(a,"<Subject 1> performs the moves in <Video 1>","h3_video")
        outb=core.assemble(b,"<Subject 1> performs the moves in <Video 1>","h3_video")
        self.assertEqual(outa[0],outb[0])
        self.assertEqual([x["kind"] for x in outa[1]],["image","image","video"])
        for left,right in zip(outa[1],outb[1]):self.assertTrue(torch.equal(left["latent"],right["latent"]))

    def test_character_swap_changes_references_without_changing_subject_id(self):
        first=core.add_pack(pack(name="Mai"),role="subject",role_id=1)
        second=core.add_pack(pack(count=2,name="Ian"),role="subject",role_id=1)
        for refs in (first,second):
            prompt,_,_,_=core.assemble(refs,"<Subject 1> stands alone.","h3_video")
            self.assertIn("<Subject 1>",prompt)
            self.assertNotIn("<Subject 2>",prompt)
        self.assertEqual(len(first["groups"][0]["entries"]),1)

    def test_outfit_alias_tracks_picture_number_in_both_positions(self):
        for outfit_first,expected in [(False,"<Picture 3>"),(True,"<Picture 1>")]:
            a=pack(count=2);o=pack(name="outfit")
            refs=core.add_pack(o,role="outfit") if outfit_first else core.add_pack(a,role="subject")
            refs=core.add_pack(a,refs,"subject") if outfit_first else core.add_pack(o,refs,"outfit")
            for alias in ("<Outfit 1>","<Picture N>"):
                text=core.assemble(refs,f"<Subject 1> wears {alias}.","h3_video")[0]
                self.assertIn(f"wears {expected}",text)
                self.assertNotIn(alias,text)

    def test_auto_outfit_and_missing_labels(self):
        refs=core.add_pack(pack(),role="subject")
        refs=core.add_pack(pack(name="outfit"),refs,"outfit")
        prompt=core.assemble(refs,"<Subject 1> stands.","h3_video")[0]
        self.assertIn("wearing the outfit in <Picture 2>",prompt)
        with self.assertRaises(ValueError):core.assemble(refs,"<Subject 2> stands.","h3_video")

    def test_motion_in_image_model_duplicate_ids_and_invalid_latents_fail(self):
        refs=core.add_pack(pack(),role="subject",role_id=1)
        with self.assertRaises(ValueError):core.add_pack(pack(),refs,"subject",1)
        with self.assertRaises(ValueError):core.validate_latent(torch.zeros(1,128,4,4),"image")
        with self.assertRaises(ValueError):core.validate_latent(torch.full((1,24,1,2,2),float('nan')),"image")
        refs=core.add_pack(pack("video"),refs,"video")
        with self.assertRaisesRegex(ValueError,"still-image references only"):core.assemble(refs,"<Subject 1>","h3_image")

    def test_legacy_image_packs_mix_in_either_order_and_keep_provenance(self):
        a=pack(target="h3_image");a['encoder_class']='image.encoder'
        b=pack(target="h3_video");b['encoder_class']='video.encoder'
        for first,second in ((a,b),(b,a)):
            refs=core.add_pack(second,core.add_pack(first,role="subject"),"outfit")
            for task in ('h3_image','h3_video'):
                compiled,blocks,items,_=core.assemble(refs,'<Subject 1> wearing <Outfit 1>',task)
                self.assertIs(blocks[0]['latent'],first['entries'][0]['latent'])
                self.assertIs(items[1]['data'],second['entries'][0]['vision'])
                self.assertEqual(refs['groups'][0]['source_target'],first['target'])
                self.assertEqual(refs['groups'][0]['encoder_class'],first['encoder_class'])
        self.assertEqual(a['target'],'h3_image')
        self.assertEqual(b['target'],'h3_video')

    def test_one_saved_still_encoding_routes_to_both_models_unchanged(self):
        vae=RecordingVAE()
        vae.first_stage_model=type('MiniMaxH3VideoVAE',(),{})()
        prepared=nodes.BExplainsH3EncodeImages().encode(vae,torch.rand(1,32,32,3),'character',reference_megapixels=0.01)[0]
        self.assertEqual(prepared['target'],'h3')
        self.assertEqual(prepared['source_target'],'h3_video')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'shared.safetensors'
            storage.save_pack(prepared,path)
            loaded=storage.load_pack(path)
        self.assertEqual(loaded['encoder_class'],prepared['encoder_class'])
        refs=core.add_pack(loaded,role='subject')
        for image in (False,True):
            clip=RecordingClip()
            result=nodes.BExplainsH3RefPackConditioning().compile(FakeModel(image),clip,refs,'<Subject 1> stands.','',64,64,22,True,False)
            self.assertIs(result[1][0][1]['minimax_refs'][0]['latent'],loaded['entries'][0]['latent'])
            args=clip.calls[0][1]
            vision=args['images'][0] if image else args['minimax_ref_items'][0]['data']
            self.assertIs(vision,loaded['entries'][0]['vision'])
        self.assertEqual(len(vae.calls),1)

    def test_image_vae_detected_without_target_and_not_overridden_by_legacy_widget(self):
        vae=RecordingVAE()
        vae.first_stage_model=type('MiniMaxH3ImageVAE',(),{})()
        prepared=nodes.BExplainsH3EncodeImages().encode(vae,torch.rand(1,32,32,3),'character',target='H3 Video')[0]
        self.assertEqual(prepared['target'],'h3')
        self.assertEqual(prepared['source_target'],'h3_image')
        self.assertEqual(len(vae.calls),1)
        self.assertEqual(vae.calls[0].shape[1:3],(1024,1024))
        for cls in (nodes.BExplainsH3EncodeImages,nodes.BExplainsH3PackLatent,nodes.BExplainsH3ImportRefMod):
            self.assertNotIn('target',cls.INPUT_TYPES()['required'])
        with self.assertRaisesRegex(ValueError,'Cannot identify'):
            nodes._vae_target(RecordingVAE())

    def test_image_prepared_pack_chains_with_motion_for_video_model(self):
        refs=core.add_pack(pack(target='h3_image'),role='subject')
        refs=core.add_pack(pack('video'),refs,'video')
        _,blocks,items,_=core.assemble(refs,'<Subject 1> follows <Video 1>','h3_video')
        self.assertEqual([b['kind'] for b in blocks],['image','video'])
        self.assertEqual([i['type'] for i in items],['image','video'])

    def test_video_encoding_resamples_pads_and_keeps_original_tail(self):
        vae=RecordingVAE()
        frames=torch.linspace(0,1,168).view(168,1,1,1).expand(-1,32,32,3)
        result=nodes.BExplainsH3EncodeVideo().encode(vae,frames,24,"dance",0.01,"pad final frame")[0]
        e=result["entries"][0]
        self.assertEqual(vae.calls[0].shape[0],175)
        self.assertEqual(e["latent"].shape[2],52)
        self.assertEqual(e["frame_count"],175)
        self.assertEqual(e["vision"].shape[0],15)
        self.assertTrue(torch.equal(vae.calls[0][-1],vae.calls[0][167]))
        self.assertEqual(e["timestamps"][-1],7.0)

    def test_multiple_images_are_separate_entries(self):
        vae=RecordingVAE()
        result=nodes.BExplainsH3EncodeImages().encode(vae,torch.rand(3,32,32,3),"character","H3 Video",0.01)[0]
        self.assertEqual(len(result["entries"]),3)
        self.assertEqual(len(vae.calls),3)

    def test_render_compilation_has_native_vision_and_latent_paths_without_vae(self):
        refs=core.add_pack(pack(),role="subject")
        refs=core.add_pack(pack("video"),refs,"video")
        clip=RecordingClip()
        model=FakeModel()
        returned,positive,negative,latent,report=nodes.BExplainsH3RefPackConditioning().compile(model,clip,refs,"<Subject 1> performs the moves in <Video 1>","",64,64,22,True)
        self.assertIs(returned.model,model.model)
        self.assertEqual(model.model_options,{})
        self.assertEqual([x['type'] for x in clip.calls[0][1]['minimax_ref_items']],["image","video"])
        self.assertEqual([x['kind'] for x in positive[0][1]['minimax_refs']],["image","video"])
        self.assertIs(positive[0][1]['minimax_refs'][1]['latent'],refs['groups'][1]['entries'][0]['latent'])
        self.assertIn("minimax_token_tags",positive[0][1])
        self.assertIn("subject 1",report)
        self.assertEqual(len(negative),1)

    def test_image_compilation_and_disabled_outfit(self):
        p=pack(target="h3_image")
        refs=core.add_pack(p,role="subject")
        self.assertIs(core.add_pack(None,refs,"outfit"),refs)
        clip=RecordingClip()
        result=nodes.BExplainsH3RefPackConditioning().compile(FakeModel(image=True),clip,refs,"<Subject 1> stands.","",64,64,5,True)
        self.assertEqual(result[3]['samples'].shape,(1,24,1,4,4))
        self.assertEqual(clip.calls[0][1]['picture_label'],"Picture %d: ")

    def test_output_length_is_independent_of_mixed_reference_lengths(self):
        refs=core.add_pack(pack(),role="subject")
        first,second=pack("video",name="long"),pack("video",name="short")
        for p,frames,t in ((first,175,52),(second,90,27)):
            e=p["entries"][0]
            e["frame_count"]=frames
            e["latent"]=torch.zeros(1,24,t,2,2)
            ix=list(range(0,frames,12))
            e["vision"]=torch.zeros(len(ix),32,32,3)
            e["timestamps"]=[i/24.0 for i in ix]
            refs=core.add_pack(p,refs,"video")
        for requested,actual,t in ((72,73,22),(209,209,62)):
            clip=RecordingClip()
            result=nodes.BExplainsH3RefPackConditioning().compile(FakeModel(),clip,refs,"<Subject 1> performs the moves in <Video 1>","",64,64,requested,True,compress_motion=True)
            self.assertEqual(result[3]["samples"][0].shape[2],t)
            blocks=result[1][0][1]["minimax_refs"]
            self.assertEqual([b["latent_t"] for b in blocks if b["kind"]=="video"],[52,27])
            self.assertIs(blocks[1]["latent"],first["entries"][0]["latent"])
            self.assertIs(blocks[2]["latent"],second["entries"][0]["latent"])
            self.assertEqual(clip.calls[0][1]["minimax_ref_items"][1]["timestamps"],first["entries"][0]["timestamps"])
            self.assertIn(f"output frames: {actual}",result[4])

    def test_portrait_canvas_need_not_be_multiple_of_32(self):
        for image,target in ((False,"h3_video"),(True,"h3_image")):
            with self.subTest(target=target):
                refs=core.add_pack(pack(target=target),role="subject")
                result=nodes.BExplainsH3RefPackConditioning().compile(FakeModel(image=image),RecordingClip(),refs,"<Subject 1> stands.","",240,432,73,True)
                z=result[3]["samples"] if image else result[3]["samples"][0]
                self.assertEqual(z.shape[-2:],(27,15))
                self.assertIn("canvas: 240x432",result[4])

    def test_default_windows_mixed_video_references_without_changing_packs(self):
        refs=core.add_pack(pack(),role="subject")
        originals=[]
        for frames,t in ((175,52),(90,27)):
            p=pack("video",name=str(frames));e=p["entries"][0]
            e.update(frame_count=frames,latent=torch.arange(t).view(1,1,t,1,1).expand(1,24,t,2,2).float())
            ix=list(range(0,frames,12))
            e.update(vision=torch.rand(len(ix),32,32,3),timestamps=[i/24 for i in ix])
            originals.append(e)
            refs=core.add_pack(p,refs,"video")
        with contextlib.redirect_stdout(io.StringIO()):
            output=nodes.BExplainsH3RefPackConditioning().compile(FakeModel(),RecordingClip(),refs,"<Subject 1> follows <Video 1>","",64,64,72,True)
        blocks=output[1][0][1]["minimax_refs"]
        self.assertEqual([b["latent_t"] for b in blocks if b["kind"]=="video"],[22,22])
        for block,original in zip(blocks[1:],originals):
            self.assertTrue(torch.equal(block["latent"],original["latent"][:,:,:22]))
        cropped=core.reference_window(refs,73)
        for group in cropped["groups"][1:]:
            e=group["entries"][0]
            self.assertEqual(e["frame_count"],73)
            self.assertEqual(e["timestamps"],[0,.5,1,1.5,2,2.5,3])
            self.assertEqual(e["vision"].shape[0],7)
        self.assertEqual([e["frame_count"] for e in originals],[175,90])
        self.assertEqual([e["latent"].shape[2] for e in originals],[52,27])

    def test_short_references_are_not_extended_and_compression_keeps_all(self):
        refs=core.add_pack(pack("video"),role="video")
        output=core.reference_window(refs,73)
        self.assertIs(output["groups"][0]["entries"][0],refs["groups"][0]["entries"][0])
        self.assertIs(core.reference_window(refs,5,compress_motion=True),refs)

    def test_outfit_and_location_aliases_support_multiple_views(self):
        for location_first in (False,True):
            with self.subTest(location_first=location_first):
                refs=None
                sequence=[("location",pack(count=2,name="hall")),("subject",pack(count=2)),("outfit",pack(count=2))] if location_first else [("subject",pack(count=2)),("outfit",pack(count=2)),("location",pack(count=2,name="hall"))]
                for role,p in sequence:refs=core.add_pack(p,refs,role)
                text,blocks,_,mapping=core.assemble(refs,"<Subject 1> wears <Outfit 1> in <Location 1>.","h3_video")
                self.assertNotIn("<Outfit 1>",text)
                self.assertNotIn("<Location 1>",text)
                expected="<Picture 1> and <Picture 2>" if location_first else "<Picture 5> and <Picture 6>"
                self.assertIn("the location shown in "+expected,text)
                self.assertIn("<Location 1>: hall",mapping)
                self.assertEqual(len(blocks),6)
                with self.assertRaises(ValueError):core.assemble(refs,"<Subject 1> stands in <Location 2>","h3_video")

    def test_node_assigns_role_ids_automatically_and_ignores_legacy_values(self):
        schema=nodes.BExplainsAddRefPack.INPUT_TYPES()
        self.assertNotIn("role_id",schema["required"])
        self.assertNotIn("role_id",schema["optional"])
        node=nodes.BExplainsAddRefPack()
        refs=node.add(pack(),"subject",role_id=99)[0]
        refs=node.add(pack(),"subject",role_id=99,references=refs)[0]
        refs=node.add(pack(),"outfit",role_id=99,references=refs)[0]
        refs=node.add(pack(),"location",role_id=99,references=refs)[0]
        self.assertEqual([(g['role'],g['role_id']) for g in refs['groups']],[("subject",1),("subject",2),("outfit",1),("location",1)])

    def test_path_traversal_and_non_h3_models_are_rejected(self):
        with self.assertRaises(ValueError):nodes._path("../outside.safetensors")
        with self.assertRaises(ValueError):core.model_target(object())

    def test_timing_wrappers_preserve_arguments_and_returned_object(self):
        timings=diagnostics.SamplingTimings()
        arg=object();value=object();calls=[]
        def inner(*args,**kwargs):
            calls.append((args,kwargs))
            return value
        def outer(*args,**kwargs):return timings.loop(inner,*args,**kwargs)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            result=timings.outer(outer,arg,seed=17)
        self.assertIs(result,value)
        self.assertIs(calls[0][0][0],arg)
        self.assertEqual(calls[0][1],{"seed":17})
        self.assertIsNotNone(timings.loop_seconds)
        self.assertIn("sampler loop",captured.getvalue())

    def test_profiling_does_not_change_conditioning_or_reference_payload(self):
        refs=core.add_pack(pack(),role="subject")
        refs=core.add_pack(pack("video"),refs,"video")
        args=(FakeModel(),RecordingClip(),refs,"<Subject 1> performs the moves in <Video 1>","",64,64,22,True)
        with contextlib.redirect_stdout(io.StringIO()):
            profiled=nodes.BExplainsH3RefPackConditioning().compile(*args,log_performance=True)
            plain=nodes.BExplainsH3RefPackConditioning().compile(*args,log_performance=False)
        for index in (1,2):
            self.assertTrue(torch.equal(profiled[index][0][0],plain[index][0][0]))
            self.assertEqual(set(profiled[index][0][1]),set(plain[index][0][1]))
        a,b=profiled[1][0][1]["minimax_refs"],plain[1][0][1]["minimax_refs"]
        for left,right in zip(a,b):self.assertIs(left["latent"],right["latent"])
        for left,right in zip(profiled[3]["samples"],plain[3]["samples"]):self.assertTrue(torch.equal(left,right))
        self.assertEqual(args[0].model_options,{})
        self.assertIn("wrappers",profiled[0].model_options["transformer_options"])
        self.assertIs(plain[0],args[0])

    def test_visual_token_count_includes_patch_padding(self):
        blocks=[{"kind":"image","latent":torch.zeros(1,24,1,3,5)}]
        items=[{"data":torch.zeros(1,32,32,3)}]
        stats=diagnostics.reference_stats(blocks,items,{"samples":torch.zeros(1,24,22,27,15)},8)
        self.assertIn("6 visual tokens",stats[0])
        self.assertIn("2,464 visual tokens",stats[1])

    def test_full_encode_refmod_import_with_original_frames(self):
        from safetensors.torch import save_file
        z=torch.zeros(1,24,7,2,2)
        path=nodes._path("legacy.safetensors")
        save_file({"latent":z},str(path),metadata={"refmod_meta":json.dumps({"kind":"video","mode":"encode"})})
        original=torch.rand(22,32,32,3)
        vae=RecordingVAE()
        imported=nodes.BExplainsH3ImportRefMod().convert("legacy.safetensors",vae,"H3 Video","legacy",original)[0]
        self.assertTrue(torch.equal(imported["entries"][0]["latent"],z))
        self.assertTrue(torch.equal(imported["entries"][0]["vision"],original[::12]))
        self.assertEqual(len(vae.calls),0)

    def test_workflow_links_and_custom_node_registration(self):
        for path in list((ROOT/"workflows").glob("*.json")) + list((ROOT.parent/"BExplains_H3_RefPacks"/"workflows").glob("*.json")):
            with self.subTest(workflow=path.name):
                flow=json.loads(path.read_text());by_id={n['id']:n for n in flow['nodes']}
                links={link[0]:link for link in flow['links']}
                for link in links.values():
                    _,source,slot,target,input_slot,typ=link
                    self.assertIn(by_id[source]['outputs'][slot]['type'],(typ,'*'))
                    self.assertIn(by_id[target]['inputs'][input_slot]['type'],(typ,'*'))
                    self.assertEqual(by_id[target]['inputs'][input_slot]['link'],link[0])
                    self.assertIn(link[0],by_id[source]['outputs'][slot]['links'])
                for n in flow['nodes']:
                    if not n['type'].startswith('BExplains'):continue
                    cls=ALL_NODE_CLASSES[n['type']]
                    schema=cls.INPUT_TYPES();inputs={i['name']:i for i in n['inputs']}
                    for name,spec in schema.get('required',{}).items():
                        if isinstance(spec[0],str) and spec[0] not in ('STRING','INT','FLOAT','BOOLEAN'):
                            self.assertIn(name,inputs)
                            self.assertIsNotNone(inputs[name]['link'])
                    self.assertEqual(len(n['outputs']),len(cls.RETURN_TYPES))

    def test_native_source_payload_parity(self):
        source=os.environ.get("H3_NATIVE_SOURCE")
        if not source:self.skipTest("Set H3_NATIVE_SOURCE to nodes_minimax_h3.py to compare against native source.")
        tree=ast.parse(Path(source).read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="MiniMaxH3ReferenceToVideo")
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="execute")
        method.decorator_list=[]
        method.returns=None
        code=compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),source,"exec")
        def set_values(cond,values):return [[t,{**m,**values}] for t,m in cond]
        env={"torch":torch,"math":__import__('math'),"FPS":24,"CANVAS_MULTIPLE":32,"REF_IMAGE_SHORT_EDGE":2048,
             "_resize":resize,"adapt_canvas":lambda w,h:(w,h),"_empty_av_latent":empty_av,
             "node_helpers":types.SimpleNamespace(conditioning_set_values=set_values),"io":types.SimpleNamespace(NodeOutput=lambda *args:args)}
        exec(code,env)
        frames=torch.rand(22,32,32,3);image=torch.rand(1,32,32,3)
        vae=RecordingVAE()
        ip=nodes.BExplainsH3EncodeImages().encode(vae,image,"character","H3 Video",0.01)[0]
        vp=nodes.BExplainsH3EncodeVideo().encode(vae,frames,24,"dance",0.01,"trim tail")[0]
        refs=core.add_pack(vp,core.add_pack(ip,role="subject"),"video")
        clip_a,clip_b=RecordingClip(),RecordingClip()
        result=nodes.BExplainsH3RefPackConditioning().compile(FakeModel(),clip_a,refs,"<Subject 1> performs the moves in <Video 1>","",64,64,22,True)
        compiled=result[4].split('\n\n',2)[-1]
        # Use precisely the compiled prompt, same image/video inputs and a recording VAE.
        expected=env['execute'](None,clip_b,clip_a.calls[0][0],64,64,22,vae=RecordingVAE(),ref_images={"ref_image_0":image},ref_videos={"ref_video_0":frames})
        self.assertEqual(clip_a.calls[0][0],clip_b.calls[0][0])
        for a,b in zip(clip_a.calls[0][1]['minimax_ref_items'],clip_b.calls[0][1]['minimax_ref_items']):
            self.assertEqual(a['type'],b['type']);self.assertTrue(torch.equal(a['data'],b['data']))
            self.assertEqual(a.get('timestamps'),b.get('timestamps'))
        for a,b in zip(result[1][0][1]['minimax_refs'],expected[0][0][1]['minimax_refs']):
            self.assertEqual(set(a),set(b))
            for k in a:
                if isinstance(a[k],torch.Tensor):self.assertTrue(torch.equal(a[k],b[k]))
                else:self.assertEqual(a[k],b[k])


if __name__=="__main__":unittest.main()
