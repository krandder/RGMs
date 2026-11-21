"""Decode and fingerprint all PNG pixels and simulated movie frames."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image
EXPECT = {
    'DEM_AtariIII': {'AtariIII_colab1_dem_'+s+'.png' for s in (
        'gameplay','attractors_basin','attractors_mdp_post_sort','generative_ai',
        'active_inference_nr','structure_learning','before_compression_rgb',
        'orbits_before','with_compression_rgb','orbits_after')},
    'DEM_drone_vi': {'DEM_drone_VI_colab2_'+s+'.png' for s in (
        'true_env_early','true_scene_post_generate','mdp_params','rgm',
        'active_inference','wayfinding')} | {'DEM_drone_VI_colab2_behaviour.mp4'},
}

def inspect(root,require_all=False):
    output=[]
    for name in ('DEM_AtariIII','DEM_drone_vi'):
        paths=sorted((root/'visualizations'/name).glob('*'))
        if require_all and (missing:=EXPECT[name]-{p.name for p in paths}):
            raise ValueError(f'{name}: missing documented output files: {sorted(missing)}')
        for path in paths:
            if path.suffix not in ('.png','.mp4'): continue
            d={'file':str(path.relative_to(root)),'nbytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            if path.suffix=='.png':
                with Image.open(path) as im:
                    d['size']=im.size; im.verify()
                with Image.open(path) as im:
                    im.load()
                    d['rgba_sha256']=hashlib.sha256(im.convert('RGBA').tobytes()).hexdigest()
            else:
                reader=imageio.get_reader(str(path),format='FFMPEG')
                try:
                    meta=reader.get_meta_data();d['duration']=meta.get('duration');d['fps']=meta.get('fps')
                    checksum=hashlib.sha256();count=0
                    for frame in reader:
                        if count==0: d['first_decoded_frame']=list(frame.shape)
                        checksum.update(frame.tobytes());count+=1
                    if not count: raise ValueError(f'empty movie: {path.name}')
                    d['decoded_frames']=count;d['frames_sha256']=checksum.hexdigest()
                finally: reader.close()
            output.append(d)
    return output
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkout',type=Path,required=True)
    parser.add_argument('--require-all',action='store_true',help='also require all 17 named outputs of the two native scripts')
    args=parser.parse_args();print(json.dumps(inspect(args.checkout,args.require_all),indent=2,allow_nan=False))
