"""Read all emitted PNGs and decode the first frame of each simulation movie."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image

def inspect(root):
    output=[]
    for name in ('DEM_AtariIII','DEM_drone_vi'):
        for path in sorted((root/'visualizations'/name).glob('*')):
            if path.suffix not in ('.png','.mp4'): continue
            d={'file':str(path.relative_to(root)),'nbytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            if path.suffix=='.png':
                with Image.open(path) as im:
                    d['size']=im.size; im.verify()
            else:
                reader=imageio.get_reader(str(path),format='FFMPEG')
                try:
                    d['first_decoded_frame']=list(reader.get_data(0).shape)
                    meta=reader.get_meta_data();d['duration']=meta.get('duration');d['fps']=meta.get('fps')
                finally: reader.close()
            output.append(d)
    return output
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkout',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(inspect(args.checkout),indent=2,allow_nan=False))
