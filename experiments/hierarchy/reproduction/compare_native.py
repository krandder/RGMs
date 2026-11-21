"""Compare two native demo runs without treating elapsed time as scientific output."""
import argparse
import json
from pathlib import Path
import re

from check_media import inspect
from summarize import receipt


def extract_games(run):
    trace=(run/'native-atari.log').read_text(errors='replace')
    return [(int(i), int(n), float(v)) for i,n,v in re.findall(
        r'game (\d+)/(\d+) wall_s=[\d.]+ F4_hits=([\d.]+)', trace)]


def compare(first,repeat,first_tree,repeat_tree):
    runs={}
    for name in ('native-atari','native-drone'):
        a=receipt(first,name)
        b=receipt(repeat,name)
        runs[name]={'first':a,'repeat':b,'both_native_markers':a['success'] and b['success']}
    clean=all(o['both_native_markers'] for o in runs.values())
    a={i['file']:i for i in inspect(first_tree,clean)}
    b={i['file']:i for i in inspect(repeat_tree,clean)}
    media=[]
    for name in sorted(a.keys() | b.keys()):
        if name not in a or name not in b:
            media.append({'file':name,'present_in_both':False})
            continue
        fields=('rgba_sha256','size') if name.endswith('.png') else ('frames_sha256','decoded_frames','first_decoded_frame','fps')
        media.append({'file':name,'present_in_both':True,'same_encoded_bytes':a[name]['sha256']==b[name]['sha256'],
                      'same_decoded_pixels':all(a[name][key]==b[name][key] for key in fields),
                      'repeat_encoded_sha256':b[name]['sha256'],'repeat_decoded_sha256':b[name][fields[0]]})
    g1,g2=extract_games(first),extract_games(repeat)
    return {'runs':runs,'media':media,'atari_game_hits':{'all_32_match':len(g1)==len(g2)==32 and g1==g2,
                'first':[v for _,_,v in g1],'repeat':[v for _,_,v in g2]}}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('first','repeat','first-tree','repeat-tree','output'): ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args()
    result=compare(args.first,args.repeat,args.first_tree,args.repeat_tree)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
