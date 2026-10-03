"""Compress comparison previews in review.html; preserve raw annotation PNGs.

This changes presentation only, never masks, arrays, site tables or coordinates.
Run after build_benchmark. Keeps the standalone review pack below 20 MB.
"""
import base64
import io
import json
import re

from PIL import Image
from analysis.morphology.build_benchmark import OUT


def main():
    path=OUT/'review.html';text=path.read_text()
    pattern=r'(<script id="benchmark-data" type="application/json">)(.*?)(</script>)'
    match=re.search(pattern,text,re.S)
    data=json.loads(match.group(2))
    for r in data['rois']:
        for key,src in r['images'].items():
            if key=='raw' or src.startswith('data:image/jpeg'):
                continue
            im=Image.open(io.BytesIO(base64.b64decode(src.split(',',1)[1]))).convert('RGB')
            out=io.BytesIO();im.save(out,format='JPEG',quality=82,optimize=True,subsampling=0)
            r['images'][key]='data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode()
        assert r['images']['raw'].startswith('data:image/png;base64,')
    payload=json.dumps(data,allow_nan=False).replace('</','<\\/')
    text=text[:match.start(2)]+payload+text[match.end(2):]
    text=text.replace('Original stored BSE values; display only.','Original stored BSE values, lossless PNG; annotation reference.')
    text=text.replace('Algorithm prediction. This is not ground truth or an expert correction.',
                      'Algorithm prediction; compressed JPEG preview. This is not ground truth or an expert correction.')
    path.write_text(text)
    assert path.stat().st_size<20_000_000,'Review HTML exceeds repository file limit'
    print(f'Review page optimised: {path.stat().st_size/1e6:.2f} MB; all annotation references remain lossless PNG.')


if __name__=='__main__':main()
