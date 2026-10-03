"""Exercise actual review-page control logic in Node with a minimal DOM model.

This verifies events/state/export guards; it is not a browser rendering test.
The fixture is synthetic and no real expert annotations are created.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from analysis.morphology.build_benchmark import OUT


def main():
    node=shutil.which('node')
    if not node:raise RuntimeError('Node required for control-logic verification')
    source=(OUT/'review.html').read_text()
    script=re.findall(r'<script>(.*?)</script>',source,re.S)[0]
    roi=dict(id='synthetic-test',batch='Synthetic',site='test',split='development',reason='unit fixture',
             box_raw_yxyx=[0,0,20,30],selection='synthetic',W=30,H=20,
             images={k:'fixture:'+k for k in ['raw','context','baseline','hysteresis','width']})
    fixture=dict(manifest_id='synthetic-ui-only',rois=[roi,{**roi,'id':'synthetic-test-2','site':'test-2'}])
    setup=r'''
const assert=require('assert'),vm=require('vm');
class Element {
 constructor(){this.value='';this.checked=false;this.disabled=false;this.textContent='';this.children=[];this.width=30;this.height=20;this.files=[]}
 append(x){this.children.push(x)}
 getContext(){return new Proxy({}, {get:()=>()=>{},set:()=>true})}
 getBoundingClientRect(){return {left:0,top:0,width:this.width,height:this.height}}
 click(){this.clicked=true}
}
const ids=['benchmark-data','error','progress','canvas','show','phase','roi','meta','selection','context','background','reviewer','notes','measurable','reviewed','prev','next','preview','view','viewhint','finish','cancel','undo','export','import'];
const elements=Object.fromEntries(ids.map(k=>[k,new Element()]));
elements['benchmark-data'].textContent=JSON.stringify(FIXTURE);elements.show.checked=true;elements.phase.value='0';elements.view.value='raw';
class MockImage {set src(v){this.value=v;if(this.onload)this.onload()}}
const storage=new Map();
const sandbox={document:{getElementById:k=>elements[k],createElement:()=>new Element()},Image:MockImage,
 localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},
 Blob:class{},URL:{createObjectURL:()=> 'fixture:export',revokeObjectURL:()=>{}},setTimeout:()=>{},console};
vm.createContext(sandbox);vm.runInContext(SCRIPT,sandbox);
const read=s=>vm.runInContext(s,sandbox);
assert.equal(elements.progress.textContent,'0 / 2 reviewed');
assert.equal(elements.preview.src,'fixture:raw');
assert(elements.prev.disabled);assert(!elements.next.disabled);
elements.reviewed.checked=true;elements.reviewed.onchange();assert(!elements.reviewed.checked);
elements.reviewer.value='synthetic test';elements.reviewer.oninput();
elements.measurable.value='false';elements.measurable.onchange();
elements.reviewed.checked=true;elements.reviewed.onchange();assert.equal(read('ann().review_status'),'reviewed');
elements.measurable.value='true';elements.measurable.onchange();assert.equal(read('ann().review_status'),'unreviewed');
elements.reviewed.checked=true;elements.reviewed.onchange();assert(!elements.reviewed.checked);
for(const [x,y] of [[3,3],[10,3],[10,10],[3,10]])elements.canvas.onclick({target:elements.canvas,clientX:x,clientY:y});
elements.finish.onclick();assert.equal(read('ann().polygons.length'),1);
elements.reviewed.checked=true;elements.reviewed.onchange();assert.equal(read('ann().review_status'),'reviewed');
elements.undo.onclick();assert.equal(read('ann().polygons.length'),0);assert.equal(read('ann().review_status'),'unreviewed');
elements.background.checked=true;elements.background.onchange();assert.equal(read('ann().background'),'solid');
elements.view.value='hysteresis';elements.view.onchange();assert.equal(elements.preview.src,'fixture:hysteresis');
elements.next.onclick();assert(elements.next.disabled);assert.equal(elements.reviewer.value,'');
elements.prev.onclick();assert.equal(elements.reviewer.value,'synthetic test');
elements.canvas.onclick({target:elements.canvas,clientX:3,clientY:3});elements.export.onclick();assert(elements.error.textContent.includes('Finish or cancel'));
console.log('Review controls passed: navigation, phase polygon/undo, review gate, invalidation, measurability, image switching, saved state and unfinished-export guard. Synthetic fixture only; browser layout not tested.');
'''
    setup=setup.replace('FIXTURE',json.dumps(fixture)).replace('SCRIPT',json.dumps(script))
    with tempfile.TemporaryDirectory(prefix='polaron-review-ui-') as td:
        p=Path(td)/'test.cjs';p.write_text(setup)
        result=subprocess.run([node,str(p)],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr.strip())
        print(result.stdout.strip())


if __name__=='__main__':main()
