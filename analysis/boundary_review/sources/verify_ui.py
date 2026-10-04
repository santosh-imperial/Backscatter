"""Exercise the raw-only page controls on a synthetic DOM; no real annotations."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def main():
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node is required for UI control verification")
    source = Path(__file__).with_name("review.html.in").read_text()
    script = re.findall(r"<script>(.*?)</script>", source, re.S)[0]
    fixture = dict(schema_version=1, packet_id="synthetic-ui-only", kind="development", rois=[
        dict(id="D01", W=30, H=20, image="fixture:raw"),
        dict(id="D02", W=30, H=20, image="fixture:raw2")])
    runner = r'''
const assert=require('assert'),vm=require('vm');
class Element {
 constructor(){this.value='';this.checked=false;this.disabled=false;this.textContent='';this.children=[];this.width=30;this.height=20;this.files=[];this.style={}}
 append(x){this.children.push(x)}
 getContext(){return new Proxy({}, {get:()=>()=>{},set:()=>true})}
 getBoundingClientRect(){return {left:0,top:0,width:this.width,height:this.height}}
 click(){this.clicked=true}
}
const ids=['review-data','error','progress','canvas','show','phase','roi','purpose','background','reviewer','notes','measurable','reviewed','prev','next','finish','cancel','undo','export','import','fit','actual','cropname','exposure','independent'];
const elements=Object.fromEntries(ids.map(k=>[k,new Element()]));
elements['review-data'].textContent=JSON.stringify(FIXTURE);elements.show.checked=true;elements.phase.value='0';
class MockImage {set src(v){this.value=v;if(this.onload)this.onload()}}
const storage=new Map(),downloads=[];
const sandbox={document:{getElementById:k=>elements[k],createElement:()=>{const e=new Element();downloads.push(e);return e}},Image:MockImage,
 localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},
 Blob:class{},URL:{createObjectURL:()=> 'fixture:export',revokeObjectURL:()=>{}},setTimeout:()=>{},console};
vm.createContext(sandbox);vm.runInContext(SCRIPT,sandbox);
const read=s=>vm.runInContext(s,sandbox);
assert.equal(elements.progress.textContent,'0 / 2 reviewed');assert(elements.prev.disabled);
elements.reviewed.checked=true;elements.reviewed.onchange();assert(!elements.reviewed.checked);
elements.reviewer.value='synthetic fixture';elements.reviewer.oninput();
elements.measurable.value='false';elements.measurable.onchange();
elements.reviewed.checked=true;elements.reviewed.onchange();assert(!elements.reviewed.checked);
elements.exposure.value='prior_exposure';elements.exposure.onchange();elements.independent.checked=true;elements.independent.onchange();
elements.reviewed.checked=true;elements.reviewed.onchange();assert.equal(read('ann().review_status'),'reviewed');
elements.measurable.value='true';elements.measurable.onchange();assert.equal(read('ann().review_status'),'unreviewed');
elements.reviewed.checked=true;elements.reviewed.onchange();assert(!elements.reviewed.checked);
for(const [x,y] of [[3,3],[10,3],[10,10],[3,10]])elements.canvas.onclick({target:elements.canvas,clientX:x,clientY:y});
elements.finish.onclick();assert.equal(read('ann().polygons.length'),1);
elements.reviewed.checked=true;elements.reviewed.onchange();assert.equal(read('ann().review_status'),'reviewed');
elements.export.onclick();assert(downloads.some(e=>e.download==='polaron_development_boundaries.json'&&e.clicked));
elements.actual.onclick();assert.equal(elements.canvas.style.width,'30px');elements.fit.onclick();assert.equal(elements.canvas.style.width,'100%');
elements.undo.onclick();assert.equal(read('ann().polygons.length'),0);assert.equal(read('ann().review_status'),'unreviewed');
elements.next.onclick();assert(elements.next.disabled);assert.equal(elements.reviewer.value,'');elements.prev.onclick();assert.equal(elements.reviewer.value,'synthetic fixture');
elements.canvas.onclick({target:elements.canvas,clientX:3,clientY:3});elements.export.onclick();assert(elements.error.textContent.includes('Finish or cancel'));
elements.cancel.onclick();
assert.throws(()=>read("validate({...state,packet_id:'wrong'})"));
assert.throws(()=>read("validate({...state,rois:[state.rois[0],state.rois[0]]})"));
assert.throws(()=>read("validate({...state,rois:state.rois.map(a=>({...a,measurable:'false'}))})"));
console.log('Synthetic UI controls passed: independent-review/exposure gates, abstention, polygon/undo, export identity, navigation, zoom, invalidation and bad-import rejection.');
'''
    runner = runner.replace("FIXTURE", json.dumps(fixture)).replace("SCRIPT", json.dumps(script))
    with tempfile.TemporaryDirectory(prefix="polaron-independent-ui-") as tmp:
        path = Path(tmp) / "test.cjs"
        path.write_text(runner)
        result = subprocess.run([node, str(path)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        print(result.stdout.strip())


if __name__ == "__main__":
    main()
