import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';

const source=await readFile(new URL('../static/renderer.js',import.meta.url),'utf8');
const {Renderer}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
function camera() {
  return Object.assign(Object.create(Renderer.prototype),{
    rotation:[0,0,0,1],orientation:'3d',center:[1,2,3],span:100,invalidate(){},
    canvas:{getBoundingClientRect:()=>({left:0,top:0,width:1000,height:1000})},
  });
}
function close(a,b,tolerance=1e-12) {
  assert.equal(a.length,b.length);
  a.forEach((value,i)=>assert.ok(Math.abs(value-b[i])<tolerance,`${a} != ${b}`));
}
function orthonormal(r) {
  const b=r.basis();
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)
    assert.ok(Math.abs(b[i].reduce((sum,v,k)=>sum+v*b[j][k],0)-(i===j?1:0))<1e-12);
  close([b[0][1]*b[1][2]-b[0][2]*b[1][1],b[0][2]*b[1][0]-b[0][0]*b[1][2],b[0][0]*b[1][1]-b[0][1]*b[1][0]],b[2]);
}

test('rim drag rolls around the viewing axis; reverse drag restores orientation',()=>{
  const r=camera(),original=r.basis().flat();
  r.rotateTrackball([950,500],[500,50]);
  close(r.basis()[2],[0,0,1]);
  close(r.basis()[0],[0,-1,0]);
  r.rotateTrackball([500,50],[950,500]);
  close(r.basis().flat(),original);
  close(r.center,[1,2,3]);assert.equal(r.span,100);
});

test('screen-local rotation has no pole and supports arbitrary roll',()=>{
  const r=camera();
  r.rotate(0,Math.PI/.006);close(r.basis()[2],[0,0,-1]);
  r.rotate(0,0,Math.PI/2/.006);orthonormal(r);
  const before=r.basis().flat();
  for(const delta of [[2*Math.PI/.006,0,0],[0,2*Math.PI/.006,0],[0,0,2*Math.PI/.006]])r.rotate(...delta);
  close(r.basis().flat(),before);
  for(let i=0;i<10000;i++)r.rotate(Math.sin(i)*4,Math.cos(i)*3,Math.sin(i*.1)*2);
  orthonormal(r);close([Math.hypot(...r.rotation)],[1]);
});

test('opposite rim points and zero movement remain finite',()=>{
  const r=camera();r.rotateTrackball([50,500],[950,500]);orthonormal(r);
  const before=r.rotation.slice();r.rotateTrackball([500,500],[500,500]);close(r.rotation,before);
});

test('orthogonal slices retain fixed bases and restore free 3D orientation',()=>{
  const r=camera();r.rotate(80,-40,30);const before=r.basis().flat();
  r.orientation='xy';close(r.basis().flat(),[1,0,0,0,1,0,0,0,1]);
  r.orientation='xz';orthonormal(r);r.orientation='yz';orthonormal(r);
  r.orientation='3d';close(r.basis().flat(),before);
});
