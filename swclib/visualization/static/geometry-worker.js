// Keep all CPU buffer assembly off the UI thread; never decimate morphology.
const neurons = new Map();
function pick(data) {
  const {base,center,right,up,forward,scaleX,scaleY,farPlane,sliceAxis,slicePosition,slab}=data.ray;
  const visible=new Set(data.visible),dot=(x,y,z,a)=>x*a[0]+y*a[1]+z*a[2];
  let best=null,bestDistance=Infinity,bestDepth=Infinity;
  function accept(distance,depth,radius,xyz,a,b=a,t=0) {
    if(distance>radius*radius||distance>bestDistance+1e-6||(Math.abs(distance-bestDistance)<1e-6&&depth>=bestDepth))return;
    bestDistance=distance;bestDepth=depth;
    best=[0,1,2].map(i=>xyz[a+i]+(xyz[b+i]-xyz[a+i])*t);
  }
  for(const [slot,n] of neurons) {
    if(!visible.has(slot))continue;
    const xyz=n.positions,projected=new Float64Array(n.count*3);
    for(let i=0;i<xyz.length;i+=3) {
      const x=xyz[i]-base[0],y=xyz[i+1]-base[1],z=xyz[i+2]-base[2];
      const px=dot(x,y,z,right)*scaleX,py=dot(x,y,z,up)*scaleY;
      const depth=dot(xyz[i]-center[0],xyz[i+1]-center[1],xyz[i+2]-center[2],forward);
      projected[i]=px;projected[i+1]=py;projected[i+2]=depth;
      if(data.points&&Math.abs(depth)<=farPlane&&(sliceAxis<0||Math.abs(xyz[i+sliceAxis]-slicePosition)<=slab))
        accept(px*px+py*py,depth,Math.max(4,data.pointSize/2),xyz,i);
    }
    if(!data.lines)continue;
    for(let i=0;i<n.edges.length;i+=2) {
      const a=n.edges[i]*3,b=n.edges[i+1]*3;
      let lo=0,hi=1;
      const clip=(v,w,min,max)=>{
        if(Math.abs(w-v)<1e-12)return v>=min&&v<=max;
        const p=(min-v)/(w-v),q=(max-v)/(w-v);
        lo=Math.max(lo,Math.min(p,q));hi=Math.min(hi,Math.max(p,q));return lo<=hi;
      };
      if(!clip(projected[a+2],projected[b+2],-farPlane,farPlane))continue;
      if(sliceAxis>=0&&!clip(xyz[a+sliceAxis],xyz[b+sliceAxis],slicePosition-slab,slicePosition+slab))continue;
      const x=projected[a],y=projected[a+1],dx=projected[b]-x,dy=projected[b+1]-y,length=dx*dx+dy*dy;
      const t=length>1e-12?Math.max(lo,Math.min(hi,-(x*dx+y*dy)/length)):(projected[a+2]<=projected[b+2]?lo:hi);
      accept((x+t*dx)**2+(y+t*dy)**2,projected[a+2]+t*(projected[b+2]-projected[a+2]),4,xyz,a,b,t);
    }
  }
  return best;
}
self.onmessage = ({data}) => {
  if (data.type === 'pick') {
    self.postMessage({type:'pick',requestId:data.requestId,revision:data.revision,target:pick(data)});return;
  }
  if (data.type === 'add') {
    const n = data.count, offset = data.offset;
    data.positions = new Float32Array(data.buffer, offset, n * 3);
    data.colors = new Uint32Array(data.buffer, offset + n * 12, n);
    data.edges = new Uint32Array(data.buffer, offset + n * 16, data.edgeCount * 2);
    neurons.set(data.slot, data);
    return;
  }
  if (data.type === 'clear') { neurons.clear(); return; }
  if (data.type === 'remove') { for (const slot of data.slots) neurons.delete(slot); return; }
  if (data.type !== 'build') return;
  let count = 0, edgeCount = 0;
  for (const n of neurons.values()) { count += n.count; edgeCount += n.edgeCount; }
  const vertices = new ArrayBuffer(count * 20);
  const floats = new Float32Array(vertices), words = new Uint32Array(vertices);
  const edges = new Uint32Array(edgeCount * 2);
  let offset = 0, eoffset = 0;
  for (const n of neurons.values()) {
    const xyz = n.positions, colors = n.colors;
    for (let i = 0; i < n.count; i++) {
      const base = (offset + i) * 5;
      floats[base] = xyz[i * 3]; floats[base + 1] = xyz[i * 3 + 1]; floats[base + 2] = xyz[i * 3 + 2];
      words[base + 3] = colors[i];
      floats[base + 4] = n.slot;
    }
    const sourceEdges = n.edges;
    for (let i = 0; i < sourceEdges.length; i++) edges[eoffset++] = sourceEdges[i] + offset;
    offset += n.count;
  }
  self.postMessage({type: 'geometry', revision: data.revision, sequence: data.sequence, vertices, edges: edges.buffer,
                    count, edgeCount, slots:[...neurons.keys()]}, [vertices, edges.buffer]);
};
