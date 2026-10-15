const swcVertex = `#version 300 es
precision highp float;
layout(location=0) in vec3 position;
layout(location=1) in vec4 topologyColor;
layout(location=2) in float neuron;
uniform vec3 center, rightAxis, upAxis, forwardAxis;
uniform float span, aspect, farPlane, pointSize, slicePosition, slab;
uniform int sliceAxis, colorMode, paletteWidth;
uniform sampler2D palette;
out vec4 color;
out float sliceDistance;
void main() {
  int id = int(neuron);
  vec4 style = texelFetch(palette, ivec2(id % paletteWidth, id / paletteWidth), 0);
  vec3 p = position - center;
  gl_Position = vec4(dot(p,rightAxis)/(span*aspect), dot(p,upAxis)/span, dot(p,forwardAxis)/farPlane, 1.);
  if (style.a < .5) gl_Position = vec4(2.,2.,2.,1.);
  gl_PointSize = pointSize;
  color = colorMode == 0 ? topologyColor : vec4(style.rgb,1.);
  sliceDistance = sliceAxis < 0 ? 0. : position[max(sliceAxis,0)] - slicePosition;
}`;
const swcFragment = `#version 300 es
precision highp float;
in vec4 color;
in float sliceDistance;
uniform bool points;
uniform float slab, opacity;
out vec4 fragColor;
void main() {
  if(abs(sliceDistance) > slab) discard;
  float alpha = opacity;
  if(points) {
    float radius = length(gl_PointCoord - .5) * 2.;
    float aa = fwidth(radius);
    alpha *= 1. - smoothstep(1.-aa, 1., radius);
    if(alpha < .01) discard;
  }
  fragColor = vec4(color.rgb, alpha);
}`;
const quadVertex = `#version 300 es
out vec2 uv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  uv = p;
  gl_Position = vec4(p*2.-1., 0., 1.);
}`;
const planeVertex = `#version 300 es
precision highp float;
uniform vec3 center, rightAxis, upAxis, forwardAxis, planeCenter, planeU, planeV;
uniform float span, aspect, farPlane, planeSize;
uniform bool edgeOn;
out vec2 planeUV;
out float edgeDistance;
void main() {
  vec2 corners[6]=vec2[6](vec2(-1.,-1.),vec2(1.,-1.),vec2(1.,1.),vec2(-1.,-1.),vec2(1.,1.),vec2(-1.,1.));
  planeUV=corners[gl_VertexID];
  vec3 p=planeCenter+(planeU*planeUV.x+planeV*planeUV.y)*planeSize-center;
  gl_Position=vec4(dot(p,rightAxis)/(span*aspect),dot(p,upAxis)/span,dot(p,forwardAxis)/farPlane,1.);
  edgeDistance=0.;
  if(edgeOn) {
    gl_Position=vec4(planeUV,0.,1.);
    vec3 screen=center+rightAxis*planeUV.x*span*aspect+upAxis*planeUV.y*span;
    edgeDistance=dot(planeCenter-screen,cross(planeU,planeV));
  }
}`;
const planeFragment = `#version 300 es
precision highp float;
in vec2 planeUV;
in float edgeDistance;
uniform bool edgeOn;
out vec4 fragColor;
void main() {
  if(edgeOn) {
    float line=1.-smoothstep(.5,1.5,abs(edgeDistance)/max(fwidth(edgeDistance),.000001));
    if(line<=0.)discard;
    fragColor=vec4(.58,.79,.88,line);return;
  }
  vec2 g=planeUV*8.;
  vec2 grid=abs(fract(g-.5)-.5)/max(fwidth(g),vec2(.0001));
  float line=1.-min(min(grid.x,grid.y),1.);
  float edge=smoothstep(.975,1.,max(abs(planeUV.x),abs(planeUV.y)));
  fragColor=vec4(.58,.79,.88,.10+.22*line+.3*edge);
}`;
const volumeHeader = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler3D;
in vec2 uv;
uniform sampler3D volume;
uniform vec3 center, rightAxis, upAxis, forwardAxis, origin, shape;
uniform float span, aspect, farPlane, black, white, gammaValue, brightness, opacity, slicePosition;
uniform int mode, sliceAxis;
uniform bool linearFiltering;
out vec4 fragColor;
float intensity(vec3 p) {
  if(linearFiltering) return texture(volume,(p-origin+.5)/shape).r;
  // Preserve interpolation on GPUs without float-texture linear filtering.
  vec3 voxel=clamp(p-origin,vec3(0.),shape-1.);
  ivec3 a=ivec3(floor(voxel)), b=min(a+1,ivec3(shape)-1);
  vec3 t=fract(voxel);
  float x00=mix(texelFetch(volume,a,0).r,texelFetch(volume,ivec3(b.x,a.y,a.z),0).r,t.x);
  float x10=mix(texelFetch(volume,ivec3(a.x,b.y,a.z),0).r,texelFetch(volume,ivec3(b.x,b.y,a.z),0).r,t.x);
  float x01=mix(texelFetch(volume,ivec3(a.x,a.y,b.z),0).r,texelFetch(volume,ivec3(b.x,a.y,b.z),0).r,t.x);
  float x11=mix(texelFetch(volume,ivec3(a.x,b.y,b.z),0).r,texelFetch(volume,b,0).r,t.x);
  return mix(mix(x00,x10,t.y),mix(x01,x11,t.y),t.z);
}
float tone(float v) { return clamp(pow(clamp((v-black)/max(white-black,.00001),0.,1.),1./gammaValue)+brightness,0.,1.); }`;
const volumeFragment = volumeHeader + `
void main() {
  vec3 base = center + rightAxis*((uv.x*2.-1.)*span*aspect) + upAxis*((uv.y*2.-1.)*span);
  vec3 ray = forwardAxis;
  vec3 start = base - ray*farPlane;
  vec3 inverseRay = 1. / mix(vec3(.000001),ray,greaterThan(abs(ray),vec3(.000001)));
  vec3 a = (origin-.5-start)*inverseRay, b = (origin+shape-.5-start)*inverseRay;
  vec3 lo = min(a,b), hi = max(a,b);
  float enter = max(max(lo.x,lo.y),lo.z), leave = min(min(hi.x,hi.y),hi.z);
  if(leave < max(enter,0.)) discard;
  float value = 0.;
  if(mode == 2) {
    float t = (slicePosition-start[sliceAxis])/ray[sliceAxis];
    if(t < enter || t > leave) discard;
    value = tone(intensity(start+ray*t));
  } else {
    float distance = leave-enter;
    int steps = min(4096, max(1,int(ceil(distance/.8))));
    float stepSize = distance/float(steps);
    float accumulated = 0.;
    for(int i=0;i<4096;i++) {
      if(i >= steps) break;
      float v = intensity(start+ray*(enter+(float(i)+.5)*stepSize));
      if(mode == 0) value = max(value,v);
      else {
        float alpha = tone(v)*.065*stepSize;
        accumulated += (1.-accumulated)*clamp(alpha,0.,1.);
        if(accumulated > .995) break;
      }
    }
    value = mode == 0 ? tone(value) : accumulated;
  }
  fragColor = vec4(vec3(value),opacity);
}`;

// Pick only two pixels on demand. Reuse the displayed float texture and tone
// mapping; no CPU volume copy, full-screen picking buffer, or extra frame draw.
const volumePickFragment = volumeHeader + `
uniform vec3 pickBase;
void main() {
  if(opacity <= 0.) discard;
  vec3 ray=forwardAxis, start=pickBase-ray*farPlane;
  vec3 inv=1./mix(vec3(.000001),ray,greaterThan(abs(ray),vec3(.000001)));
  vec3 a=(origin-.5-start)*inv, b=(origin+shape-.5-start)*inv;
  vec3 lo=min(a,b),hi=max(a,b);
  float enter=max(max(lo.x,lo.y),lo.z),leave=min(min(hi.x,hi.y),hi.z);
  if(leave<max(enter,0.))discard;
  float target=0.,best=0.;
  if(mode==2) {
    target=(slicePosition-start[sliceAxis])/ray[sliceAxis];
    if(target<enter||target>leave)discard;
    best=tone(intensity(start+ray*target));
  } else {
    int steps=min(4096,max(1,int(ceil((leave-enter)/.8))));
    float stepSize=(leave-enter)/float(steps),accumulated=0.;
    target=enter+.5*stepSize;
    for(int i=0;i<4096;i++) {
      if(i>=steps)break;
      float t=enter+(float(i)+.5)*stepSize;
      float v=intensity(start+ray*t),weight=0.;
      if(mode==0)weight=v;
      else {
        weight=(1.-accumulated)*clamp(tone(v)*.065*stepSize,0.,1.);
        accumulated+=weight;
      }
      if(weight>best){best=weight;target=t;}
      if(mode==1&&accumulated>.995)break;
    }
    if(mode==0)best=tone(best);
  }
  if(best<=.00001)discard;
  uint bits=floatBitsToUint(target);
  fragColor=gl_FragCoord.x<1.?vec4(uvec4(bits,bits>>8,bits>>16,bits>>24)&uvec4(255u))/255.:vec4(1.);
}`;

function program(gl, vertex, fragment) {
  const result = gl.createProgram();
  for (const [kind, source] of [[gl.VERTEX_SHADER, vertex], [gl.FRAGMENT_SHADER, fragment]]) {
    const shader = gl.createShader(kind);
    gl.shaderSource(shader, source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(result, shader); gl.deleteShader(shader);
  }
  gl.linkProgram(result);
  if (!gl.getProgramParameter(result, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(result));
  const uniforms = {};
  for (let i=0; i<gl.getProgramParameter(result,gl.ACTIVE_UNIFORMS); i++) {
    const u = gl.getActiveUniform(result,i); uniforms[u.name] = gl.getUniformLocation(result,u.name);
  }
  return {program:result, uniforms};
}

function multiplyQuaternion(a,b) {
  const [x,y,z,w]=a,[X,Y,Z,W]=b;
  return [w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z];
}

function axisQuaternion(axis,angle) {
  const length=Math.hypot(...axis);
  if(length<1e-15)return [0,0,0,1];
  const scale=Math.sin(angle/2)/length;
  return [...axis.map(v=>v*scale),Math.cos(angle/2)];
}

export class Renderer {
  constructor(canvas, changed, failed) {
    this.canvas = canvas; this.changed = changed;
    this.stateSerial=0;
    this.axesCanvas = document.getElementById('orientation-axes');
    this.axesContext = this.axesCanvas?.getContext('2d');
    const gl = canvas.getContext('webgl2', {antialias:true, alpha:false, powerPreference:'high-performance'});
    if (!gl) throw new Error('浏览器未启用 WebGL2。请启用硬件加速后使用 Chrome、Edge 或 Firefox。');
    this.gl = gl;
    this.linearFiltering=!!gl.getExtension('OES_texture_float_linear');
    this.swcProgram = program(gl,swcVertex,swcFragment); this.volumeProgram = program(gl,quadVertex,volumeFragment);
    this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
    this.vbo = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,this.vbo);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,20,0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1,4,gl.UNSIGNED_BYTE,true,20,12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2,1,gl.FLOAT,false,20,16);
    this.ebo=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.ebo);
    gl.bindVertexArray(null);
    this.palette=gl.createTexture(); this.paletteWidth=256;
    this.count=0; this.edgeCount=0; this.center=[0,0,0]; this.span=100; this.extent=100;
    this.rotation=multiplyQuaternion(axisQuaternion([0,1,0],-.35),axisQuaternion([1,0,0],-.3));
    this.mode='mip'; this.orientation='3d';
    this.settings={pointSize:5,points:true,lines:true,colorMode:0,swcOpacity:1,black:0,white:1,gamma:1,brightness:0,opacity:1,slab:3};
    this.slices=[0,0,0]; this.volume=null; this.cropStart=[0,0,0]; this.pending=false;
    this.drawCalls=0; this.lastFrameMs=0;
    this.setPalette([]);
    this.resizeObserver = new ResizeObserver(()=>this.invalidate()); this.resizeObserver.observe(canvas);
    this.bindControls();
    canvas.addEventListener('webglcontextlost',event=>{ event.preventDefault(); failed('显卡上下文丢失，请刷新页面并缩小图像区域。'); });
    this.invalidate();
  }
  capabilities() {
    return {max_texture_size:this.gl.getParameter(this.gl.MAX_3D_TEXTURE_SIZE),volume_mib:128};
  }
  basis() {
    if(this.orientation==='xy') return [[1,0,0],[0,1,0],[0,0,1]];
    if(this.orientation==='xz') return [[1,0,0],[0,0,1],[0,-1,0]];
    if(this.orientation==='yz') return [[0,1,0],[0,0,1],[1,0,0]];
    const [x,y,z,w]=this.rotation;
    return [[1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)],
      [2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)],
      [2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)]];
  }
  applyRotation(delta) {
    const q=multiplyQuaternion(this.rotation,delta),length=Math.hypot(...q);
    this.rotation=q.map(v=>v/length);
    this.invalidate();
  }
  rotate(dx,dy,roll=0) {
    // Screen-local rotations compose freely, including roll, without poles.
    const axis=[-dy,-dx,roll];
    this.applyRotation(axisQuaternion(axis,Math.hypot(...axis)*.006));
  }
  trackballPoint(x,y) {
    const box=this.canvas.getBoundingClientRect(),radius=Math.max(1,Math.min(box.width,box.height)*.45);
    const u=(x-box.left-box.width/2)/radius,v=(box.top+box.height/2-y)/radius;
    const length=Math.hypot(u,v);
    return length>1?[u/length,v/length,0]:[u,v,Math.sqrt(Math.max(0,1-length*length))];
  }
  rotateTrackball(from,to) {
    const a=this.trackballPoint(...from),b=this.trackballPoint(...to);
    const axis=[b[1]*a[2]-b[2]*a[1],b[2]*a[0]-b[0]*a[2],b[0]*a[1]-b[1]*a[0]];
    const dot=Math.max(-1,Math.min(1,a.reduce((sum,v,i)=>sum+v*b[i],0)));
    if(Math.hypot(...axis)<1e-12) {
      if(dot>=0)return;
      // Opposite points on the rim still define a valid half turn.
      const reference=Math.abs(a[0])<.9?[1,0,0]:[0,1,0];
      axis.splice(0,3,a[1]*reference[2]-a[2]*reference[1],a[2]*reference[0]-a[0]*reference[2],a[0]*reference[1]-a[1]*reference[0]);
    }
    this.applyRotation(axisQuaternion(axis,Math.acos(dot)));
  }
  drawAxes() {
    const canvas=this.axesCanvas,ctx=this.axesContext;
    if(!ctx)return;
    const dpr=window.devicePixelRatio||1,size=canvas.clientWidth;
    const basis=this.basis(),key=JSON.stringify([basis,dpr,size]);
    if(key===this.axesKey)return;
    this.axesKey=key;
    canvas.width=canvas.height=Math.round(size*dpr);
    ctx.setTransform(dpr*size/112,0,0,dpr*size/112,0,0);
    ctx.clearRect(0,0,112,112);
    ctx.fillStyle='#101923d9';ctx.beginPath();ctx.arc(56,56,53,0,Math.PI*2);ctx.fill();
    const [right,up,forward]=basis,colors=['#f27989','#63c3a8','#7baefa'];
    const arms=[];
    for(let axis=0;axis<3;axis++)for(const sign of [-1,1])arms.push({axis,sign,depth:forward[axis]*sign});
    arms.sort((a,b)=>b.depth-a.depth); // Distant directions underneath near ones.
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.font='bold 11px monospace';
    for(const {axis,sign,depth} of arms) {
      const x=56+right[axis]*sign*35,y=56-up[axis]*sign*35;
      ctx.globalAlpha=sign>0?1:.38;ctx.strokeStyle=colors[axis];ctx.fillStyle=colors[axis];
      ctx.lineWidth=sign>0?2:1;ctx.setLineDash(sign>0?[]:[3,3]);
      ctx.beginPath();ctx.moveTo(56,56);ctx.lineTo(x,y);ctx.stroke();ctx.setLineDash([]);
      ctx.beginPath();ctx.arc(x,y,sign>0?9:3,0,Math.PI*2);
      if(sign>0) {
        ctx.fill();ctx.fillStyle='#101923';ctx.fillText('XYZ'[axis],x,y+.5);
        // A ring distinguishes an axis pointing away along the viewing ray.
        if(depth>.99){ctx.strokeStyle='#101923';ctx.lineWidth=1;ctx.beginPath();ctx.arc(x,y,7.5,0,Math.PI*2);ctx.stroke();}
      } else ctx.stroke();
    }
    ctx.globalAlpha=1;
  }
  setGeometry(data) {
    const gl=this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER,this.vbo); gl.bufferData(gl.ARRAY_BUFFER,data.vertices,gl.STATIC_DRAW);
    gl.bindVertexArray(this.vao); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.ebo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,data.edges,gl.STATIC_DRAW); gl.bindVertexArray(null);
    this.count=data.count; this.edgeCount=data.edgeCount;this.geometrySlots=new Set(data.slots||[]);this.invalidate();
  }
  setPalette(entries) {
    const slots=entries.reduce((n,e)=>Math.max(n,e.slot+1),0);
    const gl=this.gl, height=Math.max(1,Math.ceil(slots/this.paletteWidth));
    if(height>gl.getParameter(gl.MAX_TEXTURE_SIZE)) throw new Error('神经元数量超过 GPU 调色板容量');
    const colors=new Uint8Array(this.paletteWidth*height*4);
    for(const e of entries) { colors.set(e.color,e.slot*4); colors[e.slot*4+3]=e.visible?255:0; }
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,this.palette);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,this.paletteWidth,height,0,gl.RGBA,gl.UNSIGNED_BYTE,colors);
    this.invalidate();
  }
  clearVolume() {
    if(this.volume) this.gl.deleteTexture(this.volume.texture);
    this.volume=null; this.invalidate();
  }
  setVolume(meta, bytes) {
    this.clearVolume();
    const gl=this.gl, texture=gl.createTexture();
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_3D,texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
    const filter=this.linearFiltering?gl.LINEAR:gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,filter); gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,filter);
    for(const axis of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T,gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D,axis,gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_3D,0,gl.R32F,...meta.shape,0,gl.RED,gl.FLOAT,bytes);
    if(gl.getError()!==gl.NO_ERROR) {gl.deleteTexture(texture);throw new Error('图像纹理上传失败，请缩小区域');}
    this.volume={...meta,texture};
    this.slices=meta.shape.map(n=>Math.floor((n-1)/2)); this.invalidate();
  }
  volumeOrigin() { return this.volume.origin.map((v,i)=>v-this.cropStart[i]); }
  setView(mode) {
    this.mode=mode;
    this.orientation=['xy','xz','yz'].includes(mode)?mode:'3d'; this.invalidate();
  }
  setCenter(center) {
    if(center.length!==3||center.some(v=>!Number.isFinite(v)))throw new Error('视野中心需要三个有限的 XYZ 数值');
    this.center=[...center];
    const axis={xy:2,xz:1,yz:0}[this.mode];
    if(this.volume&&axis!==undefined) {
      this.slices[axis]=Math.max(0,Math.min(this.volume.shape[axis]-1,Math.round(center[axis]-this.volumeOrigin()[axis])));
    }
    this.invalidate();
  }
  sceneBounds(bounds) {
    const boxes=bounds.filter(Boolean);
    if(this.volume) {const o=this.volumeOrigin();boxes.push([o,o.map((v,i)=>v+this.volume.shape[i]-1)]);}
    if(!boxes.length)return null;
    const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    for(const box of boxes)for(let i=0;i<3;i++){lo[i]=Math.min(lo[i],box[0][i]);hi[i]=Math.max(hi[i],box[1][i]);}
    return [lo,hi];
  }
  recenter(bounds) {
    const box=this.sceneBounds(bounds);
    this.setCenter(box?box[0].map((v,i)=>(v+box[1][i])/2):[0,0,0]);
  }
  fit(bounds) {
    const box=this.sceneBounds(bounds);
    if(!box) {this.center=[0,0,0];this.span=100;this.extent=100;this.invalidate();return;}
    const [lo,hi]=box;
    this.center=lo.map((v,i)=>(v+hi[i])/2);
    const aspect=this.canvas.clientWidth/Math.max(1,this.canvas.clientHeight);
    this.extent=Math.max(1,Math.hypot(...hi.map((v,i)=>v-lo[i])));
    this.span=this.extent*.57/Math.min(1,aspect);this.invalidate();
  }
  invalidate() {this.stateSerial++;if(!this.pending) {this.pending=true;requestAnimationFrame(()=>{this.pending=false;this.render();});}}
  screenRay(clientX,clientY) {
    const rect=this.canvas.getBoundingClientRect(),[right,up,forward]=this.basis(),aspect=this.canvas.width/this.canvas.height;
    const u=(clientX-rect.left)/rect.width*2-1,v=1-(clientY-rect.top)/rect.height*2;
    const base=this.center.map((c,i)=>c+right[i]*u*this.span*aspect+up[i]*v*this.span);
    const axis={xy:2,xz:1,yz:0}[this.mode],sliceAxis=this.volume&&axis!==undefined?axis:-1;
    return {base,center:[...this.center],right,up,forward,scaleX:rect.width/(2*this.span*aspect),scaleY:rect.height/(2*this.span),
      farPlane:Math.max(this.extent*4,this.span*4,1),sliceAxis,
      slicePosition:sliceAxis<0?0:this.volumeOrigin()[axis]+this.slices[axis],slab:this.settings.slab};
  }
  volumeUniforms(p) {
    const gl=this.gl,u=p.uniforms,s=this.settings,axis={xy:2,xz:1,yz:0}[this.mode],origin=this.volumeOrigin();
    this.common(p);gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_3D,this.volume.texture);gl.uniform1i(u.volume,1);
    gl.uniform1i(u.linearFiltering,this.linearFiltering?1:0);
    gl.uniform3fv(u.origin,origin);gl.uniform3fv(u.shape,this.volume.shape);
    gl.uniform1i(u.mode,axis!==undefined?2:this.mode==='volume'?1:0);gl.uniform1i(u.sliceAxis,axis||0);
    gl.uniform1f(u.slicePosition,axis===undefined?0:origin[axis]+this.slices[axis]);
    for(const [name,value] of Object.entries({black:s.black,white:s.white,gammaValue:s.gamma,brightness:s.brightness,opacity:s.opacity}))gl.uniform1f(u[name],value);
  }
  pickVolume(ray) {
    if(!this.volume||this.settings.opacity<=0)return null;
    const gl=this.gl,previous=gl.getParameter(gl.FRAMEBUFFER_BINDING),viewport=gl.getParameter(gl.VIEWPORT);
    const states=[gl.BLEND,gl.DEPTH_TEST,gl.DITHER].map(flag=>[flag,gl.isEnabled(flag)]);
    try {
      if(!this.pickProgram)this.pickProgram=program(gl,quadVertex,volumePickFragment);
      if(!this.pickFramebuffer) {
        this.pickFramebuffer=gl.createFramebuffer();this.pickTexture=gl.createTexture();
        gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.pickTexture);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,2,1,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
        gl.bindFramebuffer(gl.FRAMEBUFFER,this.pickFramebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,this.pickTexture,0);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER,this.pickFramebuffer);
      if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error('显卡无法执行图像目标定位');
      gl.viewport(0,0,2,1);for(const [flag] of states)gl.disable(flag);
      gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.bindVertexArray(null);
      this.volumeUniforms(this.pickProgram);gl.uniform3fv(this.pickProgram.uniforms.pickBase,ray.base);
      gl.drawArrays(gl.TRIANGLES,0,3);
      const bytes=new Uint8Array(8);gl.readPixels(0,0,2,1,gl.RGBA,gl.UNSIGNED_BYTE,bytes);
      if(!bytes[4])return null;
      const distance=new DataView(bytes.buffer).getFloat32(0,true)-ray.farPlane;
      return Number.isFinite(distance)?ray.base.map((v,i)=>v+ray.forward[i]*distance):null;
    }finally {
      gl.bindFramebuffer(gl.FRAMEBUFFER,previous);gl.viewport(...viewport);
      for(const [flag,enabled] of states)enabled?gl.enable(flag):gl.disable(flag);
    }
  }
  common(p) {
    const gl=this.gl,u=p.uniforms,[right,up,forward]=this.basis();
    gl.useProgram(p.program);
    gl.uniform3fv(u.center,this.center);gl.uniform3fv(u.rightAxis,right);gl.uniform3fv(u.upAxis,up);gl.uniform3fv(u.forwardAxis,forward);
    gl.uniform1f(u.span,this.span);gl.uniform1f(u.aspect,this.canvas.width/this.canvas.height);
    gl.uniform1f(u.farPlane,Math.max(this.extent*4,this.span*4,1));
  }
  drawHemisphere() {
    const plane=this.hemispherePlane,labels=this.hemisphereLabels;
    if(!plane?.visible){if(labels)for(const label of labels)label.hidden=true;return;}
    const gl=this.gl,box=this.hemisphereBounds?.()||[[-50,-50,-50],[50,50,50]];
    const a=plane.azimuth_deg*Math.PI/180,e=plane.elevation_deg*Math.PI/180;
    const normal=[Math.cos(e)*Math.cos(a),Math.cos(e)*Math.sin(a),Math.sin(e)];
    const u=[-Math.sin(a),Math.cos(a),0],v=[-Math.sin(e)*Math.cos(a),-Math.sin(e)*Math.sin(a),Math.cos(e)];
    const middle=box[0].map((x,i)=>(x+box[1][i])/2),dot=(x,y)=>x.reduce((sum,k,i)=>sum+k*y[i],0);
    const offset=plane.offset-dot(normal,this.cropStart),distance=offset-dot(normal,middle);
    const anchor=middle.map((x,i)=>x+normal[i]*distance),size=Math.max(1,Math.hypot(...box[1].map((x,i)=>x-box[0][i]))*.65);
    if(!this.planeProgram)this.planeProgram=program(gl,planeVertex,planeFragment);
    const p=this.planeProgram;this.common(p);gl.bindVertexArray(null);
    gl.uniform1i(p.uniforms.edgeOn,Math.abs(dot(normal,this.basis()[2]))<.0001?1:0);
    gl.uniform3fv(p.uniforms.planeCenter,anchor);gl.uniform3fv(p.uniforms.planeU,u);gl.uniform3fv(p.uniforms.planeV,v);gl.uniform1f(p.uniforms.planeSize,size);
    gl.disable(gl.DEPTH_TEST);gl.depthMask(false);gl.drawArrays(gl.TRIANGLES,0,6);gl.depthMask(true);this.drawCalls++;
    // Two small DOM labels follow real +/- normal positions. No node traversal
    // or geometry upload is needed when the plane or camera moves.
    if(labels) {
      const [right,up,forward]=this.basis(),width=this.canvas.clientWidth,height=this.canvas.clientHeight,aspect=this.canvas.width/this.canvas.height;
      const depth=dot(normal,forward),facing=Math.hypot(dot(normal,right),dot(normal,up))<.15;
      const positions=[-1,1].map(sign=>{
        const point=anchor.map((x,i)=>x+normal[i]*sign*size*.45-this.center[i]);
        return [width*(.5+dot(point,right)/(2*this.span*aspect)),height*(.5-dot(point,up)/(2*this.span))];
      });
      const overlaps=Math.hypot(positions[0][0]-positions[1][0],positions[0][1]-positions[1][1])<90;
      labels.forEach((label,i)=>{
        const sign=i?1:-1;label.hidden=false;
        label.textContent=(i?'右半脑 +n':'左半脑 −n')+(facing?(depth*sign<0?' · 近侧':' · 远侧'):'');
        let [x,y]=positions[i];if(overlaps)y+=(i?1:-1)*20;
        label.style.left=Math.max(75,Math.min(width-75,x))+'px';label.style.top=Math.max(100,Math.min(height-140,y))+'px';
      });
    }
  }
  render() {
    const begin=performance.now(),gl=this.gl,s=this.settings;
    const dpr=window.devicePixelRatio||1,w=Math.max(1,Math.round(this.canvas.clientWidth*dpr)),h=Math.max(1,Math.round(this.canvas.clientHeight*dpr));
    if(this.canvas.width!==w||this.canvas.height!==h) {this.canvas.width=w;this.canvas.height=h;}
    gl.viewport(0,0,w,h); gl.clearColor(.027,.039,.055,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    this.drawCalls=0;
    const sliceAxis={xy:2,xz:1,yz:0}[this.mode];
    let slicePosition=0;
    if(this.volume) {
      const origin=this.volumeOrigin(); slicePosition=sliceAxis===undefined?0:origin[sliceAxis]+this.slices[sliceAxis];
      const p=this.volumeProgram;this.volumeUniforms(p);
      gl.disable(gl.DEPTH_TEST);gl.bindVertexArray(null);
      gl.drawArrays(gl.TRIANGLES,0,3);this.drawCalls++;
    }
    this.drawHemisphere();
    if(this.count) {
      const p=this.swcProgram,u=p.uniforms;this.common(p);gl.bindVertexArray(this.vao);
      gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.palette);gl.uniform1i(u.palette,0);gl.uniform1i(u.paletteWidth,this.paletteWidth);
      gl.uniform1i(u.colorMode,s.colorMode);gl.uniform1f(u.pointSize,s.pointSize*dpr);
      gl.uniform1i(u.sliceAxis,this.volume&&sliceAxis!==undefined?sliceAxis:-1);gl.uniform1f(u.slicePosition,slicePosition);gl.uniform1f(u.slab,s.slab);
      gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);
      gl.uniform1f(u.opacity,s.swcOpacity);
      if(s.lines&&this.edgeCount) {gl.uniform1i(u.points,0);gl.drawElements(gl.LINES,this.edgeCount*2,gl.UNSIGNED_INT,0);this.drawCalls++;}
      if(s.points) {gl.uniform1i(u.points,1);gl.drawArrays(gl.POINTS,0,this.count);this.drawCalls++;}
      gl.bindVertexArray(null);
    }
    this.drawAxes();
    this.lastFrameMs=performance.now()-begin;
    this.changed?.();
  }
  bindControls() {
    let drag=null;
    this.canvas.addEventListener('contextmenu',e=>e.preventDefault());
    this.canvas.addEventListener('pointerdown',e=>{if(drag)return;this.onNavigate?.();this.canvas.setPointerCapture(e.pointerId);drag={id:e.pointerId,x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,button:e.button,moved:false,pan:e.button!==0||e.shiftKey,roll:e.altKey};});
    this.canvas.addEventListener('pointermove',e=>{
      if(!drag||drag.id!==e.pointerId)return;
      if(!drag.moved&&Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)<4)return;
      drag.moved=true;const from=[drag.x,drag.y],dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
      if(drag.pan||this.orientation!=='3d') {
        const [right,up]=this.basis(),scale=2*this.span/this.canvas.clientHeight;
        this.center=this.center.map((v,i)=>v-right[i]*dx*scale+up[i]*dy*scale);
      } else if(drag.roll)this.rotate(0,0,-dx);
      else this.rotateTrackball(from,[e.clientX,e.clientY]);
      this.invalidate();
    });
    this.canvas.addEventListener('pointerup',e=>{if(drag?.id!==e.pointerId)return;const click=drag.button===2&&!drag.moved;drag=null;if(click)this.onTargetMenu?.(e.clientX,e.clientY);});
    for(const event of ['pointercancel','lostpointercapture'])this.canvas.addEventListener(event,e=>{if(drag?.id===e.pointerId)drag=null;});
    this.canvas.addEventListener('wheel',e=>{e.preventDefault();this.onNavigate?.();this.span=Math.max(.05,Math.min(1e9,this.span*Math.exp(Math.max(-100,Math.min(100,e.deltaY))*.002)));this.invalidate();},{passive:false});
  }
}
