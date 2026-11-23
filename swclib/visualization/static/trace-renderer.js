// Independent GPU layers for read-only replay; all public coordinates are global XYZ.
const vertex=`#version 300 es
precision highp float;
layout(location=0) in vec3 position;
layout(location=1) in vec4 rgba;
layout(location=2) in vec2 dash;
uniform vec3 center,rightAxis,upAxis,forwardAxis;
uniform float span,aspect,farPlane,pointSize;
out vec4 color;
out vec2 pattern;
void main(){vec3 p=position-center;gl_Position=vec4(dot(p,rightAxis)/(span*aspect),dot(p,upAxis)/span,dot(p,forwardAxis)/farPlane,1.);gl_PointSize=pointSize;color=rgba;pattern=dash;}`;
const fragment=`#version 300 es
precision highp float;
in vec4 color;
in vec2 pattern;
uniform bool point;
out vec4 outputColor;
void main(){if(point&&length(gl_PointCoord-.5)>.5)discard;if(pattern.y>0.&&mod(pattern.x,pattern.y)>pattern.y*.58)discard;outputColor=color;}`;
export const traceColors=['#ffbe55','#ef72bc','#83a6ff','#adcf65','#ff8175','#70d8ff','#c898ff','#e4db75'];
function rgb(index,alpha=1){const hex=traceColors[index%traceColors.length];return [1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)/255).concat(alpha);}
function include(box,point){for(let i=0;i<3;i++){box[0][i]=Math.min(box[0][i],point[i]);box[1][i]=Math.max(box[1][i],point[i]);}}
// Use screen-space spacing so dense SWC waypoints do not become a wall of
// arrowheads. Every edge is ordered parent -> child by the replay backend.
export function drawDirectionArrows(ctx,groups,{center,right,up,span,width,height}) {
  const scale=height/(2*span),cell=18,columns=Math.ceil(width/cell),rows=Math.ceil(height/cell);
  const occupied=new Uint8Array(columns*rows);
  const project=point=>{
    const x=point[0]-center[0],y=point[1]-center[1],z=point[2]-center[2];
    return [width/2+(x*right[0]+y*right[1]+z*right[2])*scale,
            height/2-(x*up[0]+y*up[1]+z*up[2])*scale];
  };
  ctx.lineWidth=1.2;ctx.lineJoin='round';ctx.lineCap='round';
  for(const {edges,color} of groups) {
    ctx.strokeStyle=color;ctx.beginPath();
    for(const [parent,child] of edges) {
      const [ax,ay]=project(parent),[bx,by]=project(child),dx=bx-ax,dy=by-ay;
      if(Math.max(ax,bx)<0||Math.min(ax,bx)>width||Math.max(ay,by)<0||Math.min(ay,by)>height)continue;
      const length=Math.hypot(dx,dy);
      if(length<.75)continue; // Direction toward/away from the camera is ambiguous.
      let lo=0,hi=1;
      if(dx!==0){const a=-ax/dx,b=(width-ax)/dx;lo=Math.max(lo,Math.min(a,b));hi=Math.min(hi,Math.max(a,b));}
      if(dy!==0){const a=-ay/dy,b=(height-ay)/dy;lo=Math.max(lo,Math.min(a,b));hi=Math.min(hi,Math.max(a,b));}
      if(hi<=lo)continue;
      const count=Math.max(1,Math.floor(length*(hi-lo)/48)),ux=dx/length,uy=dy/length;
      for(let i=0;i<count;i++) {
        const t=lo+(hi-lo)*(i+.5)/count,x=ax+t*dx,y=ay+t*dy;
        const col=Math.floor(x/cell),row=Math.floor(y/cell);
        if(col<0||col>=columns||row<0||row>=rows||occupied[row*columns+col])continue;
        for(let r=Math.max(0,row-1);r<=Math.min(rows-1,row+1);r++)
          for(let c=Math.max(0,col-1);c<=Math.min(columns-1,col+1);c++)occupied[r*columns+c]=1;
        ctx.moveTo(x-ux*6-uy*3,y-uy*6+ux*3);
        ctx.lineTo(x,y);
        ctx.lineTo(x-ux*6+uy*3,y-uy*6-ux*3);
      }
    }
    ctx.stroke();
  }
}
function makeProgram(gl){
  const p=gl.createProgram(),shaders=[];
  for(const [type,source] of [[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]){
    const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p,s);shaders.push(s);
  }
  gl.linkProgram(p);for(const s of shaders)gl.deleteShader(s);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));
  const uniforms={};for(const name of ['center','rightAxis','upAxis','forwardAxis','span','aspect','farPlane','pointSize','point'])uniforms[name]=gl.getUniformLocation(p,name);
  return {program:p,uniforms};
}
export class TraceRenderer {
  constructor(renderer){
    this.renderer=renderer;this.gl=renderer.gl;this.program=makeProgram(this.gl);this.active=false;this.origin=[0,0,0];this.layers=[];this.showDirections=true;this.gtEdges=[];
    this.labels=document.createElement('canvas');this.labels.className='trace-labels';renderer.canvas.parentElement.append(this.labels);
  }
  layer(values){
    const gl=this.gl,vao=gl.createVertexArray(),buffer=gl.createBuffer();gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(values),gl.STATIC_DRAW);
    for(const [index,size,offset] of [[0,3,0],[1,4,12],[2,2,28]]){gl.enableVertexAttribArray(index);gl.vertexAttribPointer(index,size,gl.FLOAT,false,36,offset);}gl.bindVertexArray(null);
    return {vao,buffer,count:values.length/9};
  }
  release(layer){if(layer){this.gl.deleteBuffer(layer.buffer);this.gl.deleteVertexArray(layer.vao);}}
  append(values,a,b,color,period=0){const length=Math.hypot(...b.map((v,i)=>v-a[i]));for(const [p,t] of [[a,0],[b,length]])values.push(...p.map((v,i)=>v-this.origin[i]),...color,t,period);}
  start(gt,origin){this.release(this.gt);this.gtEdges=gt;this.origin=[...origin];this.gtBounds=[[...origin],[...origin]];const values=[];for(const [a,b] of gt){this.append(values,a,b,[.66,.72,.79,.32]);include(this.gtBounds,a);include(this.gtBounds,b);}this.gt=this.layer(values);this.active=true;}
  setDirections(enabled){this.showDirections=!!enabled;this.renderer.invalidate();}
  frame(frame){
    this.active=true;this.data=frame;for(const layer of this.layers)this.release(layer);this.layers=[];
    const traced=[],candidates=[],seed=[];
    // The backend supplies the committed tree in the same prediction frame.
    // Keep current predictions out of the cyan background (including arrows),
    // so their probability colors stay distinct after removing advance frames.
    const predicted=new Set((frame.prediction?.candidates||[]).flatMap(candidate=>candidate.actual_edges_xyz.map(edge=>JSON.stringify(edge))));
    this.tracedEdges=frame.traced_edges_xyz.filter(edge=>!predicted.has(JSON.stringify(edge)));
    this.bounds=this.gtBounds.map(point=>[...point]);include(this.bounds,frame.seed_xyz);
    for(const [a,b] of this.tracedEdges){this.append(traced,a,b,[.25,.88,.75,.9]);include(this.bounds,a);include(this.bounds,b);}
    if(frame.phase==='prediction')for(const [i,candidate] of frame.prediction.candidates.entries()){
      const points=[frame.seed_xyz,...candidate.points_xyz],forward=frame.prediction.forward_steps;
      // The faint dotted prefix makes truncated/rejected predictions inspectable.
      for(let j=1;j<points.length;j++)this.append(candidates,points[j-1],points[j],rgb(i,j<=forward?.32:.95),j<=forward?.7:2);
      for(const [a,b] of candidate.actual_edges_xyz){this.append(candidates,a,b,rgb(i));include(this.bounds,a);include(this.bounds,b);}
    }
    seed.push(...frame.seed_xyz.map((v,i)=>v-this.origin[i]),1,1,1,1,0,0);
    this.layers=[this.layer(traced),this.layer(candidates),this.layer(seed)];this.renderer.invalidate();
  }
  stop(){this.active=false;for(const layer of this.layers)this.release(layer);this.layers=[];this.release(this.gt);this.gt=null;this.gtEdges=[];this.data=null;this.labels.getContext('2d').clearRect(0,0,this.labels.width,this.labels.height);this.renderer.invalidate();}
  drawLabels(label){
    if(this.data){label(this.data.seed_xyz,'seed','#ffffff');if(this.data.phase==='prediction')for(const [i,c] of this.data.prediction.candidates.entries())label(c.points_xyz[c.points_xyz.length-1],`#${c.index} p=${c.probability.toFixed(3)}`,traceColors[i%traceColors.length]);}
  }
  draw(){
    if(!this.active)return;
    const r=this.renderer,gl=this.gl,p=this.program,u=p.uniforms;r.common(p);
    const center=r.center.map((v,i)=>v+r.cropStart[i]-this.origin[i]);gl.uniform3fv(u.center,center);gl.uniform1f(u.farPlane,1e8);gl.uniform1f(u.pointSize,10*(window.devicePixelRatio||1));
    gl.disable(gl.DEPTH_TEST);gl.uniform1i(u.point,0);
    for(const layer of [this.gt,...this.layers.slice(0,2)])if(layer?.count){gl.bindVertexArray(layer.vao);gl.drawArrays(gl.LINES,0,layer.count);r.drawCalls++;}
    const seed=this.layers[2];if(seed){gl.uniform1i(u.point,1);gl.bindVertexArray(seed.vao);gl.drawArrays(gl.POINTS,0,seed.count);r.drawCalls++;}gl.bindVertexArray(null);
    const canvas=this.labels,ctx=canvas.getContext('2d'),dpr=window.devicePixelRatio||1;
    canvas.width=r.canvas.width;canvas.height=r.canvas.height;ctx.scale(dpr,dpr);ctx.font='12px monospace';
    const width=r.canvas.clientWidth,height=r.canvas.clientHeight,[right,up]=r.basis();
    if(this.showDirections)drawDirectionArrows(ctx,[
      {edges:this.tracedEdges||[],color:'rgba(64,224,191,.95)'},
      {edges:this.gtEdges,color:'rgba(168,184,201,.8)'},
    ],{center:r.center.map((v,i)=>v+r.cropStart[i]),right,up,span:r.span,width,height});
    const label=(point,text,color)=>{const q=point.map((v,i)=>v-r.cropStart[i]-r.center[i]);const dot=a=>a.reduce((n,v,i)=>n+v*q[i],0);const x=width/2+dot(right)*height/(2*r.span),y=height/2-dot(up)*height/(2*r.span);ctx.fillStyle=color;ctx.fillText(text,x+8,y-8);};
    this.drawLabels(label);
  }
}
