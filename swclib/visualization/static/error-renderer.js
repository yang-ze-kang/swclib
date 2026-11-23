import {TraceRenderer} from './trace-renderer.js';

export function morphologyEdges(nodes) {
  const lookup=new Map(nodes.map(node=>[node.id,node]));
  return nodes.filter(node=>lookup.has(node.parent)).map(node=>[lookup.get(node.parent).xyz,node.xyz]);
}
export function affectedEdges(event,nodes) {
  if(event.kind==='break_error'||event.source==='gt')return event.missing_segments_xyz||[];
  const lookup=new Map(nodes.map(node=>[node.id,node])),ids=new Set(event.affected_subtree.node_ids),edges=[];
  for(const id of ids){
    const node=lookup.get(id);if(!node)continue;const parent=lookup.get(node.parent);
    if(id===event.start.edge_child_id)edges.push([event.start.xyz,node.xyz]);
    else if(parent&&ids.has(node.parent))edges.push([parent.xyz,node.xyz]);
  }
  return edges;
}
function bounds(points){
  const box=[[...points[0]],[...points[0]]];
  for(const p of points)for(let k=0;k<3;k++){box[0][k]=Math.min(box[0][k],p[k]);box[1][k]=Math.max(box[1][k],p[k]);}
  return box;
}
export class ErrorRenderer extends TraceRenderer {
  open(pair){
    this.pair=pair;this.predEdges=morphologyEdges(pair.pred.nodes);
    const origin=pair.gt.nodes[0].xyz;
    this.start(morphologyEdges(pair.gt.nodes),origin);
    this.sceneBounds=bounds([...pair.gt.nodes,...pair.pred.nodes].map(n=>n.xyz));
    this.select(null);
  }
  select(event){
    for(const layer of this.layers)this.release(layer);
    this.event=event;this.bounds=this.sceneBounds;
    this.data={traced_edges_xyz:this.predEdges};this.tracedEdges=this.predEdges;
    const background=[],highlight=[],markers=[];
    for(const [a,b] of this.predEdges)this.append(background,a,b,[.25,.88,.75,.62]);
    this.markers=[];this.reviewBounds=null;
    if(event){
      const edges=affectedEdges(event,this.pair.pred.nodes);
      for(const [a,b] of edges)this.append(highlight,a,b,event.kind==='false_junction'?[1,.35,.3,1]:[1,.65,.22,1]);
      this.eventBounds=bounds([event.start.xyz,...edges.flat()]);
      const witness=event.confirmation_path_xyz||[];
      for(let i=1;i<witness.length;i++)this.append(highlight,witness[i-1],witness[i],[1,.45,.85,1]);
      const estimate=event.onset_estimate;
      if(estimate?.estimated_onset){
        this.markers.push({point:estimate.estimated_onset.xyz,label:'estimated onset',color:[.3,1,.4,1],css:'#4dff66'});
        this.append(highlight,estimate.interval_start.xyz,estimate.interval_end.xyz,[.3,1,.4,1],2);
      }
      const review=event.onset_review;
      if(review?.segments?.length){
        for(const segment of review.segments)this.append(highlight,segment.start_xyz,segment.end_xyz,[1,.65,.12,1],2);
        this.reviewBounds=bounds([...review.path_xyz,...witness]);
      }
      this.markers.push({point:event.anchor.xyz,label:'anchor',color:[.9,.95,1,1],css:'#e6f2ff'},
        {point:event.start.xyz,label:(event.start_role==='matching_failure'?'mismatch':'start')+(event.trace_origin?.start?.probability!=null?' p='+event.trace_origin.start.probability.toFixed(4):''),color:[1,.9,.2,1],css:'#ffe633'});
      if(event.confirmation)this.markers.push({point:event.confirmation.xyz,label:'confirmed',color:[1,.45,.85,1],css:'#ff73d9'});
      if(review?.segments?.length)this.markers.push({point:review.start.xyz,label:'review boundary (uncertain)',color:[1,.65,.12,1],css:'#ffa61f'});
      for(const m of this.markers)markers.push(...m.point.map((v,i)=>v-this.origin[i]),...m.color,0,0);
    }
    this.layers=[this.layer(background),this.layer(highlight),this.layer(markers)];this.renderer.invalidate();
  }
  drawLabels(label){for(const m of this.markers||[])label(m.point,m.label,m.css);}
  stop(){super.stop();this.pair=null;this.event=null;this.markers=[];}
}
