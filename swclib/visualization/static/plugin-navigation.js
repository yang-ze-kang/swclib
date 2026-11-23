// One owner for scene navigation: arrows never drive two plugins at once.
export function createPluginNavigation() {
  let active=null;
  document.addEventListener('keydown',event=>{
    if(event.defaultPrevented||event.ctrlKey||event.metaKey||event.altKey||event.shiftKey)return;
    if(event.key!=='ArrowLeft'&&event.key!=='ArrowRight')return;
    const target=event.target;
    if(target?.closest?.('input,textarea,select,[contenteditable]:not([contenteditable="false"]),[role="textbox"]'))return;
    if(!active||!active.body.closest('details')?.open)return;
    if(active.step?.(event.key==='ArrowRight'?1:-1))event.preventDefault();
  });
  return body=>{
    const item={body,step:null,deactivate:null};
    return {
      setStep:fn=>{item.step=fn;},
      onDeactivate:fn=>{item.deactivate=fn;},
      activate(){
        if(active===item)return;
        const previous=active;active=item;
        if(previous){previous.body.closest('details')?.classList.remove('keyboard-active');previous.deactivate?.();}
        item.body.closest('details')?.classList.add('keyboard-active');
      },
      release(){if(active===item){active=null;item.body.closest('details')?.classList.remove('keyboard-active');}},
      isActive:()=>active===item,
    };
  };
}
