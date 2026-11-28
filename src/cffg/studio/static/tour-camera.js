// Frame graph-space geometry, including labels and items outside the current viewport.
export function createTourCamera({svg, layer, scene, byRef, drawRow, view, move, redraw}) {
  const targets = new Set();
  let region = null, marker = null, card = null, animation = 0;
  let settled = null, lastViewport = '';

  function stop() {
    cancelAnimationFrame(animation);
    animation = 0;
    settled?.();
    settled = null;
  }

  function clear() {
    stop();
    region = null;
    card = null;
    lastViewport = '';
    marker?.remove();
    marker = null;
    if (targets.size) { targets.clear(); redraw(); }
  }

  function destination(width, height, bounds=region) {
    const fit = (x,y,w,h) => {
      const pad = marker ? Math.min(32,w/8,h/8) : 0;
      return {x:x+w/2, y:y+h/2, width:w, height:h, scale:Math.min((w-2*pad)/bounds.w,(h-2*pad)/bounds.h)};
    };
    const canvas = svg.getBoundingClientRect();
    const x = Math.max(0,-canvas.left), y = Math.max(0,-canvas.top);
    const r = Math.min(width,innerWidth-canvas.left), b = Math.min(height,innerHeight-canvas.top);
    const full = fit(x,y,r-x,b-y);
    if (!card) return full;
    const cover = card.getBoundingClientRect();
    const left = Math.max(x,cover.left-canvas.left-6), top = Math.max(y,cover.top-canvas.top-6);
    const right = Math.min(r,cover.right-canvas.left+6), bottom = Math.min(b,cover.bottom-canvas.top+6);
    if (left >= right || top >= bottom) return full;
    const scale = Math.min(full.scale,marker ? 2.2 : 2);
    const dx = bounds.w*scale/2, dy = bounds.h*scale/2;
    if (full.x+dx <= left || full.x-dx >= right || full.y+dy <= top || full.y-dy >= bottom) return full;
    // Fit above, below or beside the narration, choosing the most readable scale.
    const spaces = [fit(x,y,r-x,top-y), fit(x,bottom,r-x,b-bottom),
      fit(x,y,left-x,b-y), fit(right,y,r-right,b-y)];
    return spaces.reduce((best,next) => next.scale > best.scale ? next : best);
  }

  function frame(animate, onlyIfResized=false) {
    if (!region) return;
    const {width, height, left, top, scale} = view();
    const {x, y, w, h} = region;
    const center = destination(width,height);
    const viewport = [width,height,center.x,center.y,center.width,center.height,center.scale].join(',');
    if (onlyIfResized && lastViewport === viewport) return;
    lastViewport = viewport;
    stop();
    if (center.scale <= 0) return;
    const endScale = Math.min(marker ? 2.2 : 2, center.scale);
    const cx = (center.x - left) / scale, cy = (center.y - top) / scale;
    const endX = x + w/2, endY = y + h/2;
    function place(worldX, worldY, zoom, screenX=center.x, screenY=center.y) {
      move(screenX-worldX*zoom, screenY-worldY*zoom, zoom);
    }
    if (!animate) {
      place(endX,endY,endScale);
      return;
    }
    // At mid-pan the previous view and the entire destination fit together.
    const dx = endX-cx, dy = endY-cy;
    const spanX = Math.abs(dx) + Math.max(w,2*Math.max(center.x,width-center.x)/scale);
    const spanY = Math.abs(dy) + Math.max(h,2*Math.max(center.y,height-center.y)/scale);
    const route = destination(width,height,{w:spanX,h:spanY});
    const wide = Math.min(.75*scale,.75*endScale,.88*route.scale);
    const shiftX = route.x-center.x, shiftY = route.y-center.y;
    const duration = Math.min(1450,850 + 110*Math.log2(Math.max(scale,endScale)/wide));
    const start = performance.now();
    const ease = t => t*t*(3-2*t);
    function tick(now) {
      const t = Math.max(0,(now-start)/duration);
      if (t < .3) {
        const out = ease(t/.3);
        place(cx,cy,scale*(wide/scale)**out,center.x+shiftX*out,center.y+shiftY*out);
      } else if (t < .7) {
        const pan = ease((t-.3)/.4);
        place(cx+dx*pan,cy+dy*pan,wide,route.x,route.y);
      } else if (t < 1) {
        const into = ease((t-.7)/.3);
        place(endX,endY,wide*(endScale/wide)**into,route.x-shiftX*into,route.y-shiftY*into);
      } else {
        place(endX,endY,endScale);
        stop();
        return;
      }
      animation = requestAnimationFrame(tick);
    }
    return new Promise(resolve => {
      settled = resolve;
      animation = requestAnimationFrame(tick);
    });
  }

  function focus(narration, animate) {
    card = narration;
    return frame(animate);
  }

  function show(refs) {
    clear();
    if (!refs.length) {
      region = {x:0, y:0, w:Math.max(1,scene.width), h:Math.max(1,scene.height)};
      return {target:svg, offCanvas:[], focus};
    }
    const offCanvas = [];
    for (const ref of refs) {
      let rows = byRef.get(ref) || [];
      // A compact plate's children also refer to their enclosing plate.
      if (ref.startsWith('plate:')) rows = rows.filter(row => row.item.primitive === 'PlateBox' && row.localRef === ref);
      if (!rows.length) offCanvas.push(ref);
      for (const row of rows) targets.add(row);
    }
    if (!targets.size) return {target:null, offCanvas};

    const measure = document.createElementNS(svg.namespaceURI, 'g');
    measure.setAttribute('visibility', 'hidden');
    measure.setAttribute('aria-hidden', 'true');
    for (const row of targets) {
      const glyph = drawRow(row);
      glyph.classList.add('tour-target');
      measure.append(glyph);
    }
    svg.append(measure);
    const box = measure.getBBox();
    measure.remove();
    region = {x:box.x-5, y:box.y-5, w:Math.max(1,box.width)+10, h:Math.max(1,box.height)+10};
    marker = document.createElementNS(svg.namespaceURI, 'rect');
    for (const [key, value] of Object.entries({
      class:'tour-frame', x:region.x, y:region.y, width:region.w, height:region.h,
      rx:4, fill:'none', stroke:'var(--accent)', 'stroke-width':1.5, 'vector-effect':'non-scaling-stroke',
    })) marker.setAttribute(key, value);
    layer.append(marker);
    redraw();
    return {target:marker, offCanvas, focus};
  }
  return {targets, show, clear, finish:() => frame(false), resize:() => frame(false,true)};
}
