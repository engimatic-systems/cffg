// Frame graph-space geometry, including labels and items outside the current viewport.
export function createTourCamera({svg, layer, scene, byRef, drawRow, view, move, redraw}) {
  const targets = new Set();
  let region = null, marker = null, card = null, animation = 0;

  function clear() {
    cancelAnimationFrame(animation);
    animation = 0;
    region = null;
    card = null;
    marker?.remove();
    marker = null;
    if (targets.size) { targets.clear(); redraw(); }
  }

  function destination(width, height) {
    const fit = (x,y,w,h) => {
      const pad = marker ? Math.min(32,w/8,h/8) : 0;
      return {x:x+w/2, y:y+h/2, scale:Math.min((w-2*pad)/region.w,(h-2*pad)/region.h)};
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
    const dx = region.w*scale/2, dy = region.h*scale/2;
    if (full.x+dx <= left || full.x-dx >= right || full.y+dy <= top || full.y-dy >= bottom) return full;
    // Fit above, below or beside the narration, choosing the most readable scale.
    const spaces = [fit(x,y,r-x,top-y), fit(x,bottom,r-x,b-bottom),
      fit(x,y,left-x,b-y), fit(right,y,r-right,b-y)];
    return spaces.reduce((best,next) => next.scale > best.scale ? next : best);
  }

  function frame(animate) {
    if (!region) return;
    cancelAnimationFrame(animation);
    const {width, height, left, top, scale} = view();
    const {x, y, w, h} = region;
    const center = destination(width,height);
    if (center.scale <= 0) return;
    const endScale = Math.min(marker ? 2.2 : 2, center.scale);
    const cx = (center.x - left) / scale, cy = (center.y - top) / scale;
    const endX = x + w/2, endY = y + h/2;
    function place(t) {
      const s = scale * (endScale/scale) ** t;
      move(center.x - (cx + (endX-cx)*t)*s, center.y - (cy + (endY-cy)*t)*s, s);
    }
    if (!animate || matchMedia('(prefers-reduced-motion: reduce)').matches) {
      place(1);
      return;
    }
    const start = performance.now();
    function tick(now) {
      const t = Math.min(1, Math.max(0, (now-start)/320));
      place(t*t*(3-2*t));
      animation = t < 1 ? requestAnimationFrame(tick) : 0;
    }
    animation = requestAnimationFrame(tick);
  }

  function focus(narration) {
    card = narration;
    frame(true);
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
  return {targets, show, clear, resize:() => frame(false)};
}
