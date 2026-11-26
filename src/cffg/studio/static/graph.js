// The scene owns the Forney geometry. The template supplies the meaning of each mark.
const NS = "http://www.w3.org/2000/svg";
const colors = {
  gaussian: "#79b9f4", discrete: "#b29aff", ef: "#72d3b6", mixed: "#e5a5cf",
  parameter: "#8b92a0", observed: "var(--observed, #8acfb5)",
  constraint: "var(--constraint, #dfc17a)", plain: "#c5c9d1",
};
const style = `
  .graph-item { cursor: pointer; color: var(--ink); }
  .graph-item:hover { color: #fff; }
  .graph-item.selected { color: var(--accent, #ffb18d); }
  .graph-item .face { fill: #252932; stroke: currentColor; stroke-width: 1.5; }
  .graph-item.selected .face { fill: var(--accent-bg, #423329); stroke-width: 2.8; }
  .graph-item.selected .wire { stroke-width: 3.8; }
  .graph-item text { fill: currentColor; font-family: "IBM Plex Mono", ui-monospace, monospace;
     paint-order: stroke; stroke: #1c1e23; stroke-width: 3; stroke-linejoin: round; }
  .graph-item .glyph { font-family: "STIX Two Text", Georgia, serif; font-weight: 600;
     stroke: none; font-size: 16px; }
  .graph-item .caption { fill: #d1d3da; font-size: 11px; }
  .graph-item .wire-name { font-size: 11px; }
  .graph-item .param-name { font-size: 10px; }
  .graph-item .glyph-sub { font: 10px ui-monospace, monospace; }
  .graph-item .frame { stroke: currentColor; stroke-dasharray: 6 4; fill: none; }
  .graph-item .focus-ring { display: none; stroke: var(--accent, #ffb18d); fill: none; stroke-width: 1.3; }
  .graph-item.selected .focus-ring { display: inline; }
  .graph-item .hit { stroke: transparent; stroke-width: 13; fill: none; pointer-events: stroke; }
  .graph-item .wire { fill: none; stroke: currentColor; stroke-linecap: round; stroke-linejoin: round; }
  .graph-canvas.low-detail .graph-item:not(.selected) .param-name,
  .graph-canvas.overview .graph-item:not(.selected) :is(.caption,.wire-name),
  .graph-canvas.low-detail.dense .graph-item:not(.selected) :is(.caption,.wire-name) { display: none; }
`;

function element(tag, attrs = {}, value) {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, attr]) => node.setAttribute(key, attr));
  if (value !== undefined) node.textContent = value;
  return node;
}

function family(variable) {
  if (variable?.ef) return "ef";
  if (variable?.dim) return "gaussian";
  if (variable?.cardinality) return "discrete";
  return "plain";
}

function semantics(snapshot) {
  const template = snapshot.template;
  const variables = template.variables || {};
  const factors = template.factors || {};
  const ports = template.ports || {};
  const aliases = new Map();
  const families = new Map();
  const descriptions = new Map();
  const constrained = new Set();
  const observed = new Set();
  const notes = new Map();
  for (const [name, variable] of Object.entries(variables)) {
    const ref = "var:" + name;
    const domain = family(variable);
    families.set(ref, domain);
    const description = variable.ef || (variable.dim ? "Gaussian, dimension " + variable.dim :
      "categorical, " + variable.cardinality + " states");
    descriptions.set(ref, name + " · " + description);
  }
  for (const [name, factor] of Object.entries(factors)) {
    const ref = "factor:" + name;
    const domains = new Set(Object.values(factor.edges).map(id => family(variables[id])));
    families.set(ref, domains.size === 1 ? [...domains][0] : "mixed");
    descriptions.set(ref, name + " · " + factor.type);
    for (const [role, blob] of Object.entries(factor.param_blobs || {})) {
      aliases.set("blob:" + name + "." + role, "blob:" + blob);
    }
  }
  for (const [name, port] of Object.entries(ports)) {
    families.set("port:" + name, family(variables[port.variable]));
    if (port.role === "data") {
      observed.add("var:" + port.variable);
      observed.add("port:" + name);
    }
    descriptions.set("port:" + name, name + " · " + port.role + " port");
  }
  for (const c of template.constraints || []) {
    constrained.add((c.kind === "delta" ? "var:" : "factor:") + c.target);
  }
  const meta = snapshot.metadata || {};
  const anchorRefs = new Map(Object.entries(meta.anchors || {}).map(([ref, id]) => [id, ref]));
  for (const note of meta.notes || []) {
    const ref = anchorRefs.get(note.anchor) || note.ref;
    if (ref) notes.set(ref, (notes.get(ref) || 0) + 1);
  }
  return {aliases, families, descriptions, constrained, observed, notes};
}

function roundedPath(points) {
  if (points.length < 2) return "";
  let d = "M" + points[0].join(" ");
  for (let i = 1; i < points.length - 1; i++) {
    const a = points[i - 1], b = points[i], c = points[i + 1];
    const ax = a[0] - b[0], ay = a[1] - b[1], cx = c[0] - b[0], cy = c[1] - b[1];
    const before = Math.hypot(ax, ay), after = Math.hypot(cx, cy);
    if (!before || !after) continue;
    const r = Math.min(6, before / 2, after / 2);
    d += ` L${b[0] + r * ax / before} ${b[1] + r * ay / before} Q${b[0]} ${b[1]} ${b[0] + r * cx / after} ${b[1] + r * cy / after}`;
  }
  return d + " L" + points.at(-1).join(" ");
}

function bounds(item) {
  const points = item.points || item.anchors;
  if (points) {
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    return [Math.min(...xs) - 6, Math.min(...ys) - 10, Math.max(...xs) + 6, Math.max(...ys) + 10];
  }
  if (item.primitive === "Box") return [item.x-item.w/2, item.y-item.h/2, item.x+item.w/2, item.y+item.h/2];
  if (item.w) return [item.x, item.y, item.x+item.w, item.y+item.h];
  return [item.x-8, item.y-8, item.x+8, item.y+8];
}

function intersects(a, b) {
  return a[2] >= b[0] && a[0] <= b[2] && a[3] >= b[1] && a[1] <= b[3];
}

function spatialIndex(items) {
  const bins = new Map(), wide = [];
  const cell = 256;
  const bucket = value => Math.floor(value / cell);
  for (const record of items) {
    const [l, t, r, b] = record.box.map(bucket);
    if ((r-l+1)*(b-t+1) > 64) {
      wide.push(record);
      continue;
    }
    for (let y = t; y <= b; y++) for (let x = l; x <= r; x++) {
      const key = x + ":" + y;
      if (!bins.has(key)) bins.set(key, []);
      bins.get(key).push(record);
    }
  }
  return view => {
    const [l,t,r,b] = view.map(bucket);
    if ((r-l+1)*(b-t+1) > bins.size) return items.filter(row => intersects(row.box, view));
    const found = new Set(wide);
    for (let y = t; y <= b; y++) for (let x = l; x <= r; x++) {
      for (const row of bins.get(x + ":" + y) || []) found.add(row);
    }
    return [...found].filter(row => intersects(row.box, view)).sort((a,b) => a.id-b.id);
  };
}

function drawRecord(record, model, selected) {
  const {item, ref} = record;
  const constrained = model.constrained.has(ref);
  const observed = model.observed.has(ref);
  const param = ref.startsWith("blob:");
  let domain = model.families.get(ref) || (param ? "parameter" : "plain");
  if (observed) domain = "observed";
  if (item.primitive === "CompositeBox" || item.dashed) domain = "constraint";
  const group = element("g", {"class": "graph-item" + (selected ? " selected" : ""), "data-ref": ref});
  group.style.setProperty("--ink", colors[domain]);
  group.append(element("title", {}, model.descriptions.get(ref) || ref));
  const add = (tag, attrs, text) => group.append(element(tag, attrs, text));
  const caption = (x, y, value, cls = "caption", anchor = "middle") =>
    add("text", {x, y, "class": cls, "text-anchor": anchor}, value);
  const {x, y, w, h} = item;
  switch (item.primitive) {
    case "Box": {
      const left = x - w/2, top = y - h/2;
      add("rect", {x:left-3, y:top-3, width:w+6, height:h+6, rx:3, "class":"focus-ring"});
      add("rect", {x:left, y:top, width:w, height:h, rx:2, "class":"face"});
      const [gx,gy] = item.glyph_at || [x,y];
      const glyph = item.glyph.startsWith("𝒩") ? "N" + item.glyph.slice("𝒩".length) : item.glyph;
      add("text", {x:gx, y:gy, "class":"glyph", "text-anchor":"middle", "dominant-baseline":"central"}, glyph);
      if (glyph !== "=") caption(x, top+h+13, item.name);
      if (constrained) add("path", {d:`M${left+2} ${top+9}v-7h7`, stroke:colors.constraint, fill:"none", "stroke-width":3});
      break;
    }
    case "Path": {
      const d = roundedPath(item.points);
      add("path", {d, "class":"hit"});
      const attrs = {d, "class":"wire", "stroke-width": item.kind === "internal" ? 1.1 : param ? 1.15 : 2};
      if (item.dashed || constrained) attrs["stroke-dasharray"] = "5 3";
      add("path", attrs);
      if (item.label && item.label_at) {
        const [lx, ly] = item.label_at;
        const label = param ? item.ref.split(".").at(-1) : item.label;
        caption(lx, ly, label, param ? "wire-name param-name" : "wire-name", "start");
      }
      break;
    }
    case "Terminal":
      if (item.kind === "pointmass") {
        add("rect", {x:x-3.5, y:y-3.5, width:7, height:7, fill:"currentColor"});
      } else {
        add("circle", {cx:x, cy:y, r:5, stroke:"currentColor", "stroke-width":1.6,
          fill:item.kind === "data" ? "currentColor" : "#24272d"});
        if (item.kind === "data") add("path", {d:`M${x-2.5} ${y}h5`, stroke:"#332613","stroke-width":1.5});
      }
      if (item.label) caption(x,y+16,item.label);
      break;
    case "Bead": {
      const mark = item.symbol || item.shape === "square";
      const attrs = {stroke:mark ? colors.constraint : "currentColor","stroke-width":1.5,
        fill:item.filled ? "currentColor" : "#24272d"};
      if (item.shape === "square") add("rect", {x:x-5,y:y-5,width:10,height:10,...attrs});
      else add("circle", {cx:x,cy:y,r:mark ? 5.8 : 4,...attrs});
      if (item.symbol) caption(x,y+3.2,item.symbol === "delta" ? "δ" : item.symbol,"glyph-sub");
      break;
    }
    case "Dome": {
      // Connect the two most distant anchors. Additional participating roles meet the arc.
      let [a, b] = item.anchors, longest = -1;
      for (const p of item.anchors) for (const q of item.anchors) {
        const span = Math.hypot(p[0]-q[0], p[1]-q[1]);
        if (span > longest) { a = p; b = q; longest = span; }
      }
      if (!b) break;
      let cx=item.cx, cy=item.cy;
      if (a[0] === b[0]) cx += longest * .62;
      if (a[1] === b[1]) cy += longest * .62;
      let d = `M${a} Q${cx} ${cy} ${b}`;
      for (const p of item.anchors) if (p !== a && p !== b) {
        d += ` M${p} L${(a[0]+2*cx+b[0])/4} ${(a[1]+2*cy+b[1])/4}`;
      }
      add("path", {d,fill:"none",stroke:"currentColor","stroke-width":1.2});
      break;
    }
    case "EllipsisMark":
      for (const dx of [-6,0,6]) add("circle", {cx:item.angle ? x : x+dx,cy:item.angle ? y+dx : y,r:1.6,fill:"currentColor"});
      break;
    case "PlateBox":
    case "CompositeBox":
      add("rect", {x,y,width:w,height:h,rx:4,"class":"frame","pointer-events":"stroke"});
      if (item.label) caption(x+8,y+17,item.label + "  " + item.badge,"caption","start");
      break;
  }
  if (record.noteCount) {
    const [nx,ny] = record.noteAt;
    add("path", {d:`M${nx} ${ny}h12v8h-6l-4 4v-4h-2z`,fill:"#f3c771",stroke:"#1c1e23","stroke-width":1.25});
    group.append(element("title", {}, record.noteCount + " attached notes"));
  }
  return group;
}

export function mountGraph(container, snapshot, selectCallback) {
  const scene = snapshot.scene;
  const model = semantics(snapshot);
  const svg = element("svg", {class:"graph-canvas",width:"100%",height:"100%",role:"img",
    "aria-label":(snapshot.template.name || "Model") + " factor graph"});
  Object.assign(svg.style, {display:"block", width:"100%", height:"100%", touchAction:"none", cursor:"grab",
    userSelect:"none", overflow:"hidden"});
  const layer = element("g");
  const overlay = element("g", {"pointer-events":"none","aria-hidden":"true"});
  svg.append(element("style",{},style), layer, overlay);
  container.replaceChildren(svg);
  if (!scene.items.length) {
    const label = element("text", {x:"50%",y:"50%","text-anchor":"middle",fill:"#a3a9b4",
      "font-size":14, "font-family":"system-ui, sans-serif"}, "No factors in this graph");
    svg.append(label);
  }
  const plates = scene.items.filter(i => i.primitive === "PlateBox");
  const rows = scene.items.map((item,id) => {
    let ref = item.primitive === "Path" ? model.aliases.get(item.ref) || item.ref : item.ref;
    if (ref.startsWith("gfe:")) ref = "factor:" + ref.slice(4);
    const box = bounds(item);
    if (plates.length && item.primitive !== "PlateBox") {
      const parent = plates.find(p => box[0] >= p.x && box[1] >= p.y && box[2] <= p.x+p.w && box[3] <= p.y+p.h);
      if (parent) ref = parent.ref;
    }
    return {id, item, ref, box};
  });
  const byRef = new Map();
  for (const row of rows) {
    if (!byRef.has(row.ref)) byRef.set(row.ref, []);
    byRef.get(row.ref).push(row);
  }
  for (const [ref, count] of model.notes) {
    const matching = byRef.get(ref);
    if (!matching) continue;
    const row = matching.find(r => ["Box","Terminal","PlateBox"].includes(r.item.primitive)) || matching[0];
    row.noteCount = count;
    row.noteAt = row.item.primitive === "Path" ? row.item.label_at || row.item.points[0] : [row.box[2]-4,row.box[1]-4];
  }

  const query = rows.length > 600 ? spatialIndex(rows) : () => rows;
  const pitchX = scene.pitch_x > 0 ? scene.pitch_x : 100;
  const pitchY = scene.pitch_y > 0 ? scene.pitch_y : 84;
  let visible = new Map(), selectedRef = null, frame = 0;
  let width = 1, height = 1, fitted = 1, scale = 1, left = 0, top = 0;
  let drag = null;
  const signal = new AbortController();
  const on = (name, fn, options = {}) => svg.addEventListener(name, fn, {...options,signal:signal.signal});

  function paint() {
    frame = 0;
    layer.setAttribute("transform", `translate(${left} ${top}) scale(${scale})`);
    overlay.setAttribute("transform", layer.getAttribute("transform"));
    svg.classList.toggle("low-detail", scale < 1.1);
    svg.classList.toggle("overview", scale < .57);
    const pad = 65;
    const view = [-left/scale-pad,-top/scale-pad,(width-left)/scale+pad,(height-top)/scale+pad];
    const candidates = query(view);
    // At this density, repainting wire and factor labels can cost a frame during navigation.
    svg.classList.toggle("dense", candidates.length >= 1000);
    const next = new Map();
    // Keep whole-document overviews legible with one representative factor per screen cell.
    const overview = candidates.length > 2500 && scale < .33;
    const cellSize = Math.max(12, Math.sqrt(width * height / 900));
    const cells = new Set();
    for (const row of candidates) {
      if (overview && row.ref !== selectedRef) {
        if (row.item.primitive !== "Box" && row.item.primitive !== "PlateBox") continue;
        const cell = Math.floor(row.box[0]*scale/cellSize) + ":" + Math.floor(row.box[1]*scale/cellSize);
        if (cells.has(cell)) continue;
        cells.add(cell);
      }
      next.set(row.id, visible.get(row.id) || drawRecord(row,model,row.ref === selectedRef));
    }
    for (const [key, node] of visible) if (!next.has(key)) node.remove();
    if (next.size !== visible.size || [...next.keys()].some(key => !visible.has(key))) {
      // Preserve painter's order without detaching visible elements when a boundary enters view.
      let following = null;
      for (const node of [...next.values()].reverse()) {
        if (node.parentNode !== layer || node.nextSibling !== following) {
          layer.insertBefore(node, following);
        }
        following = node;
      }
      visible = next;
    }
  }
  function redraw() {
    if (!frame) frame = requestAnimationFrame(paint);
  }
  function notifyZoom() {
    const percent = Math.round(scale/fitted*100);
    container.dispatchEvent(new CustomEvent("graphzoom", {bubbles:true,detail:{percent}}));
    return percent;
  }
  function fit() {
    fitted = Math.min(width / Math.max(scene.width,1), height / Math.max(scene.height,1), 2);
    scale = fitted;
    left = (width - scene.width*scale)/2;
    top = (height - scene.height*scale)/2;
    redraw();
    return notifyZoom();
  }
  function zoom(factor, x = width/2, y = height/2) {
    if (!(factor > 0) || !Number.isFinite(factor)) return notifyZoom();
    const previous = scale;
    scale = Math.max(fitted*.3, Math.min(Math.max(8, fitted*8), scale*factor));
    left = x - (x-left)*scale/previous;
    top = y - (y-top)*scale/previous;
    redraw();
    return notifyZoom();
  }
  function select(ref) {
    const old = selectedRef;
    selectedRef = ref || null;
    for (const key of [old,selectedRef]) for (const row of byRef.get(key) || []) {
      visible.get(row.id)?.classList.toggle("selected", row.ref === selectedRef);
    }
    const matches = byRef.get(selectedRef);
    if (!matches?.length) return;
    const view = [-left/scale,-top/scale,(width-left)/scale,(height-top)/scale];
    if (!matches.some(row => intersects(row.box,view))) {
      const [l,t,r,b] = matches[0].box;
      left = (width - (l+r)*scale)/2;
      top = (height - (t+b)*scale)/2;
    }
    redraw();
  }
  function resize() {
    const rect = svg.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return;
    const first = width === 1;
    const relative = scale/fitted;
    const cx = (width/2-left)/scale, cy = (height/2-top)/scale;
    width = rect.width;
    height = rect.height;
    if (first || (Math.abs(relative-1) < .001 && !drag)) fit();
    else {
      fitted = Math.min(width / Math.max(scene.width,1), height / Math.max(scene.height,1), 2);
      scale = relative*fitted;
      left = width/2-cx*scale; top = height/2-cy*scale;
      redraw();
      notifyZoom();
    }
  }

  function modelPoint(e) {
    const rect = svg.getBoundingClientRect();
    return [(e.clientX-rect.left-left)/scale, (e.clientY-rect.top-top)/scale];
  }
  function offset(e) {
    const [x,y] = modelPoint(e);
    const round = value => Math.sign(value)*Math.round(Math.abs(value)) || 0;
    return {ref:drag.ref, dx:round((x-drag.origin[0])/pitchX), dy:round((y-drag.origin[1])/pitchY)};
  }
  function showGhost(e) {
    const {x,y,w,h,glyph} = drag.tile;
    const {dx,dy} = offset(e);
    const nx = x+dx*pitchX, ny = y+dy*pitchY;
    if (!overlay.firstChild) {
      const line = element("line",{x1:x,y1:y,stroke:"var(--accent, #ffb18d)","stroke-dasharray":"4 4"});
      const ghost = element("g");
      ghost.append(element("rect",{x:-w/2,y:-h/2,width:w,height:h,rx:2,
        fill:"var(--accent-bg, #423329)",stroke:"var(--accent, #ffb18d)","stroke-width":2}));
      ghost.append(element("text",{"text-anchor":"middle","dominant-baseline":"central",
        "font-family":"Georgia, serif",fill:"var(--accent, #ffb18d)"},glyph));
      overlay.append(line,ghost);
    }
    overlay.firstChild.setAttribute("x2",nx);
    overlay.firstChild.setAttribute("y2",ny);
    overlay.lastChild.setAttribute("transform",`translate(${nx} ${ny})`);
  }
  function endDrag() {
    drag = null;
    overlay.replaceChildren();
    svg.style.cursor = "grab";
  }
  on("pointerdown", e => {
    if (drag || (e.button !== 0 && e.button !== 1)) return;
    container.closest("[tabindex]")?.focus({preventScroll:true});
    const target = e.target.closest?.("[data-ref]");
    const ref = target?.dataset.ref;
    const origin = modelPoint(e);
    let tile;
    if (e.button === 0 && e.shiftKey && ref?.startsWith("factor:")) {
      const box = byRef.get(ref)?.find(row => row.item.primitive === "Box");
      if (box && (target.querySelector(".face") || intersects(box.box,[...origin,...origin]))) {
        tile = box.item;
      }
    }
    drag = {id:e.pointerId,x:e.clientX,y:e.clientY,left,top,ref,origin,tile,moved:false};
    svg.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  on("pointermove", e => {
    if (!drag || drag.id !== e.pointerId) return;
    const dx = e.clientX-drag.x, dy = e.clientY-drag.y;
    if (!drag.moved && Math.hypot(dx,dy) < 4) return;
    drag.moved = true;
    if (drag.tile) {
      svg.style.cursor = "move";
      showGhost(e);
      return;
    }
    svg.style.cursor = "grabbing";
    left = drag.left+dx; top = drag.top+dy;
    redraw();
  });
  on("pointerup", e => {
    if (!drag || drag.id !== e.pointerId) return;
    const clicked = !drag.moved && e.button === 0 && drag.ref;
    const change = drag.moved && drag.tile ? offset(e) : null;
    endDrag();
    svg.releasePointerCapture(e.pointerId);
    if (change && (change.dx || change.dy)) {
      container.dispatchEvent(new CustomEvent("graphmove",{bubbles:true,detail:change}));
    } else if (clicked) {
      select(clicked);
      selectCallback?.(clicked);
    }
  });
  on("lostpointercapture", endDrag);
  on("wheel", e => {
    e.preventDefault();
    const rect = svg.getBoundingClientRect();
    const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? height : 1);
    zoom(Math.exp(-Math.max(-300,Math.min(delta,300))*.002),e.clientX-rect.left,e.clientY-rect.top);
  }, {passive:false});
  const observer = new ResizeObserver(resize);
  observer.observe(svg);
  resize();
  return {select,zoom,fit,destroy() {
    signal.abort();
    observer.disconnect();
    cancelAnimationFrame(frame);
    svg.remove();
    visible.clear();
  }};
}
