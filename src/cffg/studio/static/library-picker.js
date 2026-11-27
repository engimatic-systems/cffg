const sheetUrl = new URL("./picker.css", import.meta.url).href;
const dockKey = "cffg.studio.docks.v1";
let activeDialog;
let dockCleanup;

function loadStyles() {
  if (document.querySelector('link[data-studio-picker]')) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = sheetUrl;
  link.dataset.studioPicker = "";
  document.head.append(link);
}

function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}

const dialogContents = `
  <header class="lp-heading"><h2 id="lp-title">Open model</h2><button type="button" class="lp-close" aria-label="Close">×</button></header>
  <main class="lp-body">
    <div class="lp-search"><svg aria-hidden="true" viewBox="0 0 24 24"><circle cx="10" cy="10" r="6.1"/><path d="m14.7 14.7 5.2 5.2"/></svg>
      <input autofocus id="lp-query" type="search" role="combobox" aria-label="Search models and host files" aria-controls="lp-results" aria-expanded="true" aria-autocomplete="list" placeholder="Search models and host files" autocomplete="off">
    </div>
    <div class="lp-sources" role="group" aria-label="Search sources">
      <button type="button" data-source="all" aria-pressed="true">All sources</button>
      <button type="button" data-source="library" aria-pressed="false">Library</button>
      <button type="button" data-source="host" aria-pressed="false">Host files</button>
    </div>
    <label class="lp-compatible"><input type="checkbox" id="lp-compatible"> Hide incompatible files</label>
    <div id="lp-results" class="lp-results" role="listbox" aria-label="Matching models and files" tabindex="0"></div>
    <p class="lp-status" role="status" aria-live="polite"></p>
    <p class="lp-error" role="alert" hidden></p>
    <details class="lp-settings">
      <summary>Host search settings</summary>
      <label for="lp-roots">Indexed folders on the host <span>(one path per line)</span></label>
      <textarea id="lp-roots" rows="3" spellcheck="false" wrap="off"></textarea>
      <div class="lp-root-actions"><button type="button" data-op="save-roots">Save folders &amp; scan</button><button type="button" data-op="scan">Rescan folders</button></div>
    </details>
  </main>
  <footer class="lp-footer">
    <input id="lp-file" type="file" accept=".cffgz,.cffgstudio" hidden>
    <button type="button" data-op="upload">Import bundle…</button>
    <span class="lp-spacer"></span><button type="button" data-op="cancel">Cancel</button>
    <button type="button" data-op="open" class="accent-button" disabled>Open</button>
  </footer>`;

export async function openLibrary({api, openModel, refresh, report}) {
  loadStyles();
  if (activeDialog?.open) {
    activeDialog.querySelector("#lp-query").focus();
    return;
  }
  const dialog = node("dialog", undefined, "library-picker");
  dialog.setAttribute("aria-labelledby", "lp-title");
  dialog.innerHTML = dialogContents;
  activeDialog = dialog;
  const $ = selector => dialog.querySelector(selector);
  const search = $("#lp-query"), list = $("#lp-results"), roots = $("#lp-roots");
  const open = $('[data-op="open"]'), status = $(".lp-status"), problem = $(".lp-error");
  const fileInput = $("#lp-file");
  const compatible = $("#lp-compatible");
  compatible.checked = localStorage.getItem("studio.hideIncompatible") !== "false";
  const sourceButtons = [...dialog.querySelectorAll("[data-source]")];
  const controller = new AbortController();
  let request, debounceTimer, scanTimer, generation = 0;
  let source = "all", matches = [], visible = [], index = -1;
  let scanning = false, working = false, pending = false, dirtyRoots = false;
  const listen = (el, type, fn) => el.addEventListener(type, fn, {signal:controller.signal});

  function setProblem(text = "") {
    problem.textContent = text;
    problem.hidden = !text;
  }
  function pick(i, scroll = false) {
    if (index >= 0) list.querySelector('[aria-selected="true"]')?.setAttribute("aria-selected","false");
    index = i >= 0 && i < visible.length ? i : -1;
    const row = index < 0 ? null : document.getElementById("lp-option-" + index);
    for (const control of [search,list]) {
      if (row) control.setAttribute("aria-activedescendant", row.id);
      else control.removeAttribute("aria-activedescendant");
    }
    if (row) {
      row.setAttribute("aria-selected","true");
      if (scroll) row.scrollIntoView({block:"nearest"});
    }
    const host = visible[index]?.source === "host";
    open.textContent = host ? "Import & open" : "Open";
    open.disabled = !row || Boolean(visible[index]?.diagnostic) || pending || working;
  }
  function render(previousKey) {
    visible = matches.filter(row => source === "all" || row.source === source);
    const content = document.createDocumentFragment();
    let lastSource = "";
    let options;
    visible.forEach((result,i) => {
      if (lastSource !== result.source) {
        lastSource = result.source;
        const group = node("div", undefined, "lp-group");
        group.setAttribute("role","group");
        const heading = node("h3", result.source === "library" ? "Model library" : "Files on the host");
        heading.id = "lp-section-" + result.source;
        group.setAttribute("aria-labelledby",heading.id);
        options = node("div");
        group.append(heading,options);
        content.append(group);
      }
      const option = node("div", undefined, "lp-option");
      option.id = "lp-option-" + i;
      option.setAttribute("role","option");
      option.setAttribute("aria-selected","false");
      option.dataset.index = i;
      if (result.diagnostic) option.setAttribute("aria-disabled", "true");
      const mark = node("span", undefined, "lp-mark lp-mark-" + result.source);
      mark.setAttribute("aria-hidden","true");
      const info = node("span", undefined, "lp-info");
      info.append(node("strong",result.name));
      if (result.path) info.append(node("small",result.path));
      if (result.diagnostic) info.append(node("small",result.diagnostic));
      const type = result.source === "host" ? (result.format || "Model file") :
        ({kalman:"Linear Gaussian",rslds:"rSLDS",blank:"Blank",import:"Imported"}[result.recipe] || "Model");
      option.append(mark,info,node("span",type,"lp-kind"));
      if (result.families) option.title = Array.isArray(result.families) ? result.families.join(", ") : result.families;
      options.append(option);
    });
    list.replaceChildren(content);
    const preserve = previousKey ? visible.findIndex(r => r.key === previousKey) : -1;
    pick(preserve >= 0 ? preserve : visible.length ? 0 : -1);
    const count = visible.length;
    status.textContent = pending ? "Searching…" : count ? count + (count === 1 ? " result" : " results") :
      search.value ? "No matches for this search." : "No models or files found.";
    if (scanning) status.textContent += " Scanning host folders…";
  }

  async function results() {
    const current = ++generation;
    request?.abort();
    request = new AbortController();
    const query = "?q=" + encodeURIComponent(search.value.trim());
    const responses = await Promise.allSettled([
      api("/v1/models"+query,{signal:request.signal}),
      api("/v1/search"+query+"&compatible_only="+compatible.checked,{signal:request.signal}),
    ]);
    if (current !== generation || !dialog.open) return;
    const previous = visible[index]?.key;
    const [modelResult, hostResult] = responses;
    matches = [];
    const errors = [];
    if (modelResult.status === "fulfilled") {
      for (const model of modelResult.value) matches.push({...model,source:"library",key:"library:"+model.id});
    } else errors.push("Model library: " + (modelResult.reason?.message || String(modelResult.reason)));
    scanning = false;
    if (hostResult.status === "fulfilled") {
      const data = hostResult.value;
      for (const file of data.files) matches.push({...file,source:"host",key:"host:"+file.path});
      if (!dirtyRoots) roots.value = data.roots.join("\n");
      scanning = data.scanning;
      if (data.error) errors.push("Host scan: " + data.error);
    } else errors.push("Host files: " + (hostResult.reason?.message || String(hostResult.reason)));
    pending = false;
    list.removeAttribute("aria-busy");
    render(previous);
    if (errors.length) setProblem(errors.join("\n"));
    clearTimeout(scanTimer);
    if (scanning) scanTimer = setTimeout(results, 1500);
  }

  function searchChanged() {
    ++generation;
    request?.abort();
    clearTimeout(debounceTimer);
    clearTimeout(scanTimer);
    pending = true;
    matches = [];
    setProblem();
    list.setAttribute("aria-busy","true");
    render();
    debounceTimer = setTimeout(results,150);
  }
  function busy(value) {
    working = value;
    for (const button of dialog.querySelectorAll('button, input:not([type="file"]), textarea')) button.disabled = value;
    pick(index);
  }
  async function openResult(result, upload) {
    if (working || !result && !upload) return;
    if (result?.diagnostic) { setProblem(result.diagnostic); return; }
    setProblem();
    busy(true);
    clearTimeout(scanTimer);
    const importing = upload || result.source === "host";
    let importedId;
    try {
      status.textContent = importing ? "Importing into the model library…" : "Opening model…";
      let id = result?.id;
      if (upload) {
        const response = await fetch("/v1/import/bundle",{method:"POST",credentials:"same-origin",
          headers:{"Content-Type":"application/octet-stream"},body:upload});
        const payload = await response.json().catch(() => null);
        if (!response.ok) throw new Error(payload?.detail || "Bundle import failed (HTTP " + response.status + ")");
        id = payload.id;
      } else if (importing) {
        id = (await api("/v1/import",{method:"POST",body:{path:result.path}})).id;
      }
      if (!id) throw new Error("The library returned no model identifier.");
      if (importing) {
        importedId = id;
        await refresh();
      }
      await openModel(id);
      dialog.close();
    } catch (error) {
      const explanation = error.message || String(error);
      setProblem(importedId ? "Imported into the library. Opening failed: " + explanation : explanation);
      if (importedId) {
        matches.unshift({key:"library:"+importedId,source:"library",id:importedId,name:upload?.name || result.name,recipe:"import"});
        source = "all";
        for (const button of sourceButtons) button.setAttribute("aria-pressed",String(button.dataset.source === source));
      }
      render(importedId ? "library:"+importedId : result?.key);
      report?.(problem.textContent);
    } finally {
      fileInput.value = "";
      if (dialog.open) busy(false);
    }
  }

  async function scan(save) {
    if (working) return;
    const folders = [...new Set(roots.value.split("\n").map(path => path.trim()).filter(Boolean))];
    if (save && !folders.length) {
      roots.focus(); setProblem("Enter at least one host folder."); return;
    }
    busy(true);
    setProblem();
    try {
      await api("/v1/search/scan",{method:"POST",body:save ? {roots:folders} : {}});
      if (save) dirtyRoots = false;
      await results();
    } catch (error) {
      setProblem(error.message || String(error));
    } finally {
      if (dialog.open) busy(false);
    }
  }
  listen(search,"input",searchChanged);
  listen(compatible,"change",() => {
    localStorage.setItem("studio.hideIncompatible",String(compatible.checked));
    searchChanged();
  });
  listen(roots,"input",() => { dirtyRoots = true; });
  for (const button of sourceButtons) listen(button,"click",() => {
    source = button.dataset.source;
    for (const b of sourceButtons) b.setAttribute("aria-pressed",String(b.dataset.source === source));
    render();
    search.focus();
  });
  listen(list,"click",event => {
    const row = event.target.closest("[data-index]");
    if (row && !pending) { pick(Number(row.dataset.index)); openResult(visible[index]); }
  });
  listen(dialog,"keydown",event => {
    if (event.target !== search && event.target !== list || working) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (visible.length) pick(Math.max(0,Math.min(visible.length-1,index+(event.key === "ArrowDown" ? 1 : -1))),true);
    } else if (event.key === "Enter" && !event.isComposing) {
      event.preventDefault();
      if (!pending) openResult(visible[index]);
    }
  });
  listen(open,"click",() => openResult(visible[index]));
  listen($('[data-op="upload"]'),"click",() => fileInput.click());
  listen(fileInput,"change",() => { if (fileInput.files[0]) openResult(null,fileInput.files[0]); });
  listen($('[data-op="scan"]'),"click",() => scan(false));
  listen($('[data-op="save-roots"]'),"click",() => scan(true));
  listen(dialog,"cancel",event => { if (working) event.preventDefault(); });
  for (const button of [$(".lp-close"),$('[data-op="cancel"]')]) listen(button,"click",() => dialog.close());
  const closed = new Promise(resolve => dialog.addEventListener("close",() => {
    ++generation; request?.abort(); controller.abort(); clearTimeout(debounceTimer); clearTimeout(scanTimer);
    activeDialog = undefined; dialog.remove(); resolve();
  },{once:true}));
  document.body.append(dialog);
  dialog.showModal();
  search.focus();
  searchChanged();
  return closed;
}

export function installDockSizing() {
  loadStyles();
  if (dockCleanup) return dockCleanup;
  const workspace = document.querySelector(".workspace");
  if (!workspace) return () => {};
  const panels = {
    navigation:workspace.querySelector(".navigation-dock"),
    inspector:workspace.querySelector(".inspector-pane"),
    activity:workspace.querySelector(".activity-pane"),
    library:workspace.querySelector(".library-pane"),
  };
  if (Object.values(panels).some(panel => !panel)) return () => {};
  let sizes = {};
  try {
    const stored = JSON.parse(localStorage.getItem(dockKey));
    if (stored && typeof stored === "object") sizes = stored;
  } catch { /* Browser storage is optional. */ }
  workspace.classList.add("studio-docks");
  const bars = new Map();
  const controller = new AbortController();
  let paintFrame, drag;

  function persist() {
    try { localStorage.setItem(dockKey,JSON.stringify(sizes)); } catch { /* Session-only layout. */ }
  }
  function layoutMode() {
    return matchMedia("(max-width:590px)").matches ? "mobile" :
      matchMedia("(max-width:790px)").matches ? "tablet" : "desktop";
  }
  function range(key) {
    const w = workspace.clientWidth, h = workspace.clientHeight;
    if (key === "navigation") return [120, Math.max(120, Math.floor(w*.34))];
    if (key === "inspector") return [165, Math.max(165, Math.floor(w*.38))];
    if (key === "library") return [65, Math.max(65, panels.navigation.clientHeight-100)];
    return [86, Math.max(86, Math.min(Math.floor(h*.67),h-145))];
  }
  function change(key,pixels) {
    const [min,max] = range(key);
    const clamped = Math.round(Math.max(min,Math.min(max,pixels)));
    workspace.style.setProperty("--dock-"+key,clamped+"px");
    sizes[key] = clamped;
  }
  function measure() {
    paintFrame = undefined;
    const outer = workspace.getBoundingClientRect();
    const mode = layoutMode();
    for (const [key,bar] of bars) {
      bar.hidden = mode === "mobile" || key === "inspector" && mode !== "desktop";
      if (bar.hidden) continue;
      const rect = panels[key].getBoundingClientRect();
      const vertical = key === "navigation" || key === "inspector";
      const dimension = vertical ? rect.width : rect.height;
      let x = rect.left-outer.left, y = rect.top-outer.top;
      let width = rect.width, height = rect.height;
      if (vertical) { x += key === "navigation" ? width-4 : -4; width=8; }
      else { y += key === "library" ? height-4 : -4; height=8; }
      if (key === "activity" && mode === "tablet") { x=0; width=outer.width; }
      Object.assign(bar.style,{left:x+"px",top:y+"px",width:width+"px",height:height+"px"});
      const [min,max] = range(key);
      bar.setAttribute("aria-valuemin",min);
      bar.setAttribute("aria-valuemax",max);
      bar.setAttribute("aria-valuenow",Math.round(dimension));
      bar.setAttribute("aria-valuetext",Math.round(dimension)+" pixels");
    }
  }
  function scheduleMeasure() {
    if (!paintFrame) paintFrame=requestAnimationFrame(measure);
  }
  for (const [key,panel] of Object.entries(panels)) {
    if (Number.isFinite(sizes[key])) workspace.style.setProperty("--dock-"+key, sizes[key]+"px");
    const vertical = key === "navigation" || key === "inspector";
    const direction = key === "inspector" || key === "activity" ? -1 : 1;
    const bar = node("div",undefined,"studio-splitter");
    bar.tabIndex=0;
    bar.dataset.dock=key;
    bar.setAttribute("role","separator");
    bar.setAttribute("aria-orientation",vertical ? "vertical" : "horizontal");
    const label = "Resize " + key + " panel";
    bar.setAttribute("aria-label",label);
    bar.title=label+" · Double-click to reset";
    if (!panel.id) panel.id="studio-dock-"+key;
    bar.setAttribute("aria-controls",panel.id);
    bars.set(key,bar);
    workspace.append(bar);
    const listen=(event,fn) => bar.addEventListener(event,fn,{signal:controller.signal});
    const extent=() => panel.getBoundingClientRect()[vertical ? "width" : "height"];
    const position=e => vertical ? e.clientX : e.clientY;
    listen("keydown",e => {
      const positive = vertical ? "ArrowRight" : "ArrowDown";
      const negative = vertical ? "ArrowLeft" : "ArrowUp";
      if (![positive,negative,"Home","End"].includes(e.key)) return;
      e.preventDefault();
      const [min,max]=range(key);
      const increment=(e.shiftKey ? 40 : 12)*direction*(e.key===positive ? 1 : -1);
      change(key,e.key==="Home" ? min : e.key==="End" ? max : extent()+increment);
      persist(); scheduleMeasure();
    });
    listen("pointerdown",e => {
      if (e.button!==0 || drag) return;
      drag={id:e.pointerId,bar,at:position(e),size:extent()};
      bar.setPointerCapture(e.pointerId);
      bar.setAttribute("data-dragging","");
      bar.focus({preventScroll:true});
      e.preventDefault();
    });
    listen("pointermove",e => {
      if (drag?.id!==e.pointerId || drag.bar!==bar) return;
      change(key,drag.size+(position(e)-drag.at)*direction);
      scheduleMeasure();
    });
    listen("lostpointercapture",() => {
      if (drag?.bar===bar) { drag=undefined;bar.removeAttribute("data-dragging");persist(); }
    });
    listen("dblclick",() => {
      delete sizes[key];
      workspace.style.removeProperty("--dock-"+key);
      persist(); scheduleMeasure();
    });
  }
  const observer=new ResizeObserver(scheduleMeasure);
  observer.observe(workspace);
  Object.values(panels).forEach(panel=>observer.observe(panel));
  window.addEventListener("resize",scheduleMeasure,{signal:controller.signal});
  measure();
  dockCleanup=() => {
    controller.abort();
    observer.disconnect();
    cancelAnimationFrame(paintFrame);
    workspace.classList.remove("studio-docks");
    for (const [key,bar] of bars) {
      bar.remove(); workspace.style.removeProperty("--dock-"+key);
    }
    dockCleanup=undefined;
  };
  return dockCleanup;
}
