const intro = {
  id: "intro", title: "Getting started with CFFG Studio",
  steps: [
    {title:"Your model library", target:".library-pane", text:"Open a saved graph here. Use New to start from a blank graph, a linear Gaussian tracker, or a switching dynamical system."},
    {title:"Read a Forney graph", target:"#graph", text:"Boxes are factors. Wires carry variables. Hover a wire to follow it through equality branches. Select a mark to inspect the model element it represents."},
    {title:"Find a variable", target:"#outline", text:"Filter the outline by name. Select a variable, then press F2 to rename it or Shift+R to resize it. The inspector shows its domain and connected factors."},
    {title:"Inspect and edit", target:"#inspector", text:"Select a factor to edit its bindings or initialize its parameters. Preview shows the affected elements before you commit. Every committed graph passes compiler validation."},
    {title:"Run an experiment", target:"[data-cmd='run']", text:"Run with seeded observations or a saved tape. Use Resource profile to measure cold and warm execution, or Step through inference to watch the solver."},
    {title:"Work from the keyboard", target:"[data-cmd='palette']", text:"Open the command palette with ⌘K or Ctrl+K. Search for Keyboard shortcuts to rebind a command. Tours also lists walkthroughs that agents have written for this model."},
  ],
};

const node = (tag, value, help) => {
  const result = document.createElement(tag);
  if (value != null) result.textContent = value;
  if (help) result.dataset.help = help;
  return result;
};

function modelLink(model, tour) {
  const url = new URL(location.href);
  url.search = new URLSearchParams(tour ? {model,tour} : {model});
  url.hash = '';
  return url.href;
}

function authorPrompt(model) {
  const cli = 'uv run --extra studio cffg-studio';
  return [
    'Write and publish a guided tour of the CFFG Studio model '+JSON.stringify(model.template.name)+'.',
    'Model ID: '+model.id+'\nModel link: '+modelLink(model.id),
    'From the CFFG checkout, run '+cli+' show '+model.id+' to read the current revision, graph, and valid reference keys in metadata.anchors. Remote agents use their configured CFFG_STUDIO_URL and edit-scoped CFFG_STUDIO_TOKEN.',
    'Explain what the model represents and trace one concrete path through its factors and variables. Base each explanation on the graph. Start with an overview, then alternate single elements with groups that show how they interact.',
    'Write tour.json as an array containing one set_tour command with a new lowercase tour id, title, and ordered steps. Each step needs title and text. Omit targets for an overview, use "ref" for one reference, or "refs" for a group of up to 64 distinct references. List the inspector focus first. For stops inside a plate, set "plate" to its name and use references from GET /v1/models/'+model.id+'?plate=PLATE_NAME. Run '+cli+' capabilities for the tour schema.',
    'Preview using the revision from show in place of REVISION:\n'+cli+' preview '+model.id+' --base REVISION --commands tour.json',
    'After a successful preview, publish with the same base:\n'+cli+' apply '+model.id+' --base REVISION --commands tour.json\nIf the base has changed, read the latest graph, check the targets, and preview again before applying.',
    'Return the playback link with your actual tour id in place of TOUR_ID: '+modelLink(model.id,'TOUR_ID'),
  ].join('\n\n');
}

export function setupTours({rootModel, showTarget, clearTarget, finishTarget, currentModel, report}) {
  const dialog = node("dialog");
  dialog.id = "studio-tours";
  dialog.className = "tour-dialog";
  dialog.setAttribute("aria-labelledby", "tour-title");
  const heading = node("h2");
  heading.id = "tour-title";
  heading.tabIndex = -1;
  const progress = node("span", undefined, "tour.progress");
  progress.className = "tour-progress";
  const motionSetting = "studio.tour.animations";
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const animations = node("input", undefined, "tour.animations");
  animations.type = "checkbox";
  let preference;
  try { preference = localStorage.getItem(motionSetting); } catch {}
  animations.checked = preference == null ? !reducedMotion.matches : preference === "on";
  const motionLabel = node("label", undefined, "tour.animations");
  motionLabel.append(animations,"Animations");
  motionLabel.hidden = true;
  const header = node("header");
  header.append(progress,motionLabel);
  const body = node("div");
  const controls = node("footer");
  const ring = node("div");
  ring.className = "tour-ring";
  ring.setAttribute("aria-hidden", "true");
  const card = node("section");
  card.append(header, heading, body, controls);
  dialog.append(ring, card);
  document.body.append(dialog);
  let owner, tour, index, target, trigger, ringAnimation, generation = 0;
  function settleMotion() {
    ringAnimation?.cancel();
    finishTarget();
  }
  animations.addEventListener("change", () => {
    preference = animations.checked ? "on" : "off";
    try { localStorage.setItem(motionSetting,preference); } catch {}
    if (!animations.checked) settleMotion();
  });
  reducedMotion.addEventListener("change", () => {
    if (preference != null) return;
    animations.checked = !reducedMotion.matches;
    if (!animations.checked) settleMotion();
  });
  function action(label, help, fn, primary=false) {
    const button = node("button", label, help);
    button.type = "button";
    button.className = primary ? "accent-button" : "neutral-button";
    button.addEventListener("click", () => Promise.resolve(fn()).catch(error => report(error.message)));
    return button;
  }
  function position() {
    if (!dialog.open || !tour) return;
    const framed = target?.classList.contains('tour-frame');
    const bounds = target?.isConnected ? (framed ? target.ownerSVGElement : target).getBoundingClientRect() : null;
    ring.hidden = framed || !bounds || (bounds.width < 1 && bounds.height < 1);
    if (!ring.hidden) {
      const x = Math.max(3, bounds.left-4), y = Math.max(3, bounds.top-4);
      Object.assign(ring.style, {left:x+"px", top:y+"px",
        width:Math.max(0, Math.min(innerWidth-3,bounds.right+4)-x)+"px",
        height:Math.max(0, Math.min(innerHeight-3,bounds.bottom+4)-y)+"px"});
    }
    const leftHalf = bounds && (bounds.left + bounds.right) / 2 > innerWidth / 2;
    const upperHalf = bounds && (bounds.top + bounds.bottom) / 2 > innerHeight / 2;
    card.style.left = leftHalf ? "12px" : "auto";
    card.style.right = leftHalf ? "auto" : "12px";
    card.style.top = upperHalf ? "12px" : "auto";
    card.style.bottom = upperHalf ? "auto" : "12px";
    body.tabIndex = body.scrollHeight > body.clientHeight ? 0 : -1;
  }
  function list() {
    generation++;
    ringAnimation?.cancel();
    motionLabel.hidden = true;
    clearTarget();
    tour = null; target = null; ring.hidden = true;
    dialog.classList.remove("playing");
    card.removeAttribute("style");
    progress.textContent = "";
    heading.textContent = "Guided tours";
    heading.dataset.help = "tour.list";
    body.replaceChildren();
    body.removeAttribute("tabindex");
    for (const entry of [intro,...Object.values(owner?.metadata.tours || {})]) {
      const choice = action(entry.title, entry === intro ? "tour.intro" : "tour.model", () => start(entry));
      const count = node("small", entry.steps.length+" steps", "tour.count");
      choice.append(count);
      choice.classList.add("tour-choice");
      body.append(choice);
    }
    controls.replaceChildren(action("Close", "tour.close", () => dialog.close()));
    if (owner) {
      const copy = action("Copy agent prompt", "tour.author", async () => {
        const prompt = authorPrompt(owner);
        try {
          await navigator.clipboard.writeText(prompt);
          copy.textContent = "Copied agent prompt";
        } catch {
          let field = body.querySelector("textarea");
          if (!field) {
            field = node("textarea", undefined, "tour.prompt");
            field.readOnly = true;
            field.rows = 5;
            field.setAttribute("aria-label","Agent prompt to copy");
            body.append(field);
          }
          field.value = prompt;
          field.focus();
          field.select();
        }
      });
      controls.prepend(copy);
    }
    body.querySelector("button")?.focus();
  }
  // The introductory tour travels between workstation panels in screen coordinates.
  function moveRing(from) {
    const to = ring.getBoundingClientRect();
    const width = Math.min(innerWidth-6,Math.max(from.width,to.width)*1.12 + Math.abs(from.x-to.x)*.5);
    const height = Math.min(innerHeight-6,Math.max(from.height,to.height)*1.12 + Math.abs(from.y-to.y)*.5);
    function bounds(rect,w,h,offset) {
      const x = Math.max(3,Math.min(innerWidth-w-3,rect.x+(rect.width-w)/2));
      const y = Math.max(3,Math.min(innerHeight-h-3,rect.y+(rect.height-h)/2));
      return {left:x+"px",top:y+"px",width:w+"px",height:h+"px",easing:"ease-in-out",offset};
    }
    ringAnimation = ring.animate([bounds(from,from.width,from.height,0),bounds(from,width,height,.3),
      bounds(to,width,height,.7),bounds(to,to.width,to.height,1)], {duration:850});
  }
  async function step(next, travel=true) {
    if (!tour || next < 0 || next >= tour.steps.length) return;
    const seq = ++generation;
    const from = !ring.hidden && target?.isConnected ? ring.getBoundingClientRect() : null;
    ringAnimation?.cancel();
    clearTarget();
    index = next; target = null; ring.hidden = true;
    const stop = tour.steps[index];
    progress.textContent = tour.title+" · "+(index+1)+" of "+tour.steps.length;
    motionLabel.hidden = false;
    heading.textContent = stop.title;
    heading.dataset.help = "tour.step";
    body.replaceChildren(node("p", stop.text, "tour.step"));
    body.scrollTop = 0;
    const back = action("Back", "tour.back", () => step(index-1));
    back.disabled = index === 0;
    function hint(button, glyph, key) {
      const kbd = node("kbd",glyph);
      kbd.setAttribute("aria-hidden","true");
      button.append(kbd);
      button.setAttribute("aria-keyshortcuts",key);
      return button;
    }
    controls.replaceChildren(action("End tour", "tour.end", () => dialog.close()), hint(back,"←","ArrowLeft"),
      index === tour.steps.length-1 ? action("Finish", "tour.finish", () => dialog.close(), true)
                                    : hint(action("Next", "tour.next", () => step(index+1), true),"→","ArrowRight"));
    if (tour !== intro) {
      const copy = action("Copy link", "tour.copy", async () => {
        await navigator.clipboard.writeText(modelLink(owner.id,tour.id));
        report("Copied tour link");
      });
      controls.prepend(copy);
    }
    dialog.classList.add("playing");
    position();
    heading.focus({preventScroll:true});
    const current = () => seq === generation && dialog.open;
    const animate = () => travel && animations.checked;
    const play = (scene, allowMotion=true) => scene?.focus?.(card, allowMotion && animate());
    let result;
    try {
      result = tour === intro ? {target:document.querySelector(stop.target)}
                             : await showTarget(owner, stop, current, play, animate);
    } catch (error) {
      if (seq === generation) body.append(node("p", error.message, "tour.target-error"));
    }
    if (!current()) return;
    target = result?.target;
    if (result?.missing?.length)
      body.append(node("small","Unavailable in this revision · "+result.missing.join(", "),"tour.unframed"));
    if (result?.offCanvas?.length)
      body.append(node("small","No canvas mark · "+result.offCanvas.join(", "),"tour.unframed"));
    if (!target && !result?.missing?.length && tour !== intro && !body.querySelector('[data-help="tour.target-error"]')) {
      body.append(node("small", "This target is unavailable in the current revision.", "tour.missing-target"));
    }
    const surface = target?.ownerSVGElement || target;
    if (surface) {
      const rect = surface.getBoundingClientRect();
      if (rect.top < 0 || rect.left < 0 || rect.bottom > innerHeight || rect.right > innerWidth)
        surface.scrollIntoView({block:"nearest",inline:"nearest"});
    }
    position();
    if (from && tour === intro && !ring.hidden && travel && animations.checked) moveRing(from);
    play(result);
  }
  function start(entry) { tour = entry; return step(0); }
  async function open(id) {
    if (!dialog.open) {
      trigger = document.activeElement;
      dialog.showModal();
    }
    const seq = ++generation;
    const model = await rootModel();
    if (seq !== generation || !dialog.open) return;
    owner = model;
    const entry = id === "intro" ? intro : owner?.metadata.tours?.[id];
    if (entry) await start(entry);
    else {
      list();
      if (id) body.prepend(node("p","This tour is unavailable in the current revision.", "tour.missing"));
    }
  }
  document.addEventListener("keydown", e => {
    if (!dialog.open || !tour || e.isComposing || e.altKey || e.metaKey || e.ctrlKey || e.shiftKey) return;
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    e.stopPropagation();
    void step(index + (e.key === "ArrowRight" ? 1 : -1));
  }, true);
  dialog.addEventListener("close", () => {
    ++generation; tour = null; target = null; ring.hidden = true;
    ringAnimation?.cancel();
    clearTarget();
    if (trigger?.isConnected) trigger.focus({preventScroll:true});
  });
  addEventListener("resize", position);
  document.addEventListener("scroll", position, true);
  return {open, async changed() {
    if (!dialog.open || !owner) return;
    if (currentModel() !== owner.id) { dialog.close(); return; }
    const seq = generation;
    const latest = await rootModel();
    if (seq !== generation || !dialog.open) return;
    owner = latest;
    if (!tour) { list(); return; }
    if (tour !== intro) {
      tour = owner.metadata.tours?.[tour.id];
      if (!tour) { list(); body.prepend(node("p", "This tour was removed in the current revision.", "tour.missing")); return; }
    }
    await step(Math.min(index, tour.steps.length-1),false);
  }};
}
