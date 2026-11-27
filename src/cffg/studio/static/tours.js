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

const node = (tag, value) => {
  const result = document.createElement(tag);
  if (value != null) result.textContent = value;
  return result;
};

export function setupTours({rootModel, showTarget, currentModel, report}) {
  const dialog = node("dialog");
  dialog.id = "studio-tours";
  dialog.className = "tour-dialog";
  dialog.setAttribute("aria-labelledby", "tour-title");
  const heading = node("h2");
  heading.id = "tour-title";
  heading.tabIndex = -1;
  const progress = node("span");
  progress.className = "tour-progress";
  const body = node("div");
  const controls = node("footer");
  const ring = node("div");
  ring.className = "tour-ring";
  ring.setAttribute("aria-hidden", "true");
  const card = node("section");
  card.append(progress, heading, body, controls);
  dialog.append(ring, card);
  document.body.append(dialog);
  let owner, tour, index, target, trigger, generation = 0;
  function action(label, fn, primary=false) {
    const button = node("button", label);
    button.type = "button";
    button.className = primary ? "accent-button" : "neutral-button";
    button.addEventListener("click", () => Promise.resolve(fn()).catch(error => report(error.message)));
    return button;
  }
  function position() {
    if (!dialog.open || !tour) return;
    const bounds = target?.isConnected ? target.getBoundingClientRect() : null;
    ring.hidden = !bounds || (bounds.width < 1 && bounds.height < 1);
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
  }
  function list() {
    generation++;
    tour = null; target = null; ring.hidden = true;
    dialog.classList.remove("playing");
    card.removeAttribute("style");
    progress.textContent = "";
    heading.textContent = "Guided tours";
    body.replaceChildren();
    for (const entry of [intro,...Object.values(owner?.metadata.tours || {})]) {
      const choice = action(entry.title, () => start(entry));
      const count = node("small", entry.steps.length+" steps");
      choice.append(count);
      choice.classList.add("tour-choice");
      body.append(choice);
    }
    controls.replaceChildren(action("Close", () => dialog.close()));
    body.querySelector("button")?.focus();
  }
  async function step(next) {
    if (!tour || next < 0 || next >= tour.steps.length) return;
    const seq = ++generation;
    index = next; target = null; ring.hidden = true;
    const stop = tour.steps[index];
    progress.textContent = tour.title+" · "+(index+1)+" of "+tour.steps.length;
    heading.textContent = stop.title;
    body.replaceChildren(node("p", stop.text));
    const back = action("Back", () => step(index-1));
    back.disabled = index === 0;
    controls.replaceChildren(action("End tour", () => dialog.close()), back,
      index === tour.steps.length-1 ? action("Finish", () => dialog.close(), true)
                                    : action("Next", () => step(index+1), true));
    if (tour !== intro) {
      const copy = action("Copy link", async () => {
        const url = new URL(location.href);
        url.search = new URLSearchParams({model:owner.id,tour:tour.id}).toString();
        await navigator.clipboard.writeText(url.href);
        report("Copied tour link");
      });
      controls.prepend(copy);
    }
    dialog.classList.add("playing");
    position();
    heading.focus({preventScroll:true});
    let result;
    try {
      result = tour === intro ? document.querySelector(stop.target)
                             : await showTarget(owner, stop);
    } catch (error) {
      if (seq === generation) body.append(node("p", error.message));
    }
    if (seq !== generation || !dialog.open) return;
    target = result;
    if (!target && tour !== intro) body.append(node("small", "This target is unavailable in the current revision."));
    if (target && !target.closest("svg")) target.scrollIntoView({block:"nearest",inline:"nearest"});
    position();
  }
  function start(entry) { tour = entry; return step(0); }
  async function open(id) {
    if (!dialog.open) {
      trigger = document.activeElement;
      dialog.showModal();
    }
    const seq = ++generation;
    owner = await rootModel();
    if (seq !== generation || !dialog.open) return;
    const entry = id === "intro" ? intro : owner?.metadata.tours?.[id];
    if (entry) await start(entry);
    else {
      list();
      if (id) body.prepend(node("p","This tour is unavailable in the current revision."));
    }
  }
  dialog.addEventListener("keydown", e => {
    if (!tour || e.altKey || e.metaKey || e.ctrlKey) return;
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    void step(index + (e.key === "ArrowRight" ? 1 : -1));
  });
  dialog.addEventListener("close", () => {
    ++generation; tour = null; target = null; ring.hidden = true;
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
      if (!tour) { list(); body.prepend(node("p", "This tour was removed in the current revision.")); return; }
    }
    await step(Math.min(index, tour.steps.length-1));
  }};
}
