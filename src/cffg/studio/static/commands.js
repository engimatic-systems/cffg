const storageKey = "cffg.studio.keymap.v1";
const mac = /mac|iphone|ipad|ios/i.test(navigator.userAgentData?.platform ?? navigator.platform);
const platformMod = mac ? "Meta" : "Ctrl";
const modOrder = ["Mod", "Ctrl", "Meta", "Alt", "Shift"];
const modifiers = new Map(Object.entries({
  mod: "Mod", ctrl: "Ctrl", control: "Ctrl", cmd: "Meta", command: "Meta", meta: "Meta",
  alt: "Alt", option: "Alt", shift: "Shift",
}));
const namedKeys = new Map([
  "Enter", "Escape", "Backspace", "Delete", "Space", "Tab", "Home", "End",
  "Insert", "PageUp", "PageDown", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
].map(key => [key.toLowerCase(), key]));
for (const [alias, key] of Object.entries({ esc: "Escape", return: "Enter", spacebar: "Space",
  up: "ArrowUp", down: "ArrowDown", left: "ArrowLeft", right: "ArrowRight", plus: "+", minus: "-" })) {
  namedKeys.set(alias, key);
}

function stroke(text) {
  const parts = text.replace(/\+$/, "Plus").split("+");
  const raw = parts.pop();
  const mods = new Set();
  for (const part of parts) {
    const mod = modifiers.get(part.toLowerCase());
    if (!mod || mods.has(mod)) throw new Error(`Invalid modifier in “${text}”.`);
    mods.add(mod);
  }
  let key = namedKeys.get(raw.toLowerCase());
  if (!key && /^F(?:[1-9]|1\d|2[0-4])$/i.test(raw)) key = raw.toUpperCase();
  if (!key && /^[!-~]$/.test(raw)) key = raw.toLowerCase();
  if (!key) throw new Error(`Unknown key “${raw}”. Use a key name such as F2, Enter or ArrowLeft.`);
  if (key === "+" || key === "?") {
    mods.add("Shift");
    key = key === "+" ? "=" : "/";
  }
  if (mods.has("Mod") && (mods.has("Ctrl") || mods.has("Meta"))) {
    throw new Error("Choose Mod for a portable shortcut, or an explicit Ctrl or Meta modifier.");
  }
  return [...modOrder.filter(mod => mods.has(mod)), key].join("+");
}

function resolved(strokeText) {
  const parts = strokeText.split("+");
  const key = parts.pop();
  const mods = new Set(parts.map(mod => mod === "Mod" ? platformMod : mod));
  return [...modOrder.filter(mod => mods.has(mod)), key].join("+");
}

function eventStroke(event) {
  if (!event.key || modifiers.has(event.key.toLowerCase())) return "";
  const mods = [[event.ctrlKey, "Ctrl"], [event.metaKey, "Meta"], [event.altKey, "Alt"], [event.shiftKey, "Shift"]];
  let key = event.key === " " ? "Space" : event.key;
  // Option can turn a letter into a dead key or a symbol even while Command is held.
  if (mac && event.altKey && /^Key[A-Z]$/.test(event.code) && !/^[a-z]$/i.test(key)) {
    key = event.code.slice(3);
  }
  try {
    return resolved(stroke([...mods.filter(([held]) => held).map(([, mod]) => mod), key].join("+")));
  } catch { return ""; }
}

// These browser and window commands cannot be reliably overridden. F6 and Mod+k are Studio commands.
const reserved = new Set([
  "Tab", "Shift+Tab", "F5", "F11", "F12", "Alt+F4", "Alt+ArrowLeft", "Alt+ArrowRight",
  "Mod+l", "Mod+t", "Mod+w", "Mod+q", "Mod+n", "Mod+r", "Mod+Shift+t",
  "Mod+Shift+w", "Mod+Shift+n", "Mod+Shift+r", "Ctrl+Tab", "Ctrl+Shift+Tab",
  ...Array.from({ length: 9 }, (_, i) => "Mod+" + (i + 1)),
].map(resolved));

const panelSelector = "[data-command-context],#models,#outline,#viewport,#graph,#inspector,#activity";
function scopeName(value) {
  const scope = String(value || "global").toLowerCase().trim();
  return ({ models: "library", viewport: "graph", workspace: "global" })[scope] ?? scope;
}
function contextAt(element) {
  const target = element?.closest?.(panelSelector);
  return scopeName(target?.dataset.commandContext || target?.id);
}
function isTextField(element) {
  return element?.isContentEditable || element?.closest?.("input, textarea, select, [role=textbox]");
}
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function keyBindings(commands, overrides) {
  const entries = [];
  for (const command of commands.values()) {
    const config = overrides[command.id] ?? command;
    const context = scopeName(config.context);
    if (!/^[a-z][a-z0-9_-]*$/.test(context)) throw new Error(`Invalid context for ${command.label}.`);
    const keys = (config.keys ?? "").trim();
    const strokes = keys ? keys.split(/\s+/).map(stroke) : [];
    if (strokes.length > 2) throw new Error("Use one shortcut or a chord of two shortcuts.");
    const sequence = strokes.map(resolved);
    if (sequence.some(key => reserved.has(key))) throw new Error(`The browser reserves “${keys}” (${command.label}).`);
    if (sequence.length) {
      for (const entry of entries) {
        if (!entry.sequence.length || (context !== "global" && entry.context !== "global" && entry.context !== context)) continue;
        const length = Math.min(entry.sequence.length, sequence.length);
        if (entry.sequence.slice(0, length).join(" ") === sequence.slice(0, length).join(" ")) {
          throw new Error(`“${keys}” conflicts with ${entry.command.label} in ${entry.context} (${command.label}).`);
        }
      }
    }
    entries.push({ command, context, keys: strokes.join(" "), sequence });
  }
  return entries;
}

function readProfile(text, commands) {
  const data = JSON.parse(text);
  if (data?.version !== 1 || !data.bindings || Array.isArray(data.bindings) || typeof data.bindings !== "object") {
    throw new Error("Expected a Studio shortcut profile with version 1 and a bindings object.");
  }
  const overrides = Object.create(null);
  for (const [id, binding] of Object.entries(data.bindings)) {
    if (!commands.has(id)) throw new Error(`Unknown command “${id}”.`);
    if (typeof binding?.keys !== "string" || typeof binding?.context !== "string") {
      throw new Error(`“${id}” needs a shortcut and context string.`);
    }
    overrides[id] = { keys: binding.keys, context: binding.context };
  }
  keyBindings(commands, overrides);
  return overrides;
}
function profileText(bindings) {
  return JSON.stringify({ version: 1, bindings }, null, 2);
}

export function setupCommands(definitions, report = () => {}) {
  const commands = new Map(definitions.map(command => [command.id, command]));
  commands.set("palette", { id: "palette", label: "Command palette", keys: "Mod+k", ...commands.get("palette"), run: showPalette });
  commands.set("keys", { id: "keys", label: "Keyboard shortcuts", ...commands.get("keys"), run: showKeys });
  const palette = document.getElementById("palette");
  const input = document.getElementById("palette-input");
  const list = document.getElementById("palette-list");
  let overrides = Object.create(null);
  try {
    const stored = localStorage.getItem(storageKey);
    if (stored) overrides = readProfile(stored, commands);
  } catch (error) { report(`Using default shortcuts. ${error.message}`); }
  let bindings = keyBindings(commands, overrides);
  let opener = null;
  let active = -1;
  let matches = [];
  let pending = null;
  let timeout;
  const keyDialog = node("dialog");
  keyDialog.id = "keymap-dialog";
  keyDialog.setAttribute("aria-labelledby", "keymap-title");
  document.body.append(keyDialog);
  input.dataset.help = "palette.search";
  list.dataset.help = "palette.results";
  input.setAttribute("role", "combobox");
  input.setAttribute("aria-autocomplete", "list");
  input.setAttribute("aria-controls", list.id);
  input.setAttribute("aria-expanded", "false");
  list.setAttribute("role", "listbox");

  function clearChord() {
    clearTimeout(timeout);
    pending = null;
  }
  function enabled(command) { return !command.enabled || command.enabled(); }
  async function execute(id) {
    const command = commands.get(id);
    if (!command) { report(`Unknown command “${id}”.`); return false; }
    try {
      if (!enabled(command)) { report(`${command.label} is unavailable.`); return false; }
      await command.run();
      return true;
    } catch (error) { report(error.message ?? String(error)); return false; }
  }
  function open(dialog) {
    if (dialog.open) return true;
    if (document.querySelector("dialog[open]")) return false;
    clearChord();
    opener = document.activeElement;
    dialog.showModal();
    return true;
  }
  function close(dialog) {
    if (!dialog.open) return;
    dialog.close();
    if (opener?.isConnected) opener.focus({ preventScroll: true });
    opener = null;
    input.setAttribute("aria-expanded", "false");
  }
  for (const dialog of [palette, keyDialog]) {
    dialog.addEventListener("cancel", event => { event.preventDefault(); close(dialog); });
  }
  function highlight(index, scroll = true) {
    active = index;
    for (let i = 0; i < list.children.length; i++) {
      list.children[i].setAttribute("aria-selected", String(i === index));
    }
    const current = index < 0 ? null : list.children[index];
    if (!current) { input.removeAttribute("aria-activedescendant"); return; }
    input.setAttribute("aria-activedescendant", current.id);
    if (scroll) current.scrollIntoView({ block: "nearest" });
  }
  function runMatch(index) {
    const entry = matches[index];
    if (!entry || !enabled(entry.command)) return;
    close(palette);
    void execute(entry.command.id);
  }
  function search() {
    const words = input.value.toLowerCase().trim().split(/\s+/);
    matches = bindings.filter(({ command, context, keys }) =>
      words.every(word => `${command.label} ${command.id} ${context} ${keys}`.toLowerCase().includes(word)));
    list.replaceChildren();
    matches.forEach(({ command, context, keys }, i) => {
      const option = node("button");
      option.id = `command-option-${i}`;
      option.dataset.help = "command." + command.id;
      option.type = "button";
      option.tabIndex = -1;
      option.setAttribute("role", "option");
      option.disabled = !enabled(command);
      option.append(node("span", command.label));
      const details = [];
      if (option.disabled) details.push("Unavailable");
      if (context !== "global") details.push(context);
      if (keys) details.push(keys);
      if (details.length) {
        const detail = node("small", details.join(" · "));
        detail.dataset.help = "palette.details";
        option.append(detail);
      }
      option.addEventListener("mousedown", event => event.preventDefault());
      option.addEventListener("click", () => runMatch(i));
      list.append(option);
    });
    if (!matches.length) {
      const empty = node("p", "No matching commands.", "empty");
      empty.dataset.help = "palette.empty";
      empty.setAttribute("role", "status");
      list.append(empty);
    }
    highlight(matches.findIndex(entry => enabled(entry.command)), false);
    list.scrollTop = 0;
  }
  function showPalette() {
    if (!open(palette)) return;
    input.value = "";
    input.setAttribute("aria-expanded", "true");
    search();
    input.focus();
  }
  input.addEventListener("input", search);
  input.addEventListener("keydown", event => {
    if (event.isComposing) return;
    if (event.key === "Enter") {
      event.preventDefault();
      runMatch(active);
    } else if (["ArrowDown", "ArrowUp"].includes(event.key)) {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      for (let n = 1; n <= matches.length; n++) {
        const index = (active + n * direction + matches.length) % matches.length;
        if (enabled(matches[index].command)) { highlight(index); break; }
      }
    }
  });

  function showKeys() {
    if (keyDialog.open || !open(keyDialog)) return;
    let draft = Object.assign(Object.create(null), overrides);
    const form = node("form");
    const header = node("header", undefined, "form-header");
    const title = node("h2", "Keyboard shortcuts");
    title.id = "keymap-title";
    header.append(title);
    const fields = node("div", undefined, "detail-content");
    Object.assign(fields.style, { display: "grid", gap: "9px", maxHeight: "56dvh", overflowY: "auto" });
    const commandList = node("select");
    commandList.size = 6;
    commandList.dataset.help = "keys.command";
    const bindingInput = node("input");
    bindingInput.dataset.help = "keys.shortcut";
    bindingInput.placeholder = "Unbound";
    bindingInput.autocomplete = "off";
    bindingInput.spellcheck = false;
    const contextInput = node("input");
    contextInput.dataset.help = "keys.context";
    contextInput.setAttribute("list", "keymap-contexts");
    const contexts = node("datalist");
    contexts.id = "keymap-contexts";
    for (const name of new Set(["global", "graph", "library", "outline", "inspector", "activity", ...bindings.map(e => e.context)])) {
      const option = node("option");
      option.value = name;
      contexts.append(option);
    }
    function field(label, id, control) {
      const wrap = node("div");
      const name = node("label", label);
      name.htmlFor = control.id = id;
      name.dataset.help = control.dataset.help;
      name.style.display = "block";
      control.style.width = "100%";
      wrap.append(name, control);
      return wrap;
    }
    const hint = node("p", "Mod uses Command on Mac and Ctrl elsewhere. Use Mod+Shift+z for one shortcut or g g for a chord. Clear the shortcut to unbind it.", "empty");
    hint.id = "keymap-hint";
    hint.dataset.help = "keys.shortcut";
    bindingInput.setAttribute("aria-describedby", hint.id);
    const feedback = node("p");
    feedback.dataset.help = "keys.feedback";
    feedback.setAttribute("role", "status");
    function message(text, error = false) {
      feedback.textContent = text;
      feedback.className = error ? "error" : "empty";
    }
    fields.append(field("Command", "keymap-command", commandList), field("Shortcut", "keymap-keys", bindingInput),
      hint, field("Context", "keymap-context", contextInput), contexts);
    const tools = node("div", undefined, "tool-group");
    tools.style.flexWrap = "wrap";
    function button(label, help, action, parent) {
      const control = node("button", label, "neutral-button");
      control.dataset.help = help;
      control.type = "button";
      control.addEventListener("click", action);
      parent.append(control);
      return control;
    }
    function config(id) { return draft[id] ?? commands.get(id); }
    function loadCommand() {
      const entry = config(commandList.value);
      bindingInput.value = entry.keys ?? "";
      contextInput.value = scopeName(entry.context);
    }
    function options(selected = commands.keys().next().value) {
      commandList.replaceChildren();
      for (const command of commands.values()) {
        const option = node("option", command.label + (config(command.id).keys ? " · " + config(command.id).keys : ""));
        option.dataset.help = "command." + command.id;
        option.value = command.id;
        commandList.append(option);
      }
      commandList.value = selected;
      loadCommand();
    }
    function updateDraft() {
      const id = commandList.value;
      draft[id] = { keys: bindingInput.value.trim(), context: scopeName(contextInput.value) };
      commandList.selectedOptions[0].textContent = commands.get(id).label + (draft[id].keys ? " · " + draft[id].keys : "");
      try { keyBindings(commands, draft); message(""); }
      catch (error) { message(error.message, true); }
    }
    commandList.addEventListener("change", loadCommand);
    bindingInput.addEventListener("input", updateDraft);
    contextInput.addEventListener("input", updateDraft);
    const file = node("input");
    file.type = "file";
    file.dataset.help = "keys.import";
    file.accept = ".json,application/json";
    file.hidden = true;
    file.addEventListener("change", async () => {
      const selected = file.files[0];
      if (!selected) return;
      try {
        draft = readProfile(await selected.text(), commands);
        options(commandList.value);
        message("Profile imported. Save to apply it.");
      } catch (error) { message(error.message, true); }
      file.value = "";
    });
    button("Import JSON", "keys.import", () => file.click(), tools);
    button("Export JSON", "keys.export", () => {
      try { keyBindings(commands, draft); } catch (error) { message(error.message, true); return; }
      const url = URL.createObjectURL(new Blob([profileText(draft)], { type: "application/json" }));
      const link = node("a");
      link.href = url;
      link.download = "cffg-studio-keymap.json";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, tools);
    button("Restore defaults", "keys.restore", () => {
      draft = Object.create(null);
      options(commandList.value);
      message("Defaults restored. Save to apply them.");
    }, tools);
    fields.append(tools, file, feedback);
    const footer = node("footer", undefined, "form-footer");
    button("Cancel", "keys.cancel", () => close(keyDialog), footer);
    const save = node("button", "Save shortcuts", "accent-button");
    save.type = "submit";
    save.dataset.help = "keys.save";
    footer.append(save);
    form.addEventListener("submit", event => {
      event.preventDefault();
      try { bindings = keyBindings(commands, draft); } catch (error) { message(error.message, true); return; }
      overrides = draft;
      close(keyDialog);
      try { localStorage.setItem(storageKey, profileText(overrides)); report("Keyboard shortcuts saved."); }
      catch { report("Keyboard shortcuts applied in this tab. Browser storage is unavailable."); }
    });
    form.append(header, fields, footer);
    keyDialog.replaceChildren(form);
    options();
    commandList.focus();
  }

  document.addEventListener("click", event => {
    const control = event.target.closest?.("[data-cmd]");
    if (!control || control.disabled) return;
    event.preventDefault();
    clearChord();
    void execute(control.dataset.cmd);
  });
  document.addEventListener("keydown", event => {
    if (event.defaultPrevented || event.repeat || event.isComposing || event.getModifierState("AltGraph")) return;
    if (document.querySelector("dialog[open]") || isTextField(event.target)) { clearChord(); return; }
    if ([" ", "Enter"].includes(event.key) && !(event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) && event.target.closest?.("button,a[href]")) return;
    const key = eventStroke(event);
    if (!key) return;
    const context = contextAt(event.target);
    const applicable = bindings.filter(entry => entry.sequence.length && (entry.context === "global" || entry.context === context) && enabled(entry.command));
    if (pending) {
      const match = pending.context === context && applicable.find(entry => entry.sequence[0] === pending.first && entry.sequence[1] === key);
      clearChord();
      if (match) { event.preventDefault(); void execute(match.command.id); return; }
    }
    const candidates = applicable.filter(entry => entry.sequence[0] === key);
    if (!candidates.length) return;
    event.preventDefault();
    if (candidates[0].sequence.length === 1) { void execute(candidates[0].command.id); return; }
    pending = { first: key, context };
    timeout = setTimeout(clearChord, 1500);
  });
  window.addEventListener("blur", clearChord);
  return { execute, showPalette, showKeys };
}
