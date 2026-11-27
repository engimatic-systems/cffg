// Keep one tooltip in the same modal context as its trigger.
export function installHelp() {
  const tip = document.createElement('div');
  tip.id = 'studio-option-help';
  tip.setAttribute('role', 'tooltip');
  tip.hidden = true;

  let entries, keyHeld = false, held = false, pinned = false, dismissed = false, awake = true;
  let pointer = null, keyboard = true, owner = null, frame = 0;
  let nativeTitle = null, loading = false;
  const toggle = document.getElementById('help-gesture');
  const mac = /Mac|iPhone|iPad/.test(navigator.platform);
  toggle.querySelector('kbd').textContent = mac ? '⌥' : 'Alt';
  const enabled = () => awake && (pinned || (held && !dismissed));

  function hide() {
    if (owner) {
      const ids = (owner.getAttribute('aria-describedby') || '').split(/\s+/)
        .filter(id => id && id !== tip.id);
      if (ids.length) owner.setAttribute('aria-describedby', ids.join(' '));
      else owner.removeAttribute('aria-describedby');
      if (nativeTitle !== null && !owner.hasAttribute('title')) owner.title = nativeTitle;
    }
    nativeTitle = null;
    owner = null;
    if (!tip.hidden) tip.hidden = true;
  }

  function render() {
    if (!enabled() || !entries) return hide();
    let hit = keyboard ? document.activeElement : pointer && document.elementFromPoint(...pointer);
    if (keyboard && hit === document.body) {
      // A fresh page can already be under a stationary pointer.
      const hovered = document.querySelectorAll(':hover');
      hit = hovered[hovered.length - 1];
    }
    const target = hit?.closest('[data-help]');
    const entry = target && !hit.closest('#viewport') && entries[target.dataset.help];
    if (entry?.review !== 'approved') return hide();

    if (target !== owner) {
      hide();
      owner = target;
      const ids = (target.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean);
      target.setAttribute('aria-describedby', [...new Set([...ids, tip.id])].join(' '));
      nativeTitle = target.getAttribute('title');
      if (nativeTitle !== null) target.removeAttribute('title');
      const layer = target.closest('dialog[open]') || document.body;
      if (tip.parentNode !== layer) layer.append(tip);
    }
    const text = entry.text.replace(/^This element is (\S)/, (_, first) => first.toUpperCase());
    if (tip.textContent !== text) tip.textContent = text;
    if (tip.hidden) tip.hidden = false;
    const box = target.getBoundingClientRect();
    const [x, y] = !keyboard && pointer ? pointer : [box.left, box.bottom];
    const {width, height} = tip.getBoundingClientRect();
    const left = Math.max(8, Math.min(x + 14, innerWidth - width - 8));
    let top = y + 18;
    if (top + height > innerHeight - 8) top = Math.max(8, y - height - 14);
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }

  function schedule() {
    if (enabled() && !frame) frame = requestAnimationFrame(() => { frame = 0; render(); });
  }
  async function loadCatalog() {
    if (loading || entries) return;
    loading = true;
    try {
      const response = await fetch(new URL('./tooltips.json', import.meta.url));
      if (!response.ok) throw Error('Could not load Studio help (' + response.status + ')');
      entries = (await response.json()).entries;
      schedule();
    } catch (error) {
      console.error(error);
    } finally {
      loading = false;
    }
  }
  function update() {
    if (enabled()) schedule();
    else hide();
  }
  function setPinned(on) {
    pinned = on;
    toggle.setAttribute('aria-pressed', String(on));
  }
  function suspend() { awake = false; keyHeld = held = dismissed = false; hide(); }
  function optionKey(e) {
    return e.key === 'Alt' || e.code === 'AltLeft' || e.code === 'AltRight' ||
      (mac && (e.key === 'Option' || e.key === 'AltGraph'));
  }
  function plainOption(e) {
    return !e.ctrlKey && !e.metaKey && (mac || !e.getModifierState('AltGraph'));
  }
  function observePointer(e) {
    const wasEnabled = enabled();
    // Opening a dialog can retarget a stationary pointer after a Tab event.
    if (!pointer || e.clientX !== pointer[0] || e.clientY !== pointer[1] || e.type === 'pointerdown') {
      keyboard = false;
    }
    awake = true;
    pointer = [e.clientX, e.clientY];
    // Window focus and remote clients can consume the initial modifier keydown.
    // A boundary event caused by a DOM redraw can also omit keyboard modifiers.
    held = (e.altKey || keyHeld) && plainOption(e);
    if (!held) dismissed = false;
    if (!wasEnabled && enabled()) loadCatalog();
    update();
  }
  toggle.addEventListener('click', () => {
    setPinned(!pinned);
    dismissed = !pinned && held;
    if (pinned) loadCatalog();
    update();
  });
  document.addEventListener('pointerover', observePointer, true);
  document.addEventListener('pointermove', observePointer, true);
  document.addEventListener('pointerdown', observePointer, true);
  document.documentElement.addEventListener('pointerleave', () => { pointer = null; hide(); });
  document.addEventListener('focusin', schedule, true);
  document.addEventListener('keydown', e => {
    awake = true;
    if (e.key === 'Escape' && (pinned || (held && owner))) {
      e.preventDefault();
      e.stopPropagation();
      setPinned(false);
      dismissed = true;
      hide();
    } else if (optionKey(e) && plainOption(e)) {
      if (!e.repeat) { keyHeld = held = true; dismissed = false; loadCatalog(); update(); }
    } else if (e.key === 'Tab') {
      keyboard = true;
      held = e.altKey && plainOption(e);
      if (enabled()) loadCatalog();
      update();
    } else if (held) {
      dismissed = true;
      update();
    }
  }, true);
  document.addEventListener('keyup', e => {
    if (optionKey(e) || !e.altKey) { keyHeld = held = dismissed = false; update(); }
  }, true);
  document.addEventListener('scroll', schedule, true);
  window.addEventListener('resize', schedule);
  window.addEventListener('focus', () => { awake = true; update(); });
  window.addEventListener('blur', suspend);
  window.addEventListener('pagehide', suspend);
  document.addEventListener('visibilitychange', () => { if (document.hidden) suspend(); });
  // Live jobs and dialogs replace their controls while a pointer can stay still.
  new MutationObserver(schedule).observe(document.body, {
    childList: true, subtree: true, attributes: true, attributeFilter: ['data-help', 'open', 'hidden'],
  });
  loadCatalog();
}
