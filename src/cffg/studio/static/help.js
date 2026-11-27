// Keep one tooltip in the same modal context as its trigger.
export function installHelp() {
  const tip = document.createElement('div');
  tip.id = 'studio-option-help';
  tip.setAttribute('role', 'tooltip');
  tip.hidden = true;

  let entries, held = false, pointer = null, keyboard = true, owner = null, frame = 0;
  let nativeTitle = null, loading = false;
  const hint = document.getElementById('help-gesture');
  hint.textContent = /Mac|iPhone|iPad/.test(navigator.platform) ? '⌥ hover for help' : 'Alt + hover for help';

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
    if (!held || !entries) return hide();
    const hit = keyboard ? document.activeElement : pointer && document.elementFromPoint(...pointer);
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
    if (tip.textContent !== entry.text) tip.textContent = entry.text;
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
    if (held && !frame) frame = requestAnimationFrame(() => { frame = 0; render(); });
  }
  async function loadCatalog() {
    if (loading || entries) return;
    loading = true;
    try {
      const response = await fetch('/static/tooltips.json');
      if (!response.ok) throw Error('Could not load Studio help (' + response.status + ')');
      entries = (await response.json()).entries;
      schedule();
    } catch (error) {
      console.error(error);
    } finally {
      loading = false;
    }
  }
  function stop() { held = false; hide(); }
  const capture = true;
  document.addEventListener('pointermove', e => {
    keyboard = false;
    pointer = [e.clientX, e.clientY];
    schedule();
  }, capture);
  document.addEventListener('pointerout', e => {
    if (!e.relatedTarget) { pointer = null; hide(); }
  }, capture);
  document.addEventListener('focusin', schedule, capture);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && held && owner) {
      e.preventDefault();
      e.stopPropagation();
      stop();
    } else if (e.key === 'Alt' && !e.ctrlKey && !e.metaKey && !e.getModifierState('AltGraph')) {
      if (!e.repeat) { held = true; loadCatalog(); render(); }
    } else if (e.key === 'Tab') {
      keyboard = true;
      schedule();
    } else if (held) stop();
  }, capture);
  document.addEventListener('keyup', e => { if (e.key === 'Alt' || !e.altKey) stop(); }, capture);
  document.addEventListener('scroll', schedule, capture);
  window.addEventListener('resize', schedule);
  window.addEventListener('blur', stop);
  window.addEventListener('pagehide', stop);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  // Live jobs and dialogs replace their controls while a pointer can stay still.
  new MutationObserver(schedule).observe(document.body, {
    childList: true, subtree: true, attributes: true, attributeFilter: ['data-help', 'open', 'hidden'],
  });
  loadCatalog();
}
