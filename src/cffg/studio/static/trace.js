const svgNamespace = "http://www.w3.org/2000/svg";
const numbers = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5 });
const statusGroups = {
  running: ["running", "active", "stepping"],
  waiting: ["pending", "queued", "starting", "pausing", "resuming", "cancelling"],
  paused: ["paused"],
  complete: ["complete", "completed", "finished", "done", "success", "succeeded", "converged"],
  stopped: ["canceled", "cancelled", "interrupted", "exhausted", "budget_exhausted", "max_iterations"],
  error: ["failed", "error", "diverged", "infeasible"],
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function svg(tag, attributes) {
  const node = document.createElementNS(svgNamespace, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

function number(value) {
  return Number.isFinite(value) ? numbers.format(value) : "—";
}

function duration(ms) {
  if (!Number.isFinite(ms)) return "—";
  return ms >= 1000 ? `${number(ms / 1000)} s` : `${number(ms)} ms`;
}

function memory(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const unit = bytes >= 1024 ** 3 ? "GiB" : "MiB";
  return `${number(bytes / (unit === "GiB" ? 1024 ** 3 : 1024 ** 2))} ${unit}`;
}

function label(value) {
  return String(value ?? "").replaceAll("_", " ");
}

function energySeries(sweeps, result) {
  if (sweeps.length) {
    const samples = sweeps.map((event, i) => ({
      x: Number.isFinite(event.iteration) ? event.iteration : i + 1,
      y: event.free_energy,
    }));
    return { axis: "Sweep", samples };
  }
  // The engine records energies only for sweeps with a defined ledger.
  const energies = Array.isArray(result.free_energies) ? result.free_energies : [];
  const axis = energies.length === result.iterations ? "Sweep" : "Energy sample";
  return { axis, samples: energies.map((y, i) => ({ x: i + 1, y })) };
}

function drawChart(series, terminal) {
  const figure = el("figure", "trace-chart");
  const heading = el("figcaption", "", "Free energy");
  figure.append(heading);
  const finite = series.samples.filter(sample => Number.isFinite(sample.y));
  if (!finite.length) {
    const message = terminal ? "No finite free-energy samples were recorded." : "Waiting for a finite free-energy sample…";
    figure.append(el("p", "empty", message));
    return figure;
  }
  const latest = finite[finite.length - 1];
  const latestLabel = el("span", "trace-latest", number(latest.y));
  latestLabel.title = `Last finite sample at ${series.axis.toLowerCase()} ${latest.x}`;
  heading.append(latestLabel);
  const firstX = series.samples[0].x;
  const lastX = series.samples[series.samples.length - 1].x;
  let low = finite[0].y;
  let high = low;
  for (const { y } of finite) {
    low = Math.min(low, y);
    high = Math.max(high, y);
  }
  const chart = svg("svg", { viewBox: "0 0 320 66", role: "img", preserveAspectRatio: "none" });
  const desc = svg("title", {});
  desc.textContent = `Free energy from ${series.axis.toLowerCase()} ${firstX} to ${lastX}, ranging from ${number(low)} to ${number(high)}. Last finite value ${number(latest.y)}.`;
  chart.append(desc);
  for (const y of [5, 33, 61]) chart.append(svg("line", { x1: 3, x2: 317, y1: y, y2: y, class: "trace-grid" }));
  const scale = Math.max(Math.abs(low), Math.abs(high)) || 1;
  const normalizedLow = low / scale;
  const span = high / scale - normalizedLow;
  const path = [];
  let prior = null;
  let isolated = null;
  for (const sample of series.samples) {
    const continuous = series.axis === "Sweep" && prior !== null && sample.x === prior.x + 1 && Number.isFinite(sample.y);
    if (!continuous && isolated) chart.append(svg("circle", isolated));
    if (!Number.isFinite(sample.y)) {
      prior = null;
      isolated = null;
      continue;
    }
    const x = lastX === firstX ? 160 : 3 + (sample.x - firstX) / (lastX - firstX) * 314;
    const y = span === 0 ? 33 : 61 - ((sample.y / scale - normalizedLow) / span) * 56;
    path.push(`${continuous ? "L" : "M"}${x.toFixed(2)} ${y.toFixed(2)}`);
    isolated = continuous ? null : { cx: x, cy: y, r: 2.5, class: "trace-point" };
    prior = sample;
  }
  if (isolated) chart.append(svg("circle", isolated));
  chart.append(svg("path", { d: path.join(" "), class: "trace-line" }));
  const axis = el("div", "trace-axis");
  axis.append(el("span", "", `${series.axis} ${firstX}`));
  if (lastX !== firstX) axis.append(el("span", "", `${series.axis} ${lastX}`));
  figure.append(chart, axis);
  return figure;
}

function metrics(events, result) {
  const values = {};
  const measurements = events.filter(event => event.type === "measurement");
  const liveWarm = measurements.filter(item => item.message === "Warm execution").map(item => item.elapsed_ms);
  for (const item of [...measurements, result]) {
    for (const key of ["cold_ms", "warm_ms", "peak_rss_bytes"]) {
      if (item[key] != null) values[key] = item[key];
    }
  }
  const warm = values.warm_ms ?? liveWarm;
  const timings = Array.isArray(warm) ? warm : [warm];
  const samples = timings.filter(Number.isFinite).sort((a, b) => a - b);
  const middle = Math.floor(samples.length / 2);
  const median = samples.length % 2 ? samples[middle] : samples[middle - 1] / 2 + samples[middle] / 2;
  const list = el("dl", "trace-metrics");
  for (const [title, value] of [
    ["Cold run", duration(values.cold_ms)],
    ["Warm median", duration(median)],
    ["Peak RSS", memory(values.peak_rss_bytes)],
  ]) {
    const group = el("div");
    group.append(el("dt", "", title), el("dd", "", value));
    list.append(group);
  }
  if (samples.length) list.children[1].title = `${samples.length} warm measurements: ${samples.map(duration).join(", ")}`;
  return list;
}

function profileDetails(result) {
  const details = el("details", "profile-breakdown");
  details.append(el("summary", "", "Startup, compilation and executable costs"));
  const list = el("dl", "profile-values");
  const row = (title, value) => list.append(el("dt", "", title), el("dd", "", value));
  row("Worker startup", duration(result.process_startup_ms));
  row("Load and initialize graph", duration(result.setup_ms));
  row("Persistent state", memory(result.storage?.state_bytes));
  for (const [stage, ms] of Object.entries(result.compilation_ms ?? {})) {
    row(stage.replaceAll("_", " "), duration(ms));
  }
  for (const [name, unit] of Object.entries(result.executables ?? {})) {
    const device = unit.memory_bytes?.device;
    row(name.replaceAll("_", " "), unit.cost
      ? number(unit.cost.flops) + " FLOPs · " + memory(unit.cost["bytes accessed"]) + " traffic"
      : unit.cost_diagnostic?.message ?? "Cost unavailable");
    row("Lower / compile", duration(unit.lowering_seconds * 1000) + " / " + duration(unit.compilation_seconds * 1000));
    row("Planned temporary / peak buffers", device
      ? memory(device.temp) + " / " + memory(device.planned_peak)
      : unit.memory_diagnostic?.message ?? "Buffer information unavailable");
  }
  details.append(list);
  return details;
}

function eventRow(event) {
  const row = el("li");
  let title = label(event.type);
  let detail = "";
  if (event.type === "sweep") {
    title = `Sweep ${number(event.iteration)}`;
    detail = `FE ${number(event.free_energy)} · Δ ${number(event.delta)}`;
  } else if (event.type === "phase") {
    title = "Phase";
    detail = label(event.phase ?? event.name ?? event.message);
  } else if (event.type === "measurement") {
    title = label(event.phase ?? event.name ?? "Measurement");
    detail = label(event.message ?? event.metric);
  } else if (event.type === "finish") {
    title = "Finished";
    detail = label(event.status);
    if (Number.isFinite(event.iterations)) detail += ` · ${number(event.iterations)} sweeps`;
    if (Number.isFinite(event.residual)) detail += ` · residual ${number(event.residual)}`;
  } else if (event.type === "error") {
    row.classList.add("trace-error");
    detail = label(event.message ?? event.error ?? event.detail);
  }
  row.append(el("strong", "", title), el("span", "", detail));
  if (Number.isFinite(event.elapsed_ms)) row.append(el("small", "", duration(event.elapsed_ms)));
  return row;
}

function controlActions(status) {
  if (status === "paused") return ["resume", "step", "cancel"];
  if (["running", "active", "stepping"].includes(status)) return ["pause", "cancel"];
  if (["pending", "queued", "starting", "pausing", "resuming"].includes(status)) return ["cancel"];
  return [];
}

/** Render the latest job snapshot. Controls dispatch the corresponding service action. */
export function renderTrace(container, job, controlCallback) {
  const focus = container.contains(document.activeElement) ? document.activeElement.dataset.traceAction : null;
  if (!job) {
    container.replaceChildren(el("p", "empty", "Start a run to see its trace."));
    return;
  }
  const events = Array.isArray(job.events) ? job.events : [];
  const finish = events.findLast(event => event.type === "finish");
  const error = events.findLast(event => event.type === "error");
  const result = { ...finish, ...job.result };
  const execution = String(job.status ?? "pending").toLowerCase();
  const ended = Boolean(finish || statusGroups.complete.includes(execution));
  const rawStatus = error ? "failed" : (ended && result.status ? String(result.status).toLowerCase() : execution);
  const group = Object.keys(statusGroups).find(key => statusGroups[key].includes(rawStatus)) ?? "unknown";
  const terminal = Boolean(finish || error || ["complete", "stopped", "error"].includes(group));
  const view = el("section", "run-trace");
  view.setAttribute("aria-label", "Run timeline");
  const header = el("header", "trace-header");
  const title = el("h3", "", job.kind === "profile" ? "Resource profile" : "Run trace");
  if (job.id) title.title = String(job.id);
  const badge = el("span", "trace-status", label(rawStatus));
  badge.dataset.state = group;
  badge.tabIndex = -1;
  header.append(title, badge);
  if (job.kind === "trace" && typeof controlCallback === "function" && !terminal) {
    const controls = el("div", "trace-controls");
    for (const action of controlActions(rawStatus)) {
      const button = el("button", "", action[0].toUpperCase() + action.slice(1));
      button.type = "button";
      button.dataset.traceAction = action;
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          await controlCallback(action);
        } catch (cause) {
          const message = el("p", "trace-error", cause instanceof Error ? cause.message : String(cause));
          message.setAttribute("role", "alert");
          view.append(message);
        } finally {
          button.disabled = false;
        }
      });
      controls.append(button);
    }
    header.append(controls);
  }
  const body = el("div", "trace-body");
  const figure = drawChart(energySeries(events.filter(event => event.type === "sweep"), result), terminal);
  const recent = el("div", "trace-events");
  recent.append(el("h4", "", "Recent events"));
  const rows = events.filter(event => ["phase", "sweep", "measurement", "finish", "error"].includes(event.type));
  if (!finish && job.result && terminal) rows.push({ ...result, type: "finish" });
  if (rows.length) {
    const list = el("ol");
    for (const event of rows.slice(-8).reverse()) list.append(eventRow(event));
    recent.append(list);
  } else {
    recent.append(el("p", "empty", terminal ? "No events were recorded." : "Waiting for the first event…"));
  }
  body.append(figure, recent);
  view.append(header, body, metrics(events, result));
  if (job.kind === "profile" && job.result) view.append(profileDetails(result));
  if (job.detail && ["error", "stopped"].includes(group) && !error) view.append(el("p", "trace-error", job.detail));
  container.replaceChildren(view);
  if (focus) (view.querySelector(`[data-trace-action="${focus}"]`) ?? badge).focus({ preventScroll: true });
}
