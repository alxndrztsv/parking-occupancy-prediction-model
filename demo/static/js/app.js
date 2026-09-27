const state = {
  config: null,
  map: null,
  overlay: null,
  date: null,
  slot: 72,
  horizon: null,
  model: null,
  view: "actual",
  rows: [],
  summary: null,
  playing: false,
  timer: null,
  controller: null,
};

const COLOR_STOPS = [
  { at: 0.0, rgb: [22, 163, 74] },
  { at: 0.55, rgb: [245, 158, 11] },
  { at: 1.0, rgb: [220, 38, 38] },
];

const BINS_PER_HOUR = 6;
const WEEKDAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"];

const popovers = [];

function registerPopover(closeFn) {
  popovers.push(closeFn);
}

function closePopovers() {
  popovers.forEach((close) => close());
}

document.addEventListener("click", closePopovers);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closePopovers();
});

function occupancyColor(rate) {
  const value = Math.min(1, Math.max(0, rate == null ? 0 : rate));

  let lower = COLOR_STOPS[0];
  let upper = COLOR_STOPS[COLOR_STOPS.length - 1];
  for (let i = 0; i < COLOR_STOPS.length - 1; i += 1) {
    if (value >= COLOR_STOPS[i].at && value <= COLOR_STOPS[i + 1].at) {
      lower = COLOR_STOPS[i];
      upper = COLOR_STOPS[i + 1];
      break;
    }
  }

  const span = upper.at - lower.at || 1;
  const t = (value - lower.at) / span;
  const rgb = lower.rgb.map((c, i) => Math.round(c + t * (upper.rgb[i] - c)));

  return [...rgb, 235];
}

function slotToTime(slot) {
  const hours = Math.floor(slot / BINS_PER_HOUR);
  const minutes = (slot % BINS_PER_HOUR) * 10;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;
}

function pct(value) {
  return value == null ? "—" : `${Math.round(value * 100)}%`;
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function isoOf(year, monthIndex, day) {
  return `${year}-${pad2(monthIndex + 1)}-${pad2(day)}`;
}

function formatDate(iso) {
  const [year, month, day] = iso.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day)).toLocaleDateString("en-GB", {
    weekday: "short",
    day: "2-digit",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

function shownValue(row) {
  return state.view === "forecast"
    ? row.predicted_occupancy
    : row.occupancy_now;
}

function setStatus(message) {
  const bar = document.getElementById("status-bar");

  if (message) {
    bar.textContent = message;
    bar.hidden = false;
  } else {
    bar.hidden = true;
  }
}

async function init() {
  let response;
  try {
    response = await fetch("/api/config");
  } catch (err) {
    setStatus(`Cannot reach the API: ${err.message}`);
    return;
  }

  if (!response.ok) {
    setStatus(`/api/config failed with ${response.status}`);
    return;
  }

  state.config = await response.json();

  const cfg = state.config;
  state.date = cfg.default_date;
  state.slot = Math.round((cfg.slot_min + cfg.slot_max) / 2);
  state.horizon = cfg.default_horizon;
  state.model = cfg.default_model;

  buildControls();

  const token = cfg.mapbox_token || localStorage.getItem("mapbox_token") || "";
  if (token) {
    buildMap(token);
  } else {
    showTokenModal();
  }
}

function buildSelect(container, items, initial, onChange) {
  let value = initial;

  const button = document.createElement("button");
  button.type = "button";
  button.className = "input select-btn";
  button.setAttribute("aria-haspopup", "listbox");
  button.setAttribute("aria-expanded", "false");

  const label = document.createElement("span");
  label.className = "select-value";
  const caret = document.createElement("span");
  caret.className = "select-caret";
  button.append(label, caret);

  const menu = document.createElement("div");
  menu.className = "select-menu";
  menu.setAttribute("role", "listbox");
  menu.hidden = true;

  container.append(button, menu);

  const close = () => {
    menu.hidden = true;
    container.classList.remove("open");
    button.setAttribute("aria-expanded", "false");
  };
  const open = () => {
    menu.hidden = false;
    container.classList.add("open");
    button.setAttribute("aria-expanded", "true");
  };

  function renderOptions() {
    const current = items.find((item) => item.value === value);
    label.textContent = current ? current.label : value;
    menu.replaceChildren();

    items.forEach((item) => {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "select-option";
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", String(item.value === value));
      option.textContent = item.label;
      option.onclick = (event) => {
        event.stopPropagation();
        close();
        if (item.value !== value) {
          value = item.value;
          renderOptions();
          onChange(value);
        }
      };
      menu.appendChild(option);
    });
  }

  button.onclick = (event) => {
    event.stopPropagation();
    const wasOpen = !menu.hidden;
    closePopovers();
    if (!wasOpen) open();
  };
  menu.onclick = (event) => event.stopPropagation();
  registerPopover(close);

  renderOptions();
}

// Calendar
const calendarState = { min: null, max: null, year: 0, month: 0 };

function initCalendar(cfg) {
  calendarState.min = cfg.date_min;
  calendarState.max = cfg.date_max;

  const button = document.getElementById("date-btn");
  const panel = document.getElementById("calendar");

  const close = () => {
    panel.hidden = true;
  };
  const open = () => {
    const [year, month] = state.date.split("-").map(Number);
    calendarState.year = year;
    calendarState.month = month - 1;
    renderCalendar();
    panel.hidden = false;
  };

  button.onclick = (event) => {
    event.stopPropagation();
    const wasOpen = !panel.hidden;
    closePopovers();
    if (!wasOpen) open();
  };
  panel.onclick = (event) => event.stopPropagation();
  registerPopover(close);

  document.getElementById("cal-prev").onclick = (event) => {
    event.stopPropagation();
    stepMonth(-1);
  };
  document.getElementById("cal-next").onclick = (event) => {
    event.stopPropagation();
    stepMonth(1);
  };
}

function stepMonth(delta) {
  calendarState.month += delta;
  if (calendarState.month < 0) {
    calendarState.month = 11;
    calendarState.year -= 1;
  }
  if (calendarState.month > 11) {
    calendarState.month = 0;
    calendarState.year += 1;
  }
  renderCalendar();
}

function renderCalendar() {
  const { year, month } = calendarState;
  const currentMonth = `${year}-${pad2(month + 1)}`;

  document.getElementById("cal-title").textContent = new Date(
    Date.UTC(year, month, 1),
  ).toLocaleDateString("en-GB", {
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });
  document.getElementById("cal-prev").disabled =
    currentMonth <= calendarState.min.slice(0, 7);
  document.getElementById("cal-next").disabled =
    currentMonth >= calendarState.max.slice(0, 7);

  const grid = document.getElementById("cal-grid");
  grid.replaceChildren();

  WEEKDAYS.forEach((dow) => {
    const cell = document.createElement("span");
    cell.className = "cal-dow";
    cell.textContent = dow;
    grid.appendChild(cell);
  });

  const lead = (new Date(Date.UTC(year, month, 1)).getUTCDay() + 6) % 7;
  const daysInMonth = new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
  const today = new Date().toISOString().slice(0, 10);

  for (let i = 0; i < lead; i += 1) {
    grid.appendChild(document.createElement("span"));
  }

  for (let day = 1; day <= daysInMonth; day += 1) {
    const iso = isoOf(year, month, day);
    const dow = new Date(Date.UTC(year, month, day)).getUTCDay();

    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "cal-day";
    cell.textContent = day;
    if (iso === today) cell.classList.add("today");
    if (iso === state.date) cell.classList.add("selected");
    // Outside the data range, or a weekend: the models are weekday-only.
    cell.disabled =
      iso < calendarState.min ||
      iso > calendarState.max ||
      dow === 0 ||
      dow === 6;
    cell.onclick = () => {
      state.date = iso;
      document.getElementById("date-value").textContent = formatDate(iso);
      closePopovers();
      stopPlayback();
      fetchState();
    };
    grid.appendChild(cell);
  }
}

function buildControls() {
  const cfg = state.config;

  document.getElementById("date-value").textContent = formatDate(state.date);
  initCalendar(cfg);

  buildSelect(
    document.getElementById("model-select"),
    (cfg.models || []).map((name) => ({ value: name, label: name })),
    state.model,
    (value) => {
      state.model = value;
      fetchState();
    },
  );

  buildSelect(
    document.getElementById("horizon-select"),
    (cfg.horizons || []).map((horizon) => ({
      value: horizon,
      label: horizon.toUpperCase(),
    })),
    state.horizon,
    (value) => {
      state.horizon = value;
      fetchState();
    },
  );

  document.querySelectorAll("#view-toggle button").forEach((button) => {
    button.onclick = () => {
      document
        .querySelectorAll("#view-toggle button")
        .forEach((other) => other.classList.remove("active"));
      button.classList.add("active");
      state.view = button.dataset.view;
      render();
    };
  });

  const slider = document.getElementById("slot-slider");
  slider.min = cfg.slot_min;
  slider.max = cfg.slot_max;
  slider.step = 1;
  slider.value = state.slot;
  updateBadge();

  let debounce = null;
  slider.oninput = (event) => {
    state.slot = parseInt(event.target.value, 10);
    updateBadge();
    clearTimeout(debounce);
    debounce = setTimeout(fetchState, 80);
  };

  document.getElementById("play-btn").onclick = () => {
    if (state.playing) {
      stopPlayback();
    } else {
      startPlayback();
    }
  };
}

function updateBadge() {
  document.getElementById("time-badge").textContent = slotToTime(state.slot);
}

function showTokenModal() {
  const modal = document.getElementById("token-modal");
  modal.hidden = false;

  document.getElementById("save-token-btn").onclick = () => {
    const value = document.getElementById("token-input").value.trim();
    if (!value) {
      return;
    }
    localStorage.setItem("mapbox_token", value);
    modal.hidden = true;
    buildMap(value);
  };
}

function buildMap(token) {
  mapboxgl.accessToken = token;

  const cfg = state.config;
  state.map = new mapboxgl.Map({
    container: "map-container",
    style: cfg.mapbox_style,
    projection: "mercator",
    center: cfg.default_center,
    zoom: cfg.default_zoom,
    pitch: cfg.default_pitch,
    bearing: cfg.default_bearing,
    antialias: true,
  });

  state.overlay = new deck.MapboxOverlay({ layers: [], getTooltip });
  state.map.addControl(state.overlay);

  state.map.on("load", () => {
    fetchState();
  });
}

async function fetchState() {
  if (!state.model || !state.horizon) {
    setStatus(
      "No saved model bundles were found in models/. Run the training stage of the pipeline first.",
    );
    return;
  }

  const params = new URLSearchParams({
    date: state.date,
    slot: state.slot,
    horizon: state.horizon,
    model: state.model,
  });

  if (state.controller) {
    state.controller.abort();
  }
  state.controller = new AbortController();

  try {
    const response = await fetch(`/api/state?${params}`, {
      signal: state.controller.signal,
    });
    const body = await response.json().catch(() => null);

    if (!response.ok) {
      state.rows = [];
      state.summary = null;
      setStatus(
        body && body.detail
          ? body.detail
          : `Request failed (${response.status})`,
      );
      render();
      return;
    }

    setStatus(null);
    state.rows = body.terminals;
    state.summary = body.summary;
    render();
  } catch (err) {
    if (err.name !== "AbortError") {
      setStatus(`Cannot load state: ${err.message}`);
    }
  }
}

function getTooltip({ object }) {
  if (!object) {
    return null;
  }

  const error =
    object.absolute_error == null ? "—" : object.absolute_error.toFixed(3);

  return {
    className: "deck-tooltip",
    html: `
      <div class="tooltip-title">Terminal code: ${object.terminal_code}</div>
      <div class="tooltip-row"><span class="tooltip-key">Capacity</span><span class="tooltip-val">${object.total_spaces}</span></div>
      <div class="tooltip-row"><span class="tooltip-key">Occupancy now</span><span class="tooltip-val">${pct(object.occupancy_now)}</span></div>
      <div class="tooltip-row"><span class="tooltip-key">Forecast ${state.horizon}</span><span class="tooltip-val">${pct(object.predicted_occupancy)}</span></div>
      <div class="tooltip-row"><span class="tooltip-key">Actual ${state.horizon} later</span><span class="tooltip-val">${pct(object.actual_at_horizon)}</span></div>
      <div class="tooltip-row"><span class="tooltip-key">Absolute error</span><span class="tooltip-val">${error}</span></div>
      <div class="tooltip-row"><span class="tooltip-key">Free spaces</span><span class="tooltip-val">${object.free_spaces == null ? "—" : object.free_spaces}</span></div>
    `,
  };
}

function render() {
  if (!state.overlay) {
    return;
  }

  const layer = new deck.ColumnLayer({
    id: "terminals",
    data: state.rows,
    diskResolution: 50,
    extruded: true,
    pickable: true,
    elevationScale: 280,
    radius: 7,
    radiusUnits: "pixels",
    getPosition: (d) => [d.longitude, d.latitude],
    getElevation: (d) => shownValue(d),
    getFillColor: (d) => occupancyColor(shownValue(d)),
    updateTriggers: {
      getElevation: state.view,
      getFillColor: state.view,
    },
    transitions: {
      getElevation: { duration: 250 },
      getFillColor: { duration: 250 },
    },
  });

  state.overlay.setProps({ layers: [layer] });
  updateMetrics();
}

function updateMetrics() {
  const summary = state.summary;
  if (!summary) {
    return;
  }

  // Capacity-weighted occupancy: occupied spaces over total spaces.
  const capacity = state.rows.reduce(
    (sum, row) => sum + (row.total_spaces || 0),
    0,
  );
  const occupied = state.rows.reduce(
    (sum, row) => sum + (row.occupancy_now || 0) * (row.total_spaces || 0),
    0,
  );

  document.getElementById("stat-now").textContent = pct(
    summary.avg_occupancy_now,
  );
  document.getElementById("stat-forecast").textContent = pct(
    summary.avg_predicted,
  );
  document.getElementById("stat-mae").textContent =
    summary.mae == null ? "—" : summary.mae.toFixed(3);
  document.getElementById("stat-total").textContent =
    capacity > 0 ? pct(occupied / capacity) : "—";
  document.getElementById("stat-congested").textContent =
    `${summary.congested || 0} / ${summary.terminals || 0}`;
}

function startPlayback() {
  state.playing = true;
  document.getElementById("play-btn").textContent = "Pause";

  state.timer = setInterval(() => {
    const { slot_min: min, slot_max: max } = state.config;
    state.slot = state.slot >= max ? min : state.slot + 1;
    document.getElementById("slot-slider").value = state.slot;
    updateBadge();
    fetchState();
  }, 700);
}

function stopPlayback() {
  state.playing = false;
  clearInterval(state.timer);
  state.timer = null;
  document.getElementById("play-btn").textContent = "Play";
}

window.addEventListener("DOMContentLoaded", init);
