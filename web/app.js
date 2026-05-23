const state = {
  events: [],
  status: null,
  logs: null,
};

const els = {
  status: document.querySelector("#status"),
  stepBadge: document.querySelector("#stepBadge"),
  trainLoss: document.querySelector("#trainLoss"),
  evalLoss: document.querySelector("#evalLoss"),
  lmLoss: document.querySelector("#lmLoss"),
  pcLoss: document.querySelector("#pcLoss"),
  eventList: document.querySelector("#eventList"),
  lossChart: document.querySelector("#lossChart"),
  pcChart: document.querySelector("#pcChart"),
  configPath: document.querySelector("#configPath"),
  datasetSource: document.querySelector("#datasetSource"),
  trainBatchSize: document.querySelector("#trainBatchSize"),
  evalBatchSize: document.querySelector("#evalBatchSize"),
  gradientAccumulation: document.querySelector("#gradientAccumulation"),
  numTrainEpochs: document.querySelector("#numTrainEpochs"),
  modelArchitecture: document.querySelector("#modelArchitecture"),
  modelPreset: document.querySelector("#modelPreset"),
  modelLayers: document.querySelector("#modelLayers"),
  modelHidden: document.querySelector("#modelHidden"),
  modelHeads: document.querySelector("#modelHeads"),
  modelContext: document.querySelector("#modelContext"),
  applyModelScale: document.querySelector("#applyModelScale"),
  checkpointSelect: document.querySelector("#checkpointSelect"),
  collectDataset: document.querySelector("#collectDataset"),
  startTrain: document.querySelector("#startTrain"),
  stopTrain: document.querySelector("#stopTrain"),
  generateText: document.querySelector("#generateText"),
  promptText: document.querySelector("#promptText"),
  maxTokens: document.querySelector("#maxTokens"),
  useIrm: document.querySelector("#useIrm"),
  useContextFuzzer: document.querySelector("#useContextFuzzer"),
  irmPasses: document.querySelector("#irmPasses"),
  chunkTokens: document.querySelector("#chunkTokens"),
  generationOutput: document.querySelector("#generationOutput"),
  datasetState: document.querySelector("#datasetState"),
  trainState: document.querySelector("#trainState"),
  checkpointState: document.querySelector("#checkpointState"),
  actionMessage: document.querySelector("#actionMessage"),
  processLogs: document.querySelector("#processLogs"),
  datasetStage: document.querySelector("#datasetStage"),
  datasetPercent: document.querySelector("#datasetPercent"),
  datasetBar: document.querySelector("#datasetBar"),
  datasetDetails: document.querySelector("#datasetDetails"),
  stepProgressLabel: document.querySelector("#stepProgressLabel"),
  stepProgressPercent: document.querySelector("#stepProgressPercent"),
  stepProgressBar: document.querySelector("#stepProgressBar"),
  epochProgressLabel: document.querySelector("#epochProgressLabel"),
  epochProgressPercent: document.querySelector("#epochProgressPercent"),
  epochProgressBar: document.querySelector("#epochProgressBar"),
  passProgressLabel: document.querySelector("#passProgressLabel"),
  passProgressPercent: document.querySelector("#passProgressPercent"),
  passProgressBar: document.querySelector("#passProgressBar"),
  refreshInspector: document.querySelector("#refreshInspector"),
  openCompareModal: document.querySelector("#openCompareModal"),
  closeCompareModal: document.querySelector("#closeCompareModal"),
  compareModal: document.querySelector("#compareModal"),
  parameterCount: document.querySelector("#parameterCount"),
  int8Size: document.querySelector("#int8Size"),
  architectureSummary: document.querySelector("#architectureSummary"),
  layerStack: document.querySelector("#layerStack"),
};

async function refresh() {
  try {
    const cacheUrl = `/api/dataset/cache?config=${encodeURIComponent(els.configPath.value)}`;
    const inspectUrl = modelInspectUrl();
    const [metricsResponse, statusResponse, logsResponse, checkpointsResponse, cacheResponse, inspectResponse] = await Promise.all([
      fetch("/api/metrics", { cache: "no-store" }),
      fetch("/api/status", { cache: "no-store" }),
      fetch("/api/logs", { cache: "no-store" }),
      fetch("/api/checkpoints", { cache: "no-store" }),
      fetch(cacheUrl, { cache: "no-store" }),
      fetch(inspectUrl, { cache: "no-store" }),
    ]);
    state.events = await metricsResponse.json();
    state.status = await statusResponse.json();
    state.logs = await logsResponse.json();
    state.checkpoints = await checkpointsResponse.json();
    state.datasetCache = await cacheResponse.json();
    state.modelInspection = await inspectResponse.json();
    render();
  } catch (error) {
    els.status.textContent = "Dashboard server is reachable, but metrics could not be read";
  }
}

function render() {
  const latest = state.events[state.events.length - 1] || {};
  const latestTrain = latestValue("loss");
  const latestEval = latestValue("eval_loss");
  const latestLm = latestValue("lm_loss");
  const latestPc = latestValue("predictive_coding_loss");

  els.stepBadge.textContent = `step ${latest.step ?? 0}`;
  els.trainLoss.textContent = formatMetric(latestTrain);
  els.evalLoss.textContent = formatMetric(latestEval);
  els.lmLoss.textContent = formatMetric(latestLm);
  els.pcLoss.textContent = formatMetric(latestPc);

  const fileState = state.status?.exists ? `${state.status.bytes.toLocaleString()} bytes` : "file not created yet";
  els.status.textContent = `${state.events.length.toLocaleString()} metric events - ${fileState}`;
  renderProcesses();
  renderCheckpointOptions();
  renderModelInspection();

  drawChart(els.lossChart, [
    { label: "train", color: css("--train"), points: series("loss") },
    { label: "eval", color: css("--eval"), points: series("eval_loss") },
  ]);
  drawChart(els.pcChart, [
    { label: "pc", color: css("--pc"), points: series("predictive_coding_loss") },
    { label: "lm", color: css("--accent"), points: series("lm_loss") },
  ]);
  renderEvents();
}

function renderProcesses() {
  const processes = state.status?.processes || {};
  els.datasetState.textContent = processLabel("dataset", processes.dataset);
  els.trainState.textContent = processLabel("training", processes.train);
  els.checkpointState.textContent = checkpointLabel(state.status?.checkpoints);
  const logBits = [];
  if (state.logs?.train) logBits.push(`TRAIN\n${state.logs.train}`);
  if (state.logs?.dataset) logBits.push(`DATASET\n${state.logs.dataset}`);
  els.processLogs.textContent = logBits.join("\n\n") || "No process output yet.";
  renderDatasetProgress(state.status?.dataset_progress);
  renderTrainingProgress();
}

function processLabel(label, process) {
  if (!process) return `${label} idle`;
  if (process.running) return `${label} running pid ${process.pid}`;
  if (process.returncode === null || process.returncode === undefined) return `${label} idle`;
  return `${label} exited ${process.returncode}`;
}

function checkpointLabel(checkpoints) {
  if (!checkpoints) return "checkpoint unknown";
  if (checkpoints.final_exists) return "final checkpoint saved";
  if (checkpoints.latest_exists) return "latest checkpoint saved";
  if (checkpoints.newest_ready) return `saved ${shortPath(checkpoints.newest)}`;
  return "checkpoint not saved";
}

function renderCheckpointOptions() {
  const selected = els.checkpointSelect.value;
  const checkpoints = state.checkpoints || [];
  if (checkpoints.length === 0) {
    els.checkpointSelect.innerHTML = '<option value="">No checkpoints found</option>';
    return;
  }

  els.checkpointSelect.innerHTML = checkpoints
    .map((checkpoint) => `<option value="${escapeHtml(checkpoint.value)}">${escapeHtml(checkpoint.label)}</option>`)
    .join("");
  if (checkpoints.some((checkpoint) => checkpoint.value === selected)) {
    els.checkpointSelect.value = selected;
  }
}

function shortPath(path) {
  return String(path || "").split(/[\\/]/).slice(-1)[0] || "checkpoint";
}

function renderDatasetProgress(progress) {
  const current = progress || { status: "idle", stage: "not started", percent: 0 };
  const percent = Math.max(0, Math.min(100, Number(current.percent) || 0));
  els.datasetStage.textContent = `${current.status || "idle"} - ${current.stage || "not started"}`;
  els.datasetPercent.textContent = `${Math.round(percent)}%`;
  els.datasetBar.style.width = `${percent}%`;

  const details = [];
  if (Number.isFinite(current.train_rows)) details.push(`${current.train_rows.toLocaleString()} train rows`);
  if (Number.isFinite(current.validation_rows)) details.push(`${current.validation_rows.toLocaleString()} validation rows`);
  if (current.output_dir) details.push(current.output_dir);
  if (state.datasetCache?.exists) details.push(`cached: ${state.datasetCache.path}`);
  if (current.error) details.push(`error: ${current.error}`);
  els.datasetDetails.textContent = details.join(" - ") || "No dataset cache yet.";
}

function renderTrainingProgress() {
  const latest = state.events[state.events.length - 1] || {};
  const logText = state.logs?.train || "";
  const progress = parseLogProgress(logText);
  const currentStep = Number(latest.step) || progress.trainCurrent || 0;
  const totalSteps = progress.trainTotal || 0;
  const stepPercent = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0;
  setProgress(
    els.stepProgressLabel,
    els.stepProgressPercent,
    els.stepProgressBar,
    totalSteps > 0 ? `step ${currentStep.toLocaleString()} / ${totalSteps.toLocaleString()}` : `step ${currentStep.toLocaleString()}`,
    stepPercent,
  );

  const epoch = Number(latest.epoch);
  if (Number.isFinite(epoch)) {
    const epochIndex = Math.floor(epoch) + 1;
    const epochPercent = (epoch - Math.floor(epoch)) * 100;
    setProgress(
      els.epochProgressLabel,
      els.epochProgressPercent,
      els.epochProgressBar,
      `epoch ${epochIndex} (${epoch.toFixed(4)} total)`,
      epochPercent,
    );
  } else {
    setProgress(els.epochProgressLabel, els.epochProgressPercent, els.epochProgressBar, "epoch waiting", 0);
  }

  if (progress.active) {
    const label = progress.active.total === progress.trainTotal ? "training pass" : "eval pass";
    setProgress(
      els.passProgressLabel,
      els.passProgressPercent,
      els.passProgressBar,
      `${label} ${progress.active.current.toLocaleString()} / ${progress.active.total.toLocaleString()}`,
      (progress.active.current / progress.active.total) * 100,
    );
  } else {
    setProgress(els.passProgressLabel, els.passProgressPercent, els.passProgressBar, "active pass waiting", 0);
  }
}

function setProgress(labelEl, percentEl, barEl, label, percent) {
  const bounded = Math.max(0, Math.min(100, Number(percent) || 0));
  labelEl.textContent = label;
  percentEl.textContent = `${bounded.toFixed(1)}%`;
  barEl.style.width = `${bounded}%`;
}

function parseLogProgress(logText) {
  const matches = [...String(logText || "").matchAll(/(\d+)\/(\d+)\s*\[/g)].map((match) => ({
    current: Number(match[1]),
    total: Number(match[2]),
  }));
  const valid = matches.filter((item) => item.total > 0 && item.current <= item.total);
  if (valid.length === 0) return { active: null, trainCurrent: 0, trainTotal: 0 };
  const trainTotal = Math.max(...valid.map((item) => item.total));
  const trainMatches = valid.filter((item) => item.total === trainTotal);
  const latestTrain = trainMatches[trainMatches.length - 1] || null;
  return {
    active: valid[valid.length - 1],
    trainCurrent: latestTrain?.current || 0,
    trainTotal,
  };
}

function renderModelInspection() {
  const inspection = state.modelInspection;
  if (!inspection?.ok) {
    els.parameterCount.textContent = "--";
    els.int8Size.textContent = "--";
    els.architectureSummary.textContent = inspection?.error || "--";
    els.layerStack.innerHTML = "";
    return;
  }
  const arch = inspection.architecture;
  els.parameterCount.textContent = inspection.parameters.toLocaleString();
  els.int8Size.textContent = `${inspection.int8_size_gb.toFixed(2)} GB`;
  els.architectureSummary.textContent = `${String(arch.type || "gpt").toUpperCase()} / ${arch.layers}L / ${arch.hidden}H / ${arch.heads} groups / ctx ${arch.context}`;
  els.layerStack.innerHTML = inspection.layer_stack
    .map(
      (layer) =>
        `<tr><td>${escapeHtml(layer.name)}</td><td>${escapeHtml(layer.type)}</td><td>${escapeHtml(layer.shape)}</td><td>${Number(layer.parameters).toLocaleString()}</td></tr>`,
    )
    .join("");
}

function latestValue(key) {
  for (let index = state.events.length - 1; index >= 0; index -= 1) {
    const value = state.events[index][key];
    if (Number.isFinite(value)) return value;
  }
  return null;
}

function series(key) {
  return state.events
    .filter((event) => Number.isFinite(event[key]) && Number.isFinite(event.step))
    .map((event) => ({ x: event.step, y: event[key] }));
}

function drawChart(canvas, lines) {
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = { top: 24, right: 24, bottom: 44, left: 58 };
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fbfcf8";
  context.fillRect(0, 0, width, height);

  const allPoints = lines.flatMap((line) => line.points);
  if (allPoints.length === 0) {
    context.fillStyle = "#657069";
    context.font = "26px system-ui";
    context.textAlign = "center";
    context.fillText("Waiting for training metrics", width / 2, height / 2);
    return;
  }

  const xValues = allPoints.map((point) => point.x);
  const yValues = allPoints.map((point) => point.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  const yPad = Math.max((yMax - yMin) * 0.12, 0.01);

  const scaleX = (value) => {
    if (xMax === xMin) return padding.left;
    return padding.left + ((value - xMin) / (xMax - xMin)) * (width - padding.left - padding.right);
  };
  const scaleY = (value) => {
    const min = yMin - yPad;
    const max = yMax + yPad;
    return height - padding.bottom - ((value - min) / (max - min)) * (height - padding.top - padding.bottom);
  };

  drawGrid(context, width, height, padding, yMin - yPad, yMax + yPad);

  for (const line of lines) {
    if (line.points.length === 0) continue;
    context.strokeStyle = line.color;
    context.lineWidth = 4;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.beginPath();
    line.points.forEach((point, index) => {
      const x = scaleX(point.x);
      const y = scaleY(point.y);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  }

  context.fillStyle = "#657069";
  context.font = "20px system-ui";
  context.textAlign = "left";
  context.fillText(`step ${xMin}`, padding.left, height - 12);
  context.textAlign = "right";
  context.fillText(`step ${xMax}`, width - padding.right, height - 12);
}

function drawGrid(context, width, height, padding, yMin, yMax) {
  context.strokeStyle = "#d8ded6";
  context.lineWidth = 1;
  context.fillStyle = "#657069";
  context.font = "18px system-ui";
  context.textAlign = "right";
  context.textBaseline = "middle";

  for (let index = 0; index <= 4; index += 1) {
    const y = padding.top + (index / 4) * (height - padding.top - padding.bottom);
    const value = yMax - (index / 4) * (yMax - yMin);
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    context.fillText(value.toFixed(3), padding.left - 12, y);
  }
}

function renderEvents() {
  const rows = state.events
    .filter((event) => event.loss || event.eval_loss || event.predictive_coding_loss)
    .slice(-12)
    .reverse();

  els.eventList.innerHTML = rows
    .map((event) => {
      const bits = [];
      if (Number.isFinite(event.loss)) bits.push(`train ${formatMetric(event.loss)}`);
      if (Number.isFinite(event.eval_loss)) bits.push(`eval ${formatMetric(event.eval_loss)}`);
      if (Number.isFinite(event.predictive_coding_loss)) bits.push(`pc ${formatMetric(event.predictive_coding_loss)}`);
      return `<li><b>step ${event.step ?? 0}</b><span>${bits.join(" - ")}</span></li>`;
    })
    .join("");
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return response.json();
}

function setBusy(button, busy) {
  button.disabled = busy;
}

async function runAction(button, path, payload, successPrefix) {
  setBusy(button, true);
  els.actionMessage.textContent = "working";
  try {
    const result = await postJson(path, payload);
    els.actionMessage.textContent = result.ok ? `${successPrefix}: ${result.message || "ok"}` : `error: ${result.error}`;
    await refresh();
    return result;
  } catch (error) {
    els.actionMessage.textContent = `error: ${error.message}`;
    return { ok: false, error: error.message };
  } finally {
    setBusy(button, false);
  }
}

function formatMetric(value) {
  return Number.isFinite(value) ? value.toFixed(4) : "--";
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

els.collectDataset.addEventListener("click", () => {
  runAction(els.collectDataset, "/api/dataset/collect", { config: els.configPath.value }, "dataset");
});

els.datasetSource.addEventListener("change", () => {
  els.configPath.value = els.datasetSource.value;
  els.actionMessage.textContent = `source: ${els.datasetSource.options[els.datasetSource.selectedIndex].text}`;
  refresh();
});

els.refreshInspector.addEventListener("click", refresh);

els.openCompareModal.addEventListener("click", () => {
  els.compareModal.hidden = false;
});

els.closeCompareModal.addEventListener("click", closeCompareModal);

els.compareModal.addEventListener("click", (event) => {
  if (event.target === els.compareModal) closeCompareModal();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !els.compareModal.hidden) closeCompareModal();
});

els.modelPreset.addEventListener("change", () => {
  applyModelPreset(els.modelPreset.value);
  markModelScalePending();
});

for (const input of [els.modelArchitecture, els.modelLayers, els.modelHidden, els.modelHeads, els.modelContext]) {
  input.addEventListener("change", () => {
    if (input !== els.modelArchitecture) els.modelPreset.value = "custom";
    markModelScalePending();
  });
}

els.applyModelScale.addEventListener("click", async () => {
  setBusy(els.applyModelScale, true);
  els.actionMessage.textContent = "applying model scale";
  try {
    await refresh();
    const params = state.modelInspection?.parameters;
    els.actionMessage.textContent = Number.isFinite(params)
      ? `scale applied: ${formatCompactNumber(params)} params for next training run`
      : `scale error: ${state.modelInspection?.error || "could not inspect model"}`;
  } finally {
    setBusy(els.applyModelScale, false);
  }
});

els.startTrain.addEventListener("click", () => {
  runAction(
    els.startTrain,
    "/api/train/start",
    { config: els.configPath.value, training: trainingOverrides(), model: modelOverrides() },
    "training",
  );
});

els.stopTrain.addEventListener("click", () => {
  runAction(els.stopTrain, "/api/train/stop", {}, "training");
});

els.generateText.addEventListener("click", async () => {
  setBusy(els.generateText, true);
  els.generationOutput.textContent = "Generating...";
  try {
    const result = await postJson("/api/generate", {
      checkpoint: els.checkpointSelect.value,
      prompt: els.promptText.value,
      max_new_tokens: Number(els.maxTokens.value),
      irm: els.useIrm.checked,
      context_fuzzer: els.useContextFuzzer.checked,
      irm_passes: Number(els.irmPasses.value),
      chunk_tokens: Number(els.chunkTokens.value),
    });
    els.generationOutput.textContent = result.ok ? result.text : `Error: ${result.error}`;
  } catch (error) {
    els.generationOutput.textContent = `Error: ${error.message}`;
  } finally {
    setBusy(els.generateText, false);
  }
});

const modelPresets = {
  config: {},
  toy: { n_layer: 4, n_embd: 256, n_head: 4, n_positions: 128 },
  small: { n_layer: 12, n_embd: 768, n_head: 12, n_positions: 256 },
  medium: { n_layer: 24, n_embd: 1024, n_head: 16, n_positions: 512 },
  large: { n_layer: 24, n_embd: 1536, n_head: 16, n_positions: 512 },
  xl: { n_layer: 32, n_embd: 2048, n_head: 16, n_positions: 1024 },
};

function applyModelPreset(name) {
  const preset = modelPresets[name] || {};
  els.modelLayers.value = preset.n_layer || "";
  els.modelHidden.value = preset.n_embd || "";
  els.modelHeads.value = preset.n_head || "";
  els.modelContext.value = preset.n_positions || "";
}

function modelInspectUrl() {
  const params = new URLSearchParams({ config: els.configPath.value });
  for (const [key, value] of Object.entries(modelOverrides())) {
    params.set(key, value);
  }
  return `/api/model/inspect?${params.toString()}`;
}

function markModelScalePending() {
  els.actionMessage.textContent = "scale pending - apply to preview and use for next training run";
}

function closeCompareModal() {
  els.compareModal.hidden = true;
}

function trainingOverrides() {
  return cleanObject({
    per_device_train_batch_size: numberOrBlank(els.trainBatchSize.value),
    per_device_eval_batch_size: numberOrBlank(els.evalBatchSize.value),
    gradient_accumulation_steps: numberOrBlank(els.gradientAccumulation.value),
    num_train_epochs: numberOrBlank(els.numTrainEpochs.value),
  });
}

function modelOverrides() {
  return cleanObject({
    architecture: els.modelArchitecture.value,
    n_layer: numberOrBlank(els.modelLayers.value),
    n_embd: numberOrBlank(els.modelHidden.value),
    n_head: numberOrBlank(els.modelHeads.value),
    n_positions: numberOrBlank(els.modelContext.value),
  });
}

function numberOrBlank(value) {
  const trimmed = String(value || "").trim();
  return trimmed === "" ? "" : Number(trimmed);
}

function cleanObject(object) {
  return Object.fromEntries(Object.entries(object).filter(([, value]) => value !== ""));
}

function formatCompactNumber(value) {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

applyModelPreset(els.modelPreset.value);
refresh();
setInterval(refresh, 2000);
