const byId = (id) => document.getElementById(id);
const MAX_BATCH_ROWS = 1000;
const MAX_FILE_BYTES = 2 * 1024 * 1024;
const BATCH_CONCURRENCY = 32;
const BATCH_PREVIEW_ROWS = 50;
let batchState = null;
let batchDownloadUrl = null;

const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

async function checkService() {
  const status = byId("service-status");
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("not ready");
    status.className = "status ready";
    status.innerHTML = "<span></span>GPU service ready";
  } catch (_) {
    status.className = "status error";
    status.innerHTML = "<span></span>Service unavailable";
  }
}

function setMode(panelId) {
  document.querySelectorAll(".mode-panel").forEach((panel) => {
    panel.classList.toggle("hidden", panel.id !== panelId);
  });
  document.querySelectorAll(".mode-tab").forEach((tab) => {
    const selected = tab.dataset.panel === panelId;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
  });
}

document.querySelectorAll(".mode-tab").forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.panel));
});

function contributionRows(rows, direction) {
  if (!rows || rows.length === 0) return "<p>No active contribution in this direction.</p>";
  return rows.map((row) => {
    const smiles = row.environment_smiles || [];
    const svgs = row.environment_svg_data_urls || [];
    const environments = smiles.length ? smiles.map((environment, index) => `
      <figure class="bit-environment">
        ${svgs[index] ? `<img src="${svgs[index]}" alt="Bit ${row.bit} substructure" />` : ""}
        <figcaption><code title="${escapeHtml(environment)}">${escapeHtml(environment)}</code></figcaption>
      </figure>`).join("") : '<p class="muted">Unmapped fingerprint collision</p>';
    const collision = smiles.length > 1 ? `<span class="collision">${smiles.length} environments</span>` : "";
    return `<article class="bit-card ${direction}-bit">
      <header><strong>Bit ${row.bit}</strong>${collision}<span class="value">${row.shap_value.toFixed(5)}</span></header>
      <div class="bit-environments">${environments}</div>
    </article>`;
  }).join("");
}

function showPrediction(result) {
  if (result.status !== "ok") throw new Error("RDKit could not parse this SMILES.");
  byId("single-results").classList.remove("hidden");
  byId("canonical-smiles").textContent = result.canonical_smiles;
  const positive = result.consensus_all_antagonistic === 1;
  const card = document.querySelector(".consensus-card");
  card.classList.toggle("positive", positive);
  byId("consensus-label").textContent = positive ? "Antagonistic consensus" : "No all-three consensus";
  byId("consensus-detail").textContent = `${result.consensus_vote_count}/3 antagonistic votes · mean ${result.consensus_mean_score.toFixed(4)} · minimum ${result.consensus_min_score.toFixed(4)}`;
  const models = [
    ["rf", "Random Forest", "ECFP4 (2048-bit) · explainable branch", result.rf_antagonistic_score, result.rf_pred_label],
    ["tabpfn", "TabPFN", "ECFP4 (2048-bit) · GPU inference", result.tabpfn_antagonistic_score, result.tabpfn_pred_label],
    ["lightgbm", "LightGBM", "ECFP4 (1024-bit) + RDKit2D (200-d)", result.lightgbm_antagonistic_score, result.lightgbm_pred_label],
  ];
  byId("score-grid").innerHTML = models.map(([id, name, type, score, label]) =>
    `<article class="score ${id}" style="--score:${Math.max(0, Math.min(100, score * 100))}%">
      <div class="score-header"><p>${name}</p><span class="score-badge">label ${label}</span></div>
      <strong>${score.toFixed(4)}</strong>
      <small>${type}</small>
      <div class="score-meter" aria-hidden="true"><span></span></div>
    </article>`
  ).join("");
  const explanation = result.explanation;
  byId("structure-image").src = explanation.structure_svg_data_url || explanation.structure_png_data_url;
  byId("additivity").textContent = `RF additivity residual ${explanation.additivity_residual.toExponential(2)}`;
  const positiveRows = explanation.positive_contributions || [];
  const negativeRows = explanation.negative_contributions || [];
  const positiveSum = positiveRows.reduce((sum, row) => sum + row.shap_value, 0);
  const negativeSum = negativeRows.reduce((sum, row) => sum + row.shap_value, 0);
  const displayedSum = positiveSum + negativeSum;
  byId("shap-metrics").innerHTML = [
    ["RF base", explanation.base_value],
    ["Displayed Δ", displayedSum],
    ["RF output", explanation.rf_score],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value.toFixed(4)}</strong></div>`).join("");
  byId("positive-summary").textContent = `${positiveRows.length} bits · Σ ${positiveSum.toFixed(4)}`;
  byId("negative-summary").textContent = `${negativeRows.length} bits · Σ ${negativeSum.toFixed(4)}`;
  byId("positive-contributions").innerHTML = contributionRows(positiveRows, "positive");
  byId("negative-contributions").innerHTML = contributionRows(negativeRows, "negative");
}

function parseCsv(text) {
  const source = text.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (quoted) {
      if (character === '"' && source[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === '"' && field.length === 0) {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n" || character === "\r") {
      if (character === "\r" && source[index + 1] === "\n") index += 1;
      row.push(field);
      if (row.some((value) => value.trim() !== "")) rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  if (quoted) throw new Error("CSV contains an unterminated quoted field.");
  row.push(field);
  if (row.some((value) => value.trim() !== "")) rows.push(row);
  if (rows.length < 2) throw new Error("CSV must contain a header and at least one data row.");

  const headers = rows.shift().map((value) => value.trim());
  if (headers.some((value) => !value)) throw new Error("CSV headers cannot be empty.");
  const normalizedHeaders = headers.map((value) => value.toLowerCase());
  if (new Set(normalizedHeaders).size !== normalizedHeaders.length) {
    throw new Error("CSV headers must be unique (case-insensitive).");
  }
  const smilesColumn = normalizedHeaders.indexOf("smiles");
  if (smilesColumn < 0) throw new Error("CSV must contain a column named 'smiles'.");
  if (rows.length > MAX_BATCH_ROWS) {
    throw new Error(`CSV contains ${rows.length} rows; the limit is ${MAX_BATCH_ROWS}.`);
  }
  rows.forEach((values, index) => {
    if (values.length > headers.length) {
      throw new Error(`CSV row ${index + 2} has more fields than the header.`);
    }
    while (values.length < headers.length) values.push("");
  });
  return { headers, rows, smilesColumn };
}

function csvCell(value) {
  const text = value == null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function predictionValues(result) {
  if (!result || result.status !== "ok") {
    return [
      result?.status || "request_error",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
    ];
  }
  return [
    result.status,
    result.canonical_smiles,
    result.rf_antagonistic_score.toFixed(6),
    result.rf_pred_label,
    result.tabpfn_antagonistic_score.toFixed(6),
    result.tabpfn_pred_label,
    result.lightgbm_antagonistic_score.toFixed(6),
    result.lightgbm_pred_label,
    result.consensus_vote_count,
    result.consensus_all_antagonistic,
    result.consensus_mean_score.toFixed(6),
  ];
}

async function predictBatchRow(smiles) {
  const value = smiles.trim();
  if (!value) return { status: "missing_smiles" };
  if (value.length > 4096) return { status: "smiles_too_long" };
  try {
    const response = await fetch("/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles: value, explain: false }),
    });
    const result = await response.json();
    if (!response.ok) {
      return { status: "request_error", detail: result.detail || "Prediction failed" };
    }
    return result;
  } catch (error) {
    return { status: "request_error", detail: error.message };
  }
}

async function concurrentMap(values, concurrency, task, onProgress) {
  const results = new Array(values.length);
  let cursor = 0;
  let completed = 0;
  async function worker() {
    while (cursor < values.length) {
      const index = cursor;
      cursor += 1;
      results[index] = await task(values[index], index);
      completed += 1;
      onProgress(completed, values.length);
    }
  }
  const workerCount = Math.min(concurrency, values.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return results;
}

function showBatchResults(results) {
  const outputHeaders = [
    ...batchState.headers,
    "muor_status",
    "muor_canonical_smiles",
    "muor_rf_score",
    "muor_rf_label",
    "muor_tabpfn_score",
    "muor_tabpfn_label",
    "muor_lightgbm_score",
    "muor_lightgbm_label",
    "muor_vote_count",
    "muor_all_three_antagonistic",
    "muor_consensus_mean_score",
  ];
  const outputRows = batchState.rows.map((row, index) => [
    ...row,
    ...predictionValues(results[index]),
  ]);
  const ok = results.filter((result) => result.status === "ok").length;
  const invalid = results.length - ok;
  const consensus = results.filter(
    (result) => result.status === "ok" && result.consensus_all_antagonistic === 1,
  ).length;
  const nonAntagonistic = results.filter(
    (result) => result.status === "ok" && result.consensus_all_non_antagonistic === 1,
  ).length;
  const mixed = ok - consensus - nonAntagonistic;
  byId("batch-summary").innerHTML = [
    ["Rows", results.length],
    ["Valid", ok],
    ["All-three hits", consensus],
    ["Mixed votes", mixed],
    ["Invalid / failed", invalid],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");

  const smilesHeader = batchState.headers[batchState.smilesColumn];
  byId("batch-table-head").innerHTML = `<tr>
    <th>Row</th><th>${escapeHtml(smilesHeader)}</th><th>Status</th>
    <th>RF</th><th>TabPFN</th><th>LightGBM</th><th>Votes</th><th>All-three</th>
  </tr>`;
  byId("batch-table-body").innerHTML = results.slice(0, BATCH_PREVIEW_ROWS).map((result, index) => {
    const valid = result.status === "ok";
    const score = (value) => valid ? value.toFixed(4) : "—";
    const rawSmiles = batchState.rows[index][batchState.smilesColumn];
    return `<tr>
      <td>${index + 1}</td>
      <td class="smiles-cell" title="${escapeHtml(rawSmiles)}">${escapeHtml(rawSmiles)}</td>
      <td><span class="status-pill ${valid ? "" : "invalid"}">${escapeHtml(result.status)}</span></td>
      <td>${score(result.rf_antagonistic_score)}</td>
      <td>${score(result.tabpfn_antagonistic_score)}</td>
      <td>${score(result.lightgbm_antagonistic_score)}</td>
      <td>${valid ? `${result.consensus_vote_count}/3` : "—"}</td>
      <td class="${valid && result.consensus_all_antagonistic === 1 ? "consensus-yes" : ""}">${valid ? (result.consensus_all_antagonistic === 1 ? "Yes" : "No") : "—"}</td>
    </tr>`;
  }).join("");
  byId("batch-preview-note").textContent = results.length > BATCH_PREVIEW_ROWS
    ? `Showing the first ${BATCH_PREVIEW_ROWS} of ${results.length} rows. The downloaded CSV contains all rows and original columns.`
    : `Showing all ${results.length} rows. The downloaded CSV retains every original column.`;

  const csv = [outputHeaders, ...outputRows]
    .map((values) => values.map(csvCell).join(","))
    .join("\r\n");
  if (batchDownloadUrl) URL.revokeObjectURL(batchDownloadUrl);
  batchDownloadUrl = URL.createObjectURL(
    new Blob([`\uFEFF${csv}`], { type: "text/csv;charset=utf-8" }),
  );
  byId("batch-download").href = batchDownloadUrl;
  byId("batch-download").classList.remove("hidden");
  byId("batch-results").classList.remove("hidden");
}

byId("predict-button").addEventListener("click", async () => {
  const button = byId("predict-button");
  const message = byId("single-message");
  button.disabled = true;
  message.textContent = "Running three-model inference…";
  try {
    const response = await fetch("/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles: byId("smiles-input").value, explain: true }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Prediction failed");
    showPrediction(result);
    message.textContent = "Complete";
  } catch (error) {
    byId("single-results").classList.add("hidden");
    message.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

byId("smiles-input").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    byId("predict-button").click();
  }
});

byId("csv-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  const button = byId("batch-predict-button");
  const message = byId("batch-message");
  batchState = null;
  button.disabled = true;
  byId("batch-results").classList.add("hidden");
  byId("batch-progress").classList.add("hidden");
  if (!file) {
    message.textContent = "Select a CSV file";
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    message.textContent = "CSV exceeds the 2 MB upload limit.";
    return;
  }
  try {
    batchState = parseCsv(await file.text());
    message.textContent = `${file.name} · ${batchState.rows.length} rows ready`;
    button.disabled = false;
  } catch (error) {
    message.textContent = error.message;
  }
});

byId("batch-predict-button").addEventListener("click", async () => {
  if (!batchState) return;
  const button = byId("batch-predict-button");
  const message = byId("batch-message");
  const progress = byId("batch-progress");
  button.disabled = true;
  byId("csv-input").disabled = true;
  byId("batch-results").classList.add("hidden");
  progress.classList.remove("hidden");
  progress.style.setProperty("--progress", "0%");
  message.textContent = `Predicting 0 / ${batchState.rows.length}…`;
  try {
    const smiles = batchState.rows.map((row) => row[batchState.smilesColumn]);
    const results = await concurrentMap(
      smiles,
      BATCH_CONCURRENCY,
      predictBatchRow,
      (completed, total) => {
        message.textContent = `Predicting ${completed} / ${total}…`;
        progress.style.setProperty("--progress", `${(completed / total) * 100}%`);
      },
    );
    showBatchResults(results);
    message.textContent = `Complete · ${results.length} rows`;
  } catch (error) {
    message.textContent = error.message;
  } finally {
    button.disabled = false;
    byId("csv-input").disabled = false;
  }
});

checkService();
