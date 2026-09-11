"use strict";

const els = {
  serverState: document.getElementById("serverState"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  browseBtn: document.getElementById("browseBtn"),
  dataInput: document.getElementById("dataInput"),
  fileHint: document.getElementById("fileHint"),
  summaryType: document.getElementById("summaryType"),
  runLlm: document.getElementById("runLlm"),
  llmHint: document.getElementById("llmHint"),
  sanitize: document.getElementById("sanitize"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  progress: document.getElementById("progress"),
  progressText: document.getElementById("progressText"),
  outputPanel: document.getElementById("outputPanel"),
  errors: document.getElementById("errors"),
  outTitle: document.getElementById("outTitle"),
  outText: document.getElementById("outText"),
  outMeta: document.getElementById("outMeta"),
  toast: document.getElementById("toast"),
};

let toastTimer = null;
let pendingFile = null;

/* ---------------- helpers ---------------- */

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function toast(msg) {
  els.toast.textContent = msg;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (els.toast.hidden = true), 3200);
}

/* ---------------- input wiring ---------------- */

els.browseBtn.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  els.dropzone.classList.add("drag");
});
els.dropzone.addEventListener("dragleave", () => els.dropzone.classList.remove("drag"));
els.dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  els.dropzone.classList.remove("drag");
  if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
});

els.fileInput.addEventListener("change", () => {
  if (els.fileInput.files.length) loadFile(els.fileInput.files[0]);
});

function toBase64(u8) {
  let s = "";
  for (let i = 0; i < u8.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  }
  return btoa(s);
}

function loadFile(file) {
  if (!file) return;

  if (/\.(pdf|xlsx|xls)$/i.test(file.name)) {
    pendingFile = { name: file.name, b64: null };
    const reader = new FileReader();
    reader.onload = () => {
      pendingFile.b64 = toBase64(new Uint8Array(reader.result));
      els.fileHint.textContent = `${file.name} loaded (${(file.size / 1024).toFixed(1)} KB)`;
      onInput();
    };
    reader.readAsArrayBuffer(file);
    return;
  }

  pendingFile = null;
  const reader = new FileReader();
  reader.onload = () => {
    els.dataInput.value = String(reader.result);
    els.fileHint.textContent = `${file.name} loaded (${(file.size / 1024).toFixed(1)} KB)`;
    onInput();
  };
  reader.readAsText(file);
}

els.dataInput.addEventListener("input", onInput);
els.runLlm.addEventListener("change", () => {
  els.llmHint.textContent = els.runLlm.checked
    ? "on → LLM narrative (CPU inference can take minutes)"
    : "off → instant result";
});

function onInput() {
  els.analyzeBtn.disabled = els.dataInput.value.trim().length === 0 && !(pendingFile && pendingFile.b64);
}

/* ---------------- analyze ---------------- */

async function analyze() {
  const text = els.dataInput.value.trim();
  if (!text && !(pendingFile && pendingFile.b64)) {
    toast("Paste text or upload a file first");
    return;
  }

  hideOutput();
  showProgress(
    true,
    els.runLlm.checked ? "Running LLM — CPU inference can take minutes…" : "Computing…"
  );

  const started = performance.now();
  try {
    const common = {
      summary_type: els.summaryType.value,
      run_llm: els.runLlm.checked,
    };
    let payload;
    if (!text && pendingFile && pendingFile.b64) {
      payload = { data_b64: pendingFile.b64, filename: pendingFile.name, ...common };
    } else {
      payload = { data: text, source_type: "auto", sanitize: els.sanitize.checked, ...common };
    }

    const resp = await fetch("/api/v1/summarize/auto", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (resp.status !== 200) {
      let detail = `HTTP ${resp.status}`;
      try { detail = (await resp.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const report = await resp.json();
    render(report, (performance.now() - started) / 1000);
  } catch (err) {
    showProgress(false);
    reportError(err.message || String(err));
  }
}

function showProgress(on, text) {
  els.progress.hidden = !on;
  if (text) els.progressText.textContent = text;
}

function hideOutput() {
  els.outputPanel.hidden = true;
  els.errors.hidden = true;
}

function reportError(msg) {
  els.errors.innerHTML = "";
  els.errors.append(el("h4", null, "Analysis failed"));
  els.errors.append(el("code", null, msg));
  els.errors.hidden = false;
  els.outputPanel.hidden = false;
}

/* ---------------- rendering: summary only ---------------- */

function render(report, seconds) {
  showProgress(false);
  els.outputPanel.hidden = false;
  els.errors.hidden = true;

  const s = report.summary || {};
  let title = s.title || "Summary";
  let text = "";

  if (report.pipeline === "document") {
    const sm = s.summary;
    if (typeof sm === "string") {
      text = sm;
    } else if (sm && typeof sm === "object") {
      if (sm.raw) {
        text = sm.raw;
      } else {
        const entries = Object.entries(sm);
        if (entries.length) {
          text = entries.map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join("; ") : String(v)}`).join("\n");
        }
      }
    }
    if (!text && sm?.outline) {
      text = Array.isArray(sm.outline) ? sm.outline.join("\n") : String(sm.outline);
    }
    if (sm?.llm_error && text) {
      text = text + "\n\n(LLM call failed: " + sm.llm_error + " — showing deterministic digest.)";
    }
  } else {
    text = s.executive_summary || "";
  }

  if (!text) text = "(no summary text returned)";

  els.outTitle.textContent = title;
  els.outText.textContent = text;
  const det = report.detection || {};
  els.outMeta.textContent =
    `${report.input_type} · ${report.pipeline}${det.subtype ? " (" + det.subtype + ")" : ""}` +
    ` · mode ${report.mode} · ${seconds.toFixed(1)}s`;
}

els.analyzeBtn.addEventListener("click", analyze);

/* ---------------- init ---------------- */

(async function init() {
  onInput();
  try {
    const resp = await fetch("/api/v1/health");
    if (!resp.ok) throw new Error("bad status");
    const h = await resp.json();
    els.serverState.textContent = `server OK · ${(h.jobs.pending + h.jobs.running + h.jobs.completed) || 0} jobs`;
    els.serverState.classList.add("ok");
  } catch (_) {
    els.serverState.textContent = "server offline";
    els.serverState.classList.add("err");
  }
})();