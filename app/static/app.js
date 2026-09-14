"use strict";

const els = {
  serverState: document.getElementById("serverState"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  browseBtn: document.getElementById("browseBtn"),
  dataInput: document.getElementById("dataInput"),
  fileHint: document.getElementById("fileHint"),
  bulletPoints: document.getElementById("bulletPoints"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  outputPanel: document.getElementById("outputPanel"),
  errors: document.getElementById("errors"),
  outTitle: document.getElementById("outTitle"),
  outText: document.getElementById("outText"),
  outMeta: document.getElementById("outMeta"),
  toast: document.getElementById("toast"),
  statusLine: document.getElementById("statusLine"),
};

let toastTimer = null;
let pendingFile = null; // { name, b64 } for pdf/xlsx
let pendingText = null; // large text buffers kept OUT of the textarea

const TEXTAREA_MAX = 200 * 1024; // ~200 KB before a textarea slows down

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

function fmtSize(n) {
  if (n >= 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + " MB";
  if (n >= 1024) return (n / 1024).toFixed(1) + " KB";
  return n + " B";
}

function setStatus(msg) {
  if (!msg) {
    els.statusLine.hidden = true;
    els.statusLine.textContent = "";
    return;
  }
  els.statusLine.textContent = msg;
  els.statusLine.hidden = false;
}

function fmtElapsed(ms) {
  if (ms < 1000) return `${Math.floor(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
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
  els.fileHint.textContent = `Reading ${file.name} (${fmtSize(file.size)})…`;
  els.fileHint.classList.add("busy");

  if (/\.(pdf|xlsx|xls)$/i.test(file.name)) {
    pendingFile = { name: file.name, b64: null };
    const reader = new FileReader();
    reader.onload = () => {
      pendingFile.b64 = toBase64(new Uint8Array(reader.result));
      pendingText = null;
      els.fileHint.classList.remove("busy");
      els.fileHint.textContent = `Ready: ${file.name} (${fmtSize(file.size)})`;
      onInput();
    };
    reader.readAsArrayBuffer(file);
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    const text = String(reader.result);
    pendingFile = null;
    if (text.length <= TEXTAREA_MAX) {
      els.dataInput.value = text;
      pendingText = null;
    } else {
      els.dataInput.value = "";
      pendingText = text;
    }
    els.fileHint.classList.remove("busy");
    els.fileHint.textContent =
      `Ready: ${file.name} (${fmtSize(file.size)})` +
      (pendingText ? " · large file kept in memory" : "");
    onInput();
  };
  reader.readAsText(file);
}

els.dataInput.addEventListener("input", () => {
  pendingText = null;
  onInput();
});

function onInput() {}

/* ---------------- analyze ---------------- */

async function analyze() {
  const textFromBox = els.dataInput.value.trim();
  const text = (pendingText || textFromBox).trim();
  if (!text && !(pendingFile && pendingFile.b64)) {
    toast("Paste text or upload a file first");
    return;
  }

  setBusy(true);
  hideOutput();
  setStatus("Preparing payload…");
  const started = performance.now();

  let tick = setInterval(() => {
    setStatus(`Working… ${fmtElapsed(performance.now() - started)}`);
  }, 750);

  const clearTick = () => {
    clearInterval(tick);
    setStatus("");
  };

  const basePayload = () => {
    if (!text && pendingFile && pendingFile.b64) {
      return { data_b64: pendingFile.b64, filename: pendingFile.name };
    }
    return { data: text, source_type: "auto", sanitize: true };
  };

  const fetchReport = async (runLlm) => {
    const payload = { ...basePayload(), run_llm: runLlm };
    return fetchPost(payload, runLlm, started);
  };

  try {
    const t0 = performance.now();
    setStatus(`Uploading ${text.length || 0} chars…`);

    // Phase 1: fast deterministic digest -> render almost instantly.
    try {
      const det = await fetchReport(false);
      clearTick();
      render(det, (performance.now() - started) / 1000, "(deterministic)");
    } catch (err) {
      clearTick();
      reportError(err.message || String(err));
      return;
    }

    // Phase 2: upgrade with the AI narrative.
    setBusy(true);
    setStatus(`Generating AI narrative…`);
    try {
      const withLlm = await fetchReport(true);
      clearTick();
      render(withLlm, (performance.now() - started) / 1000, "(AI)");
    } catch (_) {
      clearTick();
      toast("AI narrative unavailable; deterministic summary shown");
    }
  } catch (err) {
    clearTick();
    reportError(err.message || String(err));
  }
}

async function fetchPost(payload, runLlm, started) {
  const resp = await fetch("/api/v1/summarize/auto", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(600000),
  });
  if (resp.status === 202) {
    const job = await resp.json();
    return pollJob(job.job_id, started);
  }
  if (resp.status !== 200) {
    let detail = `HTTP ${resp.status}`;
    try { detail = (await resp.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return resp.json();
}

async function pollJob(jobId, started) {
  while (true) {
    setStatus(`Processing on server… ${fmtElapsed(performance.now() - started)}`);
    const r = await fetch(`/api/v1/jobs/${jobId}`);
    if (!r.ok) throw new Error(`HTTP ${r.status} polling job`);
    const job = await r.json();
    if (job.status === "completed") return job.summary;
    if (job.status === "failed") throw new Error(job.error || "Background job failed");
    await new Promise((r) => setTimeout(r, 1200));
  }
}

function setBusy(busy) {
  els.analyzeBtn.disabled = busy;
  els.analyzeBtn.textContent = busy ? "Processing…" : "▶ Summarize";
}

function hideOutput() {
  els.outputPanel.hidden = true;
  els.errors.hidden = true;
}

function reportError(msg) {
  setBusy(false);
  els.errors.innerHTML = "";
  els.errors.append(el("h4", null, "Analysis failed"));
  els.errors.append(el("code", null, msg));
  els.errors.hidden = false;
  els.outputPanel.hidden = false;
}

/* ---------------- rendering: summary only ---------------- */

function render(report, seconds, source) {
  setBusy(false);
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
    // data / report pipeline — SummaryOutput shape
    text = s.executive_summary || "";
    if (report.pipeline === "report") {
      const extras = [];
      if (Array.isArray(s.key_findings) && s.key_findings.length) {
        extras.push("KEY FINDINGS\n" + s.key_findings.map(f => "  • " + f).join("\n"));
      }
      if (Array.isArray(s.recommendations) && s.recommendations.length) {
        extras.push("RECOMMENDATIONS\n" + s.recommendations.map(r => "  • " + r).join("\n"));
      }
      if (s.calculated_metrics && Object.keys(s.calculated_metrics).length) {
        const m = Object.entries(s.calculated_metrics)
          .map(([k, v]) => "  " + k + ": " + v).join("\n");
        extras.push("METRICS\n" + m);
      }
      if (extras.length) text = text + "\n\n" + extras.join("\n\n");
    }
  }

  if (!text) text = "(no summary text returned)";

  els.outTitle.textContent = title;
  els.outText.replaceChildren();
  if (els.bulletPoints.checked) {
    const items = text.split(/\n+/).map((line) => line.trim()).filter(Boolean);
    const list = el("ul", "summary-bullets");
    for (const item of items.length ? items : ["(no summary text returned)"]) {
      list.append(el("li", null, item));
    }
    els.outText.append(list);
  } else {
    els.outText.textContent = text;
  }
  const det = report.detection || {};
  els.outMeta.textContent =
    `${report.input_type} · ${report.pipeline}${det.subtype ? " (" + det.subtype + ")" : ""}` +
    ` · ${seconds.toFixed(1)}s${source ? " · " + source : ""}`;
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