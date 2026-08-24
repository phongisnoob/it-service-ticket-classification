/* IT Service Ticket Classifier UI.
   Renders responses from the real FastAPI backend only — all routing
   decisions come from the API's needs_manual_review field. */
(function () {
  "use strict";

  var MAX_LEN = 5000;   // mirrors the API's pydantic Field(max_length)
  var MIN_NON_WS = 3;   // mirrors the API's field_validator

  var els = {
    form: document.getElementById("classify-form"),
    text: document.getElementById("ticket-text"),
    hint: document.getElementById("input-hint"),
    count: document.getElementById("char-count"),
    btn: document.getElementById("submit-btn"),
    pill: document.getElementById("status-pill"),
    pillText: document.getElementById("status-text"),
    conn: document.getElementById("conn-settings"),
    keyInput: document.getElementById("api-key"),
    panel: document.getElementById("result-panel"),
    empty: document.getElementById("result-empty"),
    error: document.getElementById("result-error"),
    errorTitle: document.getElementById("error-title"),
    errorDetail: document.getElementById("error-detail"),
    body: document.getElementById("result-body"),
    category: document.getElementById("r-category"),
    verdict: document.getElementById("r-verdict"),
    verdictLabel: document.getElementById("r-verdict-label"),
    verdictNums: document.getElementById("r-verdict-nums"),
    confFill: document.getElementById("r-conf-fill"),
    thrMark: document.getElementById("r-threshold-mark"),
    thrLabel: document.getElementById("r-threshold-label"),
    nextStep: document.getElementById("r-next"),
    top3: document.getElementById("r-top3"),
    mBackend: document.getElementById("m-backend"),
    mThreshold: document.getElementById("m-threshold"),
    mSha: document.getElementById("m-sha")
  };

  function nonWsLength(s) {
    return s.replace(/\s/g, "").length;
  }

  function setInputState() {
    var value = els.text.value;
    var len = value.length;
    var valid = nonWsLength(value) >= MIN_NON_WS && len <= MAX_LEN;

    els.count.textContent = len + " / " + MAX_LEN;
    els.count.setAttribute("data-over", String(len > MAX_LEN));
    els.btn.disabled = !valid || els.btn.getAttribute("data-busy") === "1";

    if (len > 0 && nonWsLength(value) < MIN_NON_WS) {
      els.hint.textContent = "Ticket text needs at least " + MIN_NON_WS +
        " non-whitespace characters.";
    } else if (len > MAX_LEN) {
      els.hint.textContent = "Ticket text is limited to " + MAX_LEN + " characters.";
    } else {
      els.hint.textContent = "";
    }
  }

  function showError(title, detail) {
    els.empty.hidden = true;
    els.body.hidden = true;
    els.error.hidden = false;
    els.errorTitle.textContent = title;
    els.errorDetail.textContent = detail;
  }

  function showEmpty() {
    els.empty.hidden = false;
    els.body.hidden = true;
    els.error.hidden = true;
  }

  function renderPrediction(data) {
    els.empty.hidden = true;
    els.error.hidden = true;
    els.body.hidden = false;

    var confidence = Number(data.confidence);

    els.category.textContent = data.category;

    var auto = data.needs_manual_review === false;
    els.verdict.setAttribute("data-verdict", auto ? "auto" : "manual");
    els.verdictLabel.textContent = auto
      ? "\u25CF AUTO-ROUTED"
      : "\u25CF MANUAL REVIEW";
    els.verdictNums.textContent =
      "confidence " + (confidence * 100).toFixed(1) + "%" +
      " \u00B7 threshold " + (threshold * 100).toFixed(0) + "%";

    var confPct = Math.max(0, Math.min(1, confidence)) * 100;
    var thrPct = Math.max(0, Math.min(1, threshold)) * 100;
    els.confFill.style.width = confPct + "%";
    els.thrMark.style.left =
      "clamp(30px, " + thrPct + "%, calc(100% - 34px))";
    els.thrLabel.textContent = "threshold " + (threshold * 100).toFixed(0) + "%";

    els.nextStep.textContent = auto
      ? "Routed automatically to " + data.category.toLowerCase() +
        " \u2014 no action needed."
      : "Confidence is below the routing threshold \u2014 send this ticket to the triage queue.";

    els.top3.innerHTML = "";
    (data.top_3 || []).forEach(function (item, i) {
      var li = document.createElement("li");

      var name = document.createElement("span");
      name.className = "pred-name";
      var rank = document.createElement("span");
      rank.className = "pred-rank";
      rank.textContent = "#" + (i + 1);
      name.appendChild(rank);
      name.appendChild(document.createTextNode(item.category));

      var bar = document.createElement("span");
      bar.className = "pred-bar";
      var fill = document.createElement("span");
      fill.style.width = (item.probability * 100).toFixed(1) + "%";
      bar.appendChild(fill);

      var prob = document.createElement("span");
      prob.className = "pred-prob";
      prob.textContent = (item.probability * 100).toFixed(1) + "%";

      li.appendChild(name);
      li.appendChild(bar);
      li.appendChild(prob);
      els.top3.appendChild(li);
    });
  }

  function setBusy(busy) {
    els.btn.setAttribute("data-busy", busy ? "1" : "0");
    els.btn.disabled = busy || nonWsLength(els.text.value) < MIN_NON_WS;
    els.btn.textContent = busy ? "Classifying\u2026" : "Classify ticket";
    els.panel.setAttribute("aria-busy", String(busy));
  }

  function extractErrorDetail(status, payload, statusText) {
    if (payload && payload.detail) {
      if (typeof payload.detail === "string") return payload.detail;
      try { return JSON.stringify(payload.detail); } catch (e) { /* fall through */ }
    }
    return "HTTP " + status + " " + (statusText || "") +
      " \u2014 response was not JSON (proxy or gateway error?)";
  }

  els.form.addEventListener("submit", function (event) {
    event.preventDefault();
    setBusy(true);
    showEmpty();

    var headers = { "Content-Type": "application/json" };
    var key = els.keyInput.value.trim();
    if (key) headers["X-API-Key"] = key;

    fetch("/predict", {
      method: "POST",
      headers: headers,
      body: JSON.stringify({ text: els.text.value })
    })
      .then(function (response) {
        return response
          .json()
          .catch(function () { return null; })
          .then(function (payload) {
            return { ok: response.ok, status: response.status,
                     statusText: response.statusText, payload: payload };
          });
      })
      .then(function (result) {
        if (result.ok) {
          renderPrediction(result.payload);
        } else if (result.status === 401) {
          els.conn.hidden = false;
          els.keyInput.focus();
          showError(
            "This server requires an API key",
            "Open \u201CAPI key\u201D above, paste the server's key, and classify again."
          );
        } else {
          var detail = result.payload
            ? extractErrorDetail(result.status, result.payload)
            : extractErrorDetail(result.status, null, result.statusText);
          showError("Request rejected (" + result.status + ")", detail);
        }
      })
      .catch(function () {
        showError(
          "Cannot reach the model server",
          "The /predict request failed before a response arrived. Check that " +
            "the service is running and try again."
        );
      })
      .finally(function () {
        setBusy(false);
        setInputState();
      });
  });

  els.text.addEventListener("input", setInputState);

  // Focus the primary input on pointer devices only (avoids mobile keyboard pop-up).
  if (window.matchMedia && window.matchMedia("(pointer: fine)").matches) {
    els.text.focus();
  }

  // Remember the key for this tab session only.
  try {
    var saved = sessionStorage.getItem("api_key");
    if (saved) els.keyInput.value = saved;
    els.keyInput.addEventListener("change", function () {
      sessionStorage.setItem("api_key", els.keyInput.value.trim());
    });
  } catch (e) { /* storage unavailable — the key just won't persist */ }

  // Model status: /health feeds the top-bar pill and the rail card.
  fetch("/health")
    .then(function (r) { return r.json(); })
    .then(function (h) {
      if (h.status !== "ok") throw new Error("status=" + h.status);
      var sha = h.model_sha256 ? String(h.model_sha256).slice(0, 12) : "unknown";
      els.pill.setAttribute("data-state", "ok");
      els.pillText.textContent = h.model_backend + " \u00B7 sha " + sha;
      els.mBackend.textContent = h.model_backend;
      if (h.threshold !== null && h.threshold !== undefined) {
        els.mThreshold.textContent = Number(h.threshold).toFixed(2);
      }
      els.mSha.textContent = sha;
    })
    .catch(function () {
      els.pill.setAttribute("data-state", "error");
      els.pillText.textContent = "model unreachable";
    });

  // Routing performance: persisted evaluation metrics served by /model-info.
  fetch("/model-info")
    .then(function (r) { return r.json(); })
    .then(function (info) {
      if (!info || !info.available || !info.routing) return;
      var routing = info.routing;
      var cls = info.classification || {};
      var cal = info.calibration || {};

      function pct(x) { return (x * 100).toFixed(2) + "%"; }

      document.getElementById("k-coverage").textContent = pct(routing.coverage);
      document.getElementById("k-acc").textContent = pct(routing.auto_routed_accuracy);
      document.getElementById("k-review").textContent = pct(routing.manual_review_rate);
      document.getElementById("k-overall").textContent =
        pct(cls.accuracy !== undefined ? cls.accuracy : routing.overall_test_accuracy);
      document.getElementById("k-f1").textContent =
        cls.macro_f1 !== undefined ? pct(cls.macro_f1) : "\u2013";

      if (routing.total_tickets) {
        document.getElementById("perf-note").textContent =
          "Held-out test set, " + routing.total_tickets.toLocaleString() +
          " tickets (" + Number(routing.auto_routed_tickets).toLocaleString() +
          " auto-routed).";
      }

      if (cal.expected_calibration_error !== undefined) {
        document.getElementById("k-ece").textContent =
          cal.expected_calibration_error.toFixed(3);
        document.getElementById("k-brier").textContent =
          cal.top_label_brier_score.toFixed(3);
      }

      document.getElementById("perf-panel").hidden = false;
    })
    .catch(function () { /* metrics panel simply stays hidden */ });

  setInputState();
})();
