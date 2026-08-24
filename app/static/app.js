/* IT Service Ticket Classifier UI.
   Talks to the real FastAPI backend. All routing decisions come from the
   API's needs_manual_review field — this script only renders responses. */
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
    meta: document.getElementById("model-meta"),
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
    top3: document.getElementById("r-top3")
  };

  function nonWsLength(s) {
    return s.replace(/\s/g, "").length;
  }

  function setInputState() {
    var value = els.text.value;
    var len = value.length;
    var valid = nonWsLength(value) >= MIN_NON_WS && len <= MAX_LEN;

    els.count.textContent = len + " / " + MAX_LEN + " chars";
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

    els.category.textContent = data.category;

    var auto = data.needs_manual_review === false;
    els.verdict.setAttribute("data-verdict", auto ? "auto" : "manual");
    els.verdictLabel.textContent = auto
      ? "\u25CF AUTO-ROUTE"
      : "\u25CF MANUAL REVIEW";
    els.verdictNums.textContent =
      "confidence " + Number(data.confidence).toFixed(3) +
      " \u00B7 threshold " + Number(data.threshold).toFixed(2);

    var confPct = Math.max(0, Math.min(1, data.confidence)) * 100;
    var thrPct = Math.max(0, Math.min(1, data.threshold)) * 100;
    els.confFill.style.width = confPct + "%";
    els.thrMark.style.left = "calc(" + thrPct + "% - 1px)";

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
      prob.textContent = item.probability.toFixed(3);

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

  function extractErrorDetail(status, payload) {
    if (payload && payload.detail) {
      if (typeof payload.detail === "string") return payload.detail;
      try { return JSON.stringify(payload.detail); } catch (e) { /* fall through */ }
    }
    return "HTTP " + status;
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
            : "HTTP " + result.status + " " + (result.statusText || "no body") +
              " \u2014 response was not JSON (proxy or gateway error?)";
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

  // Remember the key for this tab session only.
  try {
    var saved = sessionStorage.getItem("api_key");
    if (saved) els.keyInput.value = saved;
    els.keyInput.addEventListener("change", function () {
      sessionStorage.setItem("api_key", els.keyInput.value.trim());
    });
  } catch (e) { /* storage unavailable — the key just won't persist */ }

  // Load real model metadata from /health for the masthead readout.
  fetch("/health")
    .then(function (r) { return r.json(); })
    .then(function (h) {
      if (h.status !== "ok") throw new Error("status=" + h.status);
      var sha = h.model_sha256 ? String(h.model_sha256).slice(0, 12) : "unknown";
      els.meta.textContent =
        "backend " + h.model_backend +
        " \u00B7 model sha256 " + sha +
        " \u00B7 route threshold " + Number(h.threshold).toFixed(2);
      els.meta.setAttribute("data-state", "ok");
    })
    .catch(function () {
      els.meta.textContent = "Model server unreachable \u2014 classification unavailable.";
      els.meta.setAttribute("data-state", "error");
    });

  setInputState();
})();
