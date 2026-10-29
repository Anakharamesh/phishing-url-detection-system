/**
 * Frontend JavaScript client for Phishing URL Detection System.
 */

document.addEventListener("DOMContentLoaded", () => {
  const urlInput = document.getElementById("urlInput");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const errorAlert = document.getElementById("errorAlert");
  const errorMsg = document.getElementById("errorMsg");
  const loadingBox = document.getElementById("loadingBox");
  const resultsContainer = document.getElementById("resultsContainer");

  // Overall elements
  const overallCard = document.getElementById("overallCard");
  const verdictIcon = document.getElementById("verdictIcon");
  const verdictStatus = document.getElementById("verdictStatus");
  const analyzedUrlText = document.getElementById("analyzedUrlText");
  const consensusBadge = document.getElementById("consensusBadge");
  const consensusProgress = document.getElementById("consensusProgress");
  const consensusCaption = document.getElementById("consensusCaption");

  // Models container
  const modelsGrid = document.getElementById("modelsGrid");

  // Features table body
  const featuresTableBody = document.getElementById("featuresTableBody");

  // Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  let featureMetadataMap = {};

  // 1. Fetch feature metadata
  fetch("/api/features")
    .then((res) => res.json())
    .then((data) => {
      data.forEach((item) => {
        featureMetadataMap[item.name] = item;
      });
    })
    .catch((err) => console.error("Error fetching feature metadata:", err));

  // 2. Tab switching logic
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }
    });
  });

  // 3. Quick sample chips
  const chips = document.querySelectorAll(".chip");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const sampleUrl = chip.getAttribute("data-url");
      if (sampleUrl) {
        urlInput.value = sampleUrl;
        triggerAnalysis();
      }
    });
  });

  // 4. Form submission
  analyzeBtn.addEventListener("click", triggerAnalysis);
  urlInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      triggerAnalysis();
    }
  });

  function showError(msg) {
    errorMsg.textContent = msg;
    errorAlert.style.display = "flex";
  }

  function hideError() {
    errorAlert.style.display = "none";
    errorMsg.textContent = "";
  }

  function triggerAnalysis() {
    const rawUrl = urlInput.value.trim();
    if (!rawUrl) {
      showError("Please enter a valid URL to analyze.");
      urlInput.focus();
      return;
    }

    hideError();
    resultsContainer.style.display = "none";
    loadingBox.style.display = "block";
    analyzeBtn.disabled = true;

    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: rawUrl }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned error (${res.status})`);
        }
        return res.json();
      })
      .then((data) => {
        renderResults(data);
      })
      .catch((err) => {
        showError(err.message || "Failed to analyze URL. Please check connection.");
      })
      .finally(() => {
        loadingBox.style.display = "none";
        analyzeBtn.disabled = false;
      });
  }

  function renderResults(data) {
    const isPhishing = data.is_phishing;

    // 1. Render Overall Verdict Card
    overallCard.className = `overall-card ${isPhishing ? "phishing" : "legitimate"}`;
    verdictIcon.innerHTML = isPhishing ? "⚠️" : "🛡️";
    verdictStatus.textContent = isPhishing ? "PHISHING URL DETECTED" : "LEGITIMATE URL";
    analyzedUrlText.textContent = data.url;

    consensusBadge.textContent = `${data.model_consensus_percentage}% Phishing Consensus`;
    consensusProgress.style.width = `${data.model_consensus_percentage}%`;
    consensusProgress.style.background = isPhishing
      ? "linear-gradient(90deg, #f59e0b, #ef4444)"
      : "linear-gradient(90deg, #3b82f6, #10b981)";

    consensusCaption.textContent = `${data.phishing_votes} of 4 models classified this URL as PHISHING (${data.legitimate_votes} as LEGITIMATE). Note: Consensus indicates model agreement, not a calibrated statistical probability.`;

    // 2. Render 4 Model Cards
    modelsGrid.innerHTML = "";
    data.models.forEach((model) => {
      const card = document.createElement("div");
      card.className = "model-card";

      const badgeClass = model.is_phishing ? "phishing" : "legitimate";
      const badgeText = model.prediction;

      const scoreValue = model.phishing_score !== undefined ? `${model.phishing_score}%` : "N/A";
      const marginHtml = model.decision_margin !== undefined
        ? `<div style="font-size: 0.72rem; color: var(--text-dim); margin-top: 2px;">Decision Margin: ${model.decision_margin}</div>`
        : "";

      card.innerHTML = `
        <div>
          <div class="model-header">
            <div>
              <div class="model-title">${model.name}</div>
              <div class="model-type">${model.type}</div>
            </div>
            <div class="model-tag">${model.accuracy} Acc</div>
          </div>
          <div class="model-verdict-row">
            <span class="verdict-pill ${badgeClass}">${badgeText}</span>
            <div style="text-align: right;">
              <div class="model-score">${model.metric_label}: <strong>${scoreValue}</strong></div>
              ${marginHtml}
            </div>
          </div>
        </div>
        <div class="model-footer">
          <span>Preprocessing:</span>
          <span><strong>${model.preprocessing}</strong></span>
        </div>
      `;
      modelsGrid.appendChild(card);
    });

    // 3. Render Features Table
    featuresTableBody.innerHTML = "";
    let idx = 1;
    for (const [name, val] of Object.entries(data.features)) {
      const meta = featureMetadataMap[name] || {
        category: "Lexical",
        description: "Extracted URL feature value.",
      };

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="feat-index">#${idx++}</td>
        <td class="feat-name">${name}</td>
        <td><span class="feat-cat-badge">${meta.category}</span></td>
        <td class="feat-val">${val}</td>
        <td style="font-size: 0.82rem;">${meta.description}</td>
      `;
      featuresTableBody.appendChild(tr);
    }

    // Reveal results container
    resultsContainer.style.display = "block";
    resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});
