/* ─── LexGuard Frontend Logic ────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', function () {

  /* ── DOM refs ───────────────────────────────────────────────────────────── */
  const uploadForm         = document.getElementById('uploadForm');
  const resultsRow         = document.getElementById('resultsRow');
  const summaryCard        = document.getElementById('summaryCard');
  const summaryPlaceholder = document.getElementById('summaryPlaceholder');
  const resultsList        = document.getElementById('resultsList');
  const loading            = document.getElementById('loading');
  const totalSentencesEl   = document.getElementById('totalSentences');
  const riskyCountEl       = document.getElementById('riskyCount');
  const riskPercentageEl   = document.getElementById('riskPercentage');
  const resultsBadge       = document.getElementById('resultsBadge');
  const ccicBar            = document.getElementById('ccicBar');
  const sevDistEl          = document.getElementById('sevDist');

  let riskChartInstance = null;

  /* ── Form submit ─────────────────────────────────────────────────────────── */
  uploadForm.addEventListener('submit', function (e) {
    e.preventDefault();
    const formData = new FormData(uploadForm);

    loading.style.display            = 'block';
    resultsRow.style.display         = 'none';
    summaryCard.style.display        = 'none';
    summaryPlaceholder.style.display = 'none';
    if (ccicBar)   ccicBar.style.display   = 'none';
    if (sevDistEl) sevDistEl.style.display = 'none';

    fetch('/analyze', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(data => {
        loading.style.display = 'none';
        if (data.error) {
          alert(data.error);
          summaryPlaceholder.style.display = 'flex';
          return;
        }
        displayResults(data);
      })
      .catch(err => {
        console.error('Error:', err);
        loading.style.display = 'none';
        summaryPlaceholder.style.display = 'flex';
        alert('An error occurred while analyzing the document.');
      });
  });

  /* ── Display results ─────────────────────────────────────────────────────── */
  function displayResults(data) {
    const { risks, summary, ccic, llm_available } = data;

    /* Summary stats */
    totalSentencesEl.textContent = summary.total_sentences;
    riskyCountEl.textContent     = summary.risky_count;
    riskPercentageEl.textContent = `${summary.risk_percentage}%`;
    resultsBadge.textContent     = `${summary.risky_count} Risky Clauses Found`;

    summaryCard.style.display = 'block';
    resultsRow.style.display  = 'block';

    /* CCIC info bar */
    if (ccic && ccicBar) {
      const dd = ccic.detector_distribution || {};
      ccicBar.innerHTML = `
        <strong>CCIC Mode:</strong>
        <span class="ccic-pill">${ccic.mode}</span>
        <strong>τ<sub>t</sub>:</strong><span class="ccic-pill">${ccic.transformer_threshold}</span>
        <strong>τ<sub>m</sub>:</strong><span class="ccic-pill">${ccic.ml_threshold}</span>
        <strong>T:</strong><span class="ccic-pill">${ccic.temperature}</span>
        &nbsp;|&nbsp;
        <span title="Classified by transformer">⚡ Transformer: <b>${dd.transformer || 0}</b></span>
        <span title="Classified by ML">🤖 ML: <b>${dd.ml || 0}</b></span>
        <span title="Classified by keywords">🔑 Keyword: <b>${dd.keyword || 0}</b></span>
      `;
      ccicBar.style.display = 'flex';
    }

    /* Severity distribution */
    if (summary.severity_distribution && sevDistEl) {
      const sd = summary.severity_distribution;
      sevDistEl.innerHTML = `
        <span style="font-size:.75rem;color:#64748b;margin-right:4px;">Severity:</span>
        <span class="sev-pill high">● High&nbsp;${sd.High || 0}</span>
        <span class="sev-pill medium">● Medium&nbsp;${sd.Medium || 0}</span>
        <span class="sev-pill low">● Low&nbsp;${sd.Low || 0}</span>
      `;
      sevDistEl.style.display = 'flex';
    }

    /* Risk cards */
    resultsList.innerHTML = '';
    if (risks.length === 0) {
      resultsList.innerHTML = `
        <div class="empty-state">
          <svg width="64" height="64" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.2"
              d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806
              3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438
              3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138
              3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0
              3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138
              3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438
              3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z"/>
          </svg>
          <p class="mb-0 fw-semibold">No risky clauses detected — great news!</p>
          <p class="small mt-1">This document appears to be low risk.</p>
        </div>`;
    } else {
      risks.forEach((risk, i) => {
        resultsList.appendChild(buildRiskCard(risk, i, llm_available));
      });
    }

    updateChart(risks);
  }

  /* ── Build a single risk card ────────────────────────────────────────────── */
  function buildRiskCard(risk, idx, llm_available) {
    const sev   = (risk.severity || 'Low').toLowerCase();
    const score = typeof risk.severity_score === 'number' ? risk.severity_score : null;
    const conf  = typeof risk.confidence === 'number' ? risk.confidence : null;
    const det   = risk.detector || 'keyword';

    const tierClass = det === 'transformer' ? 'tier-t' : det === 'ml' ? 'tier-m' : 'tier-k';
    const tierLabel = det === 'transformer' ? '⚡ Transformer' : det === 'ml' ? '🤖 ML' : '🔑 Keyword';

    const scorePct = score !== null ? Math.round(score * 100) : null;
    const scoreBar = score !== null ? `
      <div class="score-bar-wrap">
        <div class="score-bar-track">
          <div class="score-bar-fill ${sev}" style="width:${scorePct}%"></div>
        </div>
        <span class="score-label">${score.toFixed(3)}</span>
      </div>` : '';

    const confText = conf !== null
      ? `<span class="meta-chip">conf: ${conf.toFixed(4)}</span>` : '';

    const initExpl = llm_available ? 'Click to generate an AI explanation.' : (risk.explanation || '');

    const card = document.createElement('div');
    card.className = `risk-card sev-${sev}`;
    card.style.animationDelay = `${idx * 0.05}s`;

    card.innerHTML = `
      <div class="risk-card-header">
        <span class="risk-type-label">${risk.risk_type} Risk</span>
        <span class="severity-badge ${sev}">${risk.severity} Severity</span>
      </div>
      <p class="risk-sentence">"${escHtml(risk.original_sentence)}"</p>
      ${scoreBar}
      <div class="risk-meta">
        <span class="meta-chip ${tierClass}">${tierLabel}</span>
        ${confText}
        ${score !== null ? `<span class="meta-chip">HSE: ${score.toFixed(3)}</span>` : ''}
        <span class="toggle-hint">▾ Explain</span>
      </div>
      <div class="explanation-box" id="expl-${idx}">
        <strong>Explanation:</strong>
        <span class="expl-content">${escHtml(initExpl)}</span>
      </div>
    `;

    card.dataset.sentence = risk.original_sentence;
    card.dataset.riskType = risk.risk_type;
    card.dataset.severity = risk.severity;
    card.dataset.loaded   = llm_available ? 'false' : 'true';

    card.addEventListener('click', async () => {
      const box     = card.querySelector(`#expl-${idx}`);
      const content = card.querySelector('.expl-content');
      const hint    = card.querySelector('.toggle-hint');
      const isOpen  = box.classList.contains('open');

      box.classList.toggle('open');
      hint.textContent = isOpen ? '▾ Explain' : '▴ Hide';

      if (!isOpen && card.dataset.loaded === 'false') {
        content.innerHTML = '<span class="explanation-spinner">⏳ Generating AI explanation…</span>';
        try {
          const resp = await fetch('/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              sentence:  card.dataset.sentence,
              risk_type: card.dataset.riskType,
              severity:  card.dataset.severity,
            })
          });
          const payload = await resp.json();
          if (!resp.ok) throw new Error(payload.error || 'Failed');
          content.textContent = payload.explanation;
          card.dataset.loaded = 'true';
        } catch {
          content.textContent = 'Unable to generate explanation right now.';
          card.dataset.loaded = 'true';
        }
      }
    });

    return card;
  }

  /* ── Chart (dark-theme doughnut) ─────────────────────────────────────────── */
  function updateChart(risks) {
    const ctx = document.getElementById('riskChart');
    if (!ctx) return;

    const catCounts = {};
    risks.forEach(r => { catCounts[r.risk_type] = (catCounts[r.risk_type] || 0) + 1; });
    const labels = Object.keys(catCounts);
    const counts = Object.values(catCounts);

    if (!labels.length) return;
    if (riskChartInstance) riskChartInstance.destroy();

    riskChartInstance = new Chart(ctx.getContext('2d'), {
      type: 'doughnut',
      data: {
        labels,
        datasets: [{
          label: 'Risk Categories',
          data: counts,
          backgroundColor: ['#6366f1', '#f87171', '#fbbf24', '#34d399', '#38bdf8'],
          borderWidth: 3,
          borderColor: '#0b0f1a',
          hoverBorderColor: '#1e2a45',
          hoverOffset: 10,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '68%',
        animation: { animateRotate: true, duration: 800, easing: 'easeInOutQuart' },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              padding: 16,
              font: { size: 11, family: 'Inter, sans-serif' },
              color: '#94a3b8',
              usePointStyle: true,
              pointStyleWidth: 8,
            }
          },
          tooltip: {
            backgroundColor: 'rgba(11,15,26,0.95)',
            borderColor: 'rgba(99,102,241,0.35)',
            borderWidth: 1,
            titleColor: '#f1f5f9',
            bodyColor: '#94a3b8',
            padding: 10,
            callbacks: {
              label: c => ` ${c.label}: ${c.parsed} clause${c.parsed !== 1 ? 's' : ''}`
            }
          }
        }
      }
    });
  }

  /* ── Util ────────────────────────────────────────────────────────────────── */
  function escHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
});
