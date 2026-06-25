/* ══════════════════════════════════════════════════════════════════════
   DocIntel — Full Application
   Screens: Landing → Ingest → Workspace (DocViewer + Graph + Chat)
   Libraries: D3.js (graph), GSAP (animations)
══════════════════════════════════════════════════════════════════════ */

/* ── HuggingFace Spaces ─────────────────────────────────────────────── */
const SERVER_URL = 'https://tytonterrapin-doc-ingestion.hf.space';
const RAG_URL    = 'https://tytonterrapin-doc-rag.hf.space';

/* ── Global State ────────────────────────────────────────────────────── */
const state = {
  currentScreen: 'landing',
  file:          null,
  lastPayload:   null,
  ragReady:      false,
  chatHistory:   [],
  currentDocName: null,
  graphSim:      null,       // D3 simulation reference
  selectedNodeId: null,
  selectedChunkId: null,
  particlesAnim: null,       // particle animation frame ID
};

/* ── Helpers ─────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const escapeHtml = str => {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
};
const truncate = (str, max) => str.length > max ? str.slice(0, max) + '…' : str;

/* ══════════════════════════════════════════════════════════════════════
   PARTICLES — Canvas constellation background
══════════════════════════════════════════════════════════════════════ */
function initParticles() {
  const canvas = $('particles-canvas');
  const ctx = canvas.getContext('2d');
  let particles = [];
  const PARTICLE_COUNT = 70;
  const CONNECT_DIST = 120;
  let w, h;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  // Create particles
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 1.5 + 0.5,
      opacity: Math.random() * 0.4 + 0.1,
    });
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONNECT_DIST) {
          const alpha = (1 - dist / CONNECT_DIST) * 0.15;
          ctx.beginPath();
          ctx.strokeStyle = `rgba(0, 212, 170, ${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw particles
    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 212, 170, ${p.opacity})`;
      ctx.fill();
    }

    // Move particles
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
    }

    state.particlesAnim = requestAnimationFrame(draw);
  }

  draw();
}

/* ══════════════════════════════════════════════════════════════════════
   SCREEN MANAGEMENT
══════════════════════════════════════════════════════════════════════ */
function showScreen(name) {
  const screens = document.querySelectorAll('.screen');
  const target = $(`screen-${name}`);
  if (!target) return;

  const current = document.querySelector('.screen.active');

  if (current && current !== target) {
    // GSAP transition out
    gsap.to(current, {
      opacity: 0,
      y: -20,
      duration: 0.4,
      ease: 'power2.in',
      onComplete: () => {
        current.classList.remove('active');
        current.style.display = 'none';

        // Show target
        target.style.display = 'flex';
        target.style.opacity = '0';
        target.style.transform = 'translateY(20px)';
        target.classList.add('active');

        gsap.to(target, {
          opacity: 1,
          y: 0,
          duration: 0.5,
          ease: 'power2.out',
          onComplete: () => {
            if (name === 'workspace') {
              onWorkspaceEnter();
            }
          }
        });
      }
    });
  } else if (!current) {
    target.style.display = 'flex';
    target.style.opacity = '1';
    target.classList.add('active');
  }

  state.currentScreen = name;

  // Toggle particles visibility
  if (name === 'workspace') {
    gsap.to('#particles-canvas', { opacity: 0.15, duration: 0.5 });
  } else {
    gsap.to('#particles-canvas', { opacity: 0.6, duration: 0.5 });
  }
}

/* ══════════════════════════════════════════════════════════════════════
   LANDING PAGE ANIMATIONS
══════════════════════════════════════════════════════════════════════ */
function animateLanding() {
  const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });

  tl.from('.hero-badge', { opacity: 0, y: 20, duration: 0.6 })
    .from('.logo-large', { opacity: 0, y: 20, duration: 0.6 }, '-=0.3')
    .from('.hero-title', { opacity: 0, y: 20, duration: 0.6 }, '-=0.3')
    .from('.hero-desc',  { opacity: 0, y: 20, duration: 0.5 }, '-=0.2')
    .from('.space-card', { opacity: 0, y: 30, duration: 0.6, stagger: 0.15 }, '-=0.2')
    .from('.pipeline-section', { opacity: 0, y: 20, duration: 0.5 }, '-=0.3')
    .from('.pipeline-step', { opacity: 0, scale: 0.8, duration: 0.3, stagger: 0.06 }, '-=0.3')
    .from('.landing-footer', { opacity: 0, y: 20, duration: 0.5 }, '-=0.2');
}

/* ══════════════════════════════════════════════════════════════════════
   HEALTH PINGS
══════════════════════════════════════════════════════════════════════ */
async function pingHealth() {
  const badge = $('health-badge');
  const wsHealth = $('ws-health-s1');
  try {
    const r = await fetch(`${SERVER_URL}/health`, { signal: AbortSignal.timeout(6000) });
    const data = await r.json();
    if (data.status === 'ok') {
      badge.className = 'health-pill ok';
      badge.querySelector('.health-text').textContent = `Space 1 online · ${data.uptime_seconds}s`;
      if (wsHealth) wsHealth.className = 'ws-health ok';
    } else throw new Error();
  } catch {
    badge.className = 'health-pill err';
    badge.querySelector('.health-text').textContent = 'Space 1 unreachable';
    if (wsHealth) wsHealth.className = 'ws-health err';
  }
}

async function pingRAGHealth() {
  const badge = $('rag-health-badge');
  const wsHealth = $('ws-health-s2');
  try {
    const r = await fetch(`${RAG_URL}/health`, { signal: AbortSignal.timeout(10000) });
    const data = await r.json();
    if (data.status === 'ok') {
      badge.className = 'health-pill ok';
      badge.querySelector('.health-text').textContent = `RAG online`;
      const ragDocsBadge = $('rag-docs-badge');
      if (ragDocsBadge) ragDocsBadge.textContent = `${data.docs_ingested} doc${data.docs_ingested !== 1 ? 's' : ''} indexed`;
      state.ragReady = data.embedder_ready && data.llm_ready;
      if (wsHealth) wsHealth.className = 'ws-health ok';

      if (!data.embedder_ready || !data.llm_ready) {
        badge.className = 'health-pill warn';
        badge.querySelector('.health-text').textContent = 'RAG warming up…';
        if (wsHealth) wsHealth.className = 'ws-health warn';
      }
    } else throw new Error();
  } catch {
    badge.className = 'health-pill err';
    badge.querySelector('.health-text').textContent = 'RAG unreachable';
    state.ragReady = false;
    if (wsHealth) wsHealth.className = 'ws-health err';
  }
}

/* ══════════════════════════════════════════════════════════════════════
   FILE UPLOAD
══════════════════════════════════════════════════════════════════════ */
function initUpload() {
  const dropZone    = $('drop-zone');
  const fileInput   = $('file-input');
  const runBtn      = $('run-btn');
  const content     = $('drop-zone-content');
  const selected    = $('drop-zone-selected');
  const changeBtn   = $('change-file-btn');

  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });

  changeBtn.addEventListener('click', e => {
    e.preventDefault();
    e.stopPropagation();
    fileInput.value = '';
    if (state.fileUrl) {
      URL.revokeObjectURL(state.fileUrl);
      state.fileUrl = null;
    }
    state.file = null;
    content.style.display = '';
    selected.style.display = 'none';
    runBtn.disabled = true;
  });

  function setFile(f) {
    if (state.fileUrl) {
      URL.revokeObjectURL(state.fileUrl);
      state.fileUrl = null;
    }
    state.file = f;
    state.fileUrl = URL.createObjectURL(f);
    content.style.display = 'none';
    selected.style.display = 'flex';
    $('selected-file-name').textContent = f.name;
    $('selected-file-size').textContent = `${(f.size / 1024).toFixed(1)} KB`;
    runBtn.disabled = false;

    // Animate
    gsap.from(selected, { opacity: 0, scale: 0.95, duration: 0.3, ease: 'back.out(1.7)' });
    gsap.from(runBtn, { opacity: 0, y: 10, duration: 0.3, delay: 0.15 });
  }

  runBtn.addEventListener('click', runPipeline);
}

/* ══════════════════════════════════════════════════════════════════════
   INGESTION PIPELINE
══════════════════════════════════════════════════════════════════════ */
const INGEST_STEPS = [
  { id: 'upload',   text: 'Uploading document…' },
  { id: 'ocr',      text: 'Running OCR & layout analysis…' },
  { id: 'chunk',    text: 'Chunking & structuring content…' },
  { id: 'graph',    text: 'Building knowledge graph…' },
  { id: 'rag',      text: 'Forwarding to RAG for indexing…' },
  { id: 'done',     text: 'Ingestion complete!' },
];

function createStepElements() {
  const container = $('ingest-steps');
  container.innerHTML = '';
  INGEST_STEPS.forEach(s => {
    const el = document.createElement('div');
    el.className = 'ingest-step';
    el.id = `step-${s.id}`;
    el.innerHTML = `
      <div class="step-status-icon"></div>
      <div class="step-text">${s.text}</div>
    `;
    container.appendChild(el);
  });
}

function setStepState(stepId, stepState) {
  const el = $(`step-${stepId}`);
  if (!el) return;
  el.className = `ingest-step ${stepState}`;

  if (stepState === 'active') {
    gsap.from(el, { opacity: 0, x: -10, duration: 0.3 });
  }
}

function setProgressBar(pct) {
  const bar = $('progress-bar');
  if (pct === 'indeterminate') {
    bar.classList.add('indeterminate');
    bar.style.width = '';
  } else {
    bar.classList.remove('indeterminate');
    bar.style.width = pct + '%';
  }
}

function animateCounter(elementId, target) {
  const el = $(elementId);
  if (!el) return;
  const obj = { val: 0 };
  gsap.to(obj, {
    val: target,
    duration: 1,
    ease: 'power2.out',
    onUpdate: () => { el.textContent = Math.round(obj.val); }
  });
}

async function runPipeline() {
  if (!state.file) return;

  const runBtn = $('run-btn');
  const progressPanel = $('ingest-progress');
  const statsPanel = $('ingest-stats');
  const successPanel = $('ingest-success');

  runBtn.disabled = true;
  runBtn.style.display = 'none';
  progressPanel.style.display = 'flex';
  statsPanel.style.display = 'none';
  successPanel.style.display = 'none';

  createStepElements();
  setProgressBar('indeterminate');

  // Simulated step timing for the long API call
  let stepTimers = [];

  try {
    /* Step 1: Upload */
    setStepState('upload', 'active');

    const formData = new FormData();
    formData.append('file', state.file);

    // Start simulated sub-steps during the long wait
    stepTimers.push(setTimeout(() => {
      setStepState('upload', 'done');
      setStepState('ocr', 'active');
    }, 2000));
    stepTimers.push(setTimeout(() => {
      setStepState('ocr', 'done');
      setStepState('chunk', 'active');
    }, 6000));
    stepTimers.push(setTimeout(() => {
      setStepState('chunk', 'done');
      setStepState('graph', 'active');
    }, 10000));

    /* Actual API call to Space 1 */
    const r = await fetch(`${SERVER_URL}/ingest`, { method: 'POST', body: formData });

    // Clear simulated timers
    stepTimers.forEach(clearTimeout);
    stepTimers = [];

    if (!r.ok) {
      const errText = await r.text();
      throw new Error(`Space 1 HTTP ${r.status}: ${errText.slice(0, 200)}`);
    }

    const data = await r.json();
    state.lastPayload = data;

    // Mark all Space 1 steps done
    ['upload', 'ocr', 'chunk', 'graph'].forEach(s => setStepState(s, 'done'));
    setProgressBar(60);

    // Show stats
    statsPanel.style.display = 'grid';
    gsap.from(statsPanel, { opacity: 0, y: 10, duration: 0.4 });

    const totalPages = data.total_pages ?? 0;
    const totalChunks = data.total_chunks ?? data.chunks?.length ?? 0;
    const graphNodes = data.graph?.nodes?.length ?? 0;
    const graphEdges = data.graph?.links?.length ?? 0;

    animateCounter('s-pages', totalPages);
    animateCounter('s-chunks', totalChunks);
    animateCounter('s-nodes', graphNodes);
    animateCounter('s-edges', graphEdges);

    /* Step 2: Forward to RAG */
    setStepState('rag', 'active');
    await forwardToRAG(data);
    setStepState('rag', 'done');

    setProgressBar(100);

    /* Done */
    setStepState('done', 'active');
    await new Promise(r => setTimeout(r, 500));
    setStepState('done', 'done');

    // Show success
    successPanel.style.display = 'flex';
    gsap.from(successPanel, { opacity: 0, scale: 0.9, duration: 0.4, ease: 'back.out(1.7)' });

  } catch (e) {
    stepTimers.forEach(clearTimeout);
    setProgressBar(0);

    // Mark current active step as error
    INGEST_STEPS.forEach(s => {
      const el = $(`step-${s.id}`);
      if (el && el.classList.contains('active')) {
        setStepState(s.id, 'error');
      }
    });

    // Show error in a new step
    const errStep = document.createElement('div');
    errStep.className = 'ingest-step error';
    errStep.innerHTML = `
      <div class="step-status-icon">✗</div>
      <div class="step-text">${escapeHtml(e.message)}</div>
    `;
    $('ingest-steps').appendChild(errStep);
    gsap.from(errStep, { opacity: 0, x: -10, duration: 0.3 });

    runBtn.style.display = '';
    runBtn.disabled = false;
  }
}

/* ══════════════════════════════════════════════════════════════════════
   FORWARD TO RAG (Space 2)
══════════════════════════════════════════════════════════════════════ */
async function forwardToRAG(data) {
  if (!data.chunks || !data.chunks.length) return;

  const docName = state.file?.name ?? 'untitled';
  const payload = {
    doc_name:      docName,
    total_pages:   data.total_pages ?? 0,
    graph:         data.graph ?? { nodes: [], links: [] },
    reading_order: data.reading_order ?? [],
    chunks:        data.chunks.map(c => ({
      chunk_id:      c.chunk_id ?? '',
      doc_name:      docName,
      page_num:      c.page_num ?? 1,
      region_type:   c.region_type ?? 'text',
      text:          c.text ?? '',
      section_title: c.section_title ?? '',
      bbox:          c.bbox ?? [0, 0, 0, 0],
      confidence:    c.confidence ?? 1.0,
      char_count:    c.char_count ?? (c.text || '').length,
      table_html:    c.table_html ?? null,
      figure_path:   c.figure_path ?? null,
    })),
  };

  try {
    const r = await fetch(`${RAG_URL}/ingest`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!r.ok) {
      const errText = await r.text();
      console.warn(`RAG indexing failed: HTTP ${r.status} — ${errText.slice(0, 150)}`);
      return;
    }

    const result = await r.json();
    state.currentDocName = docName;

    const ragDocsBadge = $('rag-docs-badge');
    if (ragDocsBadge) ragDocsBadge.textContent = result.chunks_indexed + ' chunks indexed';

    enableChat();
    pingRAGHealth();

  } catch (e) {
    console.warn(`RAG forwarding error: ${e.message}`);
  }
}

/* ══════════════════════════════════════════════════════════════════════
   WORKSPACE — Entry point
══════════════════════════════════════════════════════════════════════ */
function onWorkspaceEnter() {
  const data = state.lastPayload;
  if (!data) return;

  // Set doc name — use currentDocName for restored docs
  $('ws-doc-name').textContent = state.currentDocName || state.file?.name || 'Document';

  // Hide health strip (workspace has its own)
  gsap.to('#health-strip', { opacity: 0, duration: 0.3 });

  // Render document viewer
  renderDocViewer(data);

  // Render graph
  renderGraph(data.graph, data.chunks);

  // Update health indicators
  pingHealth();
  pingRAGHealth();
}

/* ══════════════════════════════════════════════════════════════════════
   PERSISTED DOCUMENTS — Load, Restore, Delete
══════════════════════════════════════════════════════════════════════ */
async function loadPersistedDocs() {
  const panel    = $('persisted-docs-panel');
  const list     = $('persisted-docs-list');
  const empty    = $('persisted-docs-empty');
  const loading  = $('persisted-docs-loading');
  if (!panel) return;

  // Show loading state
  panel.style.display = 'block';
  list.innerHTML = '';
  empty.style.display = 'none';
  loading.style.display = 'flex';

  try {
    const r = await fetch(`${RAG_URL}/documents`, { signal: AbortSignal.timeout(10000) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();

    loading.style.display = 'none';

    if (!data.documents || data.documents.length === 0) {
      empty.style.display = 'block';
      return;
    }

    data.documents.forEach((doc, idx) => {
      const card = document.createElement('div');
      card.className = 'persisted-doc-card';
      card.style.animationDelay = `${idx * 0.07}s`;
      card.id = `persisted-doc-${idx}`;
      card.dataset.docName = doc.doc_name;

      card.innerHTML = `
        <div class="persisted-doc-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
        </div>
        <div class="persisted-doc-info">
          <div class="persisted-doc-name">${escapeHtml(doc.doc_name)}</div>
          <div class="persisted-doc-meta">
            <span>${doc.chunks} chunk${doc.chunks !== 1 ? 's' : ''}</span>
            <span class="persisted-doc-meta-sep"></span>
            <span>Ready to explore</span>
          </div>
        </div>
        <div class="persisted-doc-actions">
          <button class="persisted-doc-restore-btn" title="Restore workspace">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="1 4 1 10 7 10"/>
              <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
            </svg>
            Restore
          </button>
          <button class="persisted-doc-delete-btn" title="Delete from server">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="3 6 5 6 21 6"/>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
          </button>
        </div>
        <div class="restore-loading">
          <div class="persisted-docs-spinner"></div>
          <span>Restoring…</span>
        </div>
      `;

      // Restore button click
      card.querySelector('.persisted-doc-restore-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        restoreDocument(doc.doc_name, card);
      });

      // Delete button click
      card.querySelector('.persisted-doc-delete-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        deletePersistedDoc(doc.doc_name, card);
      });

      // Clicking the card itself also restores
      card.addEventListener('click', () => {
        restoreDocument(doc.doc_name, card);
      });

      list.appendChild(card);
    });

  } catch (e) {
    loading.style.display = 'none';
    // Don't show panel if RAG server unreachable (not an error for the user)
    panel.style.display = 'none';
    console.warn('Could not load persisted docs:', e.message);
  }
}

async function restoreDocument(docName, cardEl) {
  if (cardEl) cardEl.classList.add('restoring');

  try {
    const r = await fetch(`${RAG_URL}/documents/${encodeURIComponent(docName)}/detail`, {
      signal: AbortSignal.timeout(30000),
    });

    if (!r.ok) {
      const errText = await r.text();
      throw new Error(`HTTP ${r.status}: ${errText.slice(0, 200)}`);
    }

    const data = await r.json();

    // Populate state as if we just ingested
    state.lastPayload    = data;
    state.currentDocName = data.doc_name;
    state.file           = null;  // no local file for restored docs
    state.chatHistory    = [];

    // Enable chat since doc is already indexed in RAG
    enableChat();

    // Transition to workspace
    showScreen('workspace');

  } catch (e) {
    if (cardEl) cardEl.classList.remove('restoring');
    console.error('Failed to restore document:', e);
    alert(`Failed to restore "${docName}": ${e.message}`);
  }
}

async function deletePersistedDoc(docName, cardEl) {
  try {
    const r = await fetch(`${RAG_URL}/documents/${encodeURIComponent(docName)}`, {
      method: 'DELETE',
      signal: AbortSignal.timeout(10000),
    });

    if (!r.ok) {
      const errText = await r.text();
      throw new Error(`HTTP ${r.status}: ${errText.slice(0, 150)}`);
    }

    // Animate removal
    if (cardEl) {
      gsap.to(cardEl, {
        opacity: 0,
        height: 0,
        padding: 0,
        margin: 0,
        duration: 0.3,
        ease: 'power2.in',
        onComplete: () => {
          cardEl.remove();
          // Show empty state if no cards left
          const list = $('persisted-docs-list');
          if (list && list.children.length === 0) {
            $('persisted-docs-empty').style.display = 'block';
          }
        },
      });
    }

    // Refresh RAG health badge
    pingRAGHealth();

  } catch (e) {
    console.error('Failed to delete document:', e);
    alert(`Failed to delete "${docName}": ${e.message}`);
  }
}

/* ══════════════════════════════════════════════════════════════════════
   DOCUMENT VIEWER
══════════════════════════════════════════════════════════════════════ */
async function renderDocViewer(data) {
  const body = $('doc-viewer-body');
  const empty = $('doc-viewer-empty');
  const badge = $('doc-viewer-badge');

  let chunks = data.chunks || [];

  // Handle /ocr endpoint — flatten parsing_blocks
  if (!chunks.length && data.pages_data) {
    chunks = [];
    data.pages_data.forEach((pg, pi) => {
      (pg.parsing_blocks || []).forEach(b => {
        chunks.push({
          chunk_id:      `p${pi + 1}b${b.block_id}`,
          page_num:      pi + 1,
          region_type:   b.block_label || 'text',
          section_title: '',
          text:          b.block_content || '',
          confidence:    b.block_score ?? 1,
          char_count:    (b.block_content || '').length,
          bbox:          b.block_bbox || [0, 0, 0, 0],
        });
      });
    });
  }

  if (!chunks.length) {
    empty.style.display = 'flex';
    body.innerHTML = '';
    body.appendChild(empty);
    return;
  }

  empty.style.display = 'none';
  badge.textContent = `${chunks.length} chunks`;

  // Set container for rendered doc pages
  body.innerHTML = '<div class="doc-viewer-container" id="doc-container"></div>';
  const container = $('doc-container');

  // Lazy tooltip creation
  let tooltip = $('doc-viewer-tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.id = 'doc-viewer-tooltip';
    tooltip.className = 'doc-viewer-tooltip';
    document.body.appendChild(tooltip);
  }

  if (!state.file) {
    renderChunksFallback(chunks, container);
    return;
  }

  const isPdf = state.file.type === 'application/pdf' || state.file.name.toLowerCase().endsWith('.pdf');

  // Group chunks by page
  const chunksByPage = {};
  chunks.forEach(c => {
    const p = c.page_num ?? 1;
    if (!chunksByPage[p]) chunksByPage[p] = [];
    chunksByPage[p].push(c);
  });

  if (isPdf) {
    try {
      container.innerHTML = '<div class="empty-state"><span class="empty-text">Loading PDF pages…</span></div>';
      
      const fileUrl = state.fileUrl || URL.createObjectURL(state.file);
      if (!state.fileUrl) state.fileUrl = fileUrl;

      if (typeof window.pdfjsLib !== 'undefined') {
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
      }

      const loadingTask = pdfjsLib.getDocument(fileUrl);
      const pdf = await loadingTask.promise;
      container.innerHTML = '';

      for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
        const page = await pdf.getPage(pageNum);
        
        // Target container width to responsive fit
        const containerWidth = body.clientWidth - 40;
        const targetWidth = Math.max(Math.min(containerWidth, 600), 380);
        
        const baseViewport = page.getViewport({ scale: 1.0 });
        const scale = targetWidth / baseViewport.width;
        const viewport = page.getViewport({ scale });

        // Page wrapper
        const pageWrapper = document.createElement('div');
        pageWrapper.className = 'doc-page-wrapper';
        pageWrapper.id = `doc-page-p${pageNum}`;
        pageWrapper.style.width = `${viewport.width}px`;
        pageWrapper.style.height = `${viewport.height}px`;

        // Canvas element
        const canvas = document.createElement('canvas');
        canvas.className = 'doc-page-canvas';
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        pageWrapper.appendChild(canvas);

        // Overlay layer for bboxes
        const overlay = document.createElement('div');
        overlay.className = 'doc-page-overlay';
        pageWrapper.appendChild(overlay);

        container.appendChild(pageWrapper);

        // PDF Render
        const context = canvas.getContext('2d');
        const renderContext = {
          canvasContext: context,
          viewport: viewport
        };
        await page.render(renderContext).promise;

        // Render overlays
        const pageChunks = chunksByPage[pageNum] || [];
        pageChunks.forEach((c, idx) => {
          if (!c.bbox || c.bbox.every(v => v === 0)) return;

          const [xmin, ymin, xmax, ymax] = c.bbox;
          const left = xmin * scale;
          const top = ymin * scale;
          const width = (xmax - xmin) * scale;
          const height = (ymax - ymin) * scale;

          const box = document.createElement('div');
          box.className = 'chunk-bbox-overlay';
          box.id = `bbox-overlay-${c.chunk_id || `p${pageNum}-${idx}`}`;
          box.dataset.chunkId = c.chunk_id || '';
          box.style.left = `${left}px`;
          box.style.top = `${top}px`;
          box.style.width = `${width}px`;
          box.style.height = `${height}px`;

          // Tooltip on Hover
          box.addEventListener('mouseenter', (e) => {
            const rect = box.getBoundingClientRect();
            tooltip.style.display = 'block';
            tooltip.style.left = `${window.scrollX + rect.left + rect.width / 2 - tooltip.clientWidth / 2}px`;
            tooltip.style.top = `${window.scrollY + rect.top - tooltip.clientHeight - 10}px`;
            
            const typeColor = nodeColorByType(c.region_type);
            tooltip.innerHTML = `
              <div class="tooltip-type" style="color:${typeColor}">${c.region_type || 'text'}</div>
              <div class="tooltip-text">${escapeHtml(truncate(c.text || '', 180))}</div>
              <div class="tooltip-meta">ID: ${c.chunk_id || 'N/A'} · Page ${c.page_num}</div>
            `;
          });

          box.addEventListener('mouseleave', () => {
            tooltip.style.display = 'none';
          });

          box.addEventListener('click', () => {
            selectBBox(box);
            if (c.chunk_id) {
              highlightGraphNode(c.chunk_id);
            }
          });

          overlay.appendChild(box);
        });
      }
    } catch (err) {
      console.error('Error rendering PDF:', err);
      container.innerHTML = `<div class="empty-state"><span class="empty-text" style="color:#ef4444">Failed to load original PDF: ${escapeHtml(err.message)}</span></div>`;
      renderChunksFallback(chunks, container);
    }
  } else {
    // Render Image
    try {
      const fileUrl = state.fileUrl || URL.createObjectURL(state.file);
      if (!state.fileUrl) state.fileUrl = fileUrl;

      container.innerHTML = '';
      
      const pageWrapper = document.createElement('div');
      pageWrapper.className = 'doc-page-wrapper';
      pageWrapper.id = 'doc-page-image-wrapper';

      const img = document.createElement('img');
      img.className = 'doc-page-image';
      img.src = fileUrl;

      const overlay = document.createElement('div');
      overlay.className = 'doc-page-overlay';

      pageWrapper.appendChild(img);
      pageWrapper.appendChild(overlay);
      container.appendChild(pageWrapper);

      img.onload = () => {
        const naturalWidth = img.naturalWidth;
        const naturalHeight = img.naturalHeight;
        
        setTimeout(() => {
          const displayWidth = img.clientWidth;
          const displayHeight = img.clientHeight;
          const scale = displayWidth / naturalWidth;

          pageWrapper.style.width = `${displayWidth}px`;
          pageWrapper.style.height = `${displayHeight}px`;

          chunks.forEach((c, idx) => {
            if (!c.bbox || c.bbox.every(v => v === 0)) return;

            const [xmin, ymin, xmax, ymax] = c.bbox;
            const left = xmin * scale;
            const top = ymin * scale;
            const width = (xmax - xmin) * scale;
            const height = (ymax - ymin) * scale;

            const box = document.createElement('div');
            box.className = 'chunk-bbox-overlay';
            box.id = `bbox-overlay-${c.chunk_id || `img-${idx}`}`;
            box.dataset.chunkId = c.chunk_id || '';
            box.style.left = `${left}px`;
            box.style.top = `${top}px`;
            box.style.width = `${width}px`;
            box.style.height = `${height}px`;

            box.addEventListener('mouseenter', (e) => {
              const rect = box.getBoundingClientRect();
              tooltip.style.display = 'block';
              tooltip.style.left = `${window.scrollX + rect.left + rect.width / 2 - tooltip.clientWidth / 2}px`;
              tooltip.style.top = `${window.scrollY + rect.top - tooltip.clientHeight - 10}px`;
              
              const typeColor = nodeColorByType(c.region_type);
              tooltip.innerHTML = `
                <div class="tooltip-type" style="color:${typeColor}">${c.region_type || 'text'}</div>
                <div class="tooltip-text">${escapeHtml(truncate(c.text || '', 180))}</div>
                <div class="tooltip-meta">ID: ${c.chunk_id || 'N/A'} · Image</div>
              `;
            });

            box.addEventListener('mouseleave', () => {
              tooltip.style.display = 'none';
            });

            box.addEventListener('click', () => {
              selectBBox(box);
              if (c.chunk_id) {
                highlightGraphNode(c.chunk_id);
              }
            });

            overlay.appendChild(box);
          });
        }, 50);
      };
    } catch (err) {
      console.error('Error rendering image:', err);
      renderChunksFallback(chunks, container);
    }
  }
}

function renderChunksFallback(chunks, container) {
  container.innerHTML = '';
  const pages = {};
  chunks.forEach(c => {
    const pg = c.page_num ?? 1;
    if (!pages[pg]) pages[pg] = [];
    pages[pg].push(c);
  });

  Object.entries(pages).sort((a, b) => a[0] - b[0]).forEach(([pgNum, pgChunks]) => {
    const group = document.createElement('div');
    group.className = 'page-group';

    const header = document.createElement('div');
    header.className = 'page-group-header';
    header.textContent = `Page ${pgNum}`;
    group.appendChild(header);

    pgChunks.forEach((c, i) => {
      const card = document.createElement('div');
      card.className = 'chunk-card';
      card.dataset.chunkId = c.chunk_id || `pg${pgNum}-${i}`;
      card.dataset.nodeId = c.chunk_id || '';

      const typeClass = getTypeClass(c.region_type);

      card.innerHTML = `
        <div class="chunk-card-header">
          <span class="chunk-type-badge ${typeClass}">${c.region_type || 'text'}</span>
          <span class="chunk-id">${c.chunk_id || ''}</span>
        </div>
        ${c.section_title ? `<div class="chunk-section">${escapeHtml(c.section_title)}</div>` : ''}
        <div class="chunk-text-preview">${escapeHtml(truncate(c.text || '', 150))}</div>
      `;

      card.addEventListener('click', () => {
        document.querySelectorAll('.chunk-card.selected').forEach(x => x.classList.remove('selected'));
        card.classList.add('selected');
        state.selectedChunkId = c.chunk_id;
        if (c.chunk_id) highlightGraphNode(c.chunk_id);
      });
      group.appendChild(card);
    });

    container.appendChild(group);

    gsap.from(group.querySelectorAll('.chunk-card'), {
      opacity: 0, x: -10, duration: 0.3, stagger: 0.03, ease: 'power2.out'
    });
  });
}

function selectBBox(box) {
  document.querySelectorAll('.chunk-bbox-overlay.selected').forEach(b => b.classList.remove('selected'));
  box.classList.add('selected');
  state.selectedChunkId = box.dataset.chunkId;
}

function nodeColorByType(type) {
  const colors = {
    page:      getComputedStyle(document.documentElement).getPropertyValue('--node-page').trim() || '#a78bfa',
    paragraph: getComputedStyle(document.documentElement).getPropertyValue('--node-region').trim() || '#3b82f6',
    text:      getComputedStyle(document.documentElement).getPropertyValue('--node-region').trim() || '#3b82f6',
    title:     getComputedStyle(document.documentElement).getPropertyValue('--node-title').trim() || '#f43f5e',
    table:     getComputedStyle(document.documentElement).getPropertyValue('--node-table').trim() || '#10b981',
    figure:    getComputedStyle(document.documentElement).getPropertyValue('--node-figure').trim() || '#f59e0b',
  };
  return colors[type] || colors.text;
}

function getTypeClass(type) {
  const map = {
    paragraph: 'chunk-type-paragraph',
    title:     'chunk-type-title',
    table:     'chunk-type-table',
    figure:    'chunk-type-figure',
    text:      'chunk-type-text',
  };
  return map[type] || 'chunk-type-text';
}

function selectChunk(card) {
  // Deselect all
  document.querySelectorAll('.chunk-card.selected').forEach(c => c.classList.remove('selected'));
  card.classList.add('selected');

  const nodeId = card.dataset.nodeId;
  if (nodeId) highlightGraphNode(nodeId);
  state.selectedChunkId = card.dataset.chunkId;
}

/* ══════════════════════════════════════════════════════════════════════
   GRAPH — D3.js Force-Directed Graph
══════════════════════════════════════════════════════════════════════ */
function renderGraph(graphData, chunks) {
  const container = $('graph-container');
  const svg = d3.select('#graph-svg');
  const empty = $('graph-empty');
  const legend = $('graph-legend');
  const badge = $('graph-badge');

  // Clear previous
  svg.selectAll('*').remove();

  // Build graph data — use provided graph or generate from chunks
  let nodes = [];
  let links = [];

  if (graphData && graphData.nodes && graphData.nodes.length) {
    nodes = graphData.nodes.map(n => {
      const type = n.node_type === 'page' ? 'page' : (n.block_label || 'text');
      return {
        ...n,
        type: type,
        label: n.label || n.block_content || (type === 'page' ? `Page ${n.page_num}` : n.id)
      };
    });
    links = graphData.links.map(l => ({
      ...l,
      type: l.edge_type || 'hierarchy'
    }));
  } else if (chunks && chunks.length) {
    // Generate graph from chunks
    const pageNodes = new Set();
    chunks.forEach(c => {
      const pg = c.page_num ?? 1;
      if (!pageNodes.has(`page_${pg}`)) {
        pageNodes.add(`page_${pg}`);
        nodes.push({ id: `page_${pg}`, label: `Page ${pg}`, type: 'page', page_num: pg });
      }
      nodes.push({
        id: c.chunk_id || `chunk_${nodes.length}`,
        label: truncate(c.section_title || c.text || c.region_type || 'chunk', 30),
        type: c.region_type || 'text',
        page_num: pg,
        text: c.text,
      });
      links.push({
        source: `page_${pg}`,
        target: c.chunk_id || `chunk_${nodes.length - 1}`,
        type: 'hierarchy',
      });
    });

    // Add reading order links
    for (let i = 0; i < chunks.length - 1; i++) {
      if ((chunks[i].page_num ?? 1) === (chunks[i + 1].page_num ?? 1)) {
        links.push({
          source: chunks[i].chunk_id || `chunk_${i}`,
          target: chunks[i + 1].chunk_id || `chunk_${i + 1}`,
          type: 'reading_order',
        });
      }
    }
  }

  if (!nodes.length) {
    empty.style.display = 'flex';
    return;
  }

  empty.style.display = 'none';
  legend.style.display = '';
  badge.textContent = `${nodes.length} nodes · ${links.length} edges`;

  // SVG dimensions
  const rect = container.getBoundingClientRect();
  const width = rect.width || 800;
  const height = rect.height || 600;

  svg.attr('viewBox', [0, 0, width, height]);

  // Pre-position nodes by page in a neat grid to avoid overlapping and bursting
  const pageSet = [...new Set(nodes.map(n => n.page_num || 1))].sort((a, b) => a - b);
  const numPages = pageSet.length;
  
  // Decide grid columns and rows based on number of pages
  const cols = Math.ceil(Math.sqrt(numPages));
  const rows = Math.ceil(numPages / cols);
  
  const pageCenters = {};
  pageSet.forEach((page, index) => {
    const col = index % cols;
    const row = Math.floor(index / cols);
    const cx = numPages === 1 ? width / 2 : ((col + 0.5) / cols) * width;
    const cy = numPages === 1 ? height / 2 : ((row + 0.5) / rows) * height;
    pageCenters[page] = { x: cx, y: cy };
  });

  nodes.forEach(n => {
    const pg = n.page_num || 1;
    const center = pageCenters[pg] || { x: width / 2, y: height / 2 };
    
    if (n.type === 'page') {
      n.x = center.x;
      n.y = center.y;
    } else {
      // Scatter child nodes in a small circle around the page center
      const angle = Math.random() * Math.PI * 2;
      const radius = 30 + Math.random() * 50;
      n.x = center.x + Math.cos(angle) * radius;
      n.y = center.y + Math.sin(angle) * radius;
    }
  });

  // Color scales
  const nodeColor = d => {
    const colors = {
      page:      getComputedStyle(document.documentElement).getPropertyValue('--node-page').trim(),
      paragraph: getComputedStyle(document.documentElement).getPropertyValue('--node-region').trim(),
      text:      getComputedStyle(document.documentElement).getPropertyValue('--node-region').trim(),
      title:     getComputedStyle(document.documentElement).getPropertyValue('--node-title').trim(),
      table:     getComputedStyle(document.documentElement).getPropertyValue('--node-table').trim(),
      figure:    getComputedStyle(document.documentElement).getPropertyValue('--node-figure').trim(),
    };
    return colors[d.type] || colors.text;
  };

  const edgeColor = d => {
    const colors = {
      hierarchy:     getComputedStyle(document.documentElement).getPropertyValue('--edge-hier').trim(),
      reading_order: getComputedStyle(document.documentElement).getPropertyValue('--edge-read').trim(),
      spatial:       getComputedStyle(document.documentElement).getPropertyValue('--edge-spatial').trim(),
    };
    return colors[d.type] || '#3a4060';
  };

  const nodeRadius = d => d.type === 'page' ? 14 : 7;

  // Zoom behavior
  const g = svg.append('g');

  const zoom = d3.zoom()
    .scaleExtent([0.2, 5])
    .on('zoom', e => g.attr('transform', e.transform));

  svg.call(zoom);

  // Force simulation
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(60))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 5))
    .alphaDecay(nodes.length > 100 ? 0.04 : 0.02);

  // Pre-tick the simulation to let nodes settle before rendering (prevents bursting)
  const tickCount = Math.min(300, Math.max(150, Math.ceil(Math.log(nodes.length) * 40)));
  simulation.stop();
  for (let i = 0; i < tickCount; ++i) {
    simulation.tick();
  }

  state.graphSim = simulation;

  // Draw links
  const link = g.append('g')
    .attr('class', 'links')
    .selectAll('line')
    .data(links)
    .join('line')
    .attr('class', d => `graph-link graph-link--${d.type}`)
    .attr('stroke', d => edgeColor(d))
    .attr('stroke-width', 1.2)
    .attr('stroke-dasharray', d => d.type === 'spatial' ? '3,3' : 'none');

  // Draw nodes
  const node = g.append('g')
    .attr('class', 'nodes')
    .selectAll('g')
    .data(nodes)
    .join('g')
    .attr('class', 'graph-node')
    .call(d3.drag()
      .on('start', dragStarted)
      .on('drag', dragged)
      .on('end', dragEnded));

  // Page nodes = larger circles
  node.append('circle')
    .attr('r', d => nodeRadius(d))
    .attr('fill', d => nodeColor(d))
    .attr('stroke', d => nodeColor(d))
    .attr('stroke-width', 1.5)
    .attr('fill-opacity', d => d.type === 'page' ? 0.3 : 0.7);

  // Labels for page nodes
  node.filter(d => d.type === 'page')
    .append('text')
    .attr('class', 'graph-label')
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em')
    .attr('font-size', '9px')
    .attr('fill', '#fff')
    .text(d => d.label || d.id);

  // Tooltip handlers
  const tooltip = $('graph-tooltip');

  node.on('mouseenter', (event, d) => {
    const [x, y] = d3.pointer(event, container);
    tooltip.style.display = 'block';
    tooltip.style.left = (x + 16) + 'px';
    tooltip.style.top = (y - 10) + 'px';
    tooltip.innerHTML = `
      <div class="tooltip-type" style="color:${nodeColor(d)}">${d.type || 'node'}</div>
      <div class="tooltip-label">${escapeHtml(d.label || d.id)}</div>
      ${d.text ? `<div class="tooltip-detail">${escapeHtml(truncate(d.text, 120))}</div>` : ''}
      ${d.page_num ? `<div class="tooltip-detail">Page ${d.page_num}</div>` : ''}
    `;
  })
  .on('mousemove', (event) => {
    const [x, y] = d3.pointer(event, container);
    tooltip.style.left = (x + 16) + 'px';
    tooltip.style.top = (y - 10) + 'px';
  })
  .on('mouseleave', () => {
    tooltip.style.display = 'none';
  })
  .on('click', (event, d) => {
    selectGraphNode(d.id, node, link);
    // Highlight corresponding chunk in doc viewer
    highlightDocChunk(d.id);
  });

  // Set initial position of nodes and links from pre-ticked layout
  link
    .attr('x1', d => d.source.x)
    .attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x)
    .attr('y2', d => d.target.y);

  node.attr('transform', d => `translate(${d.x},${d.y})`);

  // Tick listener (for when simulation is restarted on drag)
  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  // Drag handlers
  function dragStarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }
  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }
  function dragEnded(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  // Edge filter toggles
  document.querySelectorAll('#graph-filters input[data-edge-type]').forEach(input => {
    input.addEventListener('change', () => {
      const type = input.dataset.edgeType;
      const visible = input.checked;
      link.filter(d => d.type === type)
        .classed('hidden', !visible);
    });
  });

  // Entrance animation — nodes fly in
  const isLargeGraph = nodes.length > 50;
  gsap.from(node.nodes(), {
    attr: { opacity: 0 },
    duration: isLargeGraph ? 0.3 : 0.5,
    stagger: isLargeGraph ? 0.002 : 0.02,
    ease: 'power2.out',
  });
}

function selectGraphNode(nodeId, nodeSelection, linkSelection) {
  state.selectedNodeId = nodeId;

  if (nodeSelection) {
    nodeSelection.classed('selected', d => d.id === nodeId);
  }
  if (linkSelection) {
    linkSelection.classed('highlighted', d => d.source.id === nodeId || d.target.id === nodeId);
  }
}

function highlightGraphNode(nodeId) {
  const svg = d3.select('#graph-svg');
  svg.selectAll('.graph-node').classed('selected', d => d.id === nodeId);
  svg.selectAll('.graph-link').classed('highlighted', d =>
    d.source.id === nodeId || d.target.id === nodeId
  );
  state.selectedNodeId = nodeId;
}

function highlightDocChunk(chunkId) {
  const card = document.querySelector(`.chunk-card[data-chunk-id="${chunkId}"]`);
  if (card) {
    document.querySelectorAll('.chunk-card.highlighted').forEach(c => c.classList.remove('highlighted'));
    card.classList.add('highlighted');
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    return;
  }

  const overlay = document.querySelector(`.chunk-bbox-overlay[data-chunk-id="${chunkId}"]`);
  if (overlay) {
    document.querySelectorAll('.chunk-bbox-overlay.selected').forEach(x => x.classList.remove('selected'));
    document.querySelectorAll('.chunk-bbox-overlay.highlighted').forEach(x => x.classList.remove('highlighted'));
    
    overlay.classList.add('highlighted');
    overlay.classList.add('selected');
    state.selectedChunkId = chunkId;

    const viewerBody = $('doc-viewer-body');
    const pageWrapper = overlay.closest('.doc-page-wrapper');
    if (pageWrapper && viewerBody) {
      const rectOverlay = overlay.getBoundingClientRect();
      const rectViewer = viewerBody.getBoundingClientRect();
      const scrollTop = viewerBody.scrollTop + rectOverlay.top - rectViewer.top - (rectViewer.height / 2) + (rectOverlay.height / 2);
      
      viewerBody.scrollTo({
        top: scrollTop,
        behavior: 'smooth'
      });
    }

    overlay.classList.remove('flash');
    void overlay.offsetWidth;
    overlay.classList.add('flash');
  }
}

/* ══════════════════════════════════════════════════════════════════════
   CHAT — RAG Q&A
══════════════════════════════════════════════════════════════════════ */
function enableChat() {
  const chatInput = $('chat-input');
  const chatSendBtn = $('chat-send-btn');
  chatInput.disabled = false;
  chatSendBtn.disabled = false;
  chatInput.placeholder = 'Ask a question about the document…';
  const chatEmpty = $('chat-empty');
  if (chatEmpty) chatEmpty.style.display = state.chatHistory.length ? 'none' : 'flex';
}

function disableChat() {
  $('chat-input').disabled = true;
  $('chat-send-btn').disabled = true;
}

function initChat() {
  const chatInput = $('chat-input');
  const chatSendBtn = $('chat-send-btn');
  const chatSourcesClose = $('chat-sources-close');

  chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuestion();
    }
  });
  chatSendBtn.addEventListener('click', sendQuestion);
  chatSourcesClose.addEventListener('click', () => {
    $('chat-sources').style.display = 'none';
  });
}

async function sendQuestion() {
  const chatInput = $('chat-input');
  const question = chatInput.value.trim();
  if (!question || question.length < 3) return;

  chatInput.value = '';
  const chatEmpty = $('chat-empty');
  if (chatEmpty) chatEmpty.style.display = 'none';

  addChatMessage('user', question);
  const thinkingEl = addChatMessage('assistant', '', true);

  disableChat();

  try {
    const body = {
      question,
      doc_name: state.currentDocName || undefined,
      top_k: 5,
    };

    const r = await fetch(`${RAG_URL}/query/stream`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });

    if (!r.ok) {
      const errText = await r.text();
      throw new Error(`HTTP ${r.status}: ${errText.slice(0, 200)}`);
    }

    if (thinkingEl && thinkingEl.parentNode) thinkingEl.remove();

    // Create an empty chat bubble for streaming response
    const wrapper = addChatMessage('assistant', '');
    const bubble = wrapper.querySelector('.chat-bubble');

    const reader = r.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let answerText = '';
    let retrievedChunks = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep last partial line

      for (const line of lines) {
        const cleaned = line.trim();
        if (!cleaned.startsWith('data: ')) continue;
        const dataStr = cleaned.slice(6);
        if (dataStr === '[DONE]') break;

        try {
          const parsed = JSON.parse(dataStr);
          if (parsed.type === 'chunks') {
            retrievedChunks = parsed.chunks;
          } else if (parsed.type === 'token') {
            answerText += parsed.text;
            // Format on-the-fly (bolding and line breaks)
            const formatted = escapeHtml(answerText)
              .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
              .replace(/\n/g, '<br>');
            bubble.innerHTML = formatted;

            // Auto-scroll chat
            const chatMessages = $('chat-messages');
            if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
          }
        } catch (err) {
          console.warn('Failed to parse SSE JSON:', dataStr, err);
        }
      }
    }

    // Finished streaming — append sources button if we retrieved chunks
    if (retrievedChunks && retrievedChunks.length) {
      const srcBtn = document.createElement('button');
      srcBtn.className = 'chat-sources-btn';
      srcBtn.textContent = `Sources (${retrievedChunks.length})`;
      srcBtn.addEventListener('click', () => showSources(retrievedChunks));
      wrapper.appendChild(srcBtn);
    }

    // Save to state history
    state.chatHistory.push({ role: 'user', content: question });
    state.chatHistory.push({ role: 'assistant', content: answerText, sources: retrievedChunks });

  } catch (e) {
    if (thinkingEl && thinkingEl.parentNode) thinkingEl.remove();
    addChatMessage('assistant', `Error: ${e.message}`, false, null, true);
  }

  enableChat();
  chatInput.focus();
}

function addChatMessage(role, content, isThinking = false, sources = null, isError = false) {
  const chatMessages = $('chat-messages');
  const wrapper = document.createElement('div');
  wrapper.className = `chat-msg chat-msg--${role}`;

  if (isThinking) {
    wrapper.classList.add('chat-msg--thinking');
    wrapper.innerHTML = `
      <div class="chat-bubble chat-bubble--assistant">
        <div class="thinking-dots">
          <span></span><span></span><span></span>
        </div>
      </div>`;
    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return wrapper;
  }

  const bubble = document.createElement('div');
  bubble.className = `chat-bubble chat-bubble--${role}`;
  if (isError) bubble.classList.add('chat-bubble--error');

  const formatted = escapeHtml(content)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
  bubble.innerHTML = formatted;
  wrapper.appendChild(bubble);

  if (sources && sources.length) {
    const srcBtn = document.createElement('button');
    srcBtn.className = 'chat-sources-btn';
    srcBtn.textContent = `Sources (${sources.length})`;
    srcBtn.addEventListener('click', () => showSources(sources));
    wrapper.appendChild(srcBtn);
  }

  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrapper;
}

function showSources(chunks) {
  const chatSourcesBody = $('chat-sources-body');
  chatSourcesBody.innerHTML = '';

  chunks.forEach((c, i) => {
    const card = document.createElement('div');
    card.className = 'source-card';
    const typeClass = getTypeClass(c.region_type);
    card.innerHTML = `
      <div class="source-header">
        <span class="source-rank">#${i + 1}</span>
        <span class="chunk-type-badge ${typeClass}">${c.region_type || 'text'}</span>
        <span class="source-meta">p.${c.page_num} · score ${c.score?.toFixed(3) ?? '—'}</span>
      </div>
      ${c.section_title ? `<div class="source-section">${escapeHtml(c.section_title)}</div>` : ''}
      <div class="source-text">${escapeHtml(c.text || '')}</div>
    `;

    // Click source → highlight in graph and doc viewer
    card.addEventListener('click', () => {
      if (c.chunk_id) {
        highlightGraphNode(c.chunk_id);
        highlightDocChunk(c.chunk_id);
      }
    });

    chatSourcesBody.appendChild(card);
  });

  $('chat-sources').style.display = 'flex';

  gsap.from(chatSourcesBody.children, {
    opacity: 0, y: 8, duration: 0.25, stagger: 0.05
  });
}

/* ══════════════════════════════════════════════════════════════════════
   NAVIGATION EVENTS
══════════════════════════════════════════════════════════════════════ */
function initNavigation() {
  // Landing → Ingest
  $('cta-btn').addEventListener('click', () => showScreen('ingest'));

  // Ingest → Landing (back)
  $('back-to-landing').addEventListener('click', () => showScreen('landing'));

  // Ingest success → Workspace
  $('view-workspace-btn').addEventListener('click', () => showScreen('workspace'));

  // Download parsed JSON payload
  $('ws-download-json').addEventListener('click', () => {
    if (!state.lastPayload) {
      alert("No document payload loaded to export.");
      return;
    }
    try {
      const jsonStr = JSON.stringify(state.lastPayload, null, 2);
      const blob = new Blob([jsonStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${state.currentDocName || 'document'}_parsed.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to export JSON:", err);
      alert("An error occurred while generating the JSON file.");
    }
  });

  // Workspace → Ingest (new doc)
  $('ws-new-doc').addEventListener('click', () => {
    // Reset state
    if (state.fileUrl) {
      URL.revokeObjectURL(state.fileUrl);
      state.fileUrl = null;
    }
    state.file = null;
    state.lastPayload = null;
    state.selectedNodeId = null;
    state.selectedChunkId = null;

    // Reset ingest screen
    const content = $('drop-zone-content');
    const selected = $('drop-zone-selected');
    const runBtn = $('run-btn');
    const progress = $('ingest-progress');
    const success = $('ingest-success');

    content.style.display = '';
    selected.style.display = 'none';
    runBtn.style.display = '';
    runBtn.disabled = true;
    progress.style.display = 'none';
    success.style.display = 'none';
    $('file-input').value = '';

    // Show health strip again
    gsap.to('#health-strip', { opacity: 1, duration: 0.3 });

    showScreen('ingest');
  });
}

/* ══════════════════════════════════════════════════════════════════════
   KEEP-ALIVE — ping both spaces every 4 min
══════════════════════════════════════════════════════════════════════ */
const KEEP_ALIVE_MS = 4 * 60 * 1000;
let keepAliveTimer = null;

async function keepAlivePing() {
  try {
    const r1 = fetch(`${SERVER_URL}/health`, { signal: AbortSignal.timeout(8000) });
    const r2 = fetch(`${RAG_URL}/health`,    { signal: AbortSignal.timeout(8000) });
    await Promise.allSettled([r1, r2]);
  } catch {
    // silent
  }
}

function startKeepAlive() {
  if (keepAliveTimer) return;
  keepAlivePing();
  keepAliveTimer = setInterval(keepAlivePing, KEEP_ALIVE_MS);
}

function stopKeepAlive() {
  if (!keepAliveTimer) return;
  clearInterval(keepAliveTimer);
  keepAliveTimer = null;
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopKeepAlive();
  else startKeepAlive();
});

/* ══════════════════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initNavigation();
  initUpload();
  initChat();
  animateLanding();
  pingHealth();
  pingRAGHealth();
  startKeepAlive();
  loadPersistedDocs();
});