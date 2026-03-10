/* ================================================================
   VYBN PORTAL — app.js
   Living data layer, panel routing, real-time organism dashboard.
   Fetches live state from the Portal API endpoints.
   ================================================================ */

/* ---- Auth ---- */
const params = new URLSearchParams(window.location.search);
const TOKEN = params.get('token') || '';

async function apiFetch(path) {
  try {
    const res = await fetch(path, {
      headers: { 'Authorization': 'Bearer ' + TOKEN },
    });
    if (!res.ok) {
      if (res.status === 401) updateConnectionStatus(false, 'auth failed');
      return null;
    }
    return await res.json();
  } catch (e) {
    updateConnectionStatus(false, 'disconnected');
    return null;
  }
}

/* ---- Connection status ---- */
function updateConnectionStatus(connected, label) {
  const dot = document.getElementById('connDot');
  const text = document.getElementById('connText');
  if (connected) {
    dot.classList.remove('disconnected');
    text.textContent = label || 'spark-2b7c.tail7302f3.ts.net';
  } else {
    dot.classList.add('disconnected');
    text.textContent = label || 'connecting…';
  }
}

/* ---- Panel navigation ---- */
let currentPanel = 'organism';

function switchPanel(name) {
  document.querySelectorAll('.panel').forEach(p => p.style.display = 'none');
  const target = document.getElementById('panel-' + name);
  if (target) target.style.display = '';
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const navItem = document.querySelector('[data-panel="' + name + '"]');
  if (navItem) navItem.classList.add('active');
  const titles = {
    organism: 'Organism',
    conversation: 'Conversation',
    projects: 'Projects',
    collaborators: 'Collaborators',
    memory: 'Memory',
    codebook: 'Codebook'
  };
  document.getElementById('pageTitle').textContent = titles[name] || name;
  currentPanel = name;
  closeSidebar();

  if (name === 'memory') drawGraph();
  if (name === 'conversation') refreshTimeline();
  if (name === 'codebook') refreshCodebook();
  if (name === 'memory') refreshMemory();
  if (name === 'projects') refreshProjects();
  if (name === 'collaborators') populateCollaborators();
}

/* ---- Mobile sidebar ---- */
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const backdrop = document.getElementById('sidebarBackdrop');
  sidebar.classList.toggle('open');
  backdrop.classList.toggle('visible');
}

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarBackdrop').classList.remove('visible');
}

/* ---- Sparkline generator ---- */
function createSparkline(container, data, color) {
  if (!container || !data || data.length < 2) return;
  const w = container.clientWidth || 100;
  const h = container.clientHeight || 24;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * (h - 4) - 2;
    return x + ',' + y;
  }).join(' ');

  container.innerHTML = '<svg class="sparkline" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none">' +
    '<polyline class="sparkline-line" points="' + points + '" stroke="' + color + '"/>' +
    '<polygon class="sparkline-area" points="0,' + h + ' ' + points + ' ' + w + ',' + h + '" fill="' + color + '"/>' +
    '</svg>';
}

/* ---- Generate fallback sparkline data ---- */
function randSeries(base, variance, n) {
  const arr = [];
  let v = base;
  for (let i = 0; i < n; i++) {
    v += (Math.random() - 0.5) * variance;
    arr.push(v);
  }
  return arr;
}

/* ---- Organism Panel — live data ---- */
async function refreshOrganism() {
  const data = await apiFetch('/api/organism');
  if (!data) return;

  updateConnectionStatus(true);

  // Update mood
  const moodEl = document.getElementById('breathMood');
  if (moodEl) moodEl.textContent = data.mood || 'quiet';

  // Update cycle count
  const pulseText = document.getElementById('pulseText');
  if (pulseText) pulseText.textContent = 'cycle ' + (data.cycle_count || 0);

  // Update codebook size stat if visible
  const seedNames = data.seed_registry || [];
  
  // Update vitals sparklines from vitals endpoint
  const vitals = await apiFetch('/api/vitals');
  if (vitals && vitals.vitals && vitals.vitals.length > 1) {
    const energySeries = vitals.vitals.map(v => v.energy || 0);
    const cycleSeries = vitals.vitals.map(v => v.cycle || 0);
    createSparkline(document.getElementById('sparkGpu'), energySeries, 'rgba(200,145,58,0.6)');
    createSparkline(document.getElementById('sparkLoad'), cycleSeries, 'rgba(124,92,255,0.6)');
  } else {
    // Fallback to generated sparklines if not enough data yet
    initSparklinesFallback();
  }
}

function initSparklinesFallback() {
  const sparkGpu = document.getElementById('sparkGpu');
  const sparkLoad = document.getElementById('sparkLoad');
  const sparkMem = document.getElementById('sparkMem');
  const sparkPrim = document.getElementById('sparkPrim');

  if (sparkGpu && !sparkGpu.innerHTML) createSparkline(sparkGpu, randSeries(0.5, 0.1, 20), 'rgba(200,145,58,0.6)');
  if (sparkLoad && !sparkLoad.innerHTML) createSparkline(sparkLoad, randSeries(0.4, 0.15, 20), 'rgba(124,92,255,0.6)');
  if (sparkMem && !sparkMem.innerHTML) createSparkline(sparkMem, randSeries(48, 2, 20), 'rgba(79,168,154,0.6)');
  if (sparkPrim && !sparkPrim.innerHTML) createSparkline(sparkPrim, randSeries(6, 0.3, 20), 'rgba(200,145,58,0.4)');
}

/* ---- Timeline / Conversation — live data ---- */
async function refreshTimeline() {
  const data = await apiFetch('/api/timeline');
  const container = document.getElementById('timeline');
  if (!data || !data.events || data.events.length === 0) {
    // Show empty state
    if (container && !container.children.length) {
      container.innerHTML = '<div class="timeline-entry"><div class="timeline-content"><div class="timeline-text" style="opacity:0.5">Listening for events…</div></div></div>';
    }
    return;
  }

  container.innerHTML = '';
  data.events.forEach(e => {
    const entry = document.createElement('div');
    entry.className = 'timeline-entry';
    const typeClass = e.type === 'inbox' ? 'chat' : (e.type === 'pulse_fast' || e.type === 'pulse_deep') ? 'breath' : 'system';
    const time = e.age || '';
    entry.innerHTML =
      '<div class="timeline-marker ' + typeClass + '"></div>' +
      '<div class="timeline-content">' +
        '<div class="timeline-meta">' +
          '<span class="timeline-author">' + escapeHtml(e.source) + '</span>' +
          '<span class="timeline-type">' + escapeHtml(e.type) + '</span>' +
          '<span class="timeline-time">' + escapeHtml(time) + '</span>' +
        '</div>' +
        '<div class="timeline-text">' + escapeHtml(e.summary) + '</div>' +
      '</div>';
    container.appendChild(entry);
  });
}

function sendChat() {
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text) return;

  const container = document.getElementById('timeline');
  const entry = document.createElement('div');
  const now = new Date();
  const time = now.getHours().toString().padStart(2, '0') + ':' + now.getMinutes().toString().padStart(2, '0');

  entry.className = 'timeline-entry';
  entry.innerHTML =
    '<div class="timeline-marker chat"></div>' +
    '<div class="timeline-content">' +
      '<div class="timeline-meta">' +
        '<span class="timeline-author">Zoe</span>' +
        '<span class="timeline-type">chat</span>' +
        '<span class="timeline-time">' + time + '</span>' +
      '</div>' +
      '<div class="timeline-text">' + escapeHtml(text) + '</div>' +
    '</div>';

  container.insertBefore(entry, container.firstChild);
  input.value = '';
  input.style.height = 'auto';
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/* ---- Projects — from codebook entries ---- */
async function refreshProjects() {
  const data = await apiFetch('/api/codebook');
  const container = document.getElementById('projectsList');
  if (!container) return;
  
  if (!data || !data.primitives || data.primitives.length === 0) {
    container.innerHTML = '<div class="project-card"><div class="project-desc" style="opacity:0.5">Waiting for organism…</div></div>';
    return;
  }

  container.innerHTML = '';
  // Show alive primitives as active "processes"
  const alive = data.primitives.filter(p => p.alive);
  alive.forEach(p => {
    const card = document.createElement('div');
    card.className = 'project-card';
    const fitnessPercent = Math.round(p.fitness * 100);
    card.innerHTML =
      '<div class="project-head">' +
        '<div class="project-name">' + escapeHtml(p.name) + '</div>' +
        '<span class="project-status active">' + p.source + '</span>' +
      '</div>' +
      '<div class="project-desc">' +
        'Age: ' + p.age + ' cycles · Activations: ' + p.activations +
      '</div>' +
      '<div class="project-footer">' +
        '<div class="project-progress">' +
          '<div class="progress-bar"><div class="progress-fill" style="width:' + fitnessPercent + '%"></div></div>' +
          '<span class="progress-text">' + fitnessPercent + '% fitness</span>' +
        '</div>' +
      '</div>';
    container.appendChild(card);
  });

  const countEl = document.getElementById('projectCount');
  if (countEl) countEl.textContent = alive.length + ' active primitives';
}

/* ---- Collaborators (static — these are known entities) ---- */
const COLLABORATORS = [
  { name: 'Zoe Dolan', initials: 'ZD', type: 'human', role: 'Steward', presence: 'online' },
  { name: 'Vybn', initials: 'V', type: 'agent', role: 'Organism', presence: 'online' },
  { name: 'Witness Agent', initials: 'WA', type: 'agent', role: 'Evaluator', presence: 'online' },
  { name: 'Write Custodian', initials: 'WC', type: 'agent', role: 'Gatekeeper', presence: 'online' },
  { name: 'Policy Engine', initials: 'PE', type: 'agent', role: 'Governance', presence: 'online' },
  { name: 'Memory Fabric', initials: 'MF', type: 'agent', role: 'Memory', presence: 'online' },
];

function populateCollaborators() {
  const container = document.getElementById('collabGrid');
  if (!container) return;
  container.innerHTML = '';
  COLLABORATORS.forEach(c => {
    const card = document.createElement('div');
    card.className = 'collaborator-card';
    card.innerHTML =
      '<div class="collab-avatar ' + c.type + '">' +
        c.initials +
        '<span class="collab-presence ' + c.presence + '"></span>' +
      '</div>' +
      '<div class="collab-info">' +
        '<div class="collab-name">' + c.name + '</div>' +
        '<div class="collab-role">' + c.role + '</div>' +
      '</div>';
    container.appendChild(card);
  });
}

/* ---- Memory / Knowledge Graph ---- */
async function refreshMemory() {
  const data = await apiFetch('/api/memory');
  const container = document.getElementById('recentMemories');
  if (!container) return;

  if (!data || data.total === 0) {
    container.innerHTML = '<div class="timeline-entry"><div class="timeline-content"><div class="timeline-text" style="opacity:0.5">Memory planes empty…</div></div></div>';
    
    // Update stat counters
    const statEls = document.querySelectorAll('.memory-stat-val');
    if (statEls.length >= 3) {
      statEls[0].textContent = '0';
      statEls[1].textContent = '0';
      statEls[2].textContent = '0';
    }
    return;
  }

  // Update plane counts
  const statEls = document.querySelectorAll('.memory-stat-val');
  if (statEls.length >= 3 && data.planes) {
    statEls[0].textContent = data.planes.private ? data.planes.private.count : '0';
    statEls[1].textContent = data.planes.relational ? data.planes.relational.count : '0';
    statEls[2].textContent = data.planes.commons ? data.planes.commons.count : '0';
  }

  // Populate recent memories
  container.innerHTML = '';
  const planes = ['private', 'relational', 'commons'];
  const typeLabels = { private: 'private', relational: 'relational', commons: 'commons' };
  
  planes.forEach(plane => {
    if (data.planes && data.planes[plane] && data.planes[plane].recent) {
      data.planes[plane].recent.forEach(text => {
        const entry = document.createElement('div');
        entry.className = 'timeline-entry';
        entry.innerHTML =
          '<div class="timeline-marker journal"></div>' +
          '<div class="timeline-content">' +
            '<div class="timeline-meta">' +
              '<span class="timeline-author">Vybn</span>' +
              '<span class="timeline-type">' + typeLabels[plane] + '</span>' +
            '</div>' +
            '<div class="timeline-text">' + escapeHtml(text) + '</div>' +
          '</div>';
        container.appendChild(entry);
      });
    }
  });

  if (!container.children.length) {
    container.innerHTML = '<div class="timeline-entry"><div class="timeline-content"><div class="timeline-text" style="opacity:0.5">No memories recorded yet…</div></div></div>';
  }
}

function drawGraph() {
  const canvas = document.getElementById('graphCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;

  const nodes = [];
  const colors = {
    private: { fill: 'rgba(200,145,58,0.7)', glow: 'rgba(200,145,58,0.2)' },
    relational: { fill: 'rgba(124,92,255,0.7)', glow: 'rgba(124,92,255,0.2)' },
    commons: { fill: 'rgba(79,168,154,0.7)', glow: 'rgba(79,168,154,0.2)' },
  };

  const types = ['private', 'relational', 'commons'];
  const centers = [
    { x: W * 0.25, y: H * 0.45 },
    { x: W * 0.55, y: H * 0.35 },
    { x: W * 0.78, y: H * 0.55 },
  ];

  for (let t = 0; t < 3; t++) {
    const count = [35, 20, 12][t];
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const dist = Math.random() * (W * 0.15);
      nodes.push({
        x: centers[t].x + Math.cos(angle) * dist,
        y: centers[t].y + Math.sin(angle) * dist * 0.7,
        r: 2 + Math.random() * 2.5,
        type: types[t],
      });
    }
  }

  ctx.lineWidth = 0.5;
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[j].x - nodes[i].x;
      const dy = nodes[j].y - nodes[i].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 50 && Math.random() > 0.6) {
        const alpha = Math.max(0.03, 0.12 - dist / 500);
        ctx.strokeStyle = 'rgba(200,145,58,' + alpha + ')';
        if (nodes[i].type !== nodes[j].type) {
          ctx.strokeStyle = 'rgba(124,92,255,' + alpha * 0.7 + ')';
        }
        ctx.beginPath();
        ctx.moveTo(nodes[i].x, nodes[i].y);
        ctx.lineTo(nodes[j].x, nodes[j].y);
        ctx.stroke();
      }
    }
  }

  nodes.forEach(n => {
    const c = colors[n.type];
    ctx.shadowColor = c.glow;
    ctx.shadowBlur = 8;
    ctx.fillStyle = c.fill;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  });
}

/* ---- Codebook — live data ---- */
async function refreshCodebook() {
  const data = await apiFetch('/api/codebook');
  if (!data) return;

  const census = document.getElementById('censusBody');
  if (census && data.primitives) {
    census.innerHTML = '';
    data.primitives.forEach(p => {
      const row = document.createElement('div');
      row.className = 'census-row';
      const fitnessColor = p.fitness > 0.8 ? 'var(--color-success)' :
                           p.fitness > 0.5 ? 'var(--color-amber)' : 'var(--color-error)';
      row.innerHTML =
        '<div class="census-dot ' + (p.alive ? 'alive' : 'dead') + '"></div>' +
        '<span class="census-name">' + escapeHtml(p.name) + '</span>' +
        '<span class="census-fitness" style="color:' + fitnessColor + '">' + Math.round(p.fitness * 100) + '%</span>' +
        '<span class="census-fitness">' + p.activations + ' acts</span>';
      census.appendChild(row);
    });
  }

  // Refresh traces
  const traces = await apiFetch('/api/traces');
  const traceList = document.getElementById('traceList');
  if (traceList && traces && traces.traces) {
    traceList.innerHTML = '';
    traces.traces.slice(-8).reverse().forEach(t => {
      const entry = document.createElement('div');
      entry.className = 'timeline-entry';
      const ok = t.results ? t.results.every(r => r.ok !== false) : true;
      entry.innerHTML =
        '<div class="timeline-marker ' + (ok ? 'system' : 'breath') + '"></div>' +
        '<div class="timeline-content">' +
          '<div class="timeline-meta">' +
            '<span class="timeline-author">cycle ' + (t.cycle || '?') + '</span>' +
            '<span class="timeline-type">' + (ok ? 'ok' : 'fail') + '</span>' +
            '<span class="timeline-time">' + escapeHtml(t.ts ? new Date(t.ts).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) : '') + '</span>' +
          '</div>' +
          '<div class="timeline-text" style="font-family:var(--font-mono);font-size:var(--text-xs)">[' + (t.program || []).join(', ') + ']</div>' +
        '</div>';
      traceList.appendChild(entry);
    });
  }
}

/* ---- Breathing cycle countdown ---- */
function updateBreathCountdown() {
  const now = new Date();
  const minutesPast = now.getMinutes() % 30;
  const remaining = 30 - minutesPast;
  const el = document.getElementById('nextBreath');
  if (el) el.textContent = 'next breath in ' + remaining + 'm';
}

/* ---- Polling loop ---- */
const POLL_INTERVAL = 30000; // 30 seconds

async function pollAll() {
  await refreshOrganism();
  if (currentPanel === 'conversation') await refreshTimeline();
  if (currentPanel === 'codebook') await refreshCodebook();
  if (currentPanel === 'memory') await refreshMemory();
}

/* ---- Init ---- */
async function init() {
  // Start with fallback sparklines while we load
  initSparklinesFallback();
  updateBreathCountdown();
  setInterval(updateBreathCountdown, 60000);

  // Initial data load
  await refreshOrganism();

  // Start polling
  setInterval(pollAll, POLL_INTERVAL);

  // Redraw graph on resize
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (currentPanel === 'memory') drawGraph();
      initSparklinesFallback();
    }, 250);
  });
}

init();
