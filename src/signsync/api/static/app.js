/**
 * SignSync browser client.
 *
 * The interface rule this file follows, from plan §16.3: never present output as
 * more certain than it is. Confidence is always shown, warnings are always
 * rendered, generated motion is labelled as generated, and a feature the server
 * says it lacks is disabled rather than offered and silently broken.
 */

import { AvatarRenderer } from './avatar.js';

const el = (id) => document.getElementById(id);

const renderer = new AvatarRenderer(el('avatar'));
let lastAnimation = null;

async function json(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `request failed: ${response.status}`);
  return body;
}

function renderWarnings(warnings) {
  const list = el('warnings');
  list.innerHTML = '';
  for (const warning of warnings || []) {
    const item = document.createElement('li');
    item.className = `warning warning--${warning.code}`;
    item.textContent = warning.message;
    list.appendChild(item);
  }
}

async function loadStatus() {
  const health = await json('/health');
  el('disclaimer').textContent = health.disclaimer;

  const capabilities = health.capabilities;
  el('capabilities').innerHTML = Object.entries(capabilities)
    .map(
      ([name, ready]) =>
        `<span class="cap ${ready ? 'cap--on' : 'cap--off'}">${name.replace(/_/g, ' ')}</span>`,
    )
    .join('');

  renderWarnings(health.warnings);
  renderer.setRig(await json('/api/rig'));
  renderer.draw();
}

async function updateLatency() {
  try {
    const metrics = await json('/api/metrics');
    const target = metrics.meets_o11 ? 'within' : 'over';
    el('latency').textContent =
      `latency ${metrics.total_p95_ms.toFixed(0)} ms p95 (${target} the 2 s target)` +
      (metrics.bottleneck ? ` — slowest: ${metrics.bottleneck}` : '');
  } catch {
    el('latency').textContent = 'latency —';
  }
}

// ---------------------------------------------------------------- Mode B

el('speak-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = el('speak-input').value.trim();
  if (!text) return;

  try {
    const result = await json('/api/english-to-sign', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    el('notation').textContent = result.notation || '(no signs)';
    renderWarnings(result.warnings);

    // Say plainly which signs are real recordings and which are generated, and
    // which words could not be signed at all (plan §8.7, §16.3).
    const notes = [];
    if (result.missing.length) notes.push(`no sign for: ${result.missing.join(', ')}`);
    if (result.generated.length) notes.push(`generated motion: ${result.generated.join(', ')}`);
    el('motion-note').textContent = notes.join(' · ');

    lastAnimation = result.animation;
    renderer.setAnimation(result.animation);
    el('replay').disabled = result.animation.frames.length === 0;
    renderer.play();
    updateLatency();
  } catch (error) {
    el('motion-note').textContent = error.message;
  }
});

el('replay').addEventListener('click', () => {
  if (!lastAnimation) return;
  renderer.setAnimation(lastAnimation);
  renderer.play();
});

setInterval(() => {
  el('now-signing').textContent = renderer.currentGloss() || '—';
}, 80);

// ---------------------------------------------------------------- Mode A

el('sign-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const glosses = el('sign-input').value.trim().toUpperCase().split(/\s+/).filter(Boolean);
  const markers = [...document.querySelectorAll('.markers input:checked')].map((i) => i.value);

  try {
    const result = await json('/api/sign-to-english', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ glosses, markers, speak: true }),
    });

    el('english').textContent = result.text || '—';
    el('frame-trace').textContent = result.frame;
    renderWarnings(result.warnings);

    const percent = Math.round(result.confidence * 100);
    el('confidence-fill').style.width = `${percent}%`;
    el('confidence-fill').className = percent < 60 ? 'low' : percent < 80 ? 'medium' : 'high';

    let label = `confidence ${percent}%`;
    if (result.needsRepeat) label += ' — ask the signer to repeat';
    if (!result.speech.audible) label += ` — no audio (${result.speech.engine})`;
    el('confidence-label').textContent = label;

    updateLatency();
  } catch (error) {
    el('english').textContent = error.message;
  }
});

loadStatus().catch((error) => {
  el('disclaimer').textContent = `Could not reach the server: ${error.message}`;
});
