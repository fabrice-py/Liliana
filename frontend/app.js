/* Liliana — interface locale.
 *
 * Le micro et la détection de fin de phrase (VAD) vivent ici, dans le
 * navigateur : c'est là que se trouve le micro, et détecter le silence côté
 * client évite un aller-retour réseau par fragment audio.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const api = (path) => `/api${path}`;

const state = {
  config: null,
  language: 'english',
  mode: 'free_conversation',
  correctionMode: 'normal',
  speed: 1.0,
  recording: false,
  busy: false,
  lastResponse: '',
  vad: { silence_threshold: 0.8, energy_threshold: 0.015, min_speech_duration: 0.3 },
};

const media = { stream: null, recorder: null, chunks: [], audioContext: null, analyser: null, raf: 0 };

/* ------------------------------------------------------------ utilitaires */

function toast(message, isError = true) {
  const element = $('toast');
  element.textContent = message;
  element.style.borderLeftColor = isError ? 'var(--err)' : 'var(--ok)';
  element.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { element.hidden = true; }, 6500);
}

/** Extrait le message lisible renvoyé par l'API (voir app/core/exceptions.py). */
async function readError(response) {
  try {
    const body = await response.json();
    const detail = body.detail ?? body;
    if (typeof detail === 'string') return detail;
    return detail.message || detail.msg || JSON.stringify(detail);
  } catch {
    return `Request failed (HTTP ${response.status}).`;
  }
}

async function request(path, options = {}) {
  const response = await fetch(api(path), options);
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

const postJSON = (path, payload) => request(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
});

function setStatus(text) { $('pipeline-status').textContent = text; }

/* ------------------------------------------------------------ conversation */

function clearEmptyState() {
  const empty = $('empty-state');
  if (empty) empty.remove();
}

function scrollDown() {
  const box = $('conversation');
  box.scrollTop = box.scrollHeight;
}

function addBubble(who, text, { replayable = false } = {}) {
  clearEmptyState();
  const bubble = document.createElement('div');
  bubble.className = `bubble ${who === 'You' ? 'user' : 'liliana'}`;

  const label = document.createElement('div');
  label.className = 'who';
  label.textContent = who;
  bubble.appendChild(label);

  const body = document.createElement('div');
  body.className = 'body';
  body.textContent = text;
  bubble.appendChild(body);

  if (replayable) addReplayButton(bubble, text);

  $('conversation').appendChild(bubble);
  scrollDown();
  return bubble;
}

function bubbleBody(bubble) {
  return bubble.querySelector('.body');
}

/** Ajoute (une seule fois) le bouton de réécoute sous une bulle de Liliana. */
function addReplayButton(bubble, text) {
  if (!text || bubble.querySelector('.replay')) return;
  const replay = document.createElement('button');
  replay.className = 'replay';
  replay.type = 'button';
  replay.textContent = '↻ Play again';
  replay.addEventListener('click', () => speakText(text));
  bubble.appendChild(replay);
}

/** Diff mot à mot (plus longue sous-séquence commune) pour surligner la correction. */
function diffWords(before, after) {
  const a = before.split(/\s+/).filter(Boolean);
  const b = after.split(/\s+/).filter(Boolean);
  const table = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));

  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      table[i][j] = a[i].toLowerCase() === b[j].toLowerCase()
        ? table[i + 1][j + 1] + 1
        : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }

  const parts = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i].toLowerCase() === b[j].toLowerCase()) {
      parts.push({ type: 'same', text: a[i] }); i += 1; j += 1;
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      parts.push({ type: 'del', text: a[i] }); i += 1;
    } else {
      parts.push({ type: 'ins', text: b[j] }); j += 1;
    }
  }
  while (i < a.length) { parts.push({ type: 'del', text: a[i] }); i += 1; }
  while (j < b.length) { parts.push({ type: 'ins', text: b[j] }); j += 1; }
  return parts;
}

function addCorrection(correction, errors) {
  if (!correction && (!errors || errors.length === 0)) return;
  clearEmptyState();

  const box = document.createElement('div');
  box.className = 'correction';

  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = 'Correction';
  box.appendChild(label);

  if (correction) {
    const line = document.createElement('div');
    for (const part of diffWords(correction.original || '', correction.corrected || '')) {
      const node = document.createElement(
        part.type === 'del' ? 'del' : part.type === 'ins' ? 'ins' : 'span',
      );
      node.textContent = `${part.text} `;
      line.appendChild(node);
    }
    box.appendChild(line);

    if (correction.explanation) {
      const why = document.createElement('div');
      why.className = 'why';
      why.textContent = correction.explanation;
      box.appendChild(why);
    }
  }

  if (errors && errors.length) {
    const tags = document.createElement('div');
    tags.className = 'tags';
    for (const error of errors) {
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = error.topic || error.type;
      tags.appendChild(tag);
    }
    box.appendChild(tags);
  }

  $('conversation').appendChild(box);
  scrollDown();
}

/* ------------------------------------------------------------------ audio */

function base64ToBlobUrl(audioBase64, mimeType) {
  const binary = atob(audioBase64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return URL.createObjectURL(new Blob([bytes], { type: mimeType || 'audio/wav' }));
}

/* File d'attente audio.
 *
 * En streaming, Liliana renvoie une piste par phrase : la première arrive
 * pendant que le modèle écrit encore la suite. On les enchaîne ici pour que
 * l'utilisateur entende une réponse continue.
 */
const audioQueue = {
  urls: [],
  playing: false,

  push(audioBase64, mimeType) {
    this.urls.push(base64ToBlobUrl(audioBase64, mimeType));
    if (!this.playing) this.playNext();
  },

  playNext() {
    const url = this.urls.shift();
    if (!url) { this.playing = false; return; }
    this.playing = true;
    const player = $('player');
    player.src = url;
    player.onended = () => { URL.revokeObjectURL(url); this.playNext(); };
    player.onerror = () => { URL.revokeObjectURL(url); this.playNext(); };
    player.play().catch(() => {
      // Lecture refusée tant que l'utilisateur n'a pas interagi avec la page.
      URL.revokeObjectURL(url);
      this.playNext();
    });
  },

  reset() {
    this.urls.forEach(URL.revokeObjectURL);
    this.urls = [];
    this.playing = false;
    const player = $('player');
    player.pause();
    player.removeAttribute('src');
  },
};

function playBase64(audioBase64, mimeType) {
  audioQueue.push(audioBase64, mimeType);
}

/* Lecture d'un flux Server-Sent Events reçu en réponse à un POST.
 * `EventSource` ne sait faire que du GET : on parse le flux à la main.
 */
async function* readServerSentEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf('\n\n');
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let name = '';
      let data = '';
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) name = line.slice(7);
        else if (line.startsWith('data: ')) data = line.slice(6);
      }
      if (name && data) {
        try {
          yield { name, data: JSON.parse(data) };
        } catch {
          // Bloc tronqué : on l'ignore plutôt que d'interrompre le flux.
        }
      }
      boundary = buffer.indexOf('\n\n');
    }
  }
}

async function speakText(text) {
  if (!$('voice-output').checked) return;
  try {
    const speech = await postJSON('/speak', { text, language: state.language, speed: state.speed });
    audioQueue.reset();
    audioQueue.push(speech.audio_base64, speech.mime_type);
  } catch (error) {
    toast(error.message);
  }
}

/* -------------------------------------------------------------------- VAD */

/** Choisit un format d'enregistrement supporté par ce navigateur. */
function pickMimeType() {
  const candidates = [
    'audio/webm;codecs=opus', 'audio/webm',
    'audio/ogg;codecs=opus', 'audio/mp4', '',
  ];
  return candidates.find((type) => !type || MediaRecorder.isTypeSupported(type)) ?? '';
}

async function ensureMicrophone() {
  if (media.stream && media.stream.active) return media.stream;
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('This browser cannot access the microphone. Try Chrome, Edge or Firefox.');
  }
  try {
    media.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (error) {
    const name = error && error.name;
    if (name === 'NotAllowedError' || name === 'SecurityError') {
      throw new Error(
        'Liliana cannot access the microphone. Allow microphone access for this page '
        + 'in your browser, and check your operating system microphone permissions.',
      );
    }
    if (name === 'NotFoundError') throw new Error('No microphone was found on this computer.');
    throw new Error(`Microphone error: ${error.message || name}`);
  }
  return media.stream;
}

/** Boucle d'analyse : met à jour le vu-mètre et coupe l'enregistrement au silence. */
function startVad(onSilence) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  media.audioContext = new AudioCtx();
  const source = media.audioContext.createMediaStreamSource(media.stream);
  media.analyser = media.audioContext.createAnalyser();
  media.analyser.fftSize = 1024;
  source.connect(media.analyser);

  const buffer = new Uint8Array(media.analyser.fftSize);
  const startedAt = performance.now();
  let lastSpeechAt = 0;
  let speechDuration = 0;
  let previousSample = startedAt;

  const tick = () => {
    if (!state.recording) return;
    media.analyser.getByteTimeDomainData(buffer);

    let sum = 0;
    for (const sample of buffer) {
      const centred = (sample - 128) / 128;
      sum += centred * centred;
    }
    const rms = Math.sqrt(sum / buffer.length);
    $('level-bar').style.width = `${Math.min(100, rms * 450)}%`;

    const now = performance.now();
    if (rms >= state.vad.energy_threshold) {
      speechDuration += (now - previousSample) / 1000;
      lastSpeechAt = now;
    }
    previousSample = now;

    const hasSpoken = speechDuration >= state.vad.min_speech_duration;
    const silentFor = lastSpeechAt ? (now - lastSpeechAt) / 1000 : 0;

    if (hasSpoken && lastSpeechAt && silentFor >= state.vad.silence_threshold) {
      onSilence();
      return;
    }
    // Garde-fou : 60 s maximum par tour, même sans silence détecté.
    if (now - startedAt > 60000) { onSilence(); return; }

    media.raf = requestAnimationFrame(tick);
  };
  media.raf = requestAnimationFrame(tick);
}

function stopVad() {
  cancelAnimationFrame(media.raf);
  $('level-bar').style.width = '0%';
  if (media.audioContext) {
    media.audioContext.close().catch(() => {});
    media.audioContext = null;
  }
}

/* ------------------------------------------------------------ tour vocal */

async function startRecording() {
  if (state.recording || state.busy) return;
  try {
    await ensureMicrophone();
  } catch (error) {
    toast(error.message);
    return;
  }

  media.chunks = [];
  const mimeType = pickMimeType();
  try {
    media.recorder = new MediaRecorder(media.stream, mimeType ? { mimeType } : undefined);
  } catch (error) {
    toast(`Cannot start the recorder: ${error.message}`);
    return;
  }

  media.recorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) media.chunks.push(event.data);
  };
  media.recorder.onstop = () => {
    stopVad();
    const blob = new Blob(media.chunks, { type: media.recorder.mimeType || 'audio/webm' });
    media.chunks = [];
    if (blob.size < 1200) {
      setStatus('Ready');
      toast('That recording was too short — try speaking a little longer.');
      return;
    }
    sendVoiceTurn(blob);
  };

  media.recorder.start();
  state.recording = true;
  $('mic-button').classList.add('recording');
  $('mic-label').textContent = 'Stop';
  setStatus('Listening…');

  if ($('hands-free').checked) startVad(stopRecording);
  else startVad(() => {});
}

function stopRecording() {
  if (!state.recording) return;
  state.recording = false;
  $('mic-button').classList.remove('recording');
  $('mic-label').textContent = 'Speak';
  setStatus('Transcribing…');
  try {
    if (media.recorder && media.recorder.state !== 'inactive') media.recorder.stop();
  } catch { /* déjà arrêté */ }
  stopVad();
}

function setBusy(busy, label) {
  state.busy = busy;
  const button = $('mic-button');
  button.disabled = busy;
  button.classList.toggle('busy', busy);
  if (label) setStatus(label);
}

function renderTurn(payload) {
  if (payload.command) {
    applyCommandFeedback(payload.command);
  }
  addBubble('Liliana', payload.response, { replayable: true });
  state.lastResponse = payload.response;
  addCorrection(payload.correction, payload.errors);

  if (payload.vocabulary && payload.vocabulary.length) {
    const words = payload.vocabulary.map((entry) => entry.word).join(', ');
    toast(`New vocabulary saved: ${words}`, false);
  }
  if (payload.speech) playBase64(payload.speech.audio_base64, payload.speech.mime_type);
  else if ($('voice-output').checked) setStatus('Answered (voice output unavailable)');
}

function applyCommandFeedback(command) {
  if (command.language && command.language !== state.language) {
    state.language = command.language;
    $('language-select').value = command.language;
  }
  if (command.mode) {
    state.mode = command.mode;
    $('mode-select').value = command.mode;
  }
  if (command.correction_mode) {
    state.correctionMode = command.correction_mode;
    $('correction-select').value = command.correction_mode;
  }
  if (command.speed) state.speed = command.speed;
}

/* Tour de conversation diffusé.
 *
 * Le texte s'écrit au fur et à mesure et la voix démarre dès la première
 * phrase, sans attendre la fin de la génération. En cas de problème sur le
 * flux, on retombe sur l'endpoint classique : mieux vaut une réponse tardive
 * que pas de réponse.
 */
async function runStreamingTurn({ path, body, headers, userBubble }) {
  audioQueue.reset();

  let answerBubble = null;
  let streamedText = '';
  let sawAnything = false;

  const response = await fetch(api(path), { method: 'POST', body, headers });
  if (!response.ok) throw new Error(await readError(response));
  if (!response.body) throw new Error('This browser cannot read streaming responses.');

  for await (const { name, data } of readServerSentEvents(response)) {
    sawAnything = true;
    switch (name) {
      case 'transcription':
        if (userBubble) bubbleBody(userBubble).textContent = data.text;
        if (!data.partial) setStatus('Thinking…');
        break;

      case 'command':
        applyCommandFeedback(data);
        break;

      case 'delta':
        if (!answerBubble) {
          answerBubble = addBubble('Liliana', '');
          setStatus('Speaking…');
        }
        streamedText += data.text;
        bubbleBody(answerBubble).textContent = streamedText;
        scrollDown();
        break;

      case 'audio':
        audioQueue.push(data.audio_base64, data.mime_type);
        break;

      case 'done':
        finishStreamedTurn(answerBubble, streamedText, data);
        return data;

      case 'error':
        if (answerBubble && !streamedText) answerBubble.remove();
        throw new Error(data.message);

      default:
        break;
    }
  }

  if (!sawAnything) throw new Error('Liliana closed the connection without answering.');
  throw new Error('The answer was cut off. Please try again.');
}

/** Réconcilie la bulle diffusée avec le tour complet reçu à la fin. */
function finishStreamedTurn(answerBubble, streamedText, payload) {
  const bubble = answerBubble || addBubble('Liliana', payload.response);
  // Le JSON final fait autorité : il peut différer légèrement du flux brut.
  if (payload.response && payload.response !== streamedText) {
    bubbleBody(bubble).textContent = payload.response;
  }
  addReplayButton(bubble, payload.response);
  state.lastResponse = payload.response;

  addCorrection(payload.correction, payload.errors);
  if (payload.vocabulary && payload.vocabulary.length) {
    toast(`New vocabulary saved: ${payload.vocabulary.map((e) => e.word).join(', ')}`, false);
  }
  scrollDown();
}

async function sendVoiceTurn(blob) {
  if (state.busy) return;
  setBusy(true, 'Transcribing…');

  const form = new FormData();
  form.append('audio', blob, 'turn.webm');
  form.append('language', state.language);
  form.append('mode', state.mode);
  form.append('correction_mode', state.correctionMode);
  form.append('speak', String($('voice-output').checked));
  form.append('speed', String(state.speed));

  const placeholder = addBubble('You', '…');
  try {
    const payload = await runStreamingTurn({
      path: '/voice/turn/stream', body: form, userBubble: placeholder,
    });
    setStatus(`Ready — ${payload.llm_elapsed}s`);
  } catch (error) {
    // Repli sur le chemin non diffusé : plus lent, mais il fonctionne.
    try {
      const payload = await postForm('/voice/turn', form);
      bubbleBody(placeholder).textContent = payload.transcription.text;
      renderTurn(payload);
      setStatus(`Ready — ${payload.llm_elapsed}s (no streaming)`);
    } catch (fallbackError) {
      placeholder.remove();
      toast(fallbackError.message || error.message);
      setStatus('Ready');
    }
  } finally {
    setBusy(false);
  }
}

async function sendTextTurn(text) {
  if (!text.trim() || state.busy) return;
  addBubble('You', text);
  setBusy(true, 'Thinking…');

  const request = {
    text,
    language: state.language,
    mode: state.mode,
    correction_mode: state.correctionMode,
    speak: $('voice-output').checked,
    speed: state.speed,
  };

  try {
    const payload = await runStreamingTurn({
      path: '/chat/turn/stream',
      body: JSON.stringify(request),
      headers: { 'Content-Type': 'application/json' },
    });
    setStatus(`Ready — ${payload.llm_elapsed}s`);
  } catch (error) {
    try {
      const payload = await postJSON('/chat/turn', request);
      renderTurn(payload);
      setStatus(`Ready — ${payload.llm_elapsed}s (no streaming)`);
    } catch (fallbackError) {
      toast(fallbackError.message || error.message);
      setStatus('Ready');
    }
  } finally {
    setBusy(false);
  }
}

async function postForm(path, form) {
  const response = await fetch(api(path), { method: 'POST', body: form });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

/* ------------------------------------------------------------- exercices */

async function newExercise() {
  const area = $('exercise-area');
  area.innerHTML = '<p class="muted">Building an exercise…</p>';
  try {
    const exercise = await postJSON('/exercise/generate', {
      language: state.language,
      topic: $('exercise-topic').value.trim() || null,
      exercise_type: $('exercise-type').value || null,
    });
    area.innerHTML = '';
    area.appendChild(renderExercise(exercise));
  } catch (error) {
    area.innerHTML = '';
    toast(error.message);
  }
}

function renderExercise(exercise) {
  const card = document.createElement('div');
  card.className = 'card';

  const title = document.createElement('h3');
  title.textContent = `${exercise.exercise_type.replace(/_/g, ' ')} · ${exercise.topic || 'practice'} · ${exercise.level}`;
  card.appendChild(title);

  const prompt = document.createElement('div');
  prompt.className = 'prompt';
  prompt.textContent = exercise.prompt;
  card.appendChild(prompt);

  let chosen = '';
  if (exercise.options && exercise.options.length) {
    const options = document.createElement('div');
    options.className = 'options';
    for (const option of exercise.options) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'option';
      button.textContent = option;
      button.addEventListener('click', () => {
        chosen = option;
        options.querySelectorAll('.option').forEach((element) => element.classList.remove('chosen'));
        button.classList.add('chosen');
      });
      options.appendChild(button);
    }
    card.appendChild(options);
  }

  const row = document.createElement('div');
  row.className = 'answer-row';
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = exercise.options?.length ? 'Or type your own answer' : 'Your answer';
  const submit = document.createElement('button');
  submit.textContent = 'Check';
  row.append(input, submit);
  card.appendChild(row);

  const feedback = document.createElement('div');
  feedback.className = 'small';
  card.appendChild(feedback);

  const check = async () => {
    const answer = input.value.trim() || chosen;
    if (!answer) { toast('Pick or type an answer first.'); return; }
    submit.disabled = true;
    feedback.textContent = 'Checking…';
    try {
      const result = await postJSON('/exercise/check', { exercise_id: exercise.id, answer });
      card.classList.add(result.is_correct ? 'correct' : 'incorrect');
      feedback.innerHTML = '';
      const verdict = document.createElement('p');
      verdict.innerHTML = `<strong>${result.is_correct ? '✓ Correct' : '✗ Not quite'}</strong> — ${result.feedback}`;
      feedback.appendChild(verdict);
      if (!result.is_correct && result.corrected) {
        const expected = document.createElement('p');
        expected.className = 'muted';
        expected.textContent = `Expected: ${result.corrected}`;
        feedback.appendChild(expected);
      }
      if (result.explanation) {
        const why = document.createElement('p');
        why.className = 'muted';
        why.textContent = result.explanation;
        feedback.appendChild(why);
      }
    } catch (error) {
      feedback.textContent = '';
      toast(error.message);
      submit.disabled = false;
    }
  };

  submit.addEventListener('click', check);
  input.addEventListener('keydown', (event) => { if (event.key === 'Enter') check(); });
  return card;
}

/* ------------------------------------------------------------ vocabulaire */

async function loadVocabulary() {
  const area = $('vocabulary-area');
  area.innerHTML = '<p class="muted">Loading…</p>';
  try {
    const data = await request(`/vocabulary/due?language=${state.language}&limit=20`);
    area.innerHTML = '';

    const summary = document.createElement('p');
    summary.className = 'muted small';
    summary.textContent = `${data.known_words} word(s) known · ${data.due.length} due for review`;
    area.appendChild(summary);

    if (!data.due.length) {
      const nothing = document.createElement('p');
      nothing.className = 'muted';
      nothing.textContent = 'Nothing to review right now. Keep talking to Liliana, or ask her to teach you a theme.';
      area.appendChild(nothing);
      return;
    }
    for (const entry of data.due) area.appendChild(renderWord(entry));
  } catch (error) {
    area.innerHTML = '';
    toast(error.message);
  }
}

function renderWord(entry) {
  const card = document.createElement('div');
  card.className = 'card';

  const row = document.createElement('div');
  row.className = 'word-row';

  const left = document.createElement('div');
  const word = document.createElement('div');
  word.className = 'word';
  word.textContent = entry.word;
  left.appendChild(word);

  const hidden = document.createElement('div');
  hidden.className = 'muted small';
  hidden.textContent = '••• tap Reveal';
  left.appendChild(hidden);
  row.appendChild(left);

  const actions = document.createElement('div');
  actions.className = 'actions';

  const reveal = document.createElement('button');
  reveal.className = 'ghost';
  reveal.textContent = 'Reveal';
  reveal.addEventListener('click', () => {
    hidden.textContent = [entry.translation, entry.example].filter(Boolean).join(' — ')
      || 'No translation stored yet.';
  });

  const listen = document.createElement('button');
  listen.className = 'ghost';
  listen.textContent = '🔊';
  listen.title = 'Hear the word';
  listen.addEventListener('click', () => speakText(entry.example || entry.word));

  const grade = async (remembered) => {
    try {
      await postJSON('/vocabulary/review', { language: state.language, word: entry.word, remembered });
      card.remove();
    } catch (error) { toast(error.message); }
  };

  const forgot = document.createElement('button');
  forgot.className = 'ghost';
  forgot.textContent = 'Forgot';
  forgot.addEventListener('click', () => grade(false));

  const knew = document.createElement('button');
  knew.className = 'primary';
  knew.textContent = 'I knew it';
  knew.addEventListener('click', () => grade(true));

  actions.append(reveal, listen, forgot, knew);
  row.appendChild(actions);
  card.appendChild(row);
  return card;
}

async function teachVocabulary() {
  const theme = $('vocabulary-theme').value.trim() || 'everyday life';
  const area = $('vocabulary-area');
  area.innerHTML = `<p class="muted">Preparing words about “${theme}”…</p>`;
  try {
    const data = await postJSON('/vocabulary/teach', { language: state.language, theme, count: 6 });
    area.innerHTML = '';
    if (!data.words.length) { area.innerHTML = '<p class="muted">Liliana could not build a list. Try another theme.</p>'; return; }
    for (const entry of data.words) {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `<div class="word-row"><div><div class="word"></div>
        <div class="muted small"></div></div></div>`;
      card.querySelector('.word').textContent = `${entry.word} — ${entry.translation}`;
      card.querySelector('.muted').textContent = entry.example;
      area.appendChild(card);
    }
  } catch (error) {
    area.innerHTML = '';
    toast(error.message);
  }
}

/* -------------------------------------------------------------- progrès */

async function loadProgress() {
  const area = $('progress-area');
  area.innerHTML = '<p class="muted">Computing…</p>';
  try {
    const data = await request('/dashboard');
    area.innerHTML = '';
    for (const entry of data.languages) area.appendChild(renderLanguageCard(entry));
  } catch (error) {
    area.innerHTML = '';
    toast(error.message);
  }
}

function renderLanguageCard(entry) {
  const languageName = (state.config?.languages || [])
    .find((item) => item.code === entry.language);

  const card = document.createElement('div');
  card.className = 'lang-card';

  const header = document.createElement('header');
  const title = document.createElement('h3');
  title.textContent = languageName ? `${languageName.flag} ${languageName.name}` : entry.language;
  const badge = document.createElement('span');
  badge.className = 'level-badge';
  badge.textContent = entry.is_estimate ? `${entry.level} (estimate)` : entry.level;
  header.append(title, badge);
  card.appendChild(header);

  for (const [skill, value] of Object.entries(entry.skills)) {
    const row = document.createElement('div');
    row.className = 'skill';
    const label = document.createElement('span');
    label.textContent = skill.charAt(0).toUpperCase() + skill.slice(1);
    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('div');
    fill.style.width = `${Math.max(0, Math.min(100, value))}%`;
    bar.appendChild(fill);
    const number = document.createElement('span');
    number.className = 'value';
    number.textContent = Math.round(value);
    row.append(label, bar, number);
    card.appendChild(row);
  }

  const totals = entry.totals;
  const stats = document.createElement('div');
  stats.className = 'stats';
  const cells = [
    [Math.round(totals.seconds_studied / 60), 'minutes studied'],
    [totals.messages, 'things you said'],
    [totals.words_learned, 'words learned'],
    [totals.errors_corrected, 'errors corrected'],
    [totals.exercises_done, 'exercises done'],
    [totals.success_rate === null ? '—' : `${totals.success_rate}%`, 'success rate'],
    [entry.reviews_due, 'reviews due'],
  ];
  for (const [value, key] of cells) {
    const stat = document.createElement('div');
    stat.className = 'stat';
    stat.innerHTML = '<div class="n"></div><div class="k"></div>';
    stat.querySelector('.n').textContent = value;
    stat.querySelector('.k').textContent = key;
    stats.appendChild(stat);
  }
  card.appendChild(stats);

  if (entry.weaknesses.length) {
    const list = document.createElement('div');
    list.className = 'weakness-list';
    const label = document.createElement('span');
    label.className = 'muted small';
    label.textContent = 'Work on:';
    list.appendChild(label);
    for (const weakness of entry.weaknesses) {
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = `${weakness.topic} (${weakness.occurrences})`;
      list.appendChild(tag);
    }
    card.appendChild(list);
  }
  return card;
}

/* --------------------------------------------------------------- modals */

function openModal(title, build) {
  $('modal-title').textContent = title;
  const body = $('modal-body');
  body.innerHTML = '';
  build(body);
  $('modal-backdrop').hidden = false;
}

function closeModal() { $('modal-backdrop').hidden = true; }

async function openAssessment() {
  openModal('Placement test', (body) => {
    body.innerHTML = '<p class="muted">Loading the test…</p>';
  });

  let test;
  try {
    test = await request(`/assessment/${state.language}`);
  } catch (error) { closeModal(); toast(error.message); return; }

  openModal(`Placement test — ${test.language_name}`, (body) => {
    const intro = document.createElement('p');
    intro.className = 'muted';
    intro.textContent = 'Ten quick questions, then two short free answers. '
      + 'Skip anything you do not know — that is part of the measurement.';
    body.appendChild(intro);

    const answers = {};
    for (const item of test.items) {
      const block = document.createElement('div');
      block.className = 'quiz-item';
      const question = document.createElement('p');
      question.textContent = item.question;
      block.appendChild(question);

      const options = document.createElement('div');
      options.className = 'options';
      for (const option of item.options) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'option';
        button.textContent = option;
        button.addEventListener('click', () => {
          answers[item.id] = option;
          options.querySelectorAll('.option').forEach((element) => element.classList.remove('chosen'));
          button.classList.add('chosen');
        });
        options.appendChild(button);
      }
      block.appendChild(options);
      body.appendChild(block);
    }

    const productions = {};
    for (const task of test.production) {
      const block = document.createElement('div');
      block.className = 'quiz-item';
      const question = document.createElement('p');
      question.textContent = task.prompt;
      const textarea = document.createElement('textarea');
      textarea.addEventListener('input', () => { productions[task.id] = textarea.value; });
      block.append(question, textarea);
      body.appendChild(block);
    }

    const submit = document.createElement('button');
    submit.className = 'primary';
    submit.textContent = 'Get my level';
    submit.addEventListener('click', async () => {
      submit.disabled = true;
      submit.textContent = 'Evaluating…';
      try {
        const result = await postJSON('/assessment', {
          language: state.language, answers, productions,
        });
        showAssessmentResult(result);
      } catch (error) {
        toast(error.message);
        submit.disabled = false;
        submit.textContent = 'Get my level';
      }
    });
    body.appendChild(submit);
  });
}

function showAssessmentResult(result) {
  openModal('Your level', (body) => {
    const level = document.createElement('p');
    level.innerHTML = `<span class="level-badge">${result.level}</span>`;
    body.appendChild(level);

    const summary = document.createElement('p');
    summary.textContent = result.summary;
    body.appendChild(summary);

    const objective = document.createElement('p');
    objective.className = 'muted small';
    objective.textContent = `Placement quiz: ${result.objective.correct}/${result.objective.total} correct.`
      + (result.llm_used ? '' : ' (Free-text answers were not graded — the language model was unavailable.)');
    body.appendChild(objective);

    for (const [skill, value] of Object.entries(result.scores)) {
      const row = document.createElement('div');
      row.className = 'skill';
      row.innerHTML = '<span></span><div class="bar"><div></div></div><span class="value"></span>';
      row.querySelector('span').textContent = skill.charAt(0).toUpperCase() + skill.slice(1);
      row.querySelector('.bar > div').style.width = `${value}%`;
      row.querySelector('.value').textContent = Math.round(value);
      body.appendChild(row);
    }

    const done = document.createElement('button');
    done.className = 'primary';
    done.textContent = 'Start learning';
    done.addEventListener('click', () => { closeModal(); loadProgress(); });
    body.appendChild(done);
  });
}

async function openLesson() {
  let plan;
  try {
    plan = await request(`/lesson?language=${state.language}&minutes=30`);
  } catch (error) { toast(error.message); return; }

  openModal('Your session plan', (body) => {
    const intro = document.createElement('p');
    intro.className = 'muted';
    intro.textContent = `${plan.total_minutes} minutes, level ${plan.level}. `
      + 'Pick a block to jump straight into it.';
    body.appendChild(intro);

    for (const block of plan.blocks) {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = '<div class="word-row"><div><div class="word"></div><div class="muted small"></div></div></div>';
      card.querySelector('.word').textContent = `${block.minutes} min — ${block.label}`;
      card.querySelector('.muted').textContent = block.focus || '';

      const go = document.createElement('button');
      go.className = 'primary';
      go.textContent = 'Start';
      go.addEventListener('click', () => {
        state.mode = block.mode;
        $('mode-select').value = block.mode;
        persistSettings();
        closeModal();
        switchTab(block.mode === 'grammar_training' ? 'exercises'
          : block.mode === 'vocabulary_training' ? 'vocabulary' : 'conversation');
      });
      card.querySelector('.word-row').appendChild(go);
      body.appendChild(card);
    }
  });
}

/* ---------------------------------------------------------------- statut */

async function refreshHealth() {
  const chip = $('status-chip');
  const text = $('status-text');
  try {
    const health = await request('/health');
    chip.dataset.health = JSON.stringify(health);
    if (health.ready) {
      chip.className = 'status-chip ok';
      text.textContent = `LOCAL · ${health.llm.model}`;
    } else {
      chip.className = 'status-chip err';
      text.textContent = !health.llm.available ? 'LLM offline' : 'STT offline';
    }
  } catch {
    chip.className = 'status-chip err';
    text.textContent = 'Server unreachable';
  }
}

function showHealthDetails() {
  let health;
  try { health = JSON.parse($('status-chip').dataset.health || '{}'); } catch { health = {}; }
  openModal('Engine status', (body) => {
    const engines = [
      ['Language model', health.llm], ['Speech-to-text', health.stt], ['Text-to-speech', health.tts],
    ];
    for (const [name, engine] of engines) {
      if (!engine) continue;
      const card = document.createElement('div');
      card.className = 'card';
      const title = document.createElement('h3');
      title.textContent = `${engine.available ? '●' : '○'} ${name} — ${engine.provider || ''} ${engine.model || ''}`;
      const detail = document.createElement('p');
      detail.className = 'muted small';
      detail.textContent = engine.detail || '';
      card.append(title, detail);
      body.appendChild(card);
    }
    const note = document.createElement('p');
    note.className = 'muted small';
    note.textContent = 'Everything runs on this computer. No conversation, audio or profile '
      + 'data leaves the machine.';
    body.appendChild(note);
  });
}

/* ----------------------------------------------------------- navigation */

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.classList.toggle('active', tab.dataset.tab === name);
  });
  document.querySelectorAll('.panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === `panel-${name}`);
  });
  if (name === 'progress') loadProgress();
  if (name === 'vocabulary') loadVocabulary();
}

async function persistSettings() {
  try {
    await postJSON('/settings', {
      language: state.language, mode: state.mode, correction_mode: state.correctionMode,
    });
  } catch (error) { toast(error.message); }
}

async function loadConversation() {
  try {
    const session = await request(
      `/session/current?language=${state.language}&mode=${state.mode}`,
    );
    const box = $('conversation');
    if (!session.messages.length) return;
    box.innerHTML = '';
    for (const message of session.messages) {
      addBubble(message.role === 'user' ? 'You' : 'Liliana', message.content,
        { replayable: message.role !== 'user' });
    }
  } catch { /* première utilisation : rien à charger */ }
}

/* -------------------------------------------------------------- démarrage */

async function boot() {
  try {
    state.config = await request('/config');
  } catch (error) {
    toast(`Liliana's server is not responding: ${error.message}`);
    return;
  }

  state.language = state.config.current.language;
  state.mode = state.config.current.mode;
  state.correctionMode = state.config.current.correction_mode || 'normal';
  state.vad = { ...state.vad, ...state.config.vad };

  const languageSelect = $('language-select');
  for (const language of state.config.languages) {
    languageSelect.add(new Option(`${language.flag} ${language.name}`, language.code));
  }
  languageSelect.value = state.language;

  const modeSelect = $('mode-select');
  for (const mode of state.config.modes) modeSelect.add(new Option(mode.label, mode.key));
  modeSelect.value = state.mode;

  const typeSelect = $('exercise-type');
  for (const type of state.config.exercise_types) {
    typeSelect.add(new Option(type.replace(/_/g, ' '), type));
  }

  $('correction-select').value = state.correctionMode;

  languageSelect.addEventListener('change', async () => {
    state.language = languageSelect.value;
    await persistSettings();
    $('conversation').innerHTML = '';
    await loadConversation();
  });
  modeSelect.addEventListener('change', async () => {
    state.mode = modeSelect.value;
    await persistSettings();
  });
  $('correction-select').addEventListener('change', async () => {
    state.correctionMode = $('correction-select').value;
    await persistSettings();
  });

  $('mic-button').addEventListener('click', () => {
    if (state.recording) stopRecording(); else startRecording();
  });

  $('text-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const input = $('text-input');
    const text = input.value;
    input.value = '';
    sendTextTurn(text);
  });

  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });
  document.querySelectorAll('[data-send]').forEach((button) => {
    button.addEventListener('click', () => sendTextTurn(button.dataset.send));
  });

  $('start-lesson')?.addEventListener('click', openLesson);
  $('run-assessment')?.addEventListener('click', openAssessment);
  $('new-exercise').addEventListener('click', newExercise);
  $('load-vocabulary').addEventListener('click', loadVocabulary);
  $('teach-vocabulary').addEventListener('click', teachVocabulary);
  $('refresh-progress').addEventListener('click', loadProgress);
  $('status-chip').addEventListener('click', showHealthDetails);
  $('modal-close').addEventListener('click', closeModal);
  $('modal-backdrop').addEventListener('click', (event) => {
    if (event.target === $('modal-backdrop')) closeModal();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeModal();
    // Espace = parler, sauf si l'on est en train d'écrire.
    if (event.code === 'Space' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) {
      event.preventDefault();
      if (state.recording) stopRecording(); else startRecording();
    }
  });

  await loadConversation();
  await refreshHealth();
  setInterval(refreshHealth, 30000);

  if (!state.config.onboarded) {
    setStatus('Ready — take the placement test to calibrate your level');
  }
}

boot();
