/* 2050 UI JS — timeline, visualizer, terminal */

(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const now = () => new Date().toLocaleTimeString();

  const navLinks = $$(".nav-link");
  const terminal = $("#terminal");
  const jsonViewer = $("#jsonViewer");
  const timelineCanvas = $("#timelineCanvas");
  const waveCanvas = $("#waveCanvas");
  const resultAudio = $("#resultAudio");
  const playBtn = $("#playBtn");
  const loaderWrap = $("#loaderWrap");

  const APP = window.APP_VARS || {};

  function log(msg) {
    if (!terminal) return;
    const el = document.createElement("div");
    el.textContent = `[${now()}] ${msg}`;
    terminal.appendChild(el);
    terminal.scrollTop = terminal.scrollHeight;
  }

  function setPanel(id) {
    $$(".panel").forEach((p) => (p.hidden = !p.classList.contains(id)));
    navLinks.forEach((a) =>
      a.classList.toggle("active", a.dataset.target === id)
    );
  }

  navLinks.forEach((a) =>
    a.addEventListener("click", (e) => {
      e.preventDefault();
      setPanel(a.dataset.target);
    })
  );

  function renderJSON(data) {
    jsonViewer.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
  }

  function renderTimeline(data) {
    if (!timelineCanvas) return;
    const cv = timelineCanvas;
    const ctx = cv.getContext("2d");

    const w = cv.clientWidth;
    const h = cv.height;
    cv.width = w * window.devicePixelRatio;
    cv.height = h * window.devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, w, h);

    if (!data.length) return;

    const duration = Math.max(...data.map((e) => e.end));
    const px = (w - 40) / duration;

    data.forEach((s) => {
      const x = 20 + s.start * px;
      const w2 = (s.end - s.start) * px;
      ctx.fillStyle = s.speaker === "Target" ? "#eee" : "#555";
      ctx.fillRect(x, 10, w2, h - 40);
    });
  }

  async function visualizer(url) {
    if (!waveCanvas || !resultAudio) return;

    resultAudio.src = url;
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;

    const src = ctx.createMediaElementSource(resultAudio);
    src.connect(analyser);
    analyser.connect(ctx.destination);

    const arr = new Uint8Array(analyser.fftSize);
    const cv = waveCanvas;
    const c = cv.getContext("2d");
    const w = cv.clientWidth;
    const h = cv.height;

    cv.width = w * devicePixelRatio;
    cv.height = h * devicePixelRatio;
    c.scale(devicePixelRatio, devicePixelRatio);

    function draw() {
      requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(arr);

      c.fillStyle = "#000";
      c.fillRect(0, 0, w, h);

      c.strokeStyle = "#ccc";
      c.beginPath();

      let x = 0;
      const slice = w / arr.length;

      for (let i = 0; i < arr.length; i++) {
        let v = arr[i] / 128.0;
        let y = (v * h) / 2;

        if (i === 0) c.moveTo(x, y);
        else c.lineTo(x, y);

        x += slice;
      }
      c.stroke();
    }
    draw();
  }

  async function loadResults() {
    if (!APP.hasResult || !APP.jsonURL) return;

    log("Fetching diarization.json…");

    const res = await fetch(APP.jsonURL);
    const data = await res.json();

    renderJSON(data);
    renderTimeline(data);

    if (APP.audioURL) {
      await visualizer(APP.audioURL);
      log("Visualizer initialized");
    }
  }

  if (playBtn) {
    playBtn.addEventListener("click", () => {
      if (resultAudio.paused) {
        resultAudio.play();
        playBtn.textContent = "Pause";
      } else {
        resultAudio.pause();
        playBtn.textContent = "Play";
      }
    });
  }

  const form = $("#uploadForm");
  if (form) {
    form.addEventListener("submit", () => {
      loaderWrap.style.visibility = "visible";
      log("Processing started");
    });
  }

  function init() {
    setPanel("panel-dashboard");
    if (APP.hasResult) {
      setPanel("panel-timeline");
      loadResults();
    }
    log("UI Ready");
  }

  init();
})();
