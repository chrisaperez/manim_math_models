/* Shared helpers for the interactive demos. Plain script (no modules) so pages
 * work from file:// as well as any static server. Everything lives on window.MM. */

(function () {
  "use strict";

  const C = {
    bg: "#000000",
    panel: "#0a0a0a",
    grid: "#222222",
    ink: "#f2f2ee",
    ink2: "#b3b3ac",
    ink3: "#78786f",
    blue: "#58c4dd",
    teal: "#5cd0b3",
    yellow: "#ffd35a",
    green: "#83c167",
    red: "#fc6255",
    purple: "#9a72ac",
  };

  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const lerp = (a, b, t) => a + (b - a) * t;

  function fmt(x, digits = 2) {
    if (!isFinite(x)) return x > 0 ? "∞" : "-∞";
    const a = Math.abs(x);
    if (a !== 0 && (a >= 1e4 || a < 10 ** -digits)) return x.toExponential(1);
    return x.toFixed(digits);
  }

  /* Crisp DPR-aware canvas bound to its CSS size. draw(ctx, w, h) is called on
   * every invalidate() and on resize. */
  function makeFig(canvas, draw) {
    const ctx = canvas.getContext("2d");
    let w = 0, h = 0;
    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0) return;
      w = rect.width;
      h = canvas.height / (canvas.width || 1) * w; // keep author aspect
      h = parseFloat(canvas.dataset.aspect ? w * canvas.dataset.aspect : rect.height) || rect.height;
      canvas.style.height = h + "px";
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      fig.invalidate();
    }
    const fig = {
      canvas, ctx,
      get w() { return w; },
      get h() { return h; },
      invalidate() { if (w > 0) draw(ctx, w, h); },
      resize,
    };
    new ResizeObserver(resize).observe(canvas);
    requestAnimationFrame(resize);
    return fig;
  }

  /* Slider control with value readout and optional annotated ticks.
   * opts: {el, label, min, max, step, value, format, ticks:[{at,color,label}], onChange} */
  function slider(opts) {
    const root = document.querySelector(opts.el);
    root.classList.add("control");
    const format = opts.format || ((v) => fmt(v, 2));

    root.innerHTML = `
      <div class="row"><span>${opts.label}</span><span class="val"></span></div>
      <div class="slider-wrap">
        <input type="range" min="${opts.min}" max="${opts.max}"
               step="${opts.step}" value="${opts.value}"
               aria-label="${opts.label.replace(/<[^>]*>/g, "")}">
      </div>`;
    const input = root.querySelector("input");
    const val = root.querySelector(".val");
    const wrap = root.querySelector(".slider-wrap");

    for (const t of opts.ticks || []) {
      const frac = (t.at - opts.min) / (opts.max - opts.min);
      const tick = document.createElement("span");
      tick.className = "tick";
      tick.style.left = `calc(8px + ${frac} * (100% - 16px))`;
      tick.style.background = t.color;
      wrap.appendChild(tick);
      if (t.label) {
        const lab = document.createElement("span");
        lab.className = "tick-label";
        lab.style.left = `calc(8px + ${frac} * (100% - 16px))`;
        lab.style.color = t.color;
        lab.textContent = t.label;
        wrap.appendChild(lab);
        root.style.paddingBottom = "14px";
      }
    }

    const api = {
      get: () => parseFloat(input.value),
      set(v, fire = true) {
        input.value = v;
        val.textContent = format(parseFloat(input.value));
        if (fire && opts.onChange) opts.onChange(api.get());
      },
    };
    input.addEventListener("input", () => api.set(api.get()));
    api.set(opts.value, false);
    return api;
  }

  /* Tooltip that follows the pointer inside a positioned parent. */
  function tooltip(parent) {
    const el = document.createElement("div");
    el.className = "tipbox";
    parent.style.position = "relative";
    parent.appendChild(el);
    return {
      show(x, y, html) {
        el.innerHTML = html;
        el.style.opacity = 1;
        const pw = parent.clientWidth;
        el.style.left = clamp(x + 14, 0, pw - el.offsetWidth - 4) + "px";
        el.style.top = y - 10 + "px";
      },
      hide() { el.style.opacity = 0; },
    };
  }

  /* Play/pause loop calling step(dtSeconds) ~30x/s. Uses setInterval rather
   * than requestAnimationFrame: embedded/occluded webviews throttle rAF to
   * zero, and these sims are cheap enough that a timer is plenty smooth. */
  function player(step) {
    let running = false, last = 0, handle = 0;
    return {
      get running() { return running; },
      play() {
        if (running) return;
        running = true;
        last = performance.now();
        handle = setInterval(() => {
          const now = performance.now();
          const dt = Math.min(0.1, (now - last) / 1000);
          last = now;
          step(dt);
        }, 33);
      },
      pause() { running = false; clearInterval(handle); },
      toggle() { this.running ? this.pause() : this.play(); },
    };
  }

  // tiny 2-vector / 2x2 matrix helpers (row-major [[a,b],[c,d]])
  const V2 = {
    rot: (t) => [[Math.cos(t), -Math.sin(t)], [Math.sin(t), Math.cos(t)]],
    mul: (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]],
    mulT: (M, v) => [M[0][0] * v[0] + M[1][0] * v[1], M[0][1] * v[0] + M[1][1] * v[1]],
    norm: (v) => Math.hypot(v[0], v[1]),
  };

  window.MM = { C, clamp, lerp, fmt, makeFig, slider, tooltip, player, V2 };
})();
