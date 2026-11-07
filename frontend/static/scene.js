(() => {
  "use strict";

  const canvas = document.getElementById("iris");
  const ctx = canvas.getContext("2d", { alpha: false });
  const light = document.createElement("canvas");
  const pen = light.getContext("2d");
  const experience = document.getElementById("experience");
  const coordinates = document.getElementById("coordinates");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const tau = Math.PI * 2;
  const steps = 24;
  let width, height, ratio, radius, centerX, centerY;
  let targetX = 0, targetY = 0, lookX = 0, lookY = 0;
  let lastFrame = 0, animationFrame = 0, startedAt = performance.now();
  let finalScene = reducedMotion.matches;
  let nextBlink = 7.2, blinkStarted = -1;
  let seed = 73619;

  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (x) => { x = clamp(x); return x * x * (3 - 2 * x); };
  const mix = (a, b, t) => a + (b - a) * t;

  // Every fiber retains its identity while the iris unfolds and reforms.
  const fibers = Array.from({ length: 1350 }, (_, i) => {
    const angle = (i / 1350) * tau + random() * .014;
    const start = .275 + random() * .045;
    const end = .76 + Math.pow(random(), .45) * .22;
    const twist = (random() - .5) * .1;
    const phase = random() * tau;
    const spread = random();
    const points = new Float32Array((steps + 1) * 3);
    for (let j = 0; j <= steps; j++) {
      const s = j / steps;
      const r = mix(start, end, s);
      const a = angle + Math.sin(s * 15 + phase) * .021 + Math.sin(s * 29 + phase) * .007 + twist * s;
      points[j * 3] = Math.cos(a) * r;
      points[j * 3 + 1] = Math.sin(a) * r;
      points[j * 3 + 2] = Math.sin(s * Math.PI) * .055 + Math.sin(a * 3 + phase) * .015;
    }
    return { angle, phase, spread, points, length: .65 + random() * .45,
      brightness: random(), width: .24 + random() * .59, offset: (random() - .5) * .028 };
  });
  const dust = Array.from({ length: 75 }, () => ({
    x: (random() - .5) * 3.8, y: (random() - .5) * 3.7,
    z: random(), phase: random() * tau,
  }));

  const tissue = document.createElement("canvas");
  tissue.width = tissue.height = 1200;
  const tissuePen = tissue.getContext("2d");
  tissuePen.translate(600, 600);
  const collarette = (a) => .43 + Math.sin(a * 11 + .6) * .022
    + Math.sin(a * 23) * .012 + Math.sin(a * 47 + 1.3) * .008;
  const irisBase = tissuePen.createRadialGradient(0, 0, 150, 0, 0, 580);
  irisBase.addColorStop(0, "#060e1b");
  irisBase.addColorStop(.12, "#3c5874");
  irisBase.addColorStop(.3, "#7c9bab");
  irisBase.addColorStop(.5, "#3f7291");
  irisBase.addColorStop(.8, "#214861");
  irisBase.addColorStop(.95, "#0c2237");
  irisBase.addColorStop(1, "#030812");
  tissuePen.fillStyle = irisBase;
  tissuePen.beginPath();tissuePen.arc(0, 0, 580, 0, tau);tissuePen.fill();

  // Cache the organic tissue so fine branching and pigment need no per-frame regeneration.
  for (let i = 0; i < 6800; i++) {
    const a = random() * tau;
    const innerZone = random() < .32;
    const start = innerZone ? .276 + random() * .06 : collarette(a) + random() * .03;
    const end = innerZone ? collarette(a) + .015 : .72 + random() * .26;
    const bend = (random() - .5) * .07;
    const phase = random() * tau;
    const shade = random();
    tissuePen.strokeStyle = shade > .86 ? `rgba(192,216,222,${.15 + random() * .23})`
      : shade > .48 ? `rgba(114,163,190,${.16 + random() * .28})`
      : shade > .2 ? `rgba(35,79,114,${.24 + random() * .35})`
      : `rgba(3,17,33,${.3 + random() * .35})`;
    tissuePen.lineWidth = .4 + random() * 1.65;
    tissuePen.beginPath();
    for (let j = 0; j <= 26; j++) {
      const s = j / 26;
      const r = mix(start, end, s) * 580;
      const angle = a + bend * s + Math.sin(s * 16 + phase) * .009
        + Math.sin(s * 37 + phase) * .003;
      const x = Math.cos(angle) * r, y = Math.sin(angle) * r;
      if (!j) tissuePen.moveTo(x, y); else tissuePen.lineTo(x, y);
    }
    tissuePen.stroke();
  }

  // Irregular crypts break the uniform radial pattern around the pupillary collar.
  for (let i = 0; i < 145; i++) {
    const a = random() * tau;
    const r = (collarette(a) + (random() - .35) * .11) * 580;
    const length = 5 + random() * 20, breadth = 1.5 + random() * 4.5;
    tissuePen.save();tissuePen.rotate(a);tissuePen.translate(r, 0);
    tissuePen.fillStyle = `rgba(2,13,27,${.25 + random() * .4})`;
    tissuePen.beginPath();tissuePen.moveTo(-length, 0);
    tissuePen.bezierCurveTo(-length * .4, -breadth, length * .6, -breadth * .8, length, 0);
    tissuePen.bezierCurveTo(length * .3, breadth * 1.6, -length * .6, breadth * .5, -length, 0);
    tissuePen.fill();tissuePen.restore();
  }
  for (let k = 0; k < 4; k++) {
    tissuePen.beginPath();
    for (let i = 0; i <= 800; i++) {
      const a = i / 800 * tau;
      const r = (collarette(a) + (k - 1.5) * .004) * 580;
      if (!i) tissuePen.moveTo(Math.cos(a) * r, Math.sin(a) * r);
      else tissuePen.lineTo(Math.cos(a) * r, Math.sin(a) * r);
    }
    tissuePen.strokeStyle = k % 2 ? "rgba(181,202,207,.17)" : "rgba(5,22,37,.32)";
    tissuePen.lineWidth = 1.3;tissuePen.stroke();
  }
  for (let i = 0; i < 32; i++) {
    const a = random() * tau, arc = .12 + random() * .5;
    const r = .72 + random() * .2;
    tissuePen.beginPath();
    for (let j = 0; j <= 45; j++) {
      const angle = a + arc * j / 45;
      const rr = (r + Math.sin(angle * 31) * .0025 + Math.sin(angle * 13) * .004) * 580;
      if (!j) tissuePen.moveTo(Math.cos(angle) * rr, Math.sin(angle) * rr);
      else tissuePen.lineTo(Math.cos(angle) * rr, Math.sin(angle) * rr);
    }
    tissuePen.strokeStyle = "rgba(2,15,30,.22)";
    tissuePen.lineWidth = 1 + random() * 2;tissuePen.stroke();
  }
  const limbalShade = tissuePen.createRadialGradient(0, 0, 478, 0, 0, 581);
  limbalShade.addColorStop(0, "rgba(1,6,15,0)");
  limbalShade.addColorStop(.64, "rgba(1,6,15,.25)");
  limbalShade.addColorStop(.88, "rgba(1,6,15,.76)");
  limbalShade.addColorStop(1, "rgba(1,5,12,1)");
  tissuePen.fillStyle = limbalShade;
  tissuePen.beginPath();tissuePen.arc(0, 0, 581, 0, tau);tissuePen.fill();

  function resize() {
    width = experience.clientWidth;
    height = experience.clientHeight;
    ratio = Math.min(devicePixelRatio || 1, 1.7);
    canvas.width = light.width = Math.round(width * ratio);
    canvas.height = light.height = Math.round(height * ratio);
    radius = width < 600 ? Math.min(width * .395, height * .235) : Math.min(width * .27, height * .315);
    centerX = width * (width < 600 ? .5 : .535);
    centerY = height * (width < 600 ? .535 : .455);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    pen.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (reducedMotion.matches) render(performance.now());
  }

  function project(x, y, z, rotationX, rotationY, zoom = 1) {
    const cy = Math.cos(rotationY), sy = Math.sin(rotationY);
    const cx = Math.cos(rotationX), sx = Math.sin(rotationX);
    const xx = x * cy + z * sy;
    const zz = z * cy - x * sy;
    const yy = y * cx - zz * sx;
    const depth = 3.8 / (3.8 - (y * sx + zz * cx));
    return [centerX + xx * radius * depth * zoom, centerY + yy * radius * depth * zoom];
  }

  function render(now) {
    const elapsed = (now - startedAt) / 1000;
    if (elapsed >= 4.2 || reducedMotion.matches) finalScene = true;
    const gathering = finalScene ? 1 : mix(.72, 1, ease(elapsed / 4.2));
    const unfurl = ease((gathering - .12) / .8);
    const settle = ease((gathering - .58) / .42);
    const morph = 1 - gathering;
    if (finalScene && experience.dataset.intro !== "complete") experience.dataset.intro = "complete";

    const follow = 1 - Math.exp(-Math.min(now - (lastFrame || now), 70) / 150);
    lookX += (targetX - lookX) * follow;
    lookY += (targetY - lookY) * follow;
    const time = reducedMotion.matches ? 0 : elapsed;
    const rx = -lookY * .29 * (1 - morph) + Math.sin(time * .27) * .012;
    const ry = lookX * .36 * (1 - morph) + Math.sin(time * .21) * .018;
    const zoom = 1 + Math.sin(time * .75) * .009;
    const pupilShiftX = lookX * .043 * settle;
    const pupilShiftY = lookY * .043 * settle;
    const tilt = unfurl * Math.PI / 2;
    const tiltCos = Math.cos(tilt), tiltSin = Math.sin(tilt);
    const winding = (1 - settle) * (1.5 + Math.sin(unfurl * Math.PI) * 1.7);
    const rotation = (time * .38 + .35) * (1 - settle);

    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = "#020306";
    ctx.fillRect(0, 0, width, height);
    const ambient = ctx.createRadialGradient(centerX, centerY, radius * .15, centerX, centerY, radius * 1.8);
    ambient.addColorStop(0, `rgba(8,24,67,${.12 + morph * .25})`);
    ambient.addColorStop(.55, `rgba(8,16,51,${.05 + morph * .22})`);
    ambient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = ambient;
    ctx.fillRect(0, 0, width, height);

    let blink = 0;
    if (finalScene && !reducedMotion.matches) {
      if (blinkStarted < 0 && elapsed >= nextBlink) blinkStarted = elapsed;
      if (blinkStarted >= 0) {
        const age = elapsed - blinkStarted;
        if (age < .105) blink = ease(age / .105);
        else if (age < .14) blink = 1;
        else if (age < .38) blink = 1 - ease((age - .14) / .24);
        else {
          nextBlink = blinkStarted + 3;
          blinkStarted = -1;
        }
      }
    }

    ctx.save();
    if (blink > 0) {
      const eye = project(0, 0, 0, rx, ry, zoom);
      const halfWidth = radius * 1.3 * zoom;
      const gap = radius * 2.65 * (1 - blink) * zoom;
      ctx.beginPath();
      ctx.moveTo(eye[0] - halfWidth, eye[1]);
      ctx.quadraticCurveTo(eye[0], eye[1] - gap, eye[0] + halfWidth, eye[1]);
      ctx.quadraticCurveTo(eye[0], eye[1] + gap, eye[0] - halfWidth, eye[1]);
      ctx.closePath();ctx.clip();
    }
    pen.clearRect(0, 0, width, height);
    pen.globalCompositeOperation = "lighter";
    pen.lineCap = "round";

    const stride = width < 600 ? 2 : 1;
    for (let i = 0; i < fibers.length; i += stride) {
      const f = fibers[i];
      const lit = f.brightness;
      const pulse = .84 + Math.sin(time * 1.3 + f.phase) * .16;
      const exposure = mix(.045, .23, settle) * pulse;
      pen.strokeStyle = lit > .96 ? `rgba(206,227,255,${.56 * exposure})`
        : lit > .72 ? `rgba(132,178,255,${.53 * exposure})`
        : lit > .23 ? `rgba(76,119,245,${.47 * exposure})` : `rgba(42,70,189,${.39 * exposure})`;
      pen.lineWidth = f.width * (radius / 255) * (1 + morph * .35);
      pen.beginPath();
      let tip, spark;
      const sparkStep = Math.floor(((time * .35 + f.spread) % 1) * steps);
      for (let j = 0; j <= steps; j++) {
        const s = j / steps;
        let ex = f.points[j * 3], ey = f.points[j * 3 + 1], ez = f.points[j * 3 + 2];
        const inner = Math.pow(1 - s, 2);
        const shimmer = Math.sin(time * .65 + s * 8 + f.phase) * .0035 * s;
        ex += Math.cos(f.angle) * shimmer;
        ey += Math.sin(f.angle) * shimmer;
        // Rotate the helix axis toward the viewer, then release its winding into radial fibers.
        const u = s * f.length;
        const eyeRadius = Math.hypot(ex, ey);
        const theta = Math.atan2(ey, ex) + winding * Math.pow(1 - s, 1.12) * tau + rotation;
        const helixRadius = .25 + f.spread * .15 + Math.sin(u * 3 + time * .35) * .035;
        const radial = mix(helixRadius, eyeRadius, unfurl);
        const axis = (-1.7 + u * 2.8 + f.offset) * (1 - settle);
        const sway = Math.sin(u * 3.6 + time * .45) * .24 * (1 - unfurl);
        const x = Math.cos(theta) * radial + sway + pupilShiftX * inner;
        const y = axis * tiltCos + Math.sin(theta) * radial * tiltSin + pupilShiftY * inner;
        const z = axis * tiltSin - Math.sin(theta) * radial * tiltCos + ez * settle;
        const p = project(x, y, z, rx, ry, zoom);
        if (j === 0) pen.moveTo(p[0], p[1]); else pen.lineTo(p[0], p[1]);
        if (j === steps) tip = p;
        if (j === sparkStep) spark = p;
      }
      pen.stroke();
      if (morph > .02 && i % 18 === 0) {
        const dot = (.6 + f.brightness * 1.6) * Math.sqrt(morph) * radius / 255;
        pen.fillStyle = `rgba(166,205,255,${morph * .68})`;
        pen.beginPath();pen.arc(tip[0], tip[1], dot, 0, tau);pen.fill();
        pen.fillStyle = `rgba(205,228,255,${morph * .75})`;
        pen.beginPath();pen.arc(spark[0], spark[1], dot * .65, 0, tau);pen.fill();
      }
    }

    const irisOpacity = ease((gathering - .62) / .38);
    if (irisOpacity > .01) {
      const center = project(pupilShiftX * .3, pupilShiftY * .3, 0, rx, ry, zoom);
      const right = project(1 + pupilShiftX * .3, pupilShiftY * .3, 0, rx, ry, zoom);
      const bottom = project(pupilShiftX * .3, 1 + pupilShiftY * .3, 0, rx, ry, zoom);
      ctx.save();
      ctx.globalAlpha = irisOpacity;
      ctx.transform(right[0] - center[0], right[1] - center[1],
        bottom[0] - center[0], bottom[1] - center[1], center[0], center[1]);
      ctx.drawImage(tissue, -600 / 580, -600 / 580, 1200 / 580, 1200 / 580);
      ctx.restore();
    }

    ctx.globalCompositeOperation = "screen";
    ctx.globalAlpha = .45;
    ctx.filter = "blur(11px)";
    ctx.drawImage(light, 0, 0, width, height);
    ctx.globalAlpha = .55;
    ctx.filter = "blur(3px)";
    ctx.drawImage(light, 0, 0, width, height);
    ctx.filter = "none";
    ctx.globalAlpha = 1;
    ctx.drawImage(light, 0, 0, width, height);

    if (irisOpacity > .01) {
      const pupil = project(pupilShiftX, pupilShiftY, .02, rx, ry, zoom);
      ctx.save();
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = irisOpacity;
      ctx.translate(pupil[0], pupil[1]);
      ctx.scale(Math.cos(ry), Math.cos(rx));
      const r = radius * .275 * zoom;
      const core = ctx.createRadialGradient(-r * .15, -r * .1, 0, 0, 0, r * 1.08);
      core.addColorStop(0, "#010205");core.addColorStop(.88, "#010307");
      core.addColorStop(.98, "#040b13");core.addColorStop(1, "#243849");
      ctx.fillStyle = core;ctx.beginPath();ctx.arc(0, 0, r * 1.02, 0, tau);ctx.fill();
      ctx.strokeStyle = "rgba(99,143,172,.25)";ctx.lineWidth = 1.3;
      ctx.shadowBlur = 5;ctx.shadowColor = "#487d9f";
      ctx.beginPath();ctx.arc(0, 0, r * 1.02, 0, tau);ctx.stroke();ctx.restore();

      ctx.save();
      ctx.globalAlpha = irisOpacity;
      ctx.globalCompositeOperation = "screen";
      const reflection = project(-.28 + lookX * .025, -.33 + lookY * .025, .15, rx, ry, zoom);
      const sheen = ctx.createRadialGradient(reflection[0], reflection[1], 0,
        reflection[0], reflection[1], radius * .32);
      sheen.addColorStop(0, "rgba(204,229,246,.1)");
      sheen.addColorStop(1, "rgba(204,229,246,0)");
      ctx.fillStyle = sheen;ctx.beginPath();
      ctx.arc(reflection[0], reflection[1], radius * .32, 0, tau);ctx.fill();
      ctx.translate(reflection[0], reflection[1]);ctx.rotate(-.45);
      ctx.shadowBlur = 8;ctx.shadowColor = "rgba(214,232,255,.25)";
      ctx.fillStyle = "rgba(230,241,255,.34)";
      ctx.beginPath();ctx.ellipse(0, 0, radius * .055, radius * .024, 0, 0, tau);ctx.fill();
      ctx.fillStyle = "rgba(230,241,255,.16)";
      ctx.beginPath();ctx.ellipse(radius * .075, radius * .035, radius * .018, radius * .01, 0, 0, tau);ctx.fill();
      ctx.restore();
    }

    if (morph > .02) {
      ctx.globalCompositeOperation = "screen";
      for (const p of dust) {
        const x = centerX + p.x * radius + Math.sin(time * .2 + p.phase) * 12;
        const y = centerY + ((p.y + time * .045 + 2.1) % 4.2 - 2.1) * radius;
        ctx.fillStyle = `rgba(185,209,255,${morph * (.15 + p.z * .6)})`;
        ctx.beginPath();ctx.arc(x, y, .4 + p.z * 1.1, 0, tau);ctx.fill();
      }
    }
    ctx.restore();
    ctx.globalCompositeOperation = "source-over";
    lastFrame = now;
  }

  function animate(now) {
    if (now - lastFrame >= 1000 / 45) render(now);
    animationFrame = requestAnimationFrame(animate);
  }
  document.addEventListener("pointermove", (event) => {
    targetX = clamp((event.clientX - centerX) / (width * .42), -1, 1);
    targetY = clamp((event.clientY - centerY) / (height * .42), -1, 1);
    coordinates.textContent = `X ${targetX.toFixed(2)} / Y ${targetY.toFixed(2)}`;
    if (reducedMotion.matches) {
      lookX = targetX;lookY = targetY;render(performance.now());
    }
  }, { passive: true });
  document.documentElement.addEventListener("pointerleave", () => { targetX = targetY = 0; });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) cancelAnimationFrame(animationFrame);
    else if (!reducedMotion.matches) animationFrame = requestAnimationFrame(animate);
  });
  reducedMotion.addEventListener("change", () => {
    cancelAnimationFrame(animationFrame);
    finalScene = true;
    render(performance.now());
    if (!reducedMotion.matches) animationFrame = requestAnimationFrame(animate);
  });
  new ResizeObserver(resize).observe(experience);
  resize();
  if (!reducedMotion.matches) animationFrame = requestAnimationFrame(animate);
})();
