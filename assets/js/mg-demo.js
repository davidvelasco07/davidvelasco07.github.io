/*
	Geometric multigrid, three small demonstrations:
	  1. relaxation as a smoother (1D model problem),
	  2. one V-cycle step by step (1D),
	  3. a 2D Poisson solver for the gravitational potential of a few masses.
	The 2D solver is cell-centred with red-black Gauss-Seidel smoothing, as in the
	multigrid solvers of Athena++ and AthenaK (Tomida & Stone 2023).
*/
(function () {
'use strict';

const $ = id => document.getElementById(id);
const C = { ink: '#2f3437', ink3: '#8a9296', line: '#e4e8e8', grid: '#eef1f1', accent: '#2f9e7e', warm: '#e08e0b', red: '#d1495b', blue: '#3b6ea5' };

function setupCanvas(cv, aspect, maxH) {
	const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
	const wcss = cv.clientWidth, hcss = Math.round(Math.max(200, Math.min(wcss * aspect, maxH || 520)));
	cv.style.height = hcss + 'px';
	if (cv.width !== Math.round(wcss * dpr) || cv.height !== Math.round(hcss * dpr)) { cv.width = Math.round(wcss * dpr); cv.height = Math.round(hcss * dpr); }
	const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
	return { ctx, W: wcss, H: hcss };
}

function rng(seed) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; }

/* ------------------------------------------------------------------ */
/* 1D model problem  -u'' = f,  u(0) = u(1) = 0,  N intervals          */
/* ------------------------------------------------------------------ */

function smooth1D(v, f, kind) {
	const N = v.length - 1, h2 = 1 / (N * N);
	if (kind === 'jacobi') {
		const o = Float64Array.from(v), w = 2 / 3;
		for (let i = 1; i < N; i++) v[i] = (1 - w) * o[i] + w * 0.5 * (o[i - 1] + o[i + 1] + h2 * f[i]);
	} else if (kind === 'rb') {
		for (let i = 1; i < N; i += 2) v[i] = 0.5 * (v[i - 1] + v[i + 1] + h2 * f[i]);
		for (let i = 2; i < N; i += 2) v[i] = 0.5 * (v[i - 1] + v[i + 1] + h2 * f[i]);
	} else {
		for (let i = 1; i < N; i++) v[i] = 0.5 * (v[i - 1] + v[i + 1] + h2 * f[i]);
	}
}
function residual1D(v, f) {
	const N = v.length - 1, r = new Float64Array(N + 1), ih2 = N * N;
	for (let i = 1; i < N; i++) r[i] = f[i] - (2 * v[i] - v[i - 1] - v[i + 1]) * ih2;
	return r;
}
function restrict1D(r) {
	const Nc = (r.length - 1) / 2, c = new Float64Array(Nc + 1);
	for (let I = 1; I < Nc; I++) c[I] = 0.25 * r[2 * I - 1] + 0.5 * r[2 * I] + 0.25 * r[2 * I + 1];
	return c;
}
function prolongAdd1D(c, v) {
	const Nc = c.length - 1;
	for (let I = 0; I < Nc; I++) { v[2 * I] += c[I]; v[2 * I + 1] += 0.5 * (c[I] + c[I + 1]); }
}
function solve1D(f) {              // Thomas algorithm
	const N = f.length - 1, u = new Float64Array(N + 1), cp = new Float64Array(N + 1), dp = new Float64Array(N + 1), h2 = 1 / (N * N);
	for (let i = 1; i < N; i++) { const m = 2 + cp[i - 1]; cp[i] = -1 / m; dp[i] = (h2 * f[i] + dp[i - 1]) / m; }
	for (let i = N - 1; i >= 1; i--) u[i] = dp[i] - cp[i] * u[i + 1];
	return u;
}
function sineSpectrum(u) {
	const N = u.length - 1, a = new Float64Array(N);
	for (let k = 1; k < N; k++) { let s = 0; for (let i = 1; i < N; i++) s += u[i] * Math.sin(k * Math.PI * i / N); a[k] = Math.abs(2 * s / N); }
	return a;
}
function initialError(N, kind) {
	const u = new Float64Array(N + 1);
	if (kind === 'random') { const r = rng(7); for (let i = 1; i < N; i++) u[i] = 1.6 * r() - 0.8; }
	else for (let i = 1; i < N; i++) { const x = i / N; u[i] = 0.36 * (Math.sin(2 * Math.PI * x) + Math.sin(9 * Math.PI * x) + Math.sin(40 * Math.PI * x)); }
	return u;
}
const maxAbs = a => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m; };

/* ------------------------------------------------------------------ */
/* Widget 1: relaxation is a smoother                                  */
/* ------------------------------------------------------------------ */

function Smoother() {
	const cv = $('sm-canvas'); if (!cv) return;
	const N = 64, zero = new Float64Array(N + 1);
	let u, sweeps = 0, timer = null;
	const PAD = { l: 12, r: 12, t: 12, b: 30 };

	function reset() { u = initialError(N, $('sm-init').value); sweeps = 0; draw(); }
	function sweep(k) { for (let i = 0; i < k; i++) smooth1D(u, zero, $('sm-kind').value); sweeps += k; draw(); }

	function draw() {
		const { ctx, W, H } = setupCanvas(cv, 0.62, 420);
		ctx.clearRect(0, 0, W, H);
		const split = Math.round(H * 0.56), X = x => PAD.l + x * (W - PAD.l - PAD.r);
		// error curve
		const Y = y => PAD.t + (1.1 - y) / 2.2 * (split - PAD.t - 8);
		ctx.strokeStyle = '#d5dbdb'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(1), Y(0)); ctx.stroke();
		ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
		for (let i = 0; i <= N; i++) i ? ctx.lineTo(X(i / N), Y(u[i])) : ctx.moveTo(X(i / N), Y(u[i]));
		ctx.stroke();
		ctx.fillStyle = C.accent; for (let i = 1; i < N; i++) { ctx.beginPath(); ctx.arc(X(i / N), Y(u[i]), 2.2, 0, 2 * Math.PI); ctx.fill(); }
		ctx.fillStyle = C.ink3; ctx.font = '12px "Source Sans Pro", sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
		ctx.fillText('error', PAD.l + 2, PAD.t);
		// spectrum
		const a = sineSpectrum(u), y0 = H - PAD.b, hmax = H - PAD.b - split - 18, bw = (W - PAD.l - PAD.r) / (N - 1);
		ctx.strokeStyle = '#d5dbdb'; ctx.beginPath(); ctx.moveTo(X(0), y0 + 0.5); ctx.lineTo(X(1), y0 + 0.5); ctx.stroke();
		for (let k = 1; k < N; k++) {
			const hgt = Math.min(1, a[k] / 0.4) * hmax;
			ctx.fillStyle = k <= N / 2 ? C.blue : C.warm;
			ctx.fillRect(PAD.l + (k - 1) * bw + 0.5, y0 - hgt, Math.max(1, bw - 1.5), hgt);
		}
		ctx.fillStyle = C.ink3; ctx.fillText('amplitude of each wave in the error', PAD.l + 2, split + 2);
		ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left'; ctx.fillText('smooth', PAD.l, H - 8);
		ctx.textAlign = 'right'; ctx.fillText('rough', W - PAD.r, H - 8);
		ctx.textAlign = 'center'; ctx.fillStyle = C.blue; ctx.fillText('visible on a grid twice as coarse', X(0.25), H - 8);
		ctx.fillStyle = C.warm; ctx.fillText('only this grid can see', X(0.75), H - 8);
		$('sm-sweeps').textContent = sweeps; $('sm-max').textContent = maxAbs(u).toFixed(3);
	}

	$('sm-1').addEventListener('click', () => sweep(1));
	$('sm-10').addEventListener('click', () => sweep(10));
	$('sm-reset').addEventListener('click', () => { stop(); reset(); });
	$('sm-kind').addEventListener('change', () => { stop(); reset(); });
	$('sm-init').addEventListener('change', () => { stop(); reset(); });
	function stop() { if (timer) { clearInterval(timer); timer = null; $('sm-play').textContent = 'Play'; } }
	$('sm-play').addEventListener('click', () => {
		if (timer) return stop();
		$('sm-play').textContent = 'Pause'; timer = setInterval(() => { sweep(1); if (sweeps >= 2000) stop(); }, 110);
	});
	new IntersectionObserver(es => { if (!es[0].isIntersecting) stop(); }).observe(cv);
	new ResizeObserver(draw).observe(cv);
	reset();
}

/* ------------------------------------------------------------------ */
/* Widget 2: one V-cycle, step by step                                 */
/* ------------------------------------------------------------------ */

function VCycle() {
	const cv = $('vc-canvas'); if (!cv) return;
	const NS = [64, 32, 16, 8, 4, 2], L = NS.length, NU = 2;
	let v, f, ex, prev, scale, state, ops, ip, cycles, history, timer = null, lastOp = null;

	function buildOps() {
		const o = [];
		for (let l = 0; l < L - 1; l++) { o.push({ t: 'smooth', l, pre: true }); o.push({ t: 'restrict', l }); }
		o.push({ t: 'solve', l: L - 1 });
		for (let l = L - 2; l >= 0; l--) { o.push({ t: 'prolong', l }); o.push({ t: 'smooth', l, pre: false }); }
		return o;
	}
	const eps = l => { const e = new Float64Array(NS[l] + 1); for (let i = 0; i <= NS[l]; i++) e[i] = v[l][i] - ex[l][i]; return e; };

	function reset() {
		v = NS.map(n => new Float64Array(n + 1)); f = NS.map(n => new Float64Array(n + 1)); ex = NS.map(n => new Float64Array(n + 1));
		v[0] = initialError(NS[0], 'random'); prev = NS.map(() => null); scale = NS.map(() => 1); state = NS.map((_, l) => l === 0 ? 'active' : 'idle');
		scale[0] = maxAbs(v[0]); ops = buildOps(); ip = 0; cycles = 0; history = [maxAbs(v[0])]; lastOp = null; draw();
	}

	function startCycle() { ip = 0; state = NS.map((_, l) => l === 0 ? 'active' : 'idle'); prev = NS.map(() => null); scale[0] = maxAbs(v[0]) || 1; }

	function step() {
		if (ip >= ops.length) startCycle();
		const op = ops[ip++]; lastOp = op; const l = op.l;
		if (op.t === 'smooth') { prev[l] = eps(l); for (let k = 0; k < NU; k++) smooth1D(v[l], f[l], 'gs'); state[l] = 'active'; }
		else if (op.t === 'restrict') {
			f[l + 1] = restrict1D(residual1D(v[l], f[l])); v[l + 1] = new Float64Array(NS[l + 1] + 1); ex[l + 1] = solve1D(f[l + 1]);
			scale[l + 1] = maxAbs(ex[l + 1]) || 1; prev[l + 1] = null; state[l] = 'wait'; state[l + 1] = 'active';
		}
		else if (op.t === 'solve') { prev[l] = eps(l); v[l] = Float64Array.from(ex[l]); }
		else if (op.t === 'prolong') { prev[l] = eps(l); prolongAdd1D(v[l + 1], v[l]); state[l + 1] = 'done'; state[l] = 'active'; }
		if (ip >= ops.length) { cycles++; history.push(maxAbs(v[0])); }
		draw();
	}

	const CAP = {
		start: 'The top row is the error on the finest grid, a mix of rough and smooth waves. Each row below is a grid with half as many cells. Press <b>Step</b>.',
		pre: n => `<b>Smooth.</b> Two Gauss–Seidel sweeps on the ${n}-cell grid remove the rough part of the error (the faint line is the error before). What is left is smooth, and relaxation would now make very slow progress.`,
		restrict: (n, m) => `<b>Restrict.</b> The equation for what is left is passed to the ${m}-cell grid. On this grid the same error spans fewer cells, so it is rough again relative to the spacing, and relaxation works on it.`,
		solve: '<b>Solve.</b> The coarsest grid has a single unknown; it is solved exactly.',
		prolong: (n, m) => `<b>Prolong and correct.</b> The solution of the ${m}-cell grid is interpolated to the ${n}-cell grid and added. The smooth error is gone; the interpolation leaves a little roughness behind.`,
		post: n => `<b>Smooth again.</b> Two more sweeps on the ${n}-cell grid remove what the interpolation introduced.`,
		done: (k, fac) => `<b>V-cycle ${k} complete.</b> The error on the finest grid is ${fac.toFixed(0)} times smaller than before the cycle, for the cost of about six sweeps on that grid. Press Step to start the next one (the top row is rescaled).`
	};

	function caption() {
		if (!lastOp) return CAP.start;
		if (ip >= ops.length) return CAP.done(cycles, history[history.length - 2] / history[history.length - 1]);
		const o = lastOp, n = NS[o.l];
		if (o.t === 'smooth') return o.pre ? CAP.pre(n) : CAP.post(n);
		if (o.t === 'restrict') return CAP.restrict(n, NS[o.l + 1]);
		if (o.t === 'solve') return CAP.solve;
		return CAP.prolong(n, NS[o.l + 1]);
	}

	function draw() {
		const { ctx, W, H } = setupCanvas(cv, 0.82, 560);
		ctx.clearRect(0, 0, W, H); ctx.font = '12px "Source Sans Pro", sans-serif';
		const sw = Math.max(86, Math.min(170, W * 0.24)), rowH = (H - 12) / L, x0 = sw + 14, x1 = W - 10;
		// schedule
		const lx = k => 16 + (sw - 32) * k / (2 * (L - 1)), ly = l => 6 + rowH * (l + 0.5);
		ctx.strokeStyle = '#c3cfcc'; ctx.lineWidth = 2; ctx.beginPath();
		for (let k = 0; k <= 2 * (L - 1); k++) { const l = k < L ? k : 2 * (L - 1) - k; k ? ctx.lineTo(lx(k), ly(l)) : ctx.moveTo(lx(k), ly(l)); }
		ctx.stroke();
		let cur = -1;
		if (lastOp && ip < ops.length + 1) {
			const o = lastOp; const down = (o.t === 'smooth' && o.pre) || o.t === 'restrict';
			const lev = o.t === 'restrict' ? o.l + 1 : o.l;
			cur = (o.t === 'solve') ? L - 1 : (down ? lev : 2 * (L - 1) - lev);
		} else cur = 0;
		for (let k = 0; k <= 2 * (L - 1); k++) {
			const l = k < L ? k : 2 * (L - 1) - k;
			ctx.beginPath(); ctx.arc(lx(k), ly(l), k === cur ? 7 : 4.5, 0, 2 * Math.PI);
			ctx.fillStyle = k === cur ? C.warm : (k < cur ? C.accent : '#c3cfcc'); ctx.fill();
		}
		// rows
		for (let l = 0; l < L; l++) {
			const yt = 6 + rowH * l, yc = yt + rowH / 2, amp = rowH * 0.38;
			if (state[l] === 'active') { ctx.fillStyle = '#eef6f3'; ctx.fillRect(x0 - 6, yt + 2, x1 - x0 + 12, rowH - 4); }
			ctx.strokeStyle = '#dfe5e4'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, yc); ctx.lineTo(x1, yc); ctx.stroke();
			ctx.fillStyle = C.ink3; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
			ctx.fillText(NS[l] + ' cells', x0, yt + 5);
			if (state[l] === 'idle') continue;
			const n = NS[l], X = i => x0 + (x1 - x0) * i / n, Y = y => yc - amp * Math.max(-1.25, Math.min(1.25, y / scale[l]));
			const line = (e, color, width) => { ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.beginPath(); for (let i = 0; i <= n; i++) i ? ctx.lineTo(X(i), Y(e[i])) : ctx.moveTo(X(i), Y(e[i])); ctx.stroke(); };
			const dim = state[l] !== 'active';
			if (prev[l] && !dim) line(prev[l], 'rgba(47,158,126,0.28)', 1.5);
			const e = eps(l); line(e, dim ? '#b7c4c1' : C.accent, 2);
			if (n <= 32) { ctx.fillStyle = dim ? '#b7c4c1' : C.accent; for (let i = 1; i < n; i++) { ctx.beginPath(); ctx.arc(X(i), Y(e[i]), n <= 8 ? 3.5 : 2.5, 0, 2 * Math.PI); ctx.fill(); } }
			ctx.fillStyle = C.ink3; ctx.textAlign = 'right'; ctx.fillText('max ' + maxAbs(e).toExponential(1), x1, yt + 5);
		}
		$('vc-cap').innerHTML = caption();
		$('vc-hist').textContent = history.map(x => x.toExponential(1)).join('  →  ');
	}

	function stop() { if (timer) { clearInterval(timer); timer = null; $('vc-run').textContent = 'Run a full cycle'; } }
	$('vc-step').addEventListener('click', () => { stop(); step(); });
	$('vc-run').addEventListener('click', () => {
		if (timer) return stop();
		$('vc-run').textContent = 'Pause'; if (ip >= ops.length) startCycle();
		timer = setInterval(() => { step(); if (ip >= ops.length) stop(); }, 650);
	});
	$('vc-reset').addEventListener('click', () => { stop(); reset(); });
	new IntersectionObserver(es => { if (!es[0].isIntersecting) stop(); }).observe(cv);
	new ResizeObserver(draw).observe(cv);
	reset();
}

/* ------------------------------------------------------------------ */
/* 2D cell-centred multigrid for  laplacian(phi) = rho,  phi = 0 on    */
/* the walls                                                           */
/* ------------------------------------------------------------------ */

function Level(n) { this.n = n; this.s = n + 2; this.h2 = 1 / (n * n); this.u = new Float64Array(this.s * this.s); this.f = new Float64Array(this.s * this.s); this.r = new Float64Array(this.s * this.s); }

function bc(L, a) {
	const n = L.n, s = L.s;
	for (let k = 1; k <= n; k++) { a[k] = -a[k + s]; a[k + (n + 1) * s] = -a[k + n * s]; a[k * s] = -a[1 + k * s]; a[n + 1 + k * s] = -a[n + k * s]; }
	a[0] = a[1 + s]; a[n + 1] = a[n + s]; a[(n + 1) * s] = a[1 + n * s]; a[n + 1 + (n + 1) * s] = a[n + n * s];
}
function smooth2D(L) {
	const n = L.n, s = L.s, u = L.u, f = L.f, h2 = L.h2;
	if (n === 1) { u[1 + s] = -h2 * f[1 + s] / 8; return; }
	for (let color = 0; color < 2; color++) {
		bc(L, u);
		for (let j = 1; j <= n; j++) for (let i = 1 + ((j + color) & 1); i <= n; i += 2) { const k = i + j * s; u[k] = 0.25 * (u[k - 1] + u[k + 1] + u[k - s] + u[k + s] - h2 * f[k]); }
	}
}
function residual2D(L) {
	const n = L.n, s = L.s, u = L.u, f = L.f, r = L.r, ih2 = 1 / L.h2; bc(L, u);
	for (let j = 1; j <= n; j++) for (let i = 1; i <= n; i++) { const k = i + j * s; r[k] = f[k] - (u[k - 1] + u[k + 1] + u[k - s] + u[k + s] - 4 * u[k]) * ih2; }
}
function restrict2D(F, src, Cc, dst) {
	const n = Cc.n, s = Cc.s, sf = F.s;
	for (let J = 1; J <= n; J++) for (let I = 1; I <= n; I++) { const k = (2 * I - 1) + (2 * J - 1) * sf; dst[I + J * s] = 0.25 * (src[k] + src[k + 1] + src[k + sf] + src[k + 1 + sf]); }
}
function prolong2D(Cc, F, add) {
	const n = Cc.n, s = Cc.s, sf = F.s, c = Cc.u, u = F.u; bc(Cc, c);
	for (let J = 1; J <= n; J++) for (let I = 1; I <= n; I++) {
		const k = I + J * s, c0 = 9 * c[k], kf = (2 * I - 1) + (2 * J - 1) * sf;
		const a = (c0 + 3 * c[k - 1] + 3 * c[k - s] + c[k - 1 - s]) / 16, b = (c0 + 3 * c[k + 1] + 3 * c[k - s] + c[k + 1 - s]) / 16;
		const d = (c0 + 3 * c[k - 1] + 3 * c[k + s] + c[k - 1 + s]) / 16, e = (c0 + 3 * c[k + 1] + 3 * c[k + s] + c[k + 1 + s]) / 16;
		if (add) { u[kf] += a; u[kf + 1] += b; u[kf + sf] += d; u[kf + 1 + sf] += e; } else { u[kf] = a; u[kf + 1] = b; u[kf + sf] = d; u[kf + 1 + sf] = e; }
	}
}
function Hierarchy(n) { this.levels = []; for (let m = n; m >= 1; m >>= 1) this.levels.push(new Level(m)); this.work = 0; }
Hierarchy.prototype.cost = function (l, c) { const r = this.levels[l].n / this.levels[0].n; this.work += c * r * r; };
Hierarchy.prototype.vcycle = function (l, nu1, nu2) {
	const Ls = this.levels, F = Ls[l];
	if (l === Ls.length - 1) { smooth2D(F); return; }
	for (let k = 0; k < nu1; k++) smooth2D(F); this.cost(l, nu1);
	residual2D(F); this.cost(l, 1);
	const Cc = Ls[l + 1]; restrict2D(F, F.r, Cc, Cc.f); Cc.u.fill(0);
	this.vcycle(l + 1, nu1, nu2);
	prolong2D(Cc, F, true); this.cost(l, 0.5);
	for (let k = 0; k < nu2; k++) smooth2D(F); this.cost(l, nu2);
};
Hierarchy.prototype.fmg = function (nu1, nu2) {
	const Ls = this.levels;
	for (let l = 0; l < Ls.length - 1; l++) restrict2D(Ls[l], Ls[l].f, Ls[l + 1], Ls[l + 1].f);
	Ls[Ls.length - 1].u.fill(0); smooth2D(Ls[Ls.length - 1]);
	for (let l = Ls.length - 2; l >= 0; l--) {
		prolong2D(Ls[l + 1], Ls[l], false);
		const saved = Ls.slice(l + 1).map(x => Float64Array.from(x.f));
		this.vcycle(l, nu1, nu2);
		Ls.slice(l + 1).forEach((x, i) => x.f.set(saved[i]));
	}
};

const MAGMA = ['#fcfdbf', '#feca8d', '#fd9668', '#f1605d', '#cd4071', '#9e2f7f', '#721f81', '#440f76', '#180f3d', '#000004'].map(h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
function magma(t) {
	t = Math.max(0, Math.min(1, t)) * (MAGMA.length - 1); const i = Math.min(MAGMA.length - 2, Math.floor(t)), w = t - i, a = MAGMA[i], b = MAGMA[i + 1];
	return [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w];
}
function diverging(t) {            // t in [-1, 1]: blue - white - red
	const a = Math.min(1, Math.abs(t)), c = t < 0 ? [59, 110, 165] : [209, 73, 91];
	return [255 + (c[0] - 255) * a, 255 + (c[1] - 255) * a, 255 + (c[2] - 255) * a];
}

/* ------------------------------------------------------------------ */
/* Widget 3: gravity in a box                                          */
/* ------------------------------------------------------------------ */

function Gravity() {
	const cv = $('gv-canvas'), ch = $('gv-chart'); if (!cv) return;
	const n = 128, IMG = 384, NU1 = 2, NU2 = 2;
	const H = new Hierarchy(n), R = new Hierarchy(n), top = H.levels[0], ref = R.levels[0];
	const masses = [{ x: 0.36, y: 0.58, m: 1.0 }, { x: 0.66, y: 0.40, m: 0.6 }, { x: 0.70, y: 0.74, m: 0.25 }];
	let mode = 'mg', view = 'phi', running = true, raf = null, drag = null, phiMin = -1, err0 = 1, hist = { gs: [], mg: [] }, dirty = true;
	const off = document.createElement('canvas'); off.width = IMG; off.height = IMG; const octx = off.getContext('2d'), img = octx.createImageData(IMG, IMG);
	const field = new Float64Array(IMG * IMG);

	function setSource() {
		const s = top.s, sig = 2.2 / n;
		for (let j = 1; j <= n; j++) for (let i = 1; i <= n; i++) {
			const x = (i - 0.5) / n, y = (j - 0.5) / n; let rho = 0;
			for (const q of masses) { const d2 = (x - q.x) * (x - q.x) + (y - q.y) * (y - q.y); rho += q.m * Math.exp(-0.5 * d2 / (sig * sig)); }
			top.f[i + j * s] = rho * 4000; ref.f[i + j * s] = rho * 4000;
		}
		R.fmg(NU1, NU2); for (let k = 0; k < 12; k++) R.vcycle(0, NU1, NU2);    // converged reference
		phiMin = 0; for (let k = 0; k < ref.u.length; k++) phiMin = Math.min(phiMin, ref.u[k]);
		restartHistory();
	}
	function error() { let e = 0; const s = top.s; for (let j = 1; j <= n; j++) for (let i = 1; i <= n; i++) { const k = i + j * s; e = Math.max(e, Math.abs(top.u[k] - ref.u[k])); } return e; }
	function restartHistory() { H.work = 0; err0 = error() || 1; hist[mode] = [[0, 1]]; dirty = true; }

	function iterate() {
		const e = error() / err0;
		if (e < 1e-10 || hist[mode].length > 4000) return false;
		if (mode === 'mg') H.vcycle(0, NU1, NU2);
		else { const k = +$('gv-sweeps').value; for (let i = 0; i < k; i++) smooth2D(top); H.work += k; }
		hist[mode].push([H.work, Math.max(1e-10, error() / err0)]); dirty = true; return true;
	}

	function render() {
		const { ctx, W, H: Hc } = setupCanvas(cv, 1, 560);
		// bilinear sampling of the cell-centred field onto the image
		const s = top.s, a = view === 'phi' ? null : ref.u, u = top.u; bc(top, u);
		for (let py = 0; py < IMG; py++) {
			const gy = (py + 0.5) / IMG * n + 0.5, j = Math.floor(gy), wy = gy - j;
			for (let px = 0; px < IMG; px++) {
				const gx = (px + 0.5) / IMG * n + 0.5, i = Math.floor(gx), wx = gx - i, k = i + j * s;
				let v = (1 - wx) * (1 - wy) * u[k] + wx * (1 - wy) * u[k + 1] + (1 - wx) * wy * u[k + s] + wx * wy * u[k + 1 + s];
				if (a) v -= (1 - wx) * (1 - wy) * a[k] + wx * (1 - wy) * a[k + 1] + (1 - wx) * wy * a[k + s] + wx * wy * a[k + 1 + s];
				field[px + py * IMG] = v;
			}
		}
		let emax = 0; if (view !== 'phi') { for (let p = 0; p < IMG * IMG; p++) emax = Math.max(emax, Math.abs(field[p])); if (emax < 1e-9 * Math.abs(phiMin)) emax = Infinity; }
		const d = img.data, NC = 14;
		for (let p = 0; p < IMG * IMG; p++) {
			let c;
			if (view === 'phi') {
				const t = field[p] / phiMin; c = magma(Math.pow(Math.max(0, t), 0.75));
				const lev = Math.floor(t * NC), px = p % IMG;
				if ((px < IMG - 1 && Math.floor(field[p + 1] / phiMin * NC) !== lev) || (p < IMG * (IMG - 1) && Math.floor(field[p + IMG] / phiMin * NC) !== lev)) c = [c[0] * 0.78, c[1] * 0.78, c[2] * 0.78];
			} else c = diverging(-field[p] / emax);
			d[4 * p] = c[0]; d[4 * p + 1] = c[1]; d[4 * p + 2] = c[2]; d[4 * p + 3] = 255;
		}
		octx.putImageData(img, 0, 0);
		ctx.imageSmoothingEnabled = true; ctx.clearRect(0, 0, W, Hc); ctx.drawImage(off, 0, 0, W, Hc);
		for (const q of masses) {
			ctx.beginPath(); ctx.arc(q.x * W, q.y * Hc, 6 + 7 * Math.sqrt(q.m), 0, 2 * Math.PI);
			ctx.fillStyle = 'rgba(255,255,255,0.92)'; ctx.fill(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.stroke();
		}
		drawChart(); dirty = false;
		const last = hist[mode][hist[mode].length - 1];
		$('gv-work').textContent = last[0].toFixed(0); $('gv-err').textContent = last[1].toExponential(1);
	}

	function drawChart() {
		const { ctx, W, H: Hc } = setupCanvas(ch, 0.42, 230), PAD = { l: 46, r: 12, t: 10, b: 32 }, XM = 400, YM = -10;
		const X = w => PAD.l + Math.min(1, w / XM) * (W - PAD.l - PAD.r), Y = e => PAD.t + Math.min(1, Math.log10(Math.max(e, 1e-12)) / YM) * (Hc - PAD.t - PAD.b);
		ctx.clearRect(0, 0, W, Hc); ctx.font = '12px "Source Sans Pro", sans-serif'; ctx.lineWidth = 1;
		ctx.strokeStyle = C.grid; ctx.fillStyle = C.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
		for (let e = 0; e >= YM; e -= 2) { const y = Y(Math.pow(10, e)); ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke(); ctx.fillText('1e' + e, PAD.l - 6, y); }
		ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
		for (let w = 0; w <= XM; w += 100) ctx.fillText(w, X(w), Hc - PAD.b + 15);
		ctx.fillText('work, in sweeps of the finest grid', (PAD.l + W - PAD.r) / 2, Hc - 4);
		const curve = (h, color, label, row) => {
			ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
			const lx = W - PAD.r - 118, ly = PAD.t + 46 + 17 * row;
			ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke(); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(label, lx + 24, ly);
			if (!h.length) return;
			ctx.beginPath(); h.forEach((p, i) => { if (p[0] > XM) return; i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])); }); ctx.stroke();
			if (h.length < 40) h.forEach(p => { if (p[0] <= XM) { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3, 0, 2 * Math.PI); ctx.fill(); } });
		};
		curve(hist.mg, C.accent, 'multigrid', 0); curve(hist.gs, C.warm, 'relaxation only', 1);
	}

	function frame() { if (running) { if (iterate()) dirty = true; } if (dirty) render(); raf = requestAnimationFrame(frame); }

	function pick(ev) { const r = cv.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width, y = (ev.clientY - r.top) / r.height; let best = null, bd = 0.07; masses.forEach(q => { const dd = Math.hypot(q.x - x, q.y - y); if (dd < bd) { bd = dd; best = q; } }); return { q: best, x, y }; }
	cv.addEventListener('pointerdown', ev => { const p = pick(ev); if (p.q) { drag = p.q; try { cv.setPointerCapture(ev.pointerId); } catch (e) {} ev.preventDefault(); } });
	cv.addEventListener('pointermove', ev => {
		const p = pick(ev); if (!drag) { cv.style.cursor = p.q ? 'grab' : 'default'; return; }
		drag.x = Math.max(0.08, Math.min(0.92, p.x)); drag.y = Math.max(0.08, Math.min(0.92, p.y)); setSource();
	});
	const end = () => { drag = null; }; cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);

	const modeBtns = Array.from(document.querySelectorAll('#gv-mode button'));
	modeBtns.forEach(b => b.addEventListener('click', () => { mode = b.dataset.mode; modeBtns.forEach(x => x.setAttribute('aria-pressed', x === b ? 'true' : 'false')); $('gv-sweeps-wrap').style.visibility = mode === 'gs' ? 'visible' : 'hidden'; restartHistory(); }));
	const viewBtns = Array.from(document.querySelectorAll('#gv-view button'));
	viewBtns.forEach(b => b.addEventListener('click', () => { view = b.dataset.view; viewBtns.forEach(x => x.setAttribute('aria-pressed', x === b ? 'true' : 'false')); dirty = true; }));
	$('gv-zero').addEventListener('click', () => { top.u.fill(0); hist = { gs: mode === 'gs' ? [] : hist.gs, mg: mode === 'mg' ? [] : hist.mg }; restartHistory(); });
	$('gv-pause').addEventListener('click', () => { running = !running; $('gv-pause').textContent = running ? 'Pause' : 'Resume'; });
	$('gv-sweeps').addEventListener('input', ev => { $('gv-sweeps-out').textContent = ev.target.value; });
	new ResizeObserver(() => { dirty = true; }).observe(cv);
	new IntersectionObserver(es => { if (es[0].isIntersecting) { if (!raf) raf = requestAnimationFrame(frame); } else if (raf) { cancelAnimationFrame(raf); raf = null; } }).observe(cv);
	setSource();
	return { H, R, error: () => error() / err0 };
}

document.addEventListener('DOMContentLoaded', () => {
	Smoother(); VCycle(); const g = Gravity();
	window.MGDemo = { g, Hierarchy, smooth2D, residual2D, smooth1D, residual1D, restrict1D, prolongAdd1D, solve1D, initialError, maxAbs };
});
})();
