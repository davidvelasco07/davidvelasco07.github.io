/*
	Spectral Difference method in 1D (linear advection and Burgers' equation), with a posteriori limiting.
	A small JavaScript transcription of the algorithm in
	Velasco-Romero, Han Veiga & Teyssier (2023), MNRAS, arXiv:2209.06597,
	following the reference Python code at github.com/davidvelasco07/spd.
*/
(function () {
'use strict';

/* ------------------------------------------------------------------ */
/* Polynomial tools                                                    */
/* ------------------------------------------------------------------ */

function legendre(n, z) {            // returns [P_n(z), P_n'(z)]
	let p1 = 1, p2 = 0;
	for (let j = 1; j <= n; j++) { const p3 = p2; p2 = p1; p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j; }
	return [p1, n * (z * p1 - p2) / (z * z - 1)];
}

function gaussLegendre(n) {          // nodes and weights on [0, 1]
	const x = [], w = [];
	for (let i = 0; i < n; i++) {
		let z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5));
		for (let it = 0; it < 100; it++) {
			const [p, dp] = legendre(n, z);
			const dz = p / dp; z -= dz;
			if (Math.abs(dz) < 1e-15) break;
		}
		const dp = legendre(n, z)[1];
		x.push((z + 1) / 2); w.push(1 / ((1 - z * z) * dp * dp));
	}
	const idx = x.map((_, i) => i).sort((a, b) => x[a] - x[b]);
	return { x: idx.map(i => x[i]), w: idx.map(i => w[i]) };
}

// p + 1 Chebyshev-Gauss points on [0, 1]
function solutionPoints(p) {
	const x = [];
	for (let i = 0; i <= p; i++) x.push((1 - Math.cos((2 * i + 1) / (2 * (p + 1)) * Math.PI)) / 2);
	return x;
}

// the two faces plus the p Gauss-Legendre points on [0, 1]
function fluxPoints(p) { return [0].concat(gaussLegendre(p).x, [1]); }

function lagrangeBasis(xFrom, j, x) {
	let v = 1;
	for (let m = 0; m < xFrom.length; m++) if (m !== j) v *= (x - xFrom[m]) / (xFrom[j] - xFrom[m]);
	return v;
}

function lagrangeBasisPrime(xFrom, j, x) {
	let s = 0;
	for (let k = 0; k < xFrom.length; k++) {
		if (k === j) continue;
		let v = 1 / (xFrom[j] - xFrom[k]);
		for (let m = 0; m < xFrom.length; m++) if (m !== j && m !== k) v *= (x - xFrom[m]) / (xFrom[j] - xFrom[m]);
		s += v;
	}
	return s;
}

function matrix(rows, cols, f) {
	const M = [];
	for (let i = 0; i < rows; i++) { const r = new Float64Array(cols); for (let j = 0; j < cols; j++) r[j] = f(i, j); M.push(r); }
	return M;
}
const lagrangeMatrix = (xTo, xFrom) => matrix(xTo.length, xFrom.length, (i, j) => lagrangeBasis(xFrom, j, xTo[i]));
const lagrangePrimeMatrix = (xTo, xFrom) => matrix(xTo.length, xFrom.length, (i, j) => lagrangeBasisPrime(xFrom, j, xTo[i]));

function matmul(A, B) {
	return matrix(A.length, B[0].length, (i, j) => { let s = 0; for (let k = 0; k < B.length; k++) s += A[i][k] * B[k][j]; return s; });
}

function inverse(A) {
	const n = A.length, M = A.map((r, i) => { const row = new Float64Array(2 * n); row.set(r); row[n + i] = 1; return row; });
	for (let c = 0; c < n; c++) {
		let piv = c;
		for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
		[M[c], M[piv]] = [M[piv], M[c]];
		const d = M[c][c];
		for (let k = 0; k < 2 * n; k++) M[c][k] /= d;
		for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; if (f !== 0) for (let k = 0; k < 2 * n; k++) M[r][k] -= f * M[c][k]; }
	}
	return M.map(r => r.slice(n));
}

// values at the solution points -> averages over the control volumes bounded by the flux points
function intFromSolMatrix(xs, xf) {
	const q = gaussLegendre(xs.length + 1);
	return matrix(xs.length, xs.length, (i, j) => {
		let s = 0;
		for (let k = 0; k < q.x.length; k++) s += q.w[k] * lagrangeBasis(xs, j, xf[i] + (xf[i + 1] - xf[i]) * q.x[k]);
		return s;
	});
}

const schemeCache = {};
function makeScheme(p) {
	if (schemeCache[p]) return schemeCache[p];
	const xs = solutionPoints(p), xf = fluxPoints(p), n = p + 1;
	const S2F = lagrangeMatrix(xf, xs);            // solution points -> flux points
	const DF = lagrangePrimeMatrix(xs, xf);        // d/dx of the flux polynomial at the solution points
	const S2M = intFromSolMatrix(xs, xf);          // point values -> control-volume averages
	const M2S = inverse(S2M);
	const M2F = matmul(S2F, M2S);                  // averages -> flux points
	const wref = [], xcref = [];
	for (let i = 0; i < n; i++) { wref.push(xf[i + 1] - xf[i]); xcref.push((xf[i] + xf[i + 1]) / 2); }
	return (schemeCache[p] = { p, n, nf: p + 2, xs, xf, S2F, DF, S2M, M2S, M2F, wref, xcref });
}

/* ------------------------------------------------------------------ */
/* Profiles (values between 1 and 2, like a density)                   */
/* ------------------------------------------------------------------ */

const wrap = x => x - Math.floor(x);
const PROFILES = {
	mixed: x => { x = wrap(x); const g = Math.exp(-0.5 * Math.pow((x - 0.25) / 0.055, 2)); return 1 + g + (x > 0.55 && x < 0.85 ? 1 : 0); },
	square: x => { x = wrap(x); return (x > 0.3 && x < 0.7) ? 2 : 1; },
	gauss: x => 1 + Math.exp(-0.5 * Math.pow((wrap(x) - 0.5) / 0.09, 2)),
	sine: x => 1.5 + 0.5 * Math.sin(2 * Math.PI * x)
};

/* ------------------------------------------------------------------ */
/* Solver on control-volume averages                                   */
/* ------------------------------------------------------------------ */

const Q3 = gaussLegendre(4);

// burgers = true solves f = u^2 / 2; the profiles stay between 1 and 2, so the wave always moves to the right
function Sim(p, N, profile, limiter, burgers) {
	const s = makeScheme(p);
	this.s = s; this.p = p; this.n = s.n; this.N = N; this.M = N * s.n; this.h = 1 / N;
	this.profile = PROFILES[profile]; this.limiter = limiter; this.tol = 1e-5; this.cfl = 0.6;
	this.f = burgers ? (v => 0.5 * v * v) : (v => v); this.amax = burgers ? 2 : 1;
	const M = this.M;
	this.w = new Float64Array(M); this.xc = new Float64Array(M);
	for (let e = 0; e < N; e++) for (let k = 0; k < s.n; k++) {
		this.w[e * s.n + k] = s.wref[k] * this.h; this.xc[e * s.n + k] = (e + s.xcref[k]) * this.h;
	}
	let wmin = Infinity; for (let i = 0; i < M; i++) wmin = Math.min(wmin, this.w[i]);
	this.dt = this.cfl * wmin / this.amax;
	this.u = new Float64Array(M); this.u1 = new Float64Array(M); this.u2 = new Float64Array(M); this.cand = new Float64Array(M);
	this.F = new Float64Array(M); this.tr = new Float64Array(N);
	this.flag = new Uint8Array(M); this.nad = new Uint8Array(M); this.shown = new Uint8Array(M);
	this.dU = new Float64Array(M); this.alpha = new Float64Array(M);
	this.t = 0; this.nflag = 0;
	this.means(0, this.u);
}

// exact control-volume averages of the profile translated by t
Sim.prototype.means = function (t, out) {
	for (let i = 0; i < this.M; i++) {
		const a = this.xc[i] - this.w[i] / 2; let v = 0;
		for (let k = 0; k < Q3.x.length; k++) v += Q3.w[k] * this.profile(a + this.w[i] * Q3.x[k] - t);
		out[i] = v;
	}
	return out;
};

Sim.prototype.highOrderFlux = function (u) {
	const { n, N, F, tr, f } = this, M2F = this.s.M2F, p = this.p;
	for (let e = 0; e < N; e++) {
		const o = e * n;
		for (let k = 1; k <= p; k++) { const r = M2F[k]; let v = 0; for (let j = 0; j < n; j++) v += r[j] * u[o + j]; F[o + k] = f(v); }
		const r = M2F[p + 1]; let v = 0; for (let j = 0; j < n; j++) v += r[j] * u[o + j]; tr[e] = v;
	}
	for (let e = 0; e < N; e++) F[e * n] = f(tr[(e + N - 1) % N]);   // upwind: the wave moves to the right
};

function minmod(a, b) { return a * b <= 0 ? 0 : (Math.abs(a) < Math.abs(b) ? a : b); }

// second-order MUSCL flux at face g, between control volumes g-1 and g
Sim.prototype.musclFlux = function (u, g) {
	const M = this.M, w = this.w, L = (g + M - 1) % M, LL = (L + M - 1) % M, R = g % M;
	const sl = (u[L] - u[LL]) / (0.5 * (w[LL] + w[L])), sr = (u[R] - u[L]) / (0.5 * (w[L] + w[R]));
	return this.f(u[L] + 0.5 * w[L] * minmod(sl, sr));
};

// numerical admissibility (discrete maximum principle), relaxed at smooth extrema
Sim.prototype.detect = function (u, c) {
	const M = this.M, flag = this.flag, tol = this.tol, w = this.w; let any = 0;
	for (let i = 0; i < M; i++) {
		const a = u[(i + M - 1) % M], b = u[i], d = u[(i + 1) % M];
		const mn = Math.min(a, b, d), mx = Math.max(a, b, d);
		flag[i] = (c[i] >= mn - Math.abs(mn) * tol && c[i] <= mx + Math.abs(mx) * tol) ? 0 : 1;
		any |= flag[i];
	}
	this.nad.set(flag);                              // outside the bounds, before the smooth-extremum test
	if (!any || this.p < 2) return any;
	const dU = this.dU, alpha = this.alpha;
	const span = i => 0.5 * (w[(i + M - 1) % M] + w[(i + 1) % M]) + w[i];
	for (let i = 0; i < M; i++) dU[i] = (c[(i + 1) % M] - c[(i + M - 1) % M]) / span(i);
	for (let i = 0; i < M; i++) {
		const im = (i + M - 1) % M, ip = (i + 1) % M;
		const dv = 0.5 * w[i] * (dU[ip] - dU[im]) / span(i);
		if (dv === 0) { alpha[i] = 1; continue; }
		const vL = dU[im] - dU[i], vR = dU[ip] - dU[i];
		const aL = Math.min(1, -(dv < 0 ? Math.max(vL, 0) : Math.min(vL, 0)) / dv);
		const aR = Math.min(1, (dv > 0 ? Math.max(vR, 0) : Math.min(vR, 0)) / dv);
		alpha[i] = Math.min(aL, aR);
	}
	any = 0;
	for (let i = 0; i < M; i++) {
		if (!flag[i]) continue;
		if (Math.min(alpha[(i + M - 1) % M], alpha[i], alpha[(i + 1) % M]) >= 1) flag[i] = 0; else any = 1;
	}
	return any;
};

// one forward-Euler stage, corrected a posteriori
Sim.prototype.stage = function (u, out) {
	const M = this.M, F = this.F, w = this.w, dt = this.dt;
	this.highOrderFlux(u);
	for (let i = 0; i < M; i++) out[i] = u[i] - dt * (F[(i + 1) % M] - F[i]) / w[i];
	if (!this.limiter || this.p === 0) return;
	if (!this.detect(u, out)) return;
	const flag = this.flag;
	for (let i = 0; i < M; i++) if (flag[i]) { this.shown[i] = 1; F[i] = this.musclFlux(u, i); F[(i + 1) % M] = this.musclFlux(u, i + 1); }
	for (let i = 0; i < M; i++) out[i] = u[i] - dt * (F[(i + 1) % M] - F[i]) / w[i];
};

// strong-stability-preserving third-order Runge-Kutta
Sim.prototype.step = function () {
	const M = this.M, u = this.u, u1 = this.u1, u2 = this.u2, c = this.cand;
	this.shown.fill(0);
	this.stage(u, u1);
	this.stage(u1, c); for (let i = 0; i < M; i++) u2[i] = 0.75 * u[i] + 0.25 * c[i];
	this.stage(u2, c); for (let i = 0; i < M; i++) u[i] = u[i] / 3 + 2 * c[i] / 3;
	this.t += this.dt;
	let nf = 0; for (let i = 0; i < M; i++) nf += this.shown[i]; this.nflag = nf;
};

Sim.prototype.l1 = function () {
	const ex = this.means(this.t, this.cand); let s = 0;
	for (let i = 0; i < this.M; i++) s += Math.abs(this.u[i] - ex[i]) * this.w[i];
	return s;
};

/* ------------------------------------------------------------------ */
/* Canvas helpers                                                      */
/* ------------------------------------------------------------------ */

const C = { ink: '#2f3437', ink3: '#8a9296', line: '#e4e8e8', accent: '#2f9e7e', warm: '#e08e0b', red: '#d1495b', blue: '#3b6ea5', grid: '#eef1f1' };

function setupCanvas(cv, aspect) {
	const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
	const wcss = cv.clientWidth, hcss = Math.round(Math.max(220, Math.min(wcss * aspect, 460)));
	cv.style.height = hcss + 'px';
	if (cv.width !== Math.round(wcss * dpr) || cv.height !== Math.round(hcss * dpr)) { cv.width = Math.round(wcss * dpr); cv.height = Math.round(hcss * dpr); }
	const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
	return { ctx, W: wcss, H: hcss };
}

function diamond(ctx, x, y, r, fill, stroke) {
	ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath();
	if (fill) { ctx.fillStyle = fill; ctx.fill(); }
	if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
}

const $ = id => document.getElementById(id);

/* ------------------------------------------------------------------ */
/* Widget 1: anatomy of an update                                      */
/* ------------------------------------------------------------------ */

function Anatomy() {
	const cv = $('anat-canvas'); if (!cv) return;
	const NE = 3, YMIN = 0.55, YMAX = 2.45, PAD = { l: 14, r: 14, t: 16, b: 46 };
	let p = 3, s = makeScheme(p), u = [], step = 1, drag = null, anim = null;

	const CAPS = [
		'',
		'<b>1 · The solution.</b> Each element stores the solution at its solution points (circles) and represents it by the polynomial through them. Nothing forces neighbouring polynomials to agree at a face. Drag any circle.',
		'<b>2 · Interpolate to the flux points.</b> The same polynomial is evaluated at the flux points (diamonds): the two faces of the element and the Gauss–Legendre points between them.',
		'<b>3 · One flux per face.</b> At a face the two elements disagree. A Riemann solver picks a single flux from the two states; for advection to the right it takes the state coming from the left. This is the only place where elements talk to each other.',
		'<b>4 · The flux polynomial.</b> Through the flux-point values, with the face values replaced by the common ones, passes a polynomial of one degree higher (dashed). It is continuous across faces, which is what makes the scheme conservative.',
		'<b>5 · Differentiate.</b> The time derivative at each solution point is minus the slope of the flux polynomial there (arrows). Press “Advance” to apply it.'
	];

	function init() {
		s = makeScheme(p); u = [];
		const off = [0.0, 0.28, -0.18];
		for (let e = 0; e < NE; e++) u.push(s.xs.map(x => 1.5 + 0.42 * Math.sin(2 * Math.PI * (e + x) / NE + 0.6) + off[e]));
	}

	function fluxData() {
		const uf = u.map(ue => s.S2F.map(r => r.reduce((a, v, j) => a + v * ue[j], 0)));
		const F = uf.map((f, e) => { const g = Array.from(f); g[0] = uf[(e + NE - 1) % NE][p + 1]; return g; });
		const du = F.map(Fe => s.DF.map(r => -r.reduce((a, v, k) => a + v * Fe[k], 0)));   // per unit element length
		return { uf, F, du };
	}

	function rk3(dt) {
		const rhs = v => { const keep = u; u = v; const d = fluxData().du; u = keep; return d; };
		const axpy = (a, b, c) => a.map((ae, e) => ae.map((x, j) => x + c * b[e][j]));
		const k1 = rhs(u), u1 = axpy(u, k1, dt);
		const k2 = rhs(u1), u2 = u.map((ue, e) => ue.map((x, j) => 0.75 * x + 0.25 * (u1[e][j] + dt * k2[e][j])));
		const k3 = rhs(u2);
		u = u.map((ue, e) => ue.map((x, j) => x / 3 + 2 * (u2[e][j] + dt * k3[e][j]) / 3));
	}

	let G = null;
	function draw() {
		const { ctx, W, H } = setupCanvas(cv, 0.5); G = { W, H };
		const X = x => PAD.l + (x / NE) * (W - PAD.l - PAD.r), Y = y => PAD.t + (YMAX - y) / (YMAX - YMIN) * (H - PAD.t - PAD.b);
		G.X = X; G.Y = Y;
		ctx.clearRect(0, 0, W, H);
		const yb = H - PAD.b + 16;
		// control volumes
		for (let e = 0; e < NE; e++) for (let k = 0; k <= p; k++) {
			ctx.fillStyle = k % 2 ? '#edf3f1' : '#dfeae6';
			ctx.fillRect(X(e + s.xf[k]), yb - 7, X(e + s.xf[k + 1]) - X(e + s.xf[k]), 14);
		}
		// element boundaries
		ctx.strokeStyle = '#cfd8d6'; ctx.lineWidth = 1;
		for (let e = 0; e <= NE; e++) { ctx.beginPath(); ctx.moveTo(X(e), PAD.t - 6); ctx.lineTo(X(e), yb + 7); ctx.stroke(); }
		for (let e = 0; e < NE; e++) for (let k = 0; k <= p + 1; k++) diamond(ctx, X(e + s.xf[k]), yb, 4.5, C.warm);
		ctx.fillStyle = C.ink3; ctx.font = '12px "Source Sans Pro", sans-serif'; ctx.textAlign = 'center';
		for (let e = 0; e < NE; e++) ctx.fillText('element ' + (e + 1), X(e + 0.5), H - 6);

		const { uf, F, du } = fluxData();
		const xs = []; for (let i = 0; i <= 40; i++) xs.push(i / 40);
		const PL = lagrangeMatrix(xs, s.xs), PF = lagrangeMatrix(xs, s.xf);

		// flux polynomial
		if (step >= 4) {
			ctx.strokeStyle = C.warm; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
			for (let e = 0; e < NE; e++) {
				ctx.beginPath();
				xs.forEach((x, i) => { const v = PF[i].reduce((a, c, k) => a + c * F[e][k], 0); i ? ctx.lineTo(X(e + x), Y(v)) : ctx.moveTo(X(e + x), Y(v)); });
				ctx.stroke();
			}
			ctx.setLineDash([]);
		}
		// solution polynomials
		ctx.strokeStyle = C.accent; ctx.lineWidth = 2.5;
		for (let e = 0; e < NE; e++) {
			ctx.beginPath();
			xs.forEach((x, i) => { const v = PL[i].reduce((a, c, j) => a + c * u[e][j], 0); i ? ctx.lineTo(X(e + x), Y(v)) : ctx.moveTo(X(e + x), Y(v)); });
			ctx.stroke();
		}
		// flux-point values
		if (step >= 2) {
			for (let e = 0; e < NE; e++) for (let k = 0; k <= p + 1; k++) {
				const atLeftFace = k === 0;
				if (step >= 3 && atLeftFace) {
					diamond(ctx, X(e), Y(uf[e][0]), 5.5, '#fff', '#b9c2c4');
					ctx.strokeStyle = '#b9c2c4'; ctx.setLineDash([2, 3]); ctx.lineWidth = 1.2;
					ctx.beginPath(); ctx.moveTo(X(e), Y(uf[e][0])); ctx.lineTo(X(e), Y(F[e][0])); ctx.stroke(); ctx.setLineDash([]);
				} else diamond(ctx, X(e + s.xf[k]), Y(uf[e][k]), 5.5, step >= 3 ? C.warm : '#fff', C.warm);
			}
			if (step >= 3) for (let e = 0; e < NE; e++) diamond(ctx, X(e), Y(F[e][0]), 6.5, C.warm, '#fff');
		}
		// time-derivative arrows
		if (step >= 5) {
			const sc = 0.085; ctx.strokeStyle = C.blue; ctx.fillStyle = C.blue; ctx.lineWidth = 2;
			for (let e = 0; e < NE; e++) for (let j = 0; j <= p; j++) {
				const x = X(e + s.xs[j]), y0 = Y(u[e][j]), y1 = Y(u[e][j] + sc * du[e][j]);
				if (Math.abs(y1 - y0) < 3) continue;
				const dir = Math.sign(y1 - y0);
				ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1 - dir * 5); ctx.stroke();
				ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x - 4.5, y1 - dir * 8); ctx.lineTo(x + 4.5, y1 - dir * 8); ctx.closePath(); ctx.fill();
			}
		}
		// solution points
		for (let e = 0; e < NE; e++) for (let j = 0; j <= p; j++) {
			ctx.beginPath(); ctx.arc(X(e + s.xs[j]), Y(u[e][j]), 6, 0, 2 * Math.PI);
			ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = C.accent; ctx.lineWidth = 2.5; ctx.stroke();
		}
		// direction of advection
		if (step >= 3) {
			ctx.strokeStyle = C.ink3; ctx.fillStyle = C.ink3; ctx.lineWidth = 1.5;
			const ax = W - PAD.r - 70, ay = PAD.t + 6;
			ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax + 44, ay); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(ax + 50, ay); ctx.lineTo(ax + 42, ay - 4); ctx.lineTo(ax + 42, ay + 4); ctx.closePath(); ctx.fill();
			ctx.textAlign = 'right'; ctx.fillText('flow', ax - 6, ay + 4);
		}
		$('anat-cap').innerHTML = CAPS[step];
	}

	function pick(ev) {
		const r = cv.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top; let best = null, bd = 22;
		for (let e = 0; e < NE; e++) for (let j = 0; j <= p; j++) {
			const d = Math.hypot(mx - G.X(e + s.xs[j]), my - G.Y(u[e][j])); if (d < bd) { bd = d; best = { e, j }; }
		}
		return best;
	}
	cv.addEventListener('pointerdown', ev => { drag = pick(ev); if (drag) { try { cv.setPointerCapture(ev.pointerId); } catch (e) {} ev.preventDefault(); } });
	cv.addEventListener('pointermove', ev => {
		if (!drag) { cv.style.cursor = pick(ev) ? 'ns-resize' : 'default'; return; }
		const r = cv.getBoundingClientRect(), my = ev.clientY - r.top;
		const y = YMAX - (my - PAD.t) / (G.H - PAD.t - PAD.b) * (YMAX - YMIN);
		u[drag.e][drag.j] = Math.max(YMIN + 0.08, Math.min(YMAX - 0.08, y)); draw();
	});
	const end = () => { drag = null; };
	cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);

	const stepBtns = Array.from(document.querySelectorAll('#anat-steps button'));
	const setStep = k => { step = k; stepBtns.forEach((b, i) => b.setAttribute('aria-pressed', i + 1 === k ? 'true' : 'false')); draw(); };
	stepBtns.forEach((b, i) => b.addEventListener('click', () => setStep(i + 1)));
	$('anat-p').addEventListener('input', ev => { p = +ev.target.value; $('anat-p-out').textContent = p; init(); draw(); });
	$('anat-reset').addEventListener('click', () => { init(); draw(); });
	$('anat-advance').addEventListener('click', () => {
		if (anim) return; if (step < 5) setStep(5);
		let k = 0; const sub = 12, total = 0.25, dt = total / (sub * 20);
		anim = setInterval(() => { for (let i = 0; i < 20; i++) rk3(dt); draw(); if (++k >= sub) { clearInterval(anim); anim = null; } }, 30);
	});
	init(); setStep(1);
	new ResizeObserver(draw).observe(cv);
}

/* ------------------------------------------------------------------ */
/* Widget 2: advection with and without limiting                       */
/* ------------------------------------------------------------------ */

function Advect() {
	const cv = $('adv-canvas'); if (!cv) return;
	const NS = [4, 8, 16, 32, 64], YMIN = 0.7, YMAX = 2.3, PAD = { l: 34, r: 12, t: 12, b: 26 };
	let sim, playing = false, raf = null, speed = 1, PM = null, acc = 0;

	function rebuild() {
		const p = +$('adv-order').value - 1, N = NS[+$('adv-n').value];
		sim = new Sim(p, N, $('adv-profile').value, $('adv-limit').checked);
		const xs = []; const m = Math.max(3, Math.min(14, Math.round(260 / sim.M) + 2 * p));
		for (let i = 0; i <= m; i++) xs.push(i / m);
		PM = { xs, P: matmul(lagrangeMatrix(xs, sim.s.xs), sim.s.M2S) };
		$('adv-order-out').textContent = p + 1; $('adv-n-out').textContent = N; $('adv-dof').textContent = sim.M;
		acc = 0; draw();
	}

	function draw() {
		const { ctx, W, H } = setupCanvas(cv, 0.5);
		const X = x => PAD.l + x * (W - PAD.l - PAD.r), Y = y => PAD.t + (YMAX - y) / (YMAX - YMIN) * (H - PAD.t - PAD.b);
		ctx.clearRect(0, 0, W, H);
		ctx.font = '12px "Source Sans Pro", sans-serif'; ctx.fillStyle = C.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
		ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
		for (const y of [1, 1.5, 2]) { ctx.beginPath(); ctx.moveTo(PAD.l, Y(y)); ctx.lineTo(W - PAD.r, Y(y)); ctx.stroke(); ctx.fillText(y.toFixed(1), PAD.l - 6, Y(y)); }
		// troubled control volumes
		const n = sim.n, M = sim.M;
		for (let i = 0; i < M; i++) if (sim.shown[i]) {
			const a = X(sim.xc[i] - sim.w[i] / 2), b = X(sim.xc[i] + sim.w[i] / 2);
			ctx.fillStyle = 'rgba(209,73,91,0.16)'; ctx.fillRect(a, PAD.t, Math.max(1, b - a), H - PAD.t - PAD.b);
			ctx.fillStyle = C.red; ctx.fillRect(a, H - PAD.b - 5, Math.max(1.5, b - a), 5);
		}
		// element boundaries
		if (sim.N <= 64) { ctx.strokeStyle = '#e9eded'; for (let e = 0; e <= sim.N; e++) { ctx.beginPath(); ctx.moveTo(X(e * sim.h), PAD.t); ctx.lineTo(X(e * sim.h), H - PAD.b); ctx.stroke(); } }
		// exact solution
		ctx.strokeStyle = '#9aa3a6'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.beginPath();
		const K = Math.round(W - PAD.l - PAD.r);
		for (let i = 0; i <= K; i++) { const x = i / K, y = sim.profile(x - sim.t); i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); }
		ctx.stroke(); ctx.setLineDash([]);
		// numerical solution: the polynomial inside each element
		ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.lineJoin = 'round';
		ctx.save(); ctx.beginPath(); ctx.rect(PAD.l, 2, W - PAD.l - PAD.r, H - PAD.b - 2); ctx.clip();
		for (let e = 0; e < sim.N; e++) {
			ctx.beginPath();
			if (sim.p === 0) { ctx.moveTo(X(e * sim.h), Y(sim.u[e])); ctx.lineTo(X((e + 1) * sim.h), Y(sim.u[e])); }
			else PM.xs.forEach((x, i) => { const r = PM.P[i]; let v = 0; for (let j = 0; j < n; j++) v += r[j] * sim.u[e * n + j]; i ? ctx.lineTo(X((e + x) * sim.h), Y(v)) : ctx.moveTo(X((e + x) * sim.h), Y(v)); });
			ctx.stroke();
		}
		ctx.restore();
		ctx.strokeStyle = '#cfd6d6'; ctx.lineWidth = 1; ctx.strokeRect(PAD.l + 0.5, PAD.t + 0.5, W - PAD.l - PAD.r - 1, H - PAD.t - PAD.b - 1);
		ctx.fillStyle = C.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillText('x', X(0.5), H - 6);
		$('adv-t').textContent = sim.t.toFixed(2);
		$('adv-err').textContent = sim.l1().toExponential(1);
		$('adv-flag').textContent = sim.limiter && sim.p > 0 ? (100 * sim.nflag / sim.M).toFixed(1) + ' %' : '–';
	}

	function frame() {
		if (!playing) return;
		acc += speed / 420;                         // one period in about seven seconds
		let k = 0; while (acc >= sim.dt && k < 4000) { sim.step(); acc -= sim.dt; k++; }
		draw(); raf = requestAnimationFrame(frame);
	}
	function setPlaying(v) { playing = v; $('adv-play').textContent = v ? 'Pause' : 'Play'; $('adv-play').classList.toggle('primary', !v); if (v) raf = requestAnimationFrame(frame); }

	$('adv-play').addEventListener('click', () => setPlaying(!playing));
	$('adv-reset').addEventListener('click', rebuild);
	['adv-order', 'adv-n'].forEach(id => $(id).addEventListener('input', rebuild));
	$('adv-profile').addEventListener('change', rebuild);
	$('adv-limit').addEventListener('change', () => { sim.limiter = $('adv-limit').checked; sim.shown.fill(0); sim.nflag = 0; draw(); });
	$('adv-speed').addEventListener('input', ev => { speed = Math.pow(2, +ev.target.value); $('adv-speed-out').textContent = (speed >= 1 ? speed : speed.toFixed(2)) + '×'; });
	new IntersectionObserver(es => { if (!es[0].isIntersecting && playing) setPlaying(false); }).observe(cv);
	new ResizeObserver(draw).observe(cv);
	rebuild();
	return { get sim() { return sim; } };
}

/* ------------------------------------------------------------------ */
/* Widget 4: one stage of the a posteriori limiting, taken apart       */
/* ------------------------------------------------------------------ */

function Fallback() {
	const cv = $('fb-canvas'); if (!cv) return;
	const N = 8, YMIN = 0.7, YMAX = 2.3, PAD = { l: 34, r: 12, t: 12, b: 46 }, GREY = '#bcc5c6';
	let sim, A = null, step = 1, PM = null, timer = null;

	const CAPS = [
		'',
		'<b>1 · Before.</b> The averages over the control volumes at the start of the stage (bars), with the polynomial they come from. Each average will change by the difference of the two fluxes at its faces (diamonds).',
		'<b>2 · A candidate.</b> The high-order update is applied everywhere, as if the solution were smooth. Next to the jumps some volumes end up above or below anything that was there before (grey: the values before).',
		'<b>3 · Check it.</b> Each candidate must stay inside the band spanned by its own old value and those of its two neighbours, widened by the tolerance. Red: outside its band, rejected. Blue: outside, but recognised as a smooth extremum and accepted.',
		'<b>4 · Replace the fluxes.</b> At the two faces of every rejected volume the high-order flux (open diamond) gives way to the flux of a second-order scheme with a limited slope (red diamond), computed from the same old values (grey). A face has one flux, so the neighbour across it gets the corrected flux too, and the scheme stays conservative.',
		'<b>5 · Redo the update.</b> The update is repeated with the corrected fluxes. The rejected volumes (red) and the neighbours that share a face with them move; everything else keeps its high-order value.'
	];

	function rebuild() {
		const p = +$('fb-order').value - 1;
		sim = new Sim(p, N, $('fb-profile').value, true); sim.tol = +$('fb-tol').value;
		const xs = [], m = 10 + 2 * p; for (let i = 0; i <= m; i++) xs.push(i / m);
		PM = { xs, P: matmul(lagrangeMatrix(xs, sim.s.xs), sim.s.M2S) };
		$('fb-order-out').textContent = p + 1;
		analyse(); draw();
	}

	// the first stage of the next time step, move by move
	function analyse() {
		const M = sim.M, u = sim.u, w = sim.w, dt = sim.dt, tol = sim.tol;
		sim.highOrderFlux(u); const FH = Float64Array.from(sim.F);
		const cand = new Float64Array(M), out = new Float64Array(M), mn = new Float64Array(M), mx = new Float64Array(M);
		for (let i = 0; i < M; i++) cand[i] = u[i] - dt * (FH[(i + 1) % M] - FH[i]) / w[i];
		for (let i = 0; i < M; i++) {
			const a = u[(i + M - 1) % M], b = u[i], d = u[(i + 1) % M], lo = Math.min(a, b, d), hi = Math.max(a, b, d);
			mn[i] = lo - Math.abs(lo) * tol; mx[i] = hi + Math.abs(hi) * tol;
		}
		sim.detect(u, cand);
		const flag = Uint8Array.from(sim.flag), nad = Uint8Array.from(sim.nad), F = Float64Array.from(FH), swap = new Uint8Array(M);
		for (let i = 0; i < M; i++) if (flag[i]) { const j = (i + 1) % M; F[i] = sim.musclFlux(u, i); F[j] = sim.musclFlux(u, i + 1); swap[i] = swap[j] = 1; }
		for (let i = 0; i < M; i++) out[i] = u[i] - dt * (F[(i + 1) % M] - F[i]) / w[i];
		const count = a => { let c = 0; for (let i = 0; i < M; i++) c += a[i]; return c; };
		A = { u: Float64Array.from(u), FH, F, cand, out, mn, mx, flag, nad, swap, nNad: count(nad), nFlag: count(flag), nSwap: count(swap) };
	}

	function draw() {
		const { ctx, W, H } = setupCanvas(cv, 0.54);
		const X = x => PAD.l + x * (W - PAD.l - PAD.r), Y = y => PAD.t + (YMAX - y) / (YMAX - YMIN) * (H - PAD.t - PAD.b);
		const M = sim.M, n = sim.n, w = sim.w, xc = sim.xc, bot = H - PAD.b, yb = bot + 17;
		const xa = i => X(xc[i] - w[i] / 2), xb = i => X(xc[i] + w[i] / 2);
		const bar = (i, v, col, lw) => { ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath(); ctx.moveTo(xa(i) + 0.5, Y(v)); ctx.lineTo(xb(i) - 0.5, Y(v)); ctx.stroke(); };
		ctx.clearRect(0, 0, W, H); ctx.lineCap = 'butt';
		ctx.font = '12px "Source Sans Pro", sans-serif'; ctx.fillStyle = C.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
		ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
		for (const y of [1, 1.5, 2]) { ctx.beginPath(); ctx.moveTo(PAD.l, Y(y)); ctx.lineTo(W - PAD.r, Y(y)); ctx.stroke(); ctx.fillText(y.toFixed(1), PAD.l - 6, Y(y)); }
		ctx.save(); ctx.beginPath(); ctx.rect(PAD.l, 2, W - PAD.l - PAD.r, bot - 2); ctx.clip();
		// rejected control volumes
		if (step >= 4) { ctx.fillStyle = 'rgba(209,73,91,0.13)'; for (let i = 0; i < M; i++) if (A.flag[i]) ctx.fillRect(xa(i), PAD.t, xb(i) - xa(i), bot - PAD.t); }
		// admissible bands
		if (step === 3 || step === 5) {
			ctx.fillStyle = step === 3 ? 'rgba(59,110,165,0.16)' : 'rgba(59,110,165,0.08)';
			for (let i = 0; i < M; i++) ctx.fillRect(xa(i) + 0.5, Y(A.mx[i]) - 1, xb(i) - xa(i) - 1, Y(A.mn[i]) - Y(A.mx[i]) + 2);
		}
		// element boundaries
		ctx.strokeStyle = '#e3e8e8'; ctx.lineWidth = 1;
		for (let e = 0; e <= N; e++) { ctx.beginPath(); ctx.moveTo(X(e * sim.h), PAD.t); ctx.lineTo(X(e * sim.h), bot); ctx.stroke(); }
		// exact solution
		ctx.strokeStyle = '#a9b1b3'; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]); ctx.beginPath();
		const K = Math.round(W - PAD.l - PAD.r);
		for (let i = 0; i <= K; i++) { const x = i / K, y = sim.profile(x - sim.t); i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); }
		ctx.stroke(); ctx.setLineDash([]);
		if (step === 1) {
			ctx.strokeStyle = 'rgba(47,158,126,0.45)'; ctx.lineWidth = 1.5;
			for (let e = 0; e < N; e++) {
				ctx.beginPath();
				PM.xs.forEach((x, i) => { const r = PM.P[i]; let v = 0; for (let j = 0; j < n; j++) v += r[j] * A.u[e * n + j]; i ? ctx.lineTo(X((e + x) * sim.h), Y(v)) : ctx.moveTo(X((e + x) * sim.h), Y(v)); });
				ctx.stroke();
			}
			for (let i = 0; i < M; i++) bar(i, A.u[i], C.accent, 3);
		} else if (step === 2) {
			for (let i = 0; i < M; i++) bar(i, A.u[i], GREY, 2);
			for (let i = 0; i < M; i++) bar(i, A.cand[i], C.accent, 3);
		} else if (step === 3) {
			for (let i = 0; i < M; i++) bar(i, A.cand[i], A.flag[i] ? C.red : (A.nad[i] ? C.blue : C.accent), A.nad[i] ? 3.5 : 3);
		} else if (step === 4) {
			for (let i = 0; i < M; i++) bar(i, A.u[i], GREY, 2.5);
			for (let i = 0; i < M; i++) {
				const x = xa(i);
				if (A.swap[i]) {
					ctx.strokeStyle = C.red; ctx.lineWidth = 1; ctx.setLineDash([2, 2]); ctx.beginPath(); ctx.moveTo(x, Y(A.FH[i])); ctx.lineTo(x, Y(A.F[i])); ctx.stroke(); ctx.setLineDash([]);
					diamond(ctx, x, Y(A.FH[i]), 5, '#fff', C.warm); diamond(ctx, x, Y(A.F[i]), 5.5, C.red, '#fff');
				} else diamond(ctx, x, Y(A.FH[i]), 3.2, C.warm);
			}
		} else {
			for (let i = 0; i < M; i++) if (Math.abs(A.cand[i] - A.out[i]) > 1e-9) bar(i, A.cand[i], GREY, 2);
			for (let i = 0; i < M; i++) bar(i, A.out[i], A.flag[i] ? C.red : C.accent, 3);
		}
		ctx.restore();
		ctx.strokeStyle = '#cfd6d6'; ctx.lineWidth = 1; ctx.strokeRect(PAD.l + 0.5, PAD.t + 0.5, W - PAD.l - PAD.r - 1, bot - PAD.t - 1);
		// the control volumes and their faces
		for (let i = 0; i < M; i++) { ctx.fillStyle = (step >= 4 && A.flag[i]) ? 'rgba(209,73,91,0.35)' : ((i % n) % 2 ? '#edf3f1' : '#dfeae6'); ctx.fillRect(xa(i), yb - 6, xb(i) - xa(i), 12); }
		for (let i = 0; i < M; i++) { const sw = step >= 4 && A.swap[i]; diamond(ctx, xa(i), yb, sw ? 4.6 : 3.2, sw ? C.red : C.warm); }
		diamond(ctx, X(1), yb, (step >= 4 && A.swap[0]) ? 4.6 : 3.2, (step >= 4 && A.swap[0]) ? C.red : C.warm);
		ctx.fillStyle = C.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillText('control volumes and their faces', X(0.5), H - 5);
		$('fb-cap').innerHTML = CAPS[step];
		$('fb-nad').textContent = A.nNad; $('fb-sed').textContent = A.nNad - A.nFlag; $('fb-flag').textContent = A.nFlag + ' of ' + M; $('fb-faces').textContent = A.nSwap;
		$('fb-t').textContent = Math.round(sim.t / sim.dt);
	}

	const stepBtns = Array.from(document.querySelectorAll('#fb-steps button'));
	const setStep = k => { step = k; stepBtns.forEach((b, i) => b.setAttribute('aria-pressed', i + 1 === k ? 'true' : 'false')); draw(); };
	stepBtns.forEach((b, i) => b.addEventListener('click', () => setStep(i + 1)));
	const advance = () => { sim.step(); analyse(); draw(); };
	const setPlaying = v => {
		if (timer) { clearInterval(timer); timer = null; }
		if (v) timer = setInterval(advance, 130);
		$('fb-play').textContent = v ? 'Pause' : 'Play';
	};
	$('fb-next').addEventListener('click', () => { setPlaying(false); advance(); });
	$('fb-play').addEventListener('click', () => setPlaying(!timer));
	$('fb-reset').addEventListener('click', () => { setPlaying(false); rebuild(); });
	$('fb-order').addEventListener('input', rebuild);
	$('fb-profile').addEventListener('change', rebuild);
	$('fb-tol').addEventListener('change', () => { sim.tol = +$('fb-tol').value; analyse(); draw(); });
	new IntersectionObserver(es => { if (!es[0].isIntersecting && timer) setPlaying(false); }).observe(cv);
	new ResizeObserver(() => { if (A) draw(); }).observe(cv);
	rebuild(); setStep(1);
	return { get sim() { return sim; }, get A() { return A; }, setStep, advance };
}

/* ------------------------------------------------------------------ */
/* Widget 5: Burgers' equation, a shock that forms on its own          */
/* ------------------------------------------------------------------ */

// reference solution: second-order MUSCL on a fine uniform mesh
function RefBurgers(K) {
	this.K = K; this.dx = 1 / K; this.dt = 0.4 * this.dx / 2; this.t = 0;
	this.u = new Float64Array(K); this.a = new Float64Array(K); this.b = new Float64Array(K); this.F = new Float64Array(K);
	for (let i = 0; i < K; i++) { const x0 = i * this.dx, x1 = x0 + this.dx; this.u[i] = 1.5 + 0.5 * (Math.cos(2 * Math.PI * x0) - Math.cos(2 * Math.PI * x1)) / (2 * Math.PI * this.dx); }
}
RefBurgers.prototype.euler = function (u, out, dt) {
	const K = this.K, F = this.F;
	for (let i = 0; i < K; i++) { const L = (i + K - 1) % K, LL = (L + K - 1) % K, v = u[L] + 0.5 * minmod(u[L] - u[LL], u[i] - u[L]); F[i] = 0.5 * v * v; }
	for (let i = 0; i < K; i++) out[i] = u[i] - dt / this.dx * (F[(i + 1) % K] - F[i]);
};
RefBurgers.prototype.advance = function (t) {
	const K = this.K, u = this.u, a = this.a, b = this.b;
	while (this.t < t - 1e-12) {
		const dt = Math.min(this.dt, t - this.t);
		this.euler(u, a, dt); this.euler(a, b, dt); for (let i = 0; i < K; i++) u[i] = 0.5 * (u[i] + b[i]);
		this.t += dt;
	}
};

function Shock() {
	const cv = $('bg-canvas'); if (!cv) return;
	const NS = [8, 16, 32, 64], YMIN = 0.7, YMAX = 2.3, TEND = 1.2, TB = 1 / Math.PI, PAD = { l: 34, r: 12, t: 12, b: 84 };
	let sim, ref, playing = false, raf = null, speed = 1, PM = null, acc = 0, hist = [], first = null, blown = false;

	function rebuild() {
		const p = +$('bg-order').value - 1, N = NS[+$('bg-n').value];
		sim = new Sim(p, N, 'sine', $('bg-limit').checked, true); ref = new RefBurgers(2000);
		const xs = [], m = Math.max(3, Math.min(14, Math.round(260 / sim.M) + 2 * p)); for (let i = 0; i <= m; i++) xs.push(i / m);
		PM = { xs, P: matmul(lagrangeMatrix(xs, sim.s.xs), sim.s.M2S) };
		$('bg-order-out').textContent = p + 1; $('bg-n-out').textContent = N;
		hist = [[0, 0]]; first = null; blown = false; acc = 0; draw();
	}

	function draw() {
		const { ctx, W, H } = setupCanvas(cv, 0.6);
		const bot = H - PAD.b, X = x => PAD.l + x * (W - PAD.l - PAD.r), Y = y => PAD.t + (YMAX - y) / (YMAX - YMIN) * (bot - PAD.t);
		const n = sim.n, M = sim.M;
		ctx.clearRect(0, 0, W, H);
		ctx.font = '12px "Source Sans Pro", sans-serif'; ctx.fillStyle = C.ink3; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
		ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
		for (const y of [1, 1.5, 2]) { ctx.beginPath(); ctx.moveTo(PAD.l, Y(y)); ctx.lineTo(W - PAD.r, Y(y)); ctx.stroke(); ctx.fillText(y.toFixed(1), PAD.l - 6, Y(y)); }
		for (let i = 0; i < M; i++) if (sim.shown[i]) {
			const a = X(sim.xc[i] - sim.w[i] / 2), b = X(sim.xc[i] + sim.w[i] / 2);
			ctx.fillStyle = 'rgba(209,73,91,0.16)'; ctx.fillRect(a, PAD.t, Math.max(1, b - a), bot - PAD.t);
			ctx.fillStyle = C.red; ctx.fillRect(a, bot - 5, Math.max(1.5, b - a), 5);
		}
		if (sim.N <= 64) { ctx.strokeStyle = '#e9eded'; for (let e = 0; e <= sim.N; e++) { ctx.beginPath(); ctx.moveTo(X(e * sim.h), PAD.t); ctx.lineTo(X(e * sim.h), bot); ctx.stroke(); } }
		ctx.save(); ctx.beginPath(); ctx.rect(PAD.l, 2, W - PAD.l - PAD.r, bot - 2); ctx.clip();
		// reference
		ctx.strokeStyle = '#9aa3a6'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.beginPath();
		const st = Math.max(1, Math.floor(ref.K / (W - PAD.l - PAD.r)));
		for (let i = 0; i < ref.K; i += st) { const x = (i + 0.5) * ref.dx; i ? ctx.lineTo(X(x), Y(ref.u[i])) : ctx.moveTo(X(x), Y(ref.u[i])); }
		ctx.stroke(); ctx.setLineDash([]);
		// numerical solution: the polynomial inside each element
		ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.lineJoin = 'round';
		for (let e = 0; e < sim.N; e++) {
			ctx.beginPath();
			if (sim.p === 0) { ctx.moveTo(X(e * sim.h), Y(sim.u[e])); ctx.lineTo(X((e + 1) * sim.h), Y(sim.u[e])); }
			else PM.xs.forEach((x, i) => { const r = PM.P[i]; let v = 0; for (let j = 0; j < n; j++) v += r[j] * sim.u[e * n + j]; i ? ctx.lineTo(X((e + x) * sim.h), Y(v)) : ctx.moveTo(X((e + x) * sim.h), Y(v)); });
			ctx.stroke();
		}
		ctx.restore();
		ctx.strokeStyle = '#cfd6d6'; ctx.lineWidth = 1; ctx.strokeRect(PAD.l + 0.5, PAD.t + 0.5, W - PAD.l - PAD.r - 1, bot - PAD.t - 1);
		ctx.fillStyle = C.ink3; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.fillText('x', X(0.72), bot + 14);
		if (blown) { ctx.fillStyle = C.red; ctx.font = '600 14px "Source Sans Pro", sans-serif'; ctx.fillText('the unlimited scheme has blown up', X(0.5), PAD.t + 22); ctx.font = '12px "Source Sans Pro", sans-serif'; }
		// history of the fraction of control volumes at second order
		const c0 = bot + 32, c1 = H - 18, TX = t => PAD.l + t / TEND * (W - PAD.l - PAD.r);
		let fmax = 0.05; for (const h of hist) fmax = Math.max(fmax, h[1]); fmax = Math.ceil(fmax * 20) / 20;
		const FY = v => c1 - v / fmax * (c1 - c0);
		ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(PAD.l, c0); ctx.lineTo(W - PAD.r, c0); ctx.moveTo(PAD.l, c1); ctx.lineTo(W - PAD.r, c1); ctx.stroke();
		ctx.strokeStyle = '#9aa3a6'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(TX(TB), c0 - 4); ctx.lineTo(TX(TB), c1); ctx.stroke(); ctx.setLineDash([]);
		ctx.fillStyle = C.ink3; ctx.textAlign = 'left'; ctx.fillText('the shock forms', TX(TB) + 5, c0 - 7);
		ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText((100 * fmax).toFixed(0) + '%', PAD.l - 6, c0); ctx.fillText('0', PAD.l - 6, c1);
		ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
		for (const t of [0, 0.4, 0.8, 1.2]) ctx.fillText(t.toFixed(1), TX(t), H - 4);
		ctx.fillText('time', TX(1.0), H - 4);
		ctx.strokeStyle = C.red; ctx.lineWidth = 1.6; ctx.beginPath();
		const hs = Math.max(1, Math.floor(hist.length / 1500));
		for (let i = 0; i < hist.length; i += hs) { const h = hist[i]; i ? ctx.lineTo(TX(h[0]), FY(h[1])) : ctx.moveTo(TX(h[0]), FY(h[1])); }
		ctx.stroke();
		ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(TX(Math.min(sim.t, TEND)), FY(hist[hist.length - 1][1]), 3, 0, 2 * Math.PI); ctx.fill();
		$('bg-t').textContent = sim.t.toFixed(2);
		$('bg-flag').textContent = sim.limiter && sim.p > 0 ? (100 * sim.nflag / sim.M).toFixed(1) + ' %' : '–';
		$('bg-first').textContent = first === null ? '–' : 't = ' + first.toFixed(2);
	}

	function advanceTo(t) {
		let k = 0;
		while (sim.t < t && sim.t < TEND && k < 6000 && !blown) {
			sim.step(); k++;
			hist.push([sim.t, sim.limiter ? sim.nflag / sim.M : 0]);
			if (first === null && sim.limiter && sim.nflag > 0) first = sim.t;
			let big = 0; for (let i = 0; i < sim.M; i++) { const v = Math.abs(sim.u[i]); if (!(v < 50)) { big = 1; break; } }
			if (big) blown = true;
		}
		ref.advance(Math.min(sim.t, TEND + 0.01));
	}
	function frame() {
		if (!playing) return;
		acc += speed / 700;                         // 1.2 time units in about 14 seconds
		if (acc >= sim.dt) { advanceTo(sim.t + acc); acc = 0; }
		draw();
		if (sim.t >= TEND || blown) { setPlaying(false); return; }
		raf = requestAnimationFrame(frame);
	}
	function setPlaying(v) { playing = v; $('bg-play').textContent = v ? 'Pause' : 'Play'; $('bg-play').classList.toggle('primary', !v); if (v) raf = requestAnimationFrame(frame); }

	$('bg-play').addEventListener('click', () => { if (!playing && (sim.t >= TEND || blown)) rebuild(); setPlaying(!playing); });
	$('bg-reset').addEventListener('click', () => { setPlaying(false); rebuild(); });
	['bg-order', 'bg-n'].forEach(id => $(id).addEventListener('input', () => { setPlaying(false); rebuild(); }));
	$('bg-limit').addEventListener('change', () => { setPlaying(false); rebuild(); });
	$('bg-speed').addEventListener('input', ev => { speed = Math.pow(2, +ev.target.value); $('bg-speed-out').textContent = (speed >= 1 ? speed : speed.toFixed(2)) + '×'; });
	new IntersectionObserver(es => { if (!es[0].isIntersecting && playing) setPlaying(false); }).observe(cv);
	new ResizeObserver(draw).observe(cv);
	rebuild();
	return { get sim() { return sim; }, get ref() { return ref; }, get hist() { return hist; }, get first() { return first; }, get blown() { return blown; }, advanceTo, rebuild };
}

/* ------------------------------------------------------------------ */
/* Widget 3: convergence                                               */
/* ------------------------------------------------------------------ */

// pure SD on point values, with a Taylor time integrator of the same order as the space discretisation
function convergenceError(p, N) {
	const s = makeScheme(p), n = s.n, M = N * n, h = 1 / N;
	let wmin = Math.min.apply(null, s.wref) * h;
	const steps = Math.ceil(1 / (0.4 * wmin)), dt = 1 / steps;
	const x = new Float64Array(M); for (let e = 0; e < N; e++) for (let j = 0; j < n; j++) x[e * n + j] = (e + s.xs[j]) * h;
	const f = PROFILES.sine; let u = Float64Array.from(x, f);
	const F = new Float64Array(n + 1), tr = new Float64Array(N);
	function rhs(v, out) {
		for (let e = 0; e < N; e++) { const r = s.S2F[p + 1]; let a = 0; for (let j = 0; j < n; j++) a += r[j] * v[e * n + j]; tr[e] = a; }
		for (let e = 0; e < N; e++) {
			const o = e * n; F[0] = tr[(e + N - 1) % N];
			for (let k = 1; k <= p + 1; k++) { const r = s.S2F[k]; let a = 0; for (let j = 0; j < n; j++) a += r[j] * v[o + j]; F[k] = a; }
			for (let j = 0; j < n; j++) { const r = s.DF[j]; let a = 0; for (let k = 0; k <= p + 1; k++) a += r[k] * F[k]; out[o + j] = -a / h; }
		}
	}
	let term = new Float64Array(M), next = new Float64Array(M);
	for (let it = 0; it < steps; it++) {
		term.set(u);
		for (let k = 1; k <= p + 1; k++) { rhs(term, next); for (let i = 0; i < M; i++) { next[i] *= dt / k; u[i] += next[i]; } const t = term; term = next; next = t; }
	}
	let err = 0; for (let i = 0; i < M; i++) err += Math.abs(u[i] - f(x[i])); return err / M;
}

function Convergence() {
	const cv = $('conv-canvas'); if (!cv) return;
	const ORDERS = [1, 2, 3, 4, 6, 8], DOFS = [24, 48, 96, 192], COLORS = ['#8a9296', '#3b6ea5', '#2f9e7e', '#e08e0b', '#d1495b', '#7b4ea3'];
	let results = ORDERS.map(() => []), running = false;
	const PAD = { l: 52, r: 62, t: 14, b: 38 }, XMIN = Math.log10(20), XMAX = Math.log10(230), YMIN = -14, YMAX = 0;

	function draw() {
		const { ctx, W, H } = setupCanvas(cv, 0.6);
		const X = d => PAD.l + (Math.log10(d) - XMIN) / (XMAX - XMIN) * (W - PAD.l - PAD.r);
		const Y = e => PAD.t + (YMAX - Math.max(YMIN, Math.log10(e))) / (YMAX - YMIN) * (H - PAD.t - PAD.b);
		ctx.clearRect(0, 0, W, H); ctx.font = '12px "Source Sans Pro", sans-serif';
		ctx.strokeStyle = C.grid; ctx.fillStyle = C.ink3; ctx.lineWidth = 1; ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
		for (let e = 0; e >= YMIN; e -= 2) { const y = Y(Math.pow(10, e)); ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke(); ctx.fillText('1e' + e, PAD.l - 6, y); }
		ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
		for (const d of DOFS) { ctx.beginPath(); ctx.moveTo(X(d), PAD.t); ctx.lineTo(X(d), H - PAD.b); ctx.stroke(); ctx.fillText(d, X(d), H - PAD.b + 16); }
		ctx.fillText('degrees of freedom', (PAD.l + W - PAD.r) / 2, H - 5);
		ctx.save(); ctx.translate(13, (PAD.t + H - PAD.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('error after one period', 0, 0); ctx.restore();
		results.forEach((r, q) => {
			if (!r.length) return;
			ctx.strokeStyle = COLORS[q]; ctx.fillStyle = COLORS[q]; ctx.lineWidth = 2;
			ctx.beginPath(); r.forEach((e, i) => i ? ctx.lineTo(X(DOFS[i]), Y(e)) : ctx.moveTo(X(DOFS[i]), Y(e))); ctx.stroke();
			r.forEach((e, i) => { ctx.beginPath(); ctx.arc(X(DOFS[i]), Y(e), 3.5, 0, 2 * Math.PI); ctx.fill(); });
			ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
			ctx.fillText('order ' + ORDERS[q], X(DOFS[r.length - 1]) + 9, Y(r[r.length - 1]));
		});
	}

	function run() {
		if (running) return; running = true; results = ORDERS.map(() => []); $('conv-run').disabled = true;
		const jobs = []; ORDERS.forEach((o, q) => DOFS.forEach(d => jobs.push([q, o, d])));
		(function next() {
			if (!jobs.length) { running = false; $('conv-run').disabled = false; $('conv-run').textContent = 'Run again'; return; }
			const [q, o, d] = jobs.shift(); results[q].push(convergenceError(o - 1, d / o)); draw(); setTimeout(next, 0);
		})();
	}
	$('conv-run').addEventListener('click', run);
	new ResizeObserver(draw).observe(cv); draw();
	return { convergenceError };
}

document.addEventListener('DOMContentLoaded', () => {
	Anatomy();
	const adv = Advect();
	const fb = Fallback();
	const shock = Shock();
	const conv = Convergence();
	window.SDDemo = { makeScheme, Sim, RefBurgers, convergenceError, adv, fb, shock, conv };
});
})();
