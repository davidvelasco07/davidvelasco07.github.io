/*
	Movie clips: play muted while on screen, pause when scrolled away, and hold
	the last frame for a few seconds before starting again.
	Hovering or focusing a clip shows the browser's controls; visitors who ask
	for reduced motion get the controls and no autoplay.
*/
(function () {
	'use strict';
	var HOLD = 3000;                                 // milliseconds on the last frame
	var clips = Array.prototype.slice.call(document.querySelectorAll('figure.movie video'));
	if (!clips.length) return;
	var calm = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
	var auto = !calm && ('IntersectionObserver' in window);

	function play(v) { var p = v.play(); if (p && p.catch) p.catch(function () { v.controls = true; }); }
	function release(v) { if (v._hold) { clearTimeout(v._hold); v._hold = null; } }

	clips.forEach(function (v) {
		v.muted = true;
		v.setAttribute('tabindex', '0');
		// The markup says "loop" so that the clips still repeat without this script; here the repeat
		// is done by hand, to put the pause in.
		v.loop = false;
		v.addEventListener('ended', function () {
			release(v);
			v._hold = setTimeout(function () { v._hold = null; if (v._away) return; v.currentTime = 0; play(v); }, HOLD);
		});
		v.addEventListener('play', function () { release(v); });
		v.addEventListener('seeking', function () { release(v); });
		if (!auto) { v.controls = true; return; }
		v.addEventListener('mouseenter', function () { v.controls = true; });
		v.addEventListener('mouseleave', function () { v.controls = false; });
		v.addEventListener('focus', function () { v.controls = true; });
		v.addEventListener('touchstart', function () { v.controls = true; }, { passive: true });
	});
	if (!auto) return;
	var io = new IntersectionObserver(function (entries) {
		entries.forEach(function (e) {
			var v = e.target;
			v._away = !e.isIntersecting;
			if (e.isIntersecting) { if (!v._hold) play(v); }      // if it is holding its last frame, let the pause finish
			else { v.pause(); if (v._hold) { release(v); v.currentTime = 0; } }
		});
	}, { threshold: 0.35 });
	clips.forEach(function (v) { io.observe(v); });
})();
