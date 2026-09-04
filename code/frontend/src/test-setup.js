import '@testing-library/jest-dom/vitest'

// jsdom implements no layout, so it has no scrollIntoView. The review table
// calls it to keep the cursor visible during a keyboard walk, which is real
// behaviour that every browser supports and jsdom does not. Stubbed rather than
// guarded in the component: a `typeof === 'function'` check there would be
// carrying a test environment's limitation into production code.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function () {}
}

// jsdom has no canvas implementation either, so getContext returns null. The
// component now guards against that, because a real browser can also refuse a
// context, but a null here would mean the drawing code is never exercised by any
// test. A recording stub keeps that path covered: the calls are what matter, not
// the pixels, and asserting on pixels would need a native canvas dependency for
// no gain.
if (typeof HTMLCanvasElement !== 'undefined') {
  HTMLCanvasElement.prototype.getContext = function () {
    const noop = () => {}
    return {
      clearRect: noop, fillRect: noop, beginPath: noop, moveTo: noop,
      lineTo: noop, stroke: noop, fill: noop, closePath: noop,
      save: noop, restore: noop, translate: noop, scale: noop,
      drawImage: noop, setTransform: noop,
      set fillStyle(_v) {}, set strokeStyle(_v) {}, set lineWidth(_v) {},
    }
  }
}
