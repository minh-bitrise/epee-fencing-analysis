import '@testing-library/jest-dom/vitest'

// jsdom implements no layout, so it has no scrollIntoView.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function () {}
}

// jsdom has no canvas implementation either, so getContext returns null.
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
