import '@testing-library/jest-dom/vitest'

// jsdom implements no layout, so it has no scrollIntoView. The review table
// calls it to keep the cursor visible during a keyboard walk, which is real
// behaviour that every browser supports and jsdom does not. Stubbed rather than
// guarded in the component: a `typeof === 'function'` check there would be
// carrying a test environment's limitation into production code.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function () {}
}
