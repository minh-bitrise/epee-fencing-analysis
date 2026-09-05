import { fireEvent, render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import Timeline from './Timeline.jsx'

/**
 * The timeline is what makes review navigable: without it, finding the next
 * proposal means scrubbing a three minute video by hand, and the scrubbing is
 * most of the manual cost the project claims to remove. So these tests are
 * about positions being right and about the strip surviving the states that
 * produce a division by zero.
 */

const row = (over = {}) => ({
  id: 'p0', kind: 'proposed', state: 'pending', at: 30, ...over,
})

const setup = (props = {}) => {
  const onSelect = vi.fn()
  const onSeek = vi.fn()
  const r = render(
    <Timeline duration={180} rows={[row()]} segments={[]} selIdx={0}
              currentTime={0} onSelect={onSelect} onSeek={onSeek} {...props} />)
  return { ...r, onSelect, onSeek }
}

describe('Timeline', () => {
  it('places a marker at its share of the bout', () => {
    // 30 s of 180 is a sixth of the way along. A marker in the wrong place is
    // worse than no marker: the user clicks it and lands somewhere else.
    const { container } = setup()
    expect(container.querySelector('.mk').style.left)
      .toBe(`${(100 * 30) / 180}%`)
  })

  it('carries the state into the marker so the strip is readable at a glance', () => {
    const { container } = setup({
      rows: [row({ id: 'a', state: 'confirmed' }),
             row({ id: 'b', state: 'rejected', at: 60 }),
             row({ id: 'c', kind: 'added', state: 'added', at: 90 })],
    })
    const classes = [...container.querySelectorAll('.mk')]
      .map((m) => m.className)
    expect(classes).toEqual(['mk confirmed', 'mk rejected', 'mk added'])
  })

  it('shades an excluded range across its own span, not a point', () => {
    const { container } = setup({
      segments: [{ id: 's0', start_s: 45, end_s: 90 }],
    })
    const seg = container.querySelector('.seg')
    expect(seg.style.left).toBe('25%')
    expect(seg.style.width).toBe('25%')
  })

  it('seeks to the point on the strip that was clicked', () => {
    const { container, onSeek } = setup()
    const strip = container.querySelector('.tl')
    strip.getBoundingClientRect = () => ({ left: 0, width: 360 })
    fireEvent.click(strip, { clientX: 180 })
    expect(onSeek).toHaveBeenCalledWith(90)
  })

  it('selects a proposal without also seeking to where the click landed', () => {
    /**
     * The marker sits inside the strip, so a click on it would otherwise fire
     * both handlers: the user would select the proposal AND jump to whatever
     * pixel they happened to hit, which is near the proposal but not on it.
     */
    const { container, onSelect, onSeek } = setup()
    const strip = container.querySelector('.tl')
    strip.getBoundingClientRect = () => ({ left: 0, width: 360 })
    fireEvent.click(container.querySelector('.mk'), { clientX: 60 })
    expect(onSelect).toHaveBeenCalledWith(0)
    expect(onSeek).not.toHaveBeenCalled()
  })

  it('moves the playhead by writing to the node, not by re-rendering', () => {
    // `timeupdate` fires several times a second. Re-rendering the whole strip
    // at that rate to move one element a few pixels made the marker list
    // flicker on every tick.
    const { container, rerender } = setup()
    rerender(
      <Timeline duration={180} rows={[row()]} segments={[]} selIdx={0}
                currentTime={45} onSelect={vi.fn()} onSeek={vi.fn()} />)
    expect(container.querySelector('.cur').style.left).toBe('25%')
  })

  it('labels the scale more coarsely on a long bout than a short one', () => {
    const long = setup({ duration: 180 })
    const short = setup({ duration: 60 })
    const ticks = (c) => [...c.querySelectorAll('.ax span')].map((s) => s.textContent)
    expect(ticks(long.container)).toContain('30s')
    expect(ticks(long.container)).not.toContain('15s')
    expect(ticks(short.container)).toContain('15s')
  })

  it('renders without dividing by zero before the duration is known', () => {
    /**
     * The bout list arrives before the touch payload that carries the duration,
     * so this renders at least once with duration 0. Every position here is a
     * percentage of it, and NaN% is silently dropped by the browser, so the
     * failure is markers stacked at the left edge rather than an error.
     */
    const { container } = setup({ duration: 0, currentTime: 0 })
    expect(container.querySelector('.tl')).not.toBeNull()
    expect(container.querySelectorAll('.ax span').length).toBeLessThan(2)
  })

  it('does not seek when the strip has no width', () => {
    // A zero-width bounding box happens while the panel is hidden or animating,
    // and dividing by it would seek to Infinity.
    const { container, onSeek } = setup()
    const strip = container.querySelector('.tl')
    strip.getBoundingClientRect = () => ({ left: 0, width: 0 })
    fireEvent.click(strip, { clientX: 40 })
    expect(onSeek).not.toHaveBeenCalled()
  })

  it('copes with a bout that has no proposals and no exclusions', () => {
    const { container } = setup({ rows: [], segments: undefined })
    expect(container.querySelectorAll('.mk').length).toBe(0)
  })
})
