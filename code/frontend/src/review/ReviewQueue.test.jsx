import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ReviewQueue, { formatElapsed } from './ReviewQueue.jsx'

/* The queue is where the project's central claim is actually cashed: that a decision costs one
   keystroke and the navigation between decisions costs nothing. */

const items = (n = 3) => Array.from({ length: n }, (_, i) => ({
  id: `p${i}`, time_s: 10 * (i + 1), detail: `confidence 0.9${i}`,
}))

const press = (key) =>
  document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }))

const setup = (over = {}) => {
  const props = {
    items: items(), kind: 'touch',
    onDecide: vi.fn(() => Promise.resolve()),
    onExit: vi.fn(), onSeek: vi.fn(),
    ...over,
  }
  return { props, ...render(<ReviewQueue {...props} />) }
}

beforeEach(() => vi.useRealTimers())
afterEach(() => vi.restoreAllMocks())

describe('ReviewQueue', () => {
  it('holds the video at the first proposal without being asked', async () => {
    // The whole point. If the user has to seek, the queue has saved nothing.
    const { props } = setup()
    await waitFor(() => expect(props.onSeek).toHaveBeenCalledWith(10))
  })

  it('seeks to the next proposal after a decision', async () => {
    const { props } = setup()
    await waitFor(() => expect(props.onSeek).toHaveBeenCalledWith(10))
    press('c')
    await waitFor(() => expect(props.onSeek).toHaveBeenCalledWith(20))
  })

  it('decides a touch on a single keystroke', async () => {
    const { props } = setup()
    press('c')
    await waitFor(() =>
      expect(props.onDecide).toHaveBeenCalledWith(
        expect.objectContaining({ id: 'p0' }), 'confirmed'))
  })

  it('rejects on a single keystroke too', async () => {
    const { props } = setup()
    press('x')
    await waitFor(() =>
      expect(props.onDecide).toHaveBeenCalledWith(
        expect.objectContaining({ id: 'p0' }), 'rejected'))
  })

  it('answers a lunge with which fencer, in one key', async () => {
    // A lunge proposal asks two questions at once. Splitting them would double
    // the keystrokes on the item type there are most of.
    const { props } = setup({ kind: 'lunge' })
    press('2')
    await waitFor(() =>
      expect(props.onDecide).toHaveBeenCalledWith(
        expect.objectContaining({ id: 'p0' }), 1))
  })

  it('shows position in the queue so the end is in sight', async () => {
    setup()
    expect(await screen.findByText('1')).toBeInTheDocument()
    expect(screen.getByText(/of 3/)).toBeInTheDocument()
  })

  it('stays on an item whose decision failed', async () => {
    /* Advancing past a decision that did not persist drops it silently, and the user has no
       way to tell which one they lost. */
    const onDecide = vi.fn(() => Promise.reject(new Error('the store said no')))
    const { props } = setup({ onDecide })
    press('c')
    await waitFor(() =>
      expect(screen.getByText(/the store said no/)).toBeInTheDocument())
    expect(props.onSeek).not.toHaveBeenCalledWith(20)
  })

  it('skips without counting the skip as a decision', async () => {
    // Counting skips would improve the rate the more questions went unanswered.
    const { props } = setup({ items: items(1) })
    press('s')
    await waitFor(() => expect(screen.getByText('Queue finished')).toBeInTheDocument())
    expect(screen.getByText('0')).toBeInTheDocument()
    expect(props.onDecide).not.toHaveBeenCalled()
  })

  it('reports the rate at the end, which is the measurement', async () => {
    setup({ items: items(1) })
    press('c')
    await waitFor(() => expect(screen.getByText('Queue finished')).toBeInTheDocument())
    expect(screen.getByText(/per decision/)).toBeInTheDocument()
  })

  it('hands the count and the elapsed time back on exit', async () => {
    const { props } = setup({ items: items(1) })
    press('c')
    await waitFor(() => expect(screen.getByText('Queue finished')).toBeInTheDocument())
    press('escape')
    await waitFor(() => expect(props.onExit).toHaveBeenCalledWith(
      expect.objectContaining({ answered: 1, skipped: 0 })))
  })

  it('leaves on escape mid-queue without losing what was answered', async () => {
    const { props } = setup()
    press('c')
    await waitFor(() => expect(props.onDecide).toHaveBeenCalled())
    press('escape')
    await waitFor(() => expect(props.onExit).toHaveBeenCalledWith(
      expect.objectContaining({ answered: 1 })))
  })

  it('says so rather than showing an empty queue', async () => {
    setup({ items: [] })
    expect(screen.getByText(/Nothing pending/)).toBeInTheDocument()
  })

  it('ignores shortcuts while the user is typing', async () => {
    const { props } = setup()
    const input = document.createElement('input')
    document.body.appendChild(input)
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'c', bubbles: true }))
    await new Promise((r) => setTimeout(r, 40))
    expect(props.onDecide).not.toHaveBeenCalled()
    input.remove()
  })
})

describe('formatElapsed', () => {
  it('reads as a clock, not a decimal', () => {
    expect(formatElapsed(0)).toBe('00:00')
    expect(formatElapsed(65)).toBe('01:05')
    expect(formatElapsed(3599)).toBe('59:59')
  })
})
