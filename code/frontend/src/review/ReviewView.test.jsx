import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ReviewView from './ReviewView.jsx'

/* The review interface, which is where the project's central claim is cashed: that confirming
   the system's proposals costs less effort than labelling from scratch. */

const BOUT = 'results_current:fencing_clip3'

const touches = (over = {}) => ({
  bout_id: BOUT,
  duration_s: 180,
  has_video: true,
  proposed: [
    { id: 'p0', time_s: 19.0, confidence: 0.9, separation_m: 2.1,
      state: 'pending', scorer: null, corrected_time_s: null },
    { id: 'p1', time_s: 35.0, confidence: 0.5, separation_m: 1.4,
      state: 'pending', scorer: null, corrected_time_s: null },
  ],
  added: [],
  unreliable_segments: [],
  reanchors: [],
  lunges: [],
  progress: { proposed: 2, reviewed: 0, confirmed: 0, rejected: 0,
              user_added: 0, unreliable_segments: 0, pending_reanchors: 0,
              complete: false },
  ...over,
})

const metrics = () => ({
  confirmed_touches: [],
  whole_recording: {
    distance_m: { mean: 2.53 },
    fencer_1: { net_displacement_m: 3.43 },
    fencer_2: { net_displacement_m: -0.43 },
    movement_basis: { source: 'raw per-frame positions' },
  },
  in_play: {
    in_play_fraction: 1.0, mean_distance_m: 2.53,
    fencer_1: { net_forward_movement_m: 3.43, closing_share_pct: 52.4 },
    fencer_2: { net_forward_movement_m: -0.43, closing_share_pct: 52.0 },
  },
  scoping_basis: 'none: no touches confirmed yet',
  movement_basis: { source: 'raw per-frame positions' },
})

let posted

// A second bout, so a test can actually CHANGE the dropdown.
const OTHER_BOUT = 'results_current:fencing_clip2'

function stubFetch({ failTouchDecision = false, touchesBody = null,
                     twoBouts = false } = {}) {
  posted = []
  global.fetch = vi.fn((url, opts) => {
    const method = opts?.method || 'GET'
    if (method !== 'GET') posted.push({ url, method, body: opts?.body })

    const ok = (body) => Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    const fail = (detail) => Promise.resolve({
      ok: false, status: 400, statusText: 'Bad Request',
      json: () => Promise.resolve({ detail }),
    })

    if (url === '/api/bouts') {
      const list = [{ bout_id: BOUT, label: BOUT, has_touches: true,
                      has_video: true, progress: {} }]
      if (twoBouts) {
        list.push({ bout_id: OTHER_BOUT, label: OTHER_BOUT, has_touches: true,
                    has_video: true, progress: {} })
      }
      return ok({ bouts: list })
    }
    if (url.includes('/decision')) {
      return failTouchDecision ? fail('the store rejected that state') : ok({ ok: true })
    }
    if (url.includes('/touches')) return ok(touchesBody || touches())
    if (url.includes('/metrics')) return ok(metrics())
    if (url.includes('/reanchor-outcomes')) return ok({ exists: false, outcomes: [] })
    if (url.includes('/summary')) return ok({ exists: false, hint: 'run the command' })
    return ok({ ok: true })
  })
}

const press = (key) =>
  document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }))

// The corrective tools moved into a tab. Opening it is now part of reaching them.
const openTools = async () => {
  fireEvent.click(await screen.findByRole('tab', { name: 'Tools' }))
}

beforeEach(() => { vi.restoreAllMocks() })
afterEach(() => { delete global.fetch })

describe('ReviewView', () => {
  it('opens a bout that can actually be reviewed', async () => {
    // Rather than whichever sorts first: several discoverable bouts have no
    // touch file at all, and opening one of those shows an empty table.
    stubFetch()
    render(<ReviewView initialBoutId={null} />)
    await waitFor(() => expect(screen.getByText('0:19.0')).toBeInTheDocument())
  })

  it('shows both proposals with what each rests on', async () => {
    stubFetch()
    const { container } = render(<ReviewView initialBoutId={BOUT} />)
    await waitFor(() => expect(screen.getByText('0:19.0')).toBeInTheDocument())
    const basis = [...container.querySelectorAll('.basis')]
      .map((el) => el.textContent).join(' | ')
    expect(basis).toContain('0.90')
    expect(basis).toContain('0.50')
  })

  it('confirms a touch on a single keystroke', async () => {
    // The effort claim depends on this.
    stubFetch()
    render(<ReviewView initialBoutId={BOUT} />)
    await waitFor(() => expect(screen.getByText('0:19.0')).toBeInTheDocument())

    press('c')
    await waitFor(() => {
      const decisions = posted.filter((p) => p.url.includes('/decision'))
      expect(decisions.length).toBe(1)
      expect(JSON.parse(decisions[0].body)).toEqual({ state: 'confirmed' })
    })
  })

  it('rejects on a single keystroke too', async () => {
    stubFetch()
    render(<ReviewView initialBoutId={BOUT} />)
    await waitFor(() => expect(screen.getByText('0:19.0')).toBeInTheDocument())

    press('x')
    await waitFor(() => {
      const d = posted.filter((p) => p.url.includes('/decision'))
      expect(JSON.parse(d[0].body)).toEqual({ state: 'rejected' })
    })
  })

  it('shows a failed decision instead of swallowing it', async () => {
    /* The defect this replaced: a rejected call became an unhandled promise rejection, so
       pressing a key against a failing server did nothing and said nothing, which is
       indistinguishable from the key not being bound. */
    stubFetch({ failTouchDecision: true })
    render(<ReviewView initialBoutId={BOUT} />)
    await waitFor(() => expect(screen.getByText('0:19.0')).toBeInTheDocument())

    press('c')
    await waitFor(() =>
      expect(screen.getByText(/the store rejected that state/)).toBeInTheDocument())
  })

  it('ignores shortcuts while the user is typing in a field', async () => {
    // The add-a-touch form takes a time in seconds.
    stubFetch()
    render(<ReviewView initialBoutId={BOUT} />)
    await waitFor(() => expect(screen.getByText('0:19.0')).toBeInTheDocument())

    const field = screen.getByPlaceholderText('time (s)')
    field.dispatchEvent(new KeyboardEvent('keydown', { key: 'c', bubbles: true }))
    await new Promise((r) => setTimeout(r, 50))
    expect(posted.filter((p) => p.url.includes('/decision'))).toHaveLength(0)
  })

  it('lists the keyboard shortcuts, since they are the whole workflow', async () => {
    stubFetch()
    const { container } = render(<ReviewView initialBoutId={BOUT} />)
    await waitFor(() => expect(screen.getByText('0:19.0')).toBeInTheDocument())
    // Matched on the hint strip itself rather than on words: "confirm" and
    // "reject" also label buttons in the table, and the strip is now a row of
    // keys with one-word glosses rather than a sentence.
    const hint = container.querySelector('.keyhint')
    expect(hint).not.toBeNull()
    for (const key of ['j', 'l', 'k', 'c', 'x', 'a', '1', '2']) {
      expect([...hint.querySelectorAll('b')].map((b) => b.textContent))
        .toContain(key)
    }
  })

  it('says a bout has no proposals rather than showing an empty table', async () => {
    stubFetch({ touchesBody: touches({
      proposed: [],
      progress: { proposed: 0, reviewed: 0, confirmed: 0, rejected: 0,
                  user_added: 0, unreliable_segments: 0,
                  pending_reanchors: 0, complete: true },
    }) })
    render(<ReviewView initialBoutId={BOUT} />)
    await waitFor(() =>
      expect(screen.getByText(/No proposals for this bout/)).toBeInTheDocument())
  })

  it('disables re-running until there is a correction to apply', async () => {
    // A reprocess with no corrections is a multi-minute job that changes nothing, so the
    // button should not invite it.
    stubFetch()
    render(<ReviewView initialBoutId={BOUT} />)
    await waitFor(() => expect(screen.getByText('0:19.0')).toBeInTheDocument())
    await openTools()
    expect(screen.getByRole('button', { name: /Re-run with corrections/ }))
      .toBeDisabled()
  })

  it('offers to read the lamps for either fencer', async () => {
    // Which fencer the green lamp belongs to cannot be inferred from the image,
    // so it is asked rather than guessed.
    stubFetch()
    render(<ReviewView initialBoutId={BOUT} />)
    await waitFor(() => expect(screen.getByText('0:19.0')).toBeInTheDocument())
    await openTools()
    expect(screen.getByText(/green belongs to/)).toBeInTheDocument()
  })

  it('keeps the review loop free of the tools that are not part of it', async () => {
    /* The restructure this replaced a stack of six equal-weight panels with. */
    stubFetch()
    render(<ReviewView initialBoutId={BOUT} />)
    await waitFor(() => expect(screen.getByText('0:19.0')).toBeInTheDocument())
    expect(screen.queryByRole('button', { name: /Re-run with corrections/ }))
      .toBeNull()
    expect(screen.queryByText(/green belongs to/)).toBeNull()
  })

  it('opens the bout tab first, since that is what a reviewer wants', async () => {
    stubFetch()
    render(<ReviewView initialBoutId={BOUT} />)
    await waitFor(() =>
      expect(screen.getByRole('tab', { name: 'Bout' }))
        .toHaveAttribute('aria-selected', 'true'))
  })
})

describe('the keyboard loop survives choosing a bout', () => {
  it('gives up focus when a bout is chosen', async () => {
    // The shortcut handler ignores every key while an input, select or textarea has focus,
    // which is right for a text box.
    stubFetch({ twoBouts: true })
    const { container } = render(<ReviewView initialBoutId={BOUT} />)
    const select = await waitFor(() => {
      const s = container.querySelector('select')
      expect(s.options.length).toBe(2)
      return s
    })
    const blur = vi.spyOn(select, 'blur')
    fireEvent.change(select, { target: { value: OTHER_BOUT } })
    expect(blur).toHaveBeenCalled()
  })
})
