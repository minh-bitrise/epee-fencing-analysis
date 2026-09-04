import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import SummaryPanel from './SummaryPanel.jsx'

/**
 * The summary is MODEL OUTPUT rendered as formatted text, which makes it the one
 * place in this interface where untrusted content becomes markup. It is also the
 * one place where being out of date is more dangerous than being absent, because
 * a stale summary reads as current.
 */

const respond = (body) => {
  global.fetch = vi.fn(() => Promise.resolve({
    ok: true, json: () => Promise.resolve(body),
  }))
}

beforeEach(() => { vi.restoreAllMocks() })
afterEach(() => { delete global.fetch })

describe('SummaryPanel', () => {
  it('does not turn model output into markup', async () => {
    // Treating generated text as trusted markup is how a prompt injection
    // becomes a script tag. React escapes children, so the guarantee comes from
    // never reaching for dangerouslySetInnerHTML; this asserts that it holds.
    respond({ exists: true, stale: false, model: 'test',
              generated_from: 'touch data',
              markdown: 'Fencer 1 <img src=x onerror="alert(1)"> closed well' })
    const { container } = render(<SummaryPanel boutId="b" />)
    await waitFor(() => expect(screen.getByText(/closed well/)).toBeInTheDocument())
    expect(container.querySelector('img')).toBeNull()
    expect(container.textContent).toContain('<img')
  })

  it('warns loudly when the summary predates its own numbers', async () => {
    // The dangerous case is not the missing summary, it is the stale one. This
    // project has already shipped one piece of confident prose describing
    // numbers that no longer existed.
    respond({ exists: true, stale: true,
              stale_reason: 'the metrics changed after this was written',
              markdown: '## Summary\nSomething confident.',
              model: 'test', generated_from: 'touch data' })
    render(<SummaryPanel boutId="b" />)
    await waitFor(() => expect(screen.getByText(/Out of date/)).toBeInTheDocument())
    expect(screen.getByText(/the metrics changed/)).toBeInTheDocument()
  })

  it('offers to generate one when none exists, and says it costs money', async () => {
    respond({ exists: false, hint: 'python3 generate_summary.py ...' })
    render(<SummaryPanel boutId="b" />)
    await waitFor(() => expect(screen.getByText(/costs a paid API call/))
      .toBeInTheDocument())
    expect(screen.getByRole('button', { name: /Generate a summary/ }))
      .toBeInTheDocument()
  })

  it('says which model wrote it and what it was built from', async () => {
    // Both qualify the text. A summary built from unreviewed detector output is
    // a different claim to one built from touches the user confirmed.
    respond({ exists: true, stale: false, markdown: 'Body.',
              model: 'claude-x', generated_from: 'the whole recording, no touch data' })
    render(<SummaryPanel boutId="b" />)
    await waitFor(() => expect(screen.getByText(/claude-x/)).toBeInTheDocument())
    expect(screen.getByText(/no touch data/)).toBeInTheDocument()
  })

  it('renders headings and bullets without a markdown library', async () => {
    respond({ exists: true, stale: false, model: 'test',
              generated_from: 'touch data',
              markdown: '## Distance\n- closed **well**\n' })
    render(<SummaryPanel boutId="b" />)
    await waitFor(() => expect(screen.getByText('Distance')).toBeInTheDocument())
    expect(screen.getByText('well').tagName).toBe('B')
  })
})
