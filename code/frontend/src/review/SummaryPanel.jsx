import { useEffect, useState } from 'react'
import { api, boutPath } from '../api.js'

/**
 * Render just enough markdown for the four headings, bullets and bold that the
 * summary prompt asks for. A full parser would be more code than the format
 * needs.
 *
 * Built as React elements rather than as an HTML string. The original page
 * escaped the text by hand before inserting it, because this is model output and
 * treating generated text as trusted markup is how a prompt injection becomes a
 * script tag. React escapes anything placed in a child position, so the same
 * guarantee comes from not reaching for `dangerouslySetInnerHTML` at all, which
 * is a harder thing to get wrong by accident later.
 */
function renderMarkdown(md) {
  return md.split('\n').map((line, i) => {
    if (/^##\s+/.test(line)) {
      return <h3 key={i} className="md-h">{line.replace(/^##\s+/, '')}</h3>
    }
    if (/^[-*]\s+/.test(line)) {
      return (
        <div key={i} style={{ margin: '0 0 4px 12px' }}>
          &bull; {bold(line.replace(/^[-*]\s+/, ''))}
        </div>
      )
    }
    return line.trim()
      ? <p key={i} style={{ margin: '0 0 7px' }}>{bold(line)}</p>
      : null
  })
}

// **bold** without an HTML round trip: split on the markers and wrap the odd
// segments, which are the ones that were between them.
function bold(text) {
  return text.split(/\*\*(.+?)\*\*/g).map((part, i) =>
    i % 2 ? <b key={i}>{part}</b> : part)
}

export default function SummaryPanel({ boutId }) {
  const [state, setState] = useState({ loading: true })

  useEffect(() => {
    let cancelled = false
    setState({ loading: true })
    api(`${boutPath(boutId)}/summary`)
      .then((r) => { if (!cancelled) setState({ loading: false, data: r }) })
      .catch((e) => { if (!cancelled) setState({ loading: false, error: e.message }) })
    return () => { cancelled = true }
  }, [boutId])

  if (state.loading) return <div className="mini">Loading.</div>
  if (state.error) return <div className="note err">{state.error}</div>

  const r = state.data
  if (!r.exists) {
    return (
      <div className="note">
        No summary for this bout yet. Generating one costs an API call, so it
        stays a command you run:
        <br /><code>{r.hint}</code>
      </div>
    )
  }

  return (
    <>
      {/* A stale summary is the dangerous case, not the missing one: it reads as
          current. This project has already shipped one piece of confident prose
          describing numbers that no longer existed. */}
      {r.stale && (
        <div className="note">
          <b>Out of date.</b> {r.stale_reason}. Regenerate before quoting
          anything from it.
        </div>
      )}
      {renderMarkdown(r.markdown)}
      <div className="mini" style={{ marginTop: 8, color: 'var(--muted)' }}>
        {r.model || 'unknown model'}, from {r.generated_from}
      </div>
    </>
  )
}
