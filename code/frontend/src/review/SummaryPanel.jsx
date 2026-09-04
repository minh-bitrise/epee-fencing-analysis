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
        No summary for this bout yet. Generating one costs a paid API call, so
        it happens only when you ask for it.
        <Generate boutId={boutId} />
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
      {r.stale && <Generate boutId={boutId} force />}
    </>
  )
}

/**
 * The button that spends money.
 *
 * Kept as an explicit action, never automatic, because each press costs a paid
 * API call. What changed when the job runner arrived is only that the user
 * presses a button instead of being handed a command line to type in a terminal,
 * which was this panel's previous answer.
 *
 * The request only QUEUES the job. The summary appears on the next load rather
 * than streaming in, which is honest about what is happening: the work is
 * running in the same single-slot queue as everything else and may be behind a
 * detection run.
 */
function Generate({ boutId, force = false }) {
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState(null)

  const go = async () => {
    setBusy(true)
    setNote(null)
    try {
      const r = await api(
        `${boutPath(boutId)}/summary/generate${force ? '?force=true' : ''}`,
        { method: 'POST' })
      setNote(`Queued, using ${r.touches_used}. Watch it on the upload tab, `
              + 'then reload this bout.')
    } catch (e) {
      setNote(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ marginTop: 9 }}>
      <button className="primary" disabled={busy} onClick={go}>
        {force ? 'Regenerate the summary' : 'Generate a summary'}
      </button>
      {note && <div className="mini" style={{ marginTop: 6 }}>{note}</div>}
    </div>
  )
}
