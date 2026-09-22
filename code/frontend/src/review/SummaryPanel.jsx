import { useCallback, useEffect, useState } from 'react'
import { api, boutPath } from '../api.js'

/* Render just enough markdown for the four headings, bullets and bold that the summary prompt
   asks for. */
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
  const [reloadKey, setReloadKey] = useState(0)

  useEffect(() => {
    let cancelled = false
    setState({ loading: true })
    api(`${boutPath(boutId)}/summary`)
      .then((r) => { if (!cancelled) setState({ loading: false, data: r }) })
      .catch((e) => { if (!cancelled) setState({ loading: false, error: e.message }) })
    return () => { cancelled = true }
  }, [boutId, reloadKey])

  const refresh = useCallback(() => setReloadKey((k) => k + 1), [])

  if (state.loading) return <div className="mini">Loading.</div>
  if (state.error) return <div className="note err">{state.error}</div>

  const r = state.data
  if (!r.exists) {
    return (
      <div className="note">
        No summary for this bout yet. Generating one costs a paid API call, so
        it happens only when you ask for it.
        <Generate boutId={boutId} onDone={refresh} />
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
      {r.stale && <Generate boutId={boutId} force onDone={refresh} />}
    </>
  )
}

/* The button that spends money. Kept as an explicit action, never automatic, because each
   press costs a paid API call. */
function Generate({ boutId, force = false, onDone }) {
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState(null)

  // WAIT HERE RATHER THAN SENDING THE USER AWAY.
  const go = async () => {
    setBusy(true)
    setNote(null)
    try {
      const r = await api(
        `${boutPath(boutId)}/summary/generate${force ? '?force=true' : ''}`,
        { method: 'POST' })
      setNote(`Writing the summary from ${r.touches_used}. This takes a few `
              + 'seconds and costs one API call.')
      const id = r.job_id
      const deadline = Date.now() + 120000
      while (id && Date.now() < deadline) {
        await new Promise((res) => setTimeout(res, 1500))
        const jobs = await api('/api/jobs')
        const job = jobs.jobs.find((j) => j.job_id === id)
        if (!job) break
        if (job.state === 'done') {
          setNote(null)
          onDone?.()
          return
        }
        if (job.state === 'failed' || job.error) {
          setNote(job.error || 'the summary job failed')
          return
        }
      }
      setNote('Still running. It will appear here when it finishes; '
              + 'the upload tab shows its progress.')
    } catch (e) {
      setNote(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ marginTop: 9 }}>
      <button className="primary" disabled={busy} onClick={go}>
        {busy ? 'Writing...' : (force ? 'Regenerate the summary' : 'Generate a summary')}
      </button>
      {note && <div className="mini" style={{ marginTop: 6 }}>{note}</div>}
    </div>
  )
}
