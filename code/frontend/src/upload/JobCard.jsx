import { useState } from 'react'
import { api, del } from '../api.js'
import PisteConfirm from './PisteConfirm.jsx'

// What each stage is called for someone who did not write the pipeline. The
// stage names are the runner's internal ones and mean little on their own.
const STAGE_LABELS = {
  piste: 'Measuring the piste region',
  detect: 'Detecting and tracking fencers',
  touches: 'Proposing touches',
  transcode: 'Preparing the video for playback',
}

function describe(job) {
  if (job.state === 'queued') {
    const n = job.queue_position
    if (n === 0 || n == null) return 'Waiting to start'
    return `Waiting, ${n} job${n === 1 ? '' : 's'} ahead`
  }
  if (job.state === 'running') {
    return STAGE_LABELS[job.stage] || 'Processing'
  }
  if (job.state === 'awaiting_piste') return 'Waiting for you to confirm the piste region'
  if (job.state === 'done') return 'Finished'
  if (job.state === 'cancelled') return 'Cancelled'
  if (job.state === 'interrupted') return 'Interrupted by a server restart'
  return 'Failed'
}

function elapsed(seconds) {
  if (seconds == null) return null
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return m ? `${m}m ${s}s` : `${s}s`
}

export default function JobCard({ job, onChanged, onOpenBout }) {
  const [busy, setBusy] = useState(false)
  const [log, setLog] = useState(null)
  const [error, setError] = useState(null)

  const act = async (fn) => {
    setBusy(true)
    setError(null)
    try { await fn(); await onChanged() }
    catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }

  const showLog = async () => {
    setBusy(true)
    try {
      const res = await fetch(`/api/jobs/${job.job_id}/log`)
      setLog(await res.text())
    } catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }

  const info = job.video_info
  const pct = job.progress?.pct ?? 0
  const live = job.state === 'running' || job.state === 'queued'
  const finished = ['done', 'failed', 'cancelled', 'interrupted'].includes(job.state)

  return (
    <div className="job">
      <header>
        <span className="name">{job.filename}</span>
        <span className={`state ${job.state}`}>{job.state.replace('_', ' ')}</span>
        <span className="spacer" />
        {job.state === 'done' && (
          <button className="primary" onClick={() => onOpenBout(job.bout_id)}>
            Open in review
          </button>
        )}
        {live && (
          <button disabled={busy}
                  onClick={() => act(() => api(`/api/jobs/${job.job_id}/cancel`,
                                               { method: 'POST' }))}>
            Cancel
          </button>
        )}
        {(finished || job.state === 'awaiting_piste') && (
          <button disabled={busy}
                  onClick={() => act(() => api(`/api/jobs/${job.job_id}`, del))}>
            Delete
          </button>
        )}
      </header>

      <div className="mini" style={{ marginBottom: 7 }}>
        {describe(job)}
        {info && ` - ${info.width}x${info.height}`}
        {info?.duration_s != null && `, ${info.duration_s}s`}
        {job.elapsed_s != null && ` - ${elapsed(job.elapsed_s)} elapsed`}
      </div>

      {job.state === 'running' && (
        <div className="bar"><div style={{ width: `${pct}%` }} /></div>
      )}

      {job.state === 'awaiting_piste' && (
        <PisteConfirm job={job} onDecided={onChanged} />
      )}

      {(job.state === 'failed' || job.state === 'interrupted') && (
        <>
          <div className="note err">{job.error}</div>
          {/* A failure here is almost always a Python traceback, and without it
              the user is told only that something exited non-zero. */}
          {job.log_tail?.length > 0 && (
            <pre className="log">{job.log_tail.join('\n')}</pre>
          )}
        </>
      )}

      {job.state === 'done' && job.piste && (
        <div className="note">
          {job.piste.polygon
            ? `Piste region ${job.piste.decision || 'measured'}: rows `
              + `${Math.round(job.piste.polygon[0][1])} to `
              + `${Math.round(job.piste.polygon[2][1])} of the frame.`
            : 'Processed without a piste region. '
              + (job.piste.reason || '')}
        </div>
      )}

      {error && <div className="note err">{error}</div>}

      <div className="row" style={{ marginTop: 7 }}>
        <button className="mini" disabled={busy} onClick={showLog}>
          {log ? 'Refresh output' : 'Show pipeline output'}
        </button>
        {log && <button className="mini" onClick={() => setLog(null)}>Hide</button>}
      </div>
      {log && <pre className="log">{log}</pre>}
    </div>
  )
}
