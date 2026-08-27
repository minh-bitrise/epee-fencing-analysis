import { useCallback, useEffect, useRef, useState } from 'react'
import { api, uploadVideo } from '../api.js'
import JobCard from './JobCard.jsx'

// How often to ask the server what the jobs are doing. Two seconds is chosen
// against what is being watched rather than by habit: the pipeline reports
// progress every hundred frames, which on this machine is a few seconds apart,
// so polling faster returns the same numbers repeatedly and polling slower makes
// the bar visibly lag the work.
const POLL_MS = 2000

// Polling stops when nothing can change. Without this the page would keep asking
// about a queue of finished jobs for as long as it was left open.
const LIVE_STATES = ['queued', 'running', 'awaiting_piste']

export default function UploadView({ onOpenBout }) {
  const [jobs, setJobs] = useState([])
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploadPct, setUploadPct] = useState(null)
  const [confirmPiste, setConfirmPiste] = useState(true)
  const fileInput = useRef(null)

  const refresh = useCallback(async () => {
    try {
      const { jobs } = await api('/api/jobs')
      setJobs(jobs)
      setError(null)
      return jobs
    } catch (e) {
      setError(e.message)
      return null
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    let timer = null

    // Chained timeouts rather than an interval. An interval fires on a fixed
    // schedule regardless of whether the previous request has come back, so a
    // slow response makes them pile up; this waits for each answer before asking
    // again.
    const tick = async () => {
      const jobs = await refresh()
      if (cancelled) return
      const live = jobs && jobs.some((j) => LIVE_STATES.includes(j.state))
      // Keep polling while jobs are live. When nothing is, poll slowly anyway,
      // because a job can be started from another tab or a terminal and this
      // page should notice.
      timer = setTimeout(tick, live ? POLL_MS : POLL_MS * 5)
    }
    tick()
    return () => { cancelled = true; if (timer) clearTimeout(timer) }
  }, [refresh])

  const send = useCallback(async (file) => {
    if (!file) return
    setError(null)
    setUploadPct(0)
    try {
      await uploadVideo(file, { confirmPiste, onProgress: setUploadPct })
      await refresh()
    } catch (e) {
      setError(e.message)
    } finally {
      setUploadPct(null)
    }
  }, [confirmPiste, refresh])

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    send(e.dataTransfer.files?.[0])
  }

  return (
    <main className="single">
      <div>
        <div className="panel">
          <h2>Upload a bout</h2>
          <div className={`drop${dragOver ? ' over' : ''}`}
               onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
               onDragLeave={() => setDragOver(false)}
               onDrop={onDrop}
               onClick={() => fileInput.current?.click()}>
            {uploadPct === null ? (
              <>
                <div><b>Drop a bout video here</b>, or click to choose one.</div>
                <div className="mini" style={{ marginTop: 6 }}>
                  MP4, MOV, M4V, AVI or MKV, up to 500 MB. Processing runs in the
                  background, so you can leave this page and come back.
                </div>
              </>
            ) : (
              <>
                <div>Uploading, {uploadPct}%</div>
                <div className="bar" style={{ marginTop: 9 }}>
                  <div style={{ width: `${uploadPct}%` }} />
                </div>
              </>
            )}
            <input ref={fileInput} type="file"
                   accept="video/mp4,video/quicktime,video/x-m4v,video/x-msvideo,video/x-matroska"
                   onChange={(e) => send(e.target.files?.[0])} />
          </div>

          <div className="row" style={{ marginTop: 11 }}>
            <label className="opt">
              <input type="checkbox" checked={confirmPiste}
                     style={{ padding: 0 }}
                     onChange={(e) => setConfirmPiste(e.target.checked)} />
              Let me confirm the piste region before processing
            </label>
          </div>
          <div className="note">
            The piste region tells the tracker which part of the frame the strip
            occupies, so it can ignore the referee and the spectators. It is
            measured from the footage first and then shown for confirmation.
            Leaving this unticked accepts whatever is measured. It matters most
            on competition footage: rebuilding the reference results without the
            regions dropped one clip from 98 per cent tracking coverage to 80.
          </div>

          {error && <div className="note err">{error}</div>}
        </div>

        <div className="panel">
          <h2>Jobs</h2>
          {jobs.length === 0
            ? <div className="mini">Nothing processed yet.</div>
            : jobs.map((job) => (
                <JobCard key={job.job_id} job={job}
                         onChanged={refresh} onOpenBout={onOpenBout} />
              ))}
        </div>
      </div>
    </main>
  )
}
