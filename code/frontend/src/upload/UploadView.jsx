import { useCallback, useEffect, useRef, useState } from 'react'
import { api, uploadVideo } from '../api.js'
import JobCard from './JobCard.jsx'
import Disclosure from '../review/Disclosure.jsx'

// How often to ask the server what the jobs are doing.
const POLL_MS = 2000

// Polling stops when nothing can change.
const LIVE_STATES = ['queued', 'running', 'awaiting_piste']

export default function UploadView({ onOpenBout }) {
  const [jobs, setJobs] = useState([])
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploadPct, setUploadPct] = useState(null)
  const [confirmPiste, setConfirmPiste] = useState(true)
  const [storage, setStorage] = useState(null)
  const [cleanup, setCleanup] = useState(null)
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

    // Chained timeouts rather than an interval.
    const tick = async () => {
      const jobs = await refresh()
      if (cancelled) return
      const live = jobs && jobs.some((j) => LIVE_STATES.includes(j.state))
      // Keep polling while jobs are live.
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
        <section className="sec">
          <h3>Upload a bout</h3>
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
          <Disclosure label="What is the piste region?">
            It tells the tracker which part of the frame the strip occupies, so
            it can ignore the referee and the spectators. It is measured from the
            footage first and then shown for confirmation; leaving this unticked
            accepts whatever is measured. It matters most on competition footage:
            rebuilding the reference results without the regions dropped one clip
            from 98 per cent tracking coverage to 80.
          </Disclosure>

          {error && <div className="note err">{error}</div>}
        </section>

        <section className="sec">
          <h3>Disk</h3>
          {!storage ? (
            <button onClick={async () => {
              try { setStorage(await api('/api/storage')) }
              catch (e) { setStorage({ error: e.message }) }
            }}>Show what is using disk</button>
          ) : storage.error ? (
            <div className="note err">{storage.error}</div>
          ) : (
            <>
              {storage.categories.map((c) => (
                <div className="stat" key={c.name}>
                  <span title={c.note}>
                    {c.name}{c.reclaimable ? '' : ' (kept)'}
                  </span>
                  <b>{c.mb} MB</b>
                </div>
              ))}
              <div className="note">
                <b>{storage.reclaimable_mb} MB</b> in {storage.reclaimable_files}{' '}
                cached video file(s) can go now: their source is gone, or they are
                older than it and would be re-encoded on next view anyway.
                {storage.old_jobs?.length > 0 && (
                  <> {storage.old_jobs.length} job record(s) are over a month
                     old; they are listed rather than removed, since each is the
                     only account of how a result was produced.</>
                )}
              </div>
              <div className="row">
                <button className="primary"
                        disabled={!storage.reclaimable_files}
                        onClick={async () => {
                          try {
                            const r = await api('/api/storage/cleanup',
                                                { method: 'POST' })
                            setStorage(null)
                            setCleanup(r)
                          } catch (e) { setStorage({ error: e.message }) }
                        }}>
                  Reclaim {storage.reclaimable_mb} MB
                </button>
                <button onClick={async () => {
                  try {
                    const r = await api('/api/storage/cleanup?everything=true',
                                        { method: 'POST' })
                    setStorage(null)
                    setCleanup(r)
                  } catch (e) { setStorage({ error: e.message }) }
                }}>Clear the whole video cache</button>
              </div>
            </>
          )}
          {cleanup && (
            <div className="note">
              Removed {cleanup.removed} file(s), freeing {cleanup.freed_mb} MB.
              {!cleanup.orphans_only
                && ' Bouts will re-encode on next view, taking a few seconds each.'}
            </div>
          )}
        </section>

        <section className="sec">
          <h3>Jobs</h3>
          {jobs.length === 0
            ? <div className="mini">Nothing processed yet.</div>
            : jobs.map((job) => (
                <JobCard key={job.job_id} job={job}
                         onChanged={refresh} onOpenBout={onOpenBout} />
              ))}
        </section>
      </div>
    </main>
  )
}
