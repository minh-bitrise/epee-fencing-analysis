import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, boutPath, del, postJSON } from '../api.js'
import { mergeRows, useBout } from './useBout.js'
import Timeline from './Timeline.jsx'
import TouchTable from './TouchTable.jsx'
import MetricsPanel from './MetricsPanel.jsx'
import SummaryPanel from './SummaryPanel.jsx'
import LungePanel from './LungePanel.jsx'

// The clips are 30 fps. Hard-coded because the API does not report frame rate
// and this only drives the step size of a keyboard nudge, so being slightly off
// on a 25 fps clip costs nothing: the label records the timestamp the video
// actually reached, not this constant.
const FRAME_S = 1 / 30

export default function ReviewView({ initialBoutId }) {
  const [bouts, setBouts] = useState([])
  const [boutId, setBoutId] = useState(initialBoutId || null)
  const [selIdx, setSelIdx] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [armedSlot, setArmedSlot] = useState(null)
  const [anchorMsg, setAnchorMsg] = useState('')
  const [exportOut, setExportOut] = useState(null)
  const [anchorOut, setAnchorOut] = useState(null)
  const [listError, setListError] = useState(null)
  const [outcomes, setOutcomes] = useState(null)
  const [scorers, setScorers] = useState(null)
  const videoRef = useRef(null)
  // Guards against keystrokes arriving mid-request. A ref rather than state
  // because the keyboard handler has to read the current value at the moment the
  // key is pressed, and a state variable captured in a closure would be stale.
  const busy = useRef(false)

  const { data, metrics, error, metricsError, reload } = useBout(boutId)
  const rows = useMemo(() => mergeRows(data), [data])

  // --- bout list ---------------------------------------------------------

  useEffect(() => {
    let cancelled = false
    api('/api/bouts').then(({ bouts }) => {
      if (cancelled) return
      setBouts(bouts)
      if (!boutId) {
        // Open a bout that can actually be reviewed, rather than whichever
        // happens to sort first.
        const best = bouts.find((b) => b.has_touches && b.has_video) || bouts[0]
        if (best) setBoutId(best.bout_id)
      }
    }).catch((e) => !cancelled && setListError(e.message))
    return () => { cancelled = true }
  }, [])                              // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { if (initialBoutId) setBoutId(initialBoutId) }, [initialBoutId])

  // Whether corrections already applied to this bout actually changed anything.
  // Fetched per bout rather than folded into the touches payload because it is a
  // property of how this bout was PRODUCED, not of the review in progress.
  useEffect(() => {
    if (!boutId) return
    let cancelled = false
    api(`${boutPath(boutId)}/reanchor-outcomes`)
      .then((r) => !cancelled && setOutcomes(r))
      .catch(() => !cancelled && setOutcomes(null))
    return () => { cancelled = true }
  }, [boutId])

  // Move the cursor to the first undecided proposal, but only when the bout
  // changes. Recomputing it on every refresh made keyboard review skip
  // proposals, because each decision triggered a reload that moved the cursor
  // out from under the next keystroke.
  useEffect(() => {
    setSelIdx(0)
    setExportOut(null)
    setAnchorOut(null)
    setOutcomes(null)
    setScorers(null)
  }, [boutId])

  const jumpedRef = useRef(null)
  useEffect(() => {
    if (!data || jumpedRef.current === boutId) return
    jumpedRef.current = boutId
    const first = rows.findIndex((r) => r.state === 'pending')
    setSelIdx(Math.max(0, first))
  }, [data, rows, boutId])

  // --- playback ----------------------------------------------------------

  const seek = useCallback((t) => {
    const v = videoRef.current
    if (!v) return
    // Land a little before the moment rather than on it. A touch is judged from
    // the approach that produced it, and starting playback on the frame of
    // contact shows the aftermath instead.
    v.currentTime = Math.max(0, t - 1.5)
    v.play().catch(() => {})
  }, [])

  const select = useCallback((i) => {
    setSelIdx(i)
    if (rows[i]) seek(rows[i].at)
  }, [rows, seek])

  // --- the four annotation actions ---------------------------------------

  const decide = useCallback(async (touchId, state, advance = false) => {
    if (busy.current) return
    busy.current = true
    try {
      await api(`${boutPath(boutId)}/touches/${touchId}/decision`,
                postJSON({ state }))
      await reload()
      if (advance) {
        // The next proposal still awaiting a decision, searching forward from
        // the cursor and then wrapping.
        setSelIdx((cur) => {
          let i = rows.findIndex((r, j) => j > cur && r.kind === 'proposed'
                                            && r.state === 'pending')
          if (i < 0) {
            i = rows.findIndex((r) => r.kind === 'proposed' && r.state === 'pending')
          }
          if (i >= 0) { seek(rows[i].at); return i }
          return cur
        })
      }
    } catch (e) {
      setListError(e.message)
    } finally {
      busy.current = false
    }
  }, [boutId, reload, rows, seek])

  // Every action routes its failures here. Without it a rejected call became an
  // unhandled promise rejection: the keyboard shortcuts were the worst case,
  // because pressing "a" to add a touch or "1" to label a lunge against a
  // failing server did nothing at all and said nothing, which is
  // indistinguishable from the key not being bound.
  const guard = useCallback(async (fn) => {
    try {
      await fn()
      setListError(null)
    } catch (e) {
      setListError(e.message)
    }
  }, [])

  const addTouchAtPlayhead = useCallback(async () => {
    const v = videoRef.current
    if (!v) return
    await guard(async () => {
      await api(`${boutPath(boutId)}/touches`,
                postJSON({ time_s: +v.currentTime.toFixed(1), scorer: 'unknown' }))
      await reload()
    })
  }, [boutId, reload, guard])

  const removeAdded = useCallback(async (id) => {
    await guard(async () => {
      await api(`${boutPath(boutId)}/touches/${id}`, del)
      await reload()
    })
  }, [boutId, reload, guard])

  const addSegment = useCallback(async (e) => {
    e.preventDefault()
    const form = new FormData(e.currentTarget)
    try {
      await api(`${boutPath(boutId)}/segments`, postJSON({
        start_s: parseFloat(form.get('start')),
        end_s: parseFloat(form.get('end')),
      }))
      e.target.reset()
      await reload()
    } catch (err) { setListError(err.message) }
  }, [boutId, reload])

  const addTouchByTime = useCallback(async (e) => {
    e.preventDefault()
    const form = new FormData(e.currentTarget)
    try {
      await api(`${boutPath(boutId)}/touches`, postJSON({
        time_s: parseFloat(form.get('time')),
        scorer: form.get('scorer'),
      }))
      e.target.reset()
      await reload()
    } catch (err) { setListError(err.message) }
  }, [boutId, reload])

  // Action 4. Arming rather than prompting, because the correction is a POSITION
  // and the only sane way to express it is to point at the fencer. The video is
  // paused on arming so the frame cannot move between the click and the
  // timestamp recorded with it.
  const armAnchor = (slot) => {
    setArmedSlot(slot)
    videoRef.current?.pause()
    setAnchorMsg(`click Fencer ${slot + 1} in the video (Esc to cancel)`)
  }

  const disarm = (msg = '') => { setArmedSlot(null); setAnchorMsg(msg) }

  const placeAnchor = useCallback(async (ev) => {
    if (armedSlot === null) return
    ev.preventDefault()
    const v = ev.currentTarget

    // Before metadata arrives videoWidth is 0, which would make the scale
    // infinite and silently record the correction at the origin. Refusing is the
    // only safe option: a correction placed at (0, 0) is worse than no
    // correction, because it looks deliberate.
    if (!v.videoWidth || !v.videoHeight) {
      disarm('the video has not loaded its size yet, nothing recorded. Try again once it plays.')
      return
    }
    const r = v.getBoundingClientRect()
    // A collapsed element gives a zero scale, and dividing by it yields NaN,
    // which passes every comparison in the bounds check below. Guard the scale,
    // not just the result: a check that silently accepts NaN is worse than none.
    if (r.width <= 0 || r.height <= 0) {
      disarm('the video is not visible, nothing recorded')
      return
    }
    // The element is scaled, and the correction has to be in SOURCE pixels
    // because that is the space the tracker works in. Letterboxing is accounted
    // for: a video shown in a box of a different aspect ratio is centred with
    // bars, and treating the element box as the image would offset every
    // correction.
    const scale = Math.min(r.width / v.videoWidth, r.height / v.videoHeight)
    const padX = (r.width - v.videoWidth * scale) / 2
    const padY = (r.height - v.videoHeight * scale) / 2
    const x = (ev.clientX - r.left - padX) / scale
    const y = (ev.clientY - r.top - padY) / scale
    if (!Number.isFinite(x) || !Number.isFinite(y) ||
        x < 0 || y < 0 || x > v.videoWidth || y > v.videoHeight) {
      disarm('that click was outside the picture, nothing recorded')
      return
    }

    const slot = armedSlot
    disarm('saving...')
    try {
      await api(`${boutPath(boutId)}/reanchor`, postJSON({
        time_s: +v.currentTime.toFixed(3), slot,
        x: Math.round(x), y: Math.round(y),
      }))
      setAnchorMsg(`recorded for Fencer ${slot + 1} at ${v.currentTime.toFixed(2)}s. `
                   + 'Takes effect only when the pipeline is rerun.')
      await reload()
    } catch (e) {
      setAnchorMsg(e.message)
    }
  }, [armedSlot, boutId, reload])

  // --- keyboard review ---------------------------------------------------
  //
  // The effort claim depends on a decision costing one keystroke, so the review
  // loop is jump, watch, judge, without touching the mouse.

  useEffect(() => {
    const onKey = async (e) => {
      const t = e.target
      // Guard the type as well as the selector: the event target is not always
      // an Element, and calling matches() on one that is not would throw and
      // silently kill every shortcut.
      if (t && typeof t.matches === 'function'
          && t.matches('input,select,textarea')) return

      if (e.key === 'Escape' && armedSlot !== null) { disarm('cancelled'); return }

      const v = videoRef.current
      const cur = rows[selIdx]
      const k = e.key.toLowerCase()

      if (k === 'l' && selIdx < rows.length - 1) select(selIdx + 1)
      else if (k === 'j' && selIdx > 0) select(selIdx - 1)
      else if (k === 'k' && v) { v.paused ? v.play() : v.pause() }
      else if (k === 'c' && cur?.kind === 'proposed') await decide(cur.id, 'confirmed', true)
      else if (k === 'x' && cur?.kind === 'proposed') await decide(cur.id, 'rejected', true)
      else if (k === 'a' && v) await addTouchAtPlayhead()
      // Frame stepping. Finding the peak of a lunge needs single-frame control:
      // at 30 fps a lunge lasts about ten frames, so scrubbing by the second is
      // useless for it and the label would land on whatever frame the mouse hit.
      else if ((k === ',' || k === '.') && v) {
        v.pause()
        v.currentTime = Math.max(0, v.currentTime + (k === '.' ? FRAME_S : -FRAME_S))
      }
      else if ((k === '1' || k === '2') && v) {
        await guard(async () => {
          await api(`${boutPath(boutId)}/lunges`, postJSON({
            time_s: +v.currentTime.toFixed(3), slot: k === '1' ? 0 : 1,
          }))
          await reload()
        })
      }
      else return
      e.preventDefault()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [rows, selIdx, armedSlot, boutId, decide, select, addTouchAtPlayhead,
      reload, guard])

  // --- exports -----------------------------------------------------------

  const exportTouches = async () => {
    setExportOut({ text: 'exporting...' })
    try {
      const r = await api(`${boutPath(boutId)}/export-touches`, { method: 'POST' })
      setExportOut({ result: r })
    } catch (e) { setExportOut({ error: e.message }) }
  }

  // Applying tracking corrections used to end in a command line for the user to
  // go and type. Action 4 changes tracking rather than interpretation, so it can
  // only take effect on a reprocess, and until the job runner existed there was
  // nowhere for that reprocess to happen but a terminal.
  const reprocess = async () => {
    setAnchorOut({ text: 'starting...' })
    try {
      const r = await api(`${boutPath(boutId)}/reprocess`, { method: 'POST' })
      setAnchorOut({ reprocess: r })
      await api(`${boutPath(boutId)}/reanchors/applied`, { method: 'POST' })
      await reload()
    } catch (e) { setAnchorOut({ error: e.message }) }
  }

  // Read the scoring lamps and propose who scored each confirmed touch. Which
  // fencer the green lamp belongs to has to come from the user: nothing in the
  // image says it, and guessing would be wrong half the time in a way that looks
  // authoritative.
  const proposeScorers = async (greenIs) => {
    setScorers({ loading: true })
    try {
      const r = await api(`${boutPath(boutId)}/propose-scorers`,
                          postJSON({ green_is: greenIs }))
      setScorers(r)
    } catch (e) {
      setScorers({ error: e.message })
    }
  }

  const exportAnchors = async () => {
    setAnchorOut({ text: 'exporting...' })
    try {
      const r = await api(`${boutPath(boutId)}/export-reanchors`, { method: 'POST' })
      setAnchorOut({ result: r })
    } catch (e) { setAnchorOut({ error: e.message }) }
  }

  // --- render ------------------------------------------------------------

  if (listError && !data) return <main className="single"><div className="panel err">{listError}</div></main>
  if (!boutId) {
    return (
      <main className="single">
        <div className="panel">
          <h2>Review</h2>
          <div className="mini">
            No processed bouts yet. Upload one on the other tab.
          </div>
        </div>
      </main>
    )
  }

  const duration = data?.duration_s || 0
  const p = data?.progress
  const reviewedPct = p?.proposed ? Math.round((100 * p.reviewed) / p.proposed) : 0
  const confirmedTimes = rows
    .filter((r) => r.state === 'confirmed' || r.kind === 'added')
    .map((r) => r.at)

  return (
    <main>
      <div>
        <div className="panel">
          <div className="row" style={{ marginBottom: 10 }}>
            <select value={boutId} onChange={(e) => setBoutId(e.target.value)}>
              {bouts.map((b) => {
                const tags = []
                if (!b.has_touches) tags.push('no touches')
                if (!b.has_video) tags.push('no video')
                return (
                  <option key={b.bout_id} value={b.bout_id}>
                    {b.label || b.bout_id}
                    {tags.length ? `  (${tags.join(', ')})` : ''}
                  </option>
                )
              })}
            </select>
            <span className="mini">{currentTime.toFixed(1)}s</span>
            <span className="spacer" style={{ flex: 1 }} />
            <span className="mini">
              {p ? `${p.reviewed}/${p.proposed} reviewed${p.complete ? ' - complete' : ''}` : ''}
            </span>
          </div>
          {listError && (
            <div className="note err" onClick={() => setListError(null)}>
              {listError} <span className="mini">(click to dismiss)</span>
            </div>
          )}

          {data?.has_video ? (
            <video ref={videoRef} controls preload="metadata"
                   src={`${boutPath(boutId)}/video`}
                   onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                   onClick={placeAnchor} />
          ) : (
            <div className="novid">
              No annotated video for this bout.<br />
              <span className="mini">Run the pipeline to produce one.</span>
            </div>
          )}

          <Timeline duration={duration} rows={rows}
                    segments={data?.unreliable_segments}
                    selIdx={selIdx} currentTime={currentTime}
                    onSelect={select}
                    onSeek={(t) => { if (videoRef.current) videoRef.current.currentTime = t }} />

          <div className="mini" style={{ marginTop: 8 }}>
            <b>j</b> / <b>l</b> previous and next, <b>k</b> play or pause,
            {' '}<b>c</b> confirm, <b>x</b> reject, <b>a</b> add a touch here,
            {' '}<b>,</b> / <b>.</b> step one frame, <b>1</b> / <b>2</b> label a lunge.
          </div>
        </div>

        <div className="panel">
          <h2>Proposed touches</h2>
          <TouchTable rows={rows} selIdx={selIdx} onSelect={select}
                      onDecide={decide} onDelete={removeAdded}
                      scorerProposals={scorers?.proposals} />

          <div className="row" style={{ marginTop: 9 }}>
            <span className="mini">Read the lamps, green belongs to:</span>
            <button onClick={() => proposeScorers('left')}>Fencer 1</button>
            <button onClick={() => proposeScorers('right')}>Fencer 2</button>
          </div>
          {scorers?.loading && <div className="mini">reading the lamps...</div>}
          {scorers?.error && <div className="note err">{scorers.error}</div>}
          {scorers?.proposals && (
            <div className="note">
              Proposed a scorer for <b>{scorers.decided}</b> of{' '}
              {scorers.total} confirmed touches, shown in the table as
              suggestions. {scorers.basis}.
              <br /><br />{scorers.note}
            </div>
          )}
          <form className="row" style={{ marginTop: 10 }} onSubmit={addTouchByTime}>
            <input name="time" type="number" step="0.1" placeholder="time (s)" required
                   style={{ width: 110 }} />
            <select name="scorer" defaultValue="unknown">
              <option value="left">left</option>
              <option value="right">right</option>
              <option value="double">double</option>
              <option value="unknown">unknown</option>
            </select>
            <button type="submit">Add a missed touch</button>
          </form>
        </div>

        <div className="panel">
          <h2>Tracking corrections</h2>
          <div className="row">
            <button onClick={() => armAnchor(0)}>Re-anchor Fencer 1</button>
            <button onClick={() => armAnchor(1)}>Re-anchor Fencer 2</button>
            <button className="primary"
                    disabled={!data?.reanchors?.length}
                    onClick={reprocess}>
              Re-run with corrections
              {data?.reanchors?.length ? ` (${data.reanchors.length})` : ''}
            </button>
            <button onClick={exportAnchors}>Export to a file</button>
          </div>
          {outcomes?.exists && (
            <div className="note">
              Of the {outcomes.total} correction(s) applied to this bout,
              {' '}<b>{outcomes.applied}</b> changed the tracker's assignment.
              {outcomes.applied < outcomes.total && (
                <> The rest were overruled by the ordinary matching, which is
                   this action's designed behaviour: nothing is force-assigned,
                   so a correction the matcher disagrees with has no effect.
                   {outcomes.outcomes.filter(o => o.outcome !== 'applied')
                     .map(o => (
                       <div key={`${o.frame}-${o.slot}`} className="mini">
                         {o.time_s.toFixed(2)}s, Fencer {o.slot + 1}: {o.outcome}
                       </div>
                     ))}
                </>
              )}
            </div>
          )}
          {anchorMsg && <div className="note">{anchorMsg}</div>}
          {anchorOut?.error && <div className="note err">{anchorOut.error}</div>}
          {anchorOut?.text && <div className="mini">{anchorOut.text}</div>}
          {anchorOut?.reprocess && (
            <div className="note">
              Re-running with <b>{anchorOut.reprocess.corrections}</b>{' '}
              correction(s){anchorOut.reprocess.piste_config
                ? `, carrying over ${anchorOut.reprocess.piste_config}` : ''}.
              {' '}Watch it on the upload tab. The result arrives as a separate
              bout, so you can open this one beside it and see whether the
              correction helped.
            </div>
          )}
          {anchorOut?.result && (
            <div className="note">
              Wrote <b>{anchorOut.result.corrections}</b> correction(s) to{' '}
              <code>{anchorOut.result.path}</code>. Rerun the pipeline with them:
              <br /><code>{anchorOut.result.next}</code>
              <br /><br />
              <button onClick={markApplied}>I have rerun it</button>
            </div>
          )}

          <form className="row" style={{ marginTop: 11 }} onSubmit={addSegment}>
            <input name="start" type="number" step="0.1" placeholder="from (s)" required
                   style={{ width: 100 }} />
            <input name="end" type="number" step="0.1" placeholder="to (s)" required
                   style={{ width: 100 }} />
            <button type="submit">Exclude range</button>
          </form>
          {data?.unreliable_segments?.map((s) => (
            <div className="stat" key={s.id}>
              <span>{s.start_s.toFixed(1)}-{s.end_s.toFixed(1)}s excluded</span>
              <button onClick={() => guard(async () => {
                await api(`${boutPath(boutId)}/segments/${s.id}`, del)
                await reload()
              })}>remove</button>
            </div>
          ))}
        </div>

        <div className="panel">
          <h2>Generated summary</h2>
          <SummaryPanel boutId={boutId} />
        </div>

        <LungePanel boutId={boutId} lunges={data?.lunges || []}
                    touchTimes={confirmedTimes} onChanged={reload} />
      </div>

      <div>
        <div className="panel">
          <h2>Review progress</h2>
          <div className="bar"><div style={{ width: `${reviewedPct}%` }} /></div>
          {p && (
            <div className="mini" style={{ marginTop: 7 }}>
              <div className="stat"><span>confirmed</span><b>{p.confirmed}</b></div>
              <div className="stat"><span>rejected</span><b>{p.rejected}</b></div>
              <div className="stat"><span>added by hand</span><b>{p.user_added}</b></div>
              <div className="stat"><span>excluded ranges</span><b>{p.unreliable_segments}</b></div>
              <div className="stat"><span>re-anchors pending</span><b>{p.pending_reanchors}</b></div>
            </div>
          )}
          <button style={{ marginTop: 9, width: '100%' }} onClick={exportTouches}>
            Export confirmed touches
          </button>
          {exportOut?.text && <div className="mini">{exportOut.text}</div>}
          {exportOut?.error && <div className="note err">{exportOut.error}</div>}
          {exportOut?.result && (
            <div className="note">
              Wrote <b>{exportOut.result.touches}</b> touches to{' '}
              <code>{exportOut.result.path}</code>
              {!exportOut.result.review_complete && (
                <><br /><b>Review is not finished</b>, so pending proposals are
                  missing and the count is a lower bound.</>
              )}
              <br /><br />Regenerate the summary from it:
              <br /><code>{exportOut.result.next}</code>
            </div>
          )}
        </div>

        <div className="panel">
          <h2>Metrics</h2>
          {metricsError
            ? <div className="note">{metricsError}</div>
            : <MetricsPanel metrics={metrics} />}
        </div>

        {error && <div className="panel err">{error}</div>}
      </div>
    </main>
  )
}
