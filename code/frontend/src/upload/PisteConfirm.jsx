import { useEffect, useRef, useState } from 'react'
import { api, postJSON } from '../api.js'

/**
 * Show the measured piste region over a frame of the video, and let the user
 * agree with it, drag its edges, or drop it.
 *
 * WHY THE USER CONFIRMS RATHER THAN DRAWS. The obvious design is an empty frame
 * and a drawing tool. This project has already established that it does not
 * work: a polygon placed by eye on the broadcast clip admitted the adjacent
 * piste and raised the count of physically impossible distance readings from 53
 * to 252, while the headline coverage figure went up, so it looked like an
 * improvement. The regions that work were measured, by sampling frames and
 * finding where the fencers' feet actually fall. The server does that
 * measurement, and this shows the result with the numbers behind it.
 *
 * WHY THE USER IS ASKED AT ALL. The measurement rests on an assumption that can
 * fail: that the two people standing closest together at the same depth are the
 * fencers. On footage with a referee standing on the strip, or a second bout in
 * shot, it can be wrong, and a person spots that instantly. Confirming takes
 * seconds. Discovering it afterwards costs the whole processing run.
 *
 * WHY ONLY THE TOP AND BOTTOM EDGES MOVE. The region the pipeline needs is a
 * horizontal band: the strip runs across the frame, and every measured polygon
 * spans the full width. Offering arbitrary vertices would offer precision the
 * measurement does not have and invite exactly the freehand placement this
 * exists to avoid.
 */
export default function PisteConfirm({ job, onDecided }) {
  const measured = job.piste || {}
  const [top, setTop] = useState(measured.polygon?.[0]?.[1] ?? 0)
  const [bottom, setBottom] = useState(measured.polygon?.[2]?.[1] ?? 0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(null)
  const canvasRef = useRef(null)
  const frameRef = useRef(null)

  const [frameW, frameH] = measured.frame_size || [
    job.video_info?.width || 1, job.video_info?.height || 1]

  // Redraw whenever an edge moves. The canvas is sized in source pixels and
  // stretched by CSS to match the image, so every coordinate here is in the same
  // space as the polygon the pipeline will receive and nothing has to be scaled.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.width = frameW
    canvas.height = frameH
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, frameW, frameH)

    // Everything outside the band is dimmed rather than the band being outlined.
    // The question the user is answering is which people get ignored, so showing
    // the excluded area directly answers it.
    ctx.fillStyle = 'rgba(10, 12, 16, 0.62)'
    ctx.fillRect(0, 0, frameW, top)
    ctx.fillRect(0, bottom, frameW, frameH - bottom)

    ctx.strokeStyle = '#4da3ff'
    ctx.lineWidth = Math.max(2, frameH / 220)
    ctx.beginPath()
    ctx.moveTo(0, top); ctx.lineTo(frameW, top)
    ctx.moveTo(0, bottom); ctx.lineTo(frameW, bottom)
    ctx.stroke()
  }, [top, bottom, frameW, frameH])

  const yFromEvent = (e) => {
    const rect = canvasRef.current.getBoundingClientRect()
    if (rect.height <= 0) return null
    const y = ((e.clientY - rect.top) / rect.height) * frameH
    return Math.max(0, Math.min(frameH, y))
  }

  const onPointerDown = (e) => {
    const y = yFromEvent(e)
    if (y == null) return
    // Grab whichever edge is nearer, so there is nothing to aim at precisely.
    setDragging(Math.abs(y - top) <= Math.abs(y - bottom) ? 'top' : 'bottom')
    e.currentTarget.setPointerCapture(e.pointerId)
  }

  const onPointerMove = (e) => {
    if (!dragging) return
    const y = yFromEvent(e)
    if (y == null) return
    // The edges are not allowed to cross. A band of negative height would be
    // accepted by the polygon check and would reject every detection in the
    // frame, producing an empty result that looks like a tracking failure.
    if (dragging === 'top') setTop(Math.min(y, bottom - 10))
    else setBottom(Math.max(y, top + 10))
  }

  const decide = async (body) => {
    setBusy(true)
    setError(null)
    try {
      await api(`/api/jobs/${job.job_id}/piste`, postJSON(body))
      await onDecided()
    } catch (e) {
      setError(e.message)
      setBusy(false)
    }
  }

  const moved = Math.abs(top - (measured.polygon?.[0]?.[1] ?? 0)) > 1 ||
                Math.abs(bottom - (measured.polygon?.[2]?.[1] ?? 0)) > 1

  const m = measured.measurements || {}

  return (
    <div style={{ marginTop: 9 }}>
      <div className="piste-frame" ref={frameRef}>
        <img src={`/api/jobs/${job.job_id}/frame`} alt="a frame from the bout" />
        <canvas ref={canvasRef}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={() => setDragging(null)}
                onPointerCancel={() => setDragging(null)} />
      </div>

      <div className="note">
        The shaded areas will be ignored. People detected there, typically the
        referee and the crowd, cannot be mistaken for a fencer.
        {' '}Drag either edge to adjust.
        {m.fencer_samples != null && (
          <>
            <br /><br />
            Measured from {m.frames_sampled} sampled frames:
            {' '}{m.fencer_samples} fencer detections, with feet between rows
            {' '}{m.fencer_feet_y_min} and {m.fencer_feet_y_max}.
            {m.other_groups?.length > 0 && (
              <> A separate group of {m.other_groups[0].samples} detections sits
                 at rows {m.other_groups[0].feet_y[0]} to
                 {' '}{m.other_groups[0].feet_y[1]} and has been excluded.</>
            )}
          </>
        )}
        {measured.confident === false && (
          <><br /><br /><b>Check this one.</b> {measured.reason}</>
        )}
      </div>

      {error && <div className="note err">{error}</div>}

      <div className="row" style={{ marginTop: 9 }}>
        <button className="primary" disabled={busy}
                onClick={() => decide(moved
                  ? { polygon: [[0, top], [frameW, top],
                                [frameW, bottom], [0, bottom]] }
                  : {})}>
          {moved ? 'Use my adjusted region' : 'Looks right, process it'}
        </button>
        <button disabled={busy} onClick={() => decide({ skip: true })}>
          Process without a region
        </button>
      </div>
      <div className="mini" style={{ marginTop: 5, color: 'var(--muted)' }}>
        Skip it only if the two fencers are the only people on camera. With
        anyone else in shot, the tracker can lock onto them instead.
      </div>
    </div>
  )
}
