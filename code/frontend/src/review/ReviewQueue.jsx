import { useCallback, useEffect, useRef, useState } from 'react'
import { formatTime } from './time.js'

/* Walks the user through pending proposals one at a time, seeking the video to each and taking
   a single keystroke as the answer. */

const pad = (n) => String(Math.floor(n)).padStart(2, '0')

export const formatElapsed = (s) =>
  `${pad(s / 60)}:${pad(s % 60)}`

export default function ReviewQueue({ items, kind, onDecide, onExit, onSeek,
                                      startedAt }) {
  const [idx, setIdx] = useState(0)
  const [answered, setAnswered] = useState(0)
  const [skipped, setSkipped] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [now, setNow] = useState(() => Date.now())
  const begun = useRef(startedAt || Date.now())

  const total = items.length
  const item = items[idx]

  // A ticking clock rather than an elapsed time computed once at the end,
  // because a user who cannot see the timer running has no way to tell whether
  // the measurement includes the ten minutes they spent making coffee.
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])

  const elapsed = Math.max(0, (now - begun.current) / 1000)

  // Seek whenever the cursor lands on an item, including the first.
  useEffect(() => {
    if (item && onSeek) onSeek(item.time_s)
  }, [item, onSeek])

  const advance = useCallback((wasAnswer) => {
    if (wasAnswer) setAnswered((n) => n + 1)
    else setSkipped((n) => n + 1)
    setIdx((i) => i + 1)
  }, [])

  const answer = useCallback(async (decision) => {
    if (busy || !item) return
    setBusy(true)
    setError(null)
    try {
      await onDecide(item, decision)
      advance(true)
    } catch (e) {
      // Left on the same item deliberately.
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }, [busy, item, onDecide, advance])

  useEffect(() => {
    const onKey = (e) => {
      const t = e.target
      if (t && typeof t.matches === 'function'
          && t.matches('input,select,textarea')) return
      const k = e.key.toLowerCase()

      if (k === 'escape') { onExit({ answered, skipped, elapsed }); return }
      if (k === 's') { advance(false); e.preventDefault(); return }
      if (!item) return

      if (kind === 'touch') {
        if (k === 'c') answer('confirmed')
        else if (k === 'x') answer('rejected')
        else return
      } else {
        // A lunge proposal answers two questions at once: is it a lunge, and whose.
        if (k === '1') answer(0)
        else if (k === '2') answer(1)
        else if (k === 'x') answer('rejected')
        else return
      }
      e.preventDefault()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [item, kind, answer, advance, onExit, answered, skipped, elapsed])

  if (!total) {
    return (
      <div className="queue">
        <div className="note">Nothing pending to review here.</div>
        <button onClick={() => onExit({ answered: 0, skipped: 0, elapsed: 0 })}>
          Close
        </button>
      </div>
    )
  }

  if (!item) {
    const rate = answered ? elapsed / answered : null
    return (
      <div className="queue done">
        <h3>Queue finished</h3>
        <div className="queue-figures">
          <div><b>{answered}</b><span>decided</span></div>
          <div><b>{skipped}</b><span>skipped</span></div>
          <div><b>{formatElapsed(elapsed)}</b><span>elapsed</span></div>
          <div><b>{rate ? rate.toFixed(1) + 's' : '-'}</b>
               <span>per decision</span></div>
        </div>
        <div className="note">
          Skipped items are excluded from the rate. They are not decisions, and
          counting them would improve the figure the more questions went
          unanswered.
        </div>
        <button className="primary"
                onClick={() => onExit({ answered, skipped, elapsed })}>
          Done
        </button>
      </div>
    )
  }

  return (
    <div className="queue">
      <div className="queue-head">
        <b>{idx + 1}</b> of {total}
        <span className="queue-clock">{formatElapsed(elapsed)}</span>
      </div>
      <div className="queue-bar">
        <i style={{ width: `${(idx / total) * 100}%` }} />
      </div>

      <div className="queue-item">
        <span className="queue-time">{formatTime(item.time_s)}</span>
        {item.detail && <span className="queue-detail">{item.detail}</span>}
      </div>

      {error && <div className="note err">{error}</div>}

      <div className="row queue-actions">
        {kind === 'touch' ? (
          <>
            <button className="primary" disabled={busy}
                    onClick={() => answer('confirmed')}>Touch (c)</button>
            <button disabled={busy}
                    onClick={() => answer('rejected')}>Not a touch (x)</button>
          </>
        ) : (
          <>
            <button className="primary" disabled={busy}
                    onClick={() => answer(0)}>Fencer 1 (1)</button>
            <button className="primary" disabled={busy}
                    onClick={() => answer(1)}>Fencer 2 (2)</button>
            <button disabled={busy}
                    onClick={() => answer('rejected')}>Not a lunge (x)</button>
          </>
        )}
        <button disabled={busy} onClick={() => advance(false)}>Skip (s)</button>
        <button onClick={() => onExit({ answered, skipped, elapsed })}>
          Stop (esc)
        </button>
      </div>

      <div className="mini">
        The video is held at each proposal, so the only thing left to do is
        judge it. {answered} decided, {skipped} skipped.
      </div>
    </div>
  )
}
