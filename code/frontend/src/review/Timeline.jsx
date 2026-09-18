import { useEffect, useRef } from 'react'
import { formatClock, formatTime } from './time.js'

/**
 * The whole bout on one strip: every proposed touch as a marker, every excluded
 * range shaded, and the playhead.
 *
 * This is what makes review navigable. Without it, finding the next proposal
 * means scrubbing a three-minute video by hand, and the claim that assisted
 * review costs less effort than manual review would not survive that: the
 * scrubbing is most of the manual cost.
 */
export default function Timeline({ duration, rows, segments, selIdx,
                                   currentTime, onSelect, onSeek }) {
  const ref = useRef(null)

  // The playhead is moved by writing to the DOM node rather than by re-rendering
  // with the current time in state. `timeupdate` fires several times a second,
  // and re-rendering the whole timeline at that rate to move one element by a
  // few pixels makes the marker list flicker on every tick.
  const cursorRef = useRef(null)
  useEffect(() => {
    if (cursorRef.current && duration) {
      cursorRef.current.style.left = `${(100 * currentTime) / duration}%`
    }
  }, [currentTime, duration])

  const clickAt = (e) => {
    if (!duration) return
    const r = ref.current.getBoundingClientRect()
    if (r.width <= 0) return
    onSeek((duration * (e.clientX - r.left)) / r.width)
  }

  const step = duration > 150 ? 30 : 15
  const ticks = []
  for (let s = 0; s <= duration; s += step) ticks.push(s)

  return (
    <>
      <div className="tl" ref={ref} onClick={clickAt}>
        {(segments || []).map((s) => (
          <div key={s.id} className="seg" style={{
            left: `${(100 * s.start_s) / duration}%`,
            width: `${(100 * (s.end_s - s.start_s)) / duration}%`,
          }} />
        ))}
        {rows.map((r, i) => (
          <div key={`${r.kind}-${r.id}`}
               className={`mk ${r.kind === 'added' ? 'added' : r.state}`}
               style={{ left: `${(100 * r.at) / duration}%` }}
               title={`${formatTime(r.at)} ${r.state}`}
               onClick={(e) => { e.stopPropagation(); onSelect(i) }} />
        ))}
        <div className="cur" ref={cursorRef} />
        {/* Nested inside the strip, matching the ported stylesheet's `.tl .ax`
            rule, which positions the scale against the strip rather than the
            page. */}
        <div className="ax">
          {ticks.map((s) => (
            <span key={s} style={{ left: `${(100 * s) / duration}%` }}>{formatClock(s)}</span>
          ))}
        </div>
      </div>
    </>
  )
}
