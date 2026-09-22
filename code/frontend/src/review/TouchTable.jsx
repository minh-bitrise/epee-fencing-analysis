import { useEffect, useRef } from 'react'
import { formatTime } from './time.js'

// Buttons, in the order the two fencers appear on screen.
const SIDES = [['left', 'F1', 'Fencer 1 scored'],
               ['right', 'F2', 'Fencer 2 scored'],
               ['double', 'both', 'both scored, a double']]

/* The proposals, with what each one rests on. */
export default function TouchTable({ rows, selIdx, onSelect, onDecide, onDelete,
                                     scorerProposals, onScorer }) {
  // Proposed scorers, keyed by the time they belong to.
  const proposedFor = (at) => scorerProposals?.find(
    (p) => Math.abs(p.time_s - at) < 0.3)
  const selRef = useRef(null)

  // Keep the cursor visible during a keyboard walk.
  useEffect(() => {
    selRef.current?.scrollIntoView({ block: 'nearest' })
  }, [selIdx])

  if (rows.length === 0) {
    return <div className="mini">No proposals for this bout.</div>
  }

  return (
    <div className="touches">
      {rows.map((r, i) => {
        const sel = i === selIdx
        const ref = sel ? selRef : null
        const added = r.kind === 'added'
        const state = added ? 'added' : r.state
        const lamp = added ? null : proposedFor(r.at)

        return (
          <div key={`${r.kind}-${r.id}`} ref={ref}
               className={`touch ${state}${sel ? ' sel' : ''}`}
               onClick={() => onSelect(i)}>
            <span className="t">{formatTime(r.at)}</span>

            <span className={`pill ${state}`}>{state}</span>

            {/* THE LAMP'S READING, ALONGSIDE THE STATE AND NOT INSTEAD OF THE
                ANSWER. It sits here rather than inside the who cell so that it
                stays visible after the user has decided, which is the point:
                the two side by side show where the system agreed with the
                person and where it did not, and that comparison is the
                evidence behind the attribution figures in the evaluation.
                Folding it away once answered would hide exactly the cases
                worth looking at.

                While nothing is recorded it is also the fastest way to record
                something, since pressing it accepts it. */}
            {!added && state === 'confirmed' && lamp && lamp.proposed !== 'unknown' && (
              <button type="button"
                      className={`lamp${r.scorer
                        ? (r.scorer === lamp.proposed ? ' agrees' : ' differs') : ''}`}
                      disabled={!!r.scorer}
                      title={r.scorer
                        ? (r.scorer === lamp.proposed
                            ? 'the lamp agrees with you'
                            : `the lamp read ${lamp.proposed}, you recorded ${r.scorer}`)
                        : `green delta ${lamp.green_delta}, red delta `
                          + `${lamp.red_delta}. Press to accept.`}
                      onClick={() => !r.scorer && onScorer(r.id, lamp.proposed)}>
                lamp {lamp.proposed}
              </button>
            )}

            {/* Who scored, as the user has it. Confirm and reject answer
                whether the touch happened; these answer who it belongs to, and
                until they existed a touch could not be attributed at all. */}
            <span className="who">
              {added ? r.scorer : state !== 'confirmed'
                ? <span className="none">-</span>
                : SIDES.map(([value, label, hint]) => (
                    <button key={value} type="button"
                            className={`sc${r.scorer === value ? ' on' : ''}`}
                            title={r.scorer === value ? `${hint}. Press to clear.` : hint}
                            onClick={() => onScorer(r.id, r.scorer === value ? null : value)}>
                      {label}
                    </button>
                  ))}
            </span>

            {/* What the proposal rests on. The user is adjudicating the
                system's guess, and a guess without its basis is an assertion. */}
            <span className="basis">
              {added ? 'added by hand' : (
                <>conf {r.confidence.toFixed(2)}
                  {r.separation_m != null
                    && <> &middot; sep {r.separation_m.toFixed(2)}m</>}</>
              )}
            </span>

            <span className="acts">
              {added ? (
                <button onClick={(e) => { e.stopPropagation(); onDelete(r.id) }}>
                  remove
                </button>
              ) : (
                <>
                  <button onClick={(e) => {
                    e.stopPropagation(); onDecide(r.id, 'confirmed')
                  }}>confirm</button>
                  <button onClick={(e) => {
                    e.stopPropagation(); onDecide(r.id, 'rejected')
                  }}>reject</button>
                </>
              )}
            </span>
          </div>
        )
      })}
    </div>
  )
}
