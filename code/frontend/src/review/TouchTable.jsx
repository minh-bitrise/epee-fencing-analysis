import { useEffect, useRef } from 'react'
import { formatTime } from './time.js'

// Buttons, in the order the two fencers appear on screen.
const SIDES = [['left', 'F1', 'Fencer 1 scored'],
               ['right', 'F2', 'Fencer 2 scored'],
               ['double', 'both', 'both scored, a double']]

/**
 * The proposals, with what each one rests on.
 *
 * Confidence and the separation that triggered the proposal are shown beside
 * every row because the user is being asked to adjudicate the system's guess,
 * and a guess presented without its basis is just an assertion. A low-confidence
 * proposal next to a large separation is a different thing to judge than a
 * high-confidence one, and the user cannot tell them apart otherwise.
 *
 * WHY ROWS AND NOT A TABLE. As a six-column table every cell had equal weight,
 * so fourteen proposals were fourteen identical bands of small text and finding
 * the one under review meant reading. Each proposal is now one row with a
 * hierarchy inside it: the timestamp first because that is what identifies it,
 * the state as a coloured pill because that is what the user is changing, and
 * the evidence in smaller muted text because it is consulted rather than
 * scanned. The columns that carried "conf" and "sep" headings are gone: three
 * words of heading for a number that is self-describing once labelled inline.
 */
export default function TouchTable({ rows, selIdx, onSelect, onDecide, onDelete,
                                     scorerProposals, onScorer }) {
  // Proposed scorers, keyed by the time they belong to. Matched on time rather
  // than id because the proposal is made against the CONFIRMED touch list, which
  // merges detector proposals and hand-added touches under different ids.
  const proposedFor = (at) => scorerProposals?.find(
    (p) => Math.abs(p.time_s - at) < 0.3)
  const selRef = useRef(null)

  // Keep the cursor visible during a keyboard walk. Without this the selection
  // moves off the bottom of the panel after a few decisions and the user is
  // reviewing a row they cannot see.
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
        const scorer = added ? r.scorer : r.scorer ?? proposedFor(r.at)

        return (
          <div key={`${r.kind}-${r.id}`} ref={ref}
               className={`touch ${state}${sel ? ' sel' : ''}`}
               onClick={() => onSelect(i)}>
            <span className="t">{formatTime(r.at)}</span>

            <span className={`pill ${state}`}>{state}</span>

            {/* WHO SCORED, AND HOW IT GETS RECORDED. The lamp reading was
                previously shown and then stranded: a proposal appeared in
                italics and there was nothing to press, so a touch could never
                be attributed at all and the three scoring figures on the
                profile could never be populated from the interface. The
                backend had accepted a scorer on a decision all along; the
                client simply never sent one.

                The suggestion stays visibly a suggestion. Pressing it accepts
                it, which is the same one-press economy as confirming a touch,
                and the three buttons are there for when it is wrong or absent. */}
            <span className="who">
              {added ? r.scorer : state !== 'confirmed'
                ? <span className="none">-</span>
                : (
                  <>
                    {SIDES.map(([value, label, hint]) => (
                      <button key={value} type="button"
                              className={`sc${r.scorer === value ? ' on' : ''}`}
                              title={r.scorer === value ? `${hint}. Press to clear.` : hint}
                              onClick={() => onScorer(r.id, r.scorer === value ? null : value)}>
                        {label}
                      </button>
                    ))}
                    {!r.scorer && scorer && scorer.proposed !== 'unknown' && (
                      <button type="button" className="suggest"
                              title={`green delta ${scorer.green_delta}, `
                                     + `red delta ${scorer.red_delta}. `
                                     + 'Press to accept.'}
                              onClick={() => onScorer(r.id, scorer.proposed)}>
                        {scorer.proposed}?
                      </button>
                    )}
                  </>
                )}
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
