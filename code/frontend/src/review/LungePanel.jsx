import { useState } from 'react'
import { formatTime } from './time.js'
import { api, boutPath, del } from '../api.js'

// Scored or missed is derived, not asked for. A lunge that scored is one whose
// peak sits shortly before an awarded touch. One second is generous on purpose:
// the question here is which lunges to compare against, not precise attribution.
const SCORED_WINDOW_S = 1.0

export default function LungePanel({ boutId, lunges, touchTimes, onChanged,
                                    onQueue }) {
  const [proposed, setProposed] = useState(null)

  const askForProposals = async (slot) => {
    setProposed({ loading: true })
    try {
      const r = await api(`${boutPath(boutId)}/propose-lunges?slot=${slot}`,
                          { method: 'POST' })
      setProposed(r)
    } catch (e) {
      setProposed({ error: e.message })
    }
  }
  const scored = (l) => touchTimes.some(
    (t) => t >= l.time_s - 0.2 && t <= l.time_s + SCORED_WINDOW_S)

  // With no confirmed touches there is nothing to derive from, and every lunge
  // would be reported as "missed". That is not the same as unknown, and showing
  // it would be a quiet falsehood of exactly the kind this project keeps finding
  // in its own metrics, so say so instead.
  const canDerive = touchTimes.length > 0
  const label = (l) => (!canDerive ? 'scored?' : scored(l) ? 'scored' : 'missed')

  const n = lunges.length
  const nScored = lunges.filter(scored).length

  // No panel wrapper and no heading: this now sits inside a titled section in
  // the Tools tab, and a box inside a box was most of what made the interface
  // look like a stack of announcements.
  return (
    <>
      <div className="mini">
        Pause on maximum extension and press <code>1</code> or <code>2</code>;
        {' '}<code>,</code> and <code>.</code> step a frame.
      </div>
      <div className="mini">
        {n === 0 ? 'none yet' : canDerive ? (
          <><b>{n}</b> labelled - {nScored} scored, {n - nScored} missed</>
        ) : (
          <><b>{n}</b> labelled - scored or missed unknown, because this bout has
            no confirmed touches to compare against</>
        )}
      </div>
      <div className="row" style={{ marginTop: 9 }}>
        <span className="mini">Propose more for:</span>
        <button onClick={() => askForProposals(0)}>Fencer 1</button>
        <button onClick={() => askForProposals(1)}>Fencer 2</button>
      </div>
      {proposed?.loading && <div className="mini">calibrating...</div>}
      {proposed?.error && <div className="note err">{proposed.error}</div>}
      {proposed?.proposals && (
        <div className="note">
          Calibrated on {proposed.calibrated_on} of your labels for Fencer
          {' '}{proposed.slot + 1}, threshold {proposed.threshold}.
          {' '}<b>{proposed.proposals.length}</b> more to look at:
          <div className="mini" style={{ marginTop: 5 }}>
            {proposed.proposals.slice(0, 12).map((p) => (
              <span key={p.time_s} style={{ marginRight: 9 }}>
                {formatTime(p.time_s)} ({p.margin}x)
              </span>
            ))}
            {proposed.proposals.length > 12
              && `... and ${proposed.proposals.length - 12} more`}
          </div>
          {onQueue && proposed.proposals.length > 0 && (
            <div className="row" style={{ marginTop: 9 }}>
              <button className="primary"
                      onClick={() => onQueue(proposed.slot, proposed.proposals)}>
                Go through them one at a time
              </button>
            </div>
          )}
          <br />{proposed.note}
        </div>
      )}
      {lunges.map((l) => (
        <div className="stat" key={l.id}>
          <span>{formatTime(l.time_s)} - F{l.slot + 1} {label(l)}</span>
          <button onClick={async () => {
            // Guarded here rather than left to bubble: this is a delete, and a
            // silent failure looks exactly like a successful one until the list
            // fails to change.
            try {
              await api(`${boutPath(boutId)}/lunges/${l.id}`, del)
              await onChanged()
            } catch (e) {
              setProposed({ error: e.message })
            }
          }}>remove</button>
        </div>
      ))}
    </>
  )
}
