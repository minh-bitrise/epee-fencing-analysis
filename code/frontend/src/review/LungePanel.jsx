import { api, boutPath, del } from '../api.js'

// Scored or missed is derived, not asked for. A lunge that scored is one whose
// peak sits shortly before an awarded touch. One second is generous on purpose:
// the question here is which lunges to compare against, not precise attribution.
const SCORED_WINDOW_S = 1.0

export default function LungePanel({ boutId, lunges, touchTimes, onChanged }) {
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

  return (
    <div className="panel">
      <h2>Lunge labels</h2>
      <div className="note">
        For evaluating the pose model, not for correcting it. Pause on the frame
        of maximum extension and press <code>1</code> or <code>2</code>; use
        {' '}<code>,</code> and <code>.</code> to step a frame at a time. Whether
        a lunge scored is worked out from the touch labels, so it does not need
        marking. Around 30 is enough to settle whether stance carries a usable
        signal.
      </div>
      <div className="mini">
        {n === 0 ? 'none yet' : canDerive ? (
          <><b>{n}</b> labelled - {nScored} scored, {n - nScored} missed</>
        ) : (
          <><b>{n}</b> labelled - scored or missed unknown, because this bout has
            no confirmed touches to compare against</>
        )}
      </div>
      {lunges.map((l) => (
        <div className="stat" key={l.id}>
          <span>{l.time_s.toFixed(2)}s - F{l.slot + 1} {label(l)}</span>
          <button onClick={async () => {
            await api(`${boutPath(boutId)}/lunges/${l.id}`, del)
            await onChanged()
          }}>remove</button>
        </div>
      ))}
    </div>
  )
}
