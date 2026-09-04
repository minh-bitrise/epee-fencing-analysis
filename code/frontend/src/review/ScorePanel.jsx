import Disclosure from './Disclosure.jsx'

/**
 * The scoreline: how it moved, and where on the strip the touches were scored.
 *
 * WHY THIS IS SEPARATE FROM THE PROFILE. Everything here is derived from the
 * touches the user confirmed, and nothing from tracking. That means it survives
 * the swap check that withholds every per-fencer axis: a bout the tracker could
 * not follow still has a score, and suppressing it would be withholding
 * something the system did not get wrong.
 *
 * WHY LEAD CHANGES AND TIME LEADING. A 5-4 bout that one fencer led throughout
 * and a 5-4 bout that changed hands four times are the same scoreline and
 * different bouts, and the second is the one worth talking about. The final
 * score cannot tell them apart and these two numbers can.
 */
const pct = (v) => (v == null ? '-' : `${v.toFixed(0)}%`)

export default function ScorePanel({ score, zones }) {
  if (!score) return null
  if (!score.available) {
    return <div className="mini">{score.reason}.</div>
  }

  const t = score.time_leading_pct
  const total = zones?.available
    ? zones.fencer_1.reduce((a, b) => a + b, 0)
      + zones.fencer_2.reduce((a, b) => a + b, 0)
    : 0

  return (
    <>
      <div className="scoreline">
        <span className="s1">{score.final.fencer_1}</span>
        <span className="dash">-</span>
        <span className="s2">{score.final.fencer_2}</span>
      </div>

      <div className="stat"><span>lead changes</span>
        <b>{score.lead_changes}</b></div>
      <div className="stat"><span>F1 ahead</span><b>{pct(t?.[1])}</b></div>
      <div className="stat"><span>F2 ahead</span><b>{pct(t?.[2])}</b></div>
      <div className="stat"><span>level</span>
        <b>{score.level_s.toFixed(0)}s</b></div>

      {score.unattributed > 0 && (
        <div className="note">
          <b>{score.unattributed}</b> touch
          {score.unattributed === 1 ? '' : 'es'} with no scorer, not counted.
        </div>
      )}

      {zones?.available && total > 0 && (
        <>
          <h3 className="sub">Where they scored</h3>
          <table className="zones">
            <thead>
              <tr><th /><th>own third</th><th>middle</th><th>far third</th></tr>
            </thead>
            <tbody>
              <tr><th>Fencer 1</th>
                {zones.fencer_1.map((n, i) => <td key={i}>{n}</td>)}</tr>
              <tr><th>Fencer 2</th>
                {zones.fencer_2.map((n, i) => <td key={i}>{n}</td>)}</tr>
            </tbody>
          </table>
          <Disclosure label="How are these measured?">
            Measured from each fencer's own end, so "far third" means the same
            for both. Thirds rather than five zones: the metre scale is derived
            per clip, and finer bands would be narrower than the measurement
            behind them.
          </Disclosure>
        </>
      )}
    </>
  )
}
