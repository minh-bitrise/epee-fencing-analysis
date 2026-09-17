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
 * WHY PACE IS HERE AND NOT ON THE RADAR. Touches per minute is derived from
 * the same confirmed list, so it belongs on the same side of that line. It is
 * deliberately not a radar axis: both fencers are divided by the same bout
 * duration, so scoring a rate against the other fencer would reproduce the
 * scoring-share axis exactly, and a radar with the same number twice reads as
 * better evidenced than it is. As an absolute figure it says what share cannot,
 * which is whether this was a bout of fourteen touches or of three.
 *
 * WHY LEAD CHANGES AND TIME LEADING. A 5-4 bout that one fencer led throughout
 * and a 5-4 bout that changed hands four times are the same scoreline and
 * different bouts, and the second is the one worth talking about. The final
 * score cannot tell them apart and these two numbers can.
 */
export default function ScorePanel({ score, zones, pace }) {
  if (!score) return null
  if (!score.available) {
    return <div className="mini">{score.reason}.</div>
  }

  const t = score.time_leading_pct || {}
  const a = t[1] ?? 0
  const b = t[2] ?? 0
  const level = 100 - a - b
  const total = zones?.available
    ? zones.fencer_1.reduce((x, y) => x + y, 0)
      + zones.fencer_2.reduce((x, y) => x + y, 0)
    : 0

  return (
    <>
      {/* A scoreline of 0-0 over touches nobody has attributed is not a
          measured draw, and printing it large with the explanation three lines
          below is printing the wrong thing first. score_progression has carried
          a comment since it was written saying an unattributed touch "is not a
          zero-zero event, it is an unknown one"; this is where that stopped
          being true on the screen. A partial attribution still shows: 2-1 with
          one touch unattributed is a real scoreline as far as it goes. */}
      {score.final.fencer_1 === 0 && score.final.fencer_2 === 0
       && score.unattributed > 0
        ? <div className="scoreline unknown">
            <span className="dash">score not known</span>
          </div>
        : <div className="scoreline">
            <span className="s1">{score.final.fencer_1}</span>
            <span className="dash">-</span>
            <span className="s2">{score.final.fencer_2}</span>
          </div>}

      {/* Time ahead as one bar rather than two rows reading "F1 ahead" and
          "F2 ahead". The quantity is a split of a whole, so a split of a whole
          is what it should look like; as a pair of percentages the reader has to
          reconstruct that the two plus the level time make the bout. */}
      <div className="split" title="share of the bout each fencer led">
        <i className="a" style={{ width: `${a}%` }} />
        <i className="lv" style={{ width: `${level}%` }} />
        <i className="b" style={{ width: `${b}%` }} />
      </div>
      <div className="split-key">
        <span><i className="sw f1" />ahead {a.toFixed(0)}%</span>
        <span className="mid">level {score.level_s.toFixed(0)}s</span>
        <span>{b.toFixed(0)}% ahead<i className="sw f2" /></span>
      </div>

      <div className="stat"><span>lead changes</span>
        <b>{score.lead_changes}</b></div>

      {pace?.available && (
        <>
          <div className="stat"><span>touches per minute</span>
            <b>{pace.per_min.toFixed(2)}</b></div>
          <div className="mini">
            {/* A null split is not a zero one. With no scorer recorded anywhere
                the per-fencer rates are unknown, and printing 0.00 twice would
                read as two fencers who scored nothing. */}
            {pace.fencer_1_per_min == null || pace.fencer_2_per_min == null
              ? 'No scorer recorded on any touch, so the split between the two '
                + 'fencers is not known.'
              : <>
                  {pace.fencer_1_per_min.toFixed(2)} and{' '}
                  {pace.fencer_2_per_min.toFixed(2)} per fencer, doubles counted
                  half to each.
                  {pace.unattributed > 0 && ' The two do not sum to the bout rate'
                    + ' because a touch with no scorer is counted in neither.'}
                </>}
          </div>
        </>
      )}

      {score.unattributed > 0 && (
        <div className="note">
          <b>{score.unattributed}</b> touch
          {score.unattributed === 1 ? '' : 'es'} with no scorer, not counted.
        </div>
      )}

      {zones?.available && total > 0 && (
        <>
          <h3 className="sub">Where they scored</h3>
          <table className="compare">
            <thead>
              <tr><th /><th>own third</th><th>middle</th><th>far third</th></tr>
            </thead>
            <tbody>
              <tr><th><i className="sw f1" />Fencer 1</th>
                {zones.fencer_1.map((n, i) => <td key={i}>{n}</td>)}</tr>
              <tr><th><i className="sw f2" />Fencer 2</th>
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
