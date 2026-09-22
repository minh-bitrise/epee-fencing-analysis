import { useEffect, useState } from 'react'
import { api } from '../api.js'
import Disclosure from './Disclosure.jsx'
import ScorePanel from './ScorePanel.jsx'

/* How the two fencers in one bout compare, on six axes, drawn as a radar. */

const R = 78
const CX = 108
const CY = 96
// Parity sits at 50, so a fencer who did none of something still has a visible
// point rather than one buried in the centre where it cannot be read.
const FLOOR = 8

const point = (i, n, score) => {
  const a = (Math.PI * 2 * i) / n - Math.PI / 2
  const r = FLOOR + (R - FLOOR) * ((score ?? 50) / 100)
  return [CX + r * Math.cos(a), CY + r * Math.sin(a)]
}

const path = (axes, side) =>
  axes.map((a, i) => point(i, axes.length, a[side].score).join(',')).join(' ')

// A share and a distance are both two-decimal numbers, so the unit is what stops "58.30" and
// "1.67" being read on the same scale.
const DECIMALS = { m: 2, '%': 1, '/min': 1 }

const fmt = (v, unit) => {
  if (v == null) return 'not measured'
  const dp = DECIMALS[unit]
  const n = dp == null ? String(Math.round(v)) : v.toFixed(dp)
  return unit ? `${n}${unit === '%' ? '' : ' '}${unit}` : n
}

/* `scoreOnly` renders the scoreline without the radar. */
export default function ProfilePanel({ boutId, boutPath, refreshKey,
                                       scoreOnly = false }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let live = true
    setData(null)
    setError(null)
    api(`${boutPath}/profile`)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e.message))
    return () => { live = false }
  }, [boutId, boutPath, refreshKey])

  if (error) return <div className="note err">{error}</div>
  if (!data) return <div className="mini">Loading.</div>

  if (scoreOnly) return <ScorePanel score={data.score} zones={data.zones}
                                    pace={data.pace} />

  if (!data.available) {
    return (
      <>
        <div className="note">
          <b>No profile for this bout.</b> {data.reason}
          {data.left_share_pct != null && (
            <> Fencer 1 was on the left in {data.left_share_pct} per cent of
               frames; on footage where the tracking held it is 100.</>
          )}
        </div>
        {/* The scoreline comes from confirmed touches, not from tracking, so it
            is still valid on a bout whose per-fencer axes are not. */}
        <ScorePanel score={data.score} pace={data.pace} />
      </>
    )
  }

  const axes = data.axes
  const n = axes.length

  return (
    <div className="profile">
      {/* The viewBox carries margin on every side rather than relying on
          overflow, because the axis labels sit OUTSIDE the outer ring and the
          side column clips them at its edge. Measured: "Ground used" ran off a
          348 px panel. */}
      <svg viewBox="-38 -8 292 212" role="img"
           aria-label="how the two fencers compare on six measures">
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <polygon key={f} className="rings"
            points={axes.map((_, i) => point(i, n, f * 100).join(',')).join(' ')}
            style={{ opacity: f === 0.5 ? 0.55 : 0.22 }} />
        ))}
        {/* An unmeasured axis is drawn at parity, because there is nowhere
            honest to put it. That makes it indistinguishable from a genuine tie
            in the SHAPE, so the spoke itself carries the distinction: dashed
            means nothing has been measured on this one yet. Without it the
            radar reads as complete and the caveat lives only in the table. */}
        {axes.map((a, i) => {
          const [x, y] = point(i, n, 100)
          const absent = a.fencer_1.score == null && a.fencer_2.score == null
          return <line key={i} x1={CX} y1={CY} x2={x} y2={y}
                       className={absent ? 'spoke absent' : 'spoke'} />
        })}

        <polygon className="f1" points={path(axes, 'fencer_1')} />
        <polygon className="f2" points={path(axes, 'fencer_2')} />

        {axes.map((a, i) => {
          const [x, y] = point(i, n, 118)
          const absent = a.fencer_1.score == null && a.fencer_2.score == null
          return (
            <text key={a.key} x={x} y={y}
                  className={absent ? 'axis-label absent' : 'axis-label'}
                  textAnchor={x < CX - 4 ? 'end' : x > CX + 4 ? 'start' : 'middle'}
                  dominantBaseline="middle">{a.label}</text>
          )
        })}
      </svg>

      <div className="profile-key">
        <span><i className="sw f1" /> Fencer 1</span>
        <span><i className="sw f2" /> Fencer 2</span>
      </div>

      <table className="profile-table">
        <tbody>
          {axes.map((a) => (
            <tr key={a.key}
                className={a.fencer_1.score == null ? 'absent' : undefined}>
              <th title={a.explains}>{a.label}</th>
              <td>{fmt(a.fencer_1.value, a.unit)}</td>
              <td>{fmt(a.fencer_2.value, a.unit)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {data.confirmed_touches === 0 ? (
        <div className="note">
          <b>No touches confirmed yet</b>, so three axes are undrawn.
        </div>
      ) : (
        <div className="mini">
          Resting on {data.confirmed_touches} confirmed touch
          {data.confirmed_touches === 1 ? '' : 'es'}.
        </div>
      )}
      <Disclosure label="How to read this">
        {data.note} Scoped to {data.scoping}.
      </Disclosure>
    </div>
  )
}
