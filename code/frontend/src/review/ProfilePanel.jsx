import { useEffect, useState } from 'react'
import { api } from '../api.js'
import ScorePanel from './ScorePanel.jsx'

/**
 * How the two fencers in one bout compare, on six axes, drawn as a radar.
 *
 * WHY THE AXES COMPARE THE PAIR AND NOT A POPULATION. A radar needs a scale, and
 * the honest one does not exist here: eight fencer-bouts from one club and one
 * broadcast is not a population, and normalising against it would invent a
 * typical epeeist. Each axis is therefore this fencer's share of the pair's
 * total, where the midpoint is parity. The measured value sits beside every
 * axis so the shape is never the only thing on offer.
 *
 * WHY AN AXIS CAN BE ABSENT. A fencer with no confirmed lunges has not been
 * measured lunging rarely, they have not been measured at all. Drawing that as
 * zero would collapse a spoke and read as a finding, so an axis with no data is
 * drawn at parity in grey and named in the legend as not yet measured.
 *
 * WHY THE WHOLE PANEL CAN REFUSE. Every axis is per-fencer and so assumes the
 * tracker kept the two apart. On the broadcast clip it did not, swapping them 14
 * times, and a radar drawn from that describes the tracker while looking exactly
 * as convincing as a real one.
 */

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

// A share and a distance are both two-decimal numbers, so the unit is what
// stops "58.30" and "1.67" being read on the same scale.
// Precision follows the unit rather than the value. Metres are read to the
// centimetre because that is the scale the fencers move at; a share to one
// decimal, because "58.30%" claims a precision the fourteen touches behind it
// do not have.
const DECIMALS = { m: 2, '%': 1, '/min': 1 }

const fmt = (v, unit) => {
  if (v == null) return 'not measured'
  const dp = DECIMALS[unit]
  const n = dp == null ? String(Math.round(v)) : v.toFixed(dp)
  return unit ? `${n}${unit === '%' ? '' : ' '}${unit}` : n
}

export default function ProfilePanel({ boutId, boutPath, refreshKey }) {
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
        <ScorePanel score={data.score} />
      </>
    )
  }

  const axes = data.axes
  const n = axes.length

  return (
    <div className="profile">
      <svg viewBox="0 0 216 192" role="img"
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

      <ScorePanel score={data.score} zones={data.zones} />

      <div className="note">
        {data.note}
        {' '}Scoped to {data.scoping}.
        {data.confirmed_touches === 0 ? (
          <> <b>No touches confirmed yet</b>, so the scoring, range and run axes
             are undrawn. Confirm some touches and they fill in.</>
        ) : (
          <> Resting on {data.confirmed_touches} confirmed touch
             {data.confirmed_touches === 1 ? '' : 'es'}.</>
        )}
      </div>
    </div>
  )
}
