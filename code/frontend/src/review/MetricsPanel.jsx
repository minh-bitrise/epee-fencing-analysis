import Disclosure from './Disclosure.jsx'

// Signed metres, so the sign has to be shown: +1.2 m toward the opponent and
// -1.2 m away are opposite tactical readings and "1.2 m" is neither.
const signedM = (v) => (v > 0 ? '+' : '') + v.toFixed(1) + ' m'

// Net displacement below this magnitude means the fencer finished where they started, and the
// sign is not meaningful there.
const NET_SIGN_FLOOR_M = 1.0

// Both movement figures get the floor treatment, because both are sums of the
// same per-frame signed motion and so share the instability at small magnitudes.
const floored = (v) =>
  Math.abs(v) < NET_SIGN_FLOOR_M ? 'no net change' : signedM(v)

const closing = (v) => (v != null ? v.toFixed(1) + '%' : '-')

function Tile({ value, label }) {
  return <div className="tile"><b>{value}</b><span>{label}</span></div>
}

/* The bout's numbers. WHY THE PER-FENCER FIGURES ARE A TABLE AND NOT A LIST. */
export default function MetricsPanel({ metrics }) {
  if (!metrics) return <div className="mini">Loading.</div>
  const w = metrics.whole_recording
  const i = metrics.in_play

  const rows = [
    ['net displacement', floored(w.fencer_1.net_displacement_m),
                         floored(w.fencer_2.net_displacement_m)],
    ['forward movement', floored(i.fencer_1.net_forward_movement_m),
                         floored(i.fencer_2.net_forward_movement_m)],
    ['closing share', closing(i.fencer_1.closing_share_pct),
                      closing(i.fencer_2.closing_share_pct)],
  ]

  return (
    <>
      {/* The three figures that describe the bout rather than either fencer, as
          tiles. They were the first three lines of a nine-line list, where they
          read as three more of the same. */}
      <div className="tiles">
        <Tile value={metrics.confirmed_touches.length} label="touches" />
        <Tile value={`${Math.round(100 * i.in_play_fraction)}%`} label="in play" />
        <Tile value={i.mean_distance_m != null
                       ? `${i.mean_distance_m.toFixed(2)} m` : '-'}
              label="mean gap" />
      </div>

      <table className="compare">
        <thead>
          <tr>
            <th />
            <th><i className="sw f1" />Fencer 1</th>
            <th><i className="sw f2" />Fencer 2</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, a, b]) => (
            <tr key={label}>
              <th>{label}</th><td>{a}</td><td>{b}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* A warning is printed; an explanation is folded. The two used to share
          one grey block, which meant a real data-quality problem arrived at the
          bottom of a paragraph about what "net displacement" means. */}
      {w.data_quality_warnings && (
        <div className="note err">
          <b>Warning.</b> {w.data_quality_warnings.join(' ')}
        </div>
      )}

      <Disclosure label="What these mean">
        Net displacement is where a fencer finished relative to where they
        started, positive toward the opponent, and it is the reliable figure.
        Forward movement in play sums the same motion over playing time only, so
        it is not a displacement and can be larger: ground gained in a phrase is
        given back walking to the guard line. Closing share is the proportion of
        moving frames spent reducing the distance, accurate to a few points.
        Cumulative push and pull totals are not shown because they measure the
        smoothing, not the fencer.
        <div className="mini" style={{ marginTop: 7 }}>
          Scoped to {metrics.scoping_basis}.
          {' '}Movement from {metrics.movement_basis.source}.
          {' '}Mean gap over the whole recording is
          {' '}{w.distance_m.mean.toFixed(2)} m.
        </div>
      </Disclosure>
    </>
  )
}
