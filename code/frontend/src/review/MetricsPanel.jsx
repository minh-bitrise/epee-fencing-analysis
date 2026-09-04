// Signed metres, so the sign has to be shown: +1.2 m toward the opponent and
// -1.2 m away are opposite tactical readings and "1.2 m" is neither.
const signedM = (v) => (v > 0 ? '+' : '') + v.toFixed(1) + ' m'

// Net displacement below this magnitude means the fencer finished where they
// started, and the sign is not meaningful there. Fencer 2 on the club clip
// measures +0.43 m from raw positions and -0.94 m from smoothed endpoints: the
// two readings agree on the substance and disagree on the sign, so printing one
// of them with a sign would invent a direction the data does not support.
const NET_SIGN_FLOOR_M = 1.0

// Both movement figures get the floor treatment, because both are sums of the
// same per-frame signed motion and so share the instability at small magnitudes.
const floored = (v) =>
  Math.abs(v) < NET_SIGN_FLOOR_M ? 'no net change' : signedM(v)

const closing = (v) => (v != null ? v.toFixed(1) + '%' : '-')

function Stat({ label, value }) {
  return <div className="stat"><span>{label}</span><b>{value}</b></div>
}

/**
 * The bout's numbers.
 *
 * Movement is shown as net displacement and closing share, never as cumulative
 * push and pull totals. Path length sums the size of every frame's movement, so
 * tracking noise accumulates into it and never cancels: re-measuring the same
 * footage under median smoothing windows from 1 to 121 frames moved the totals
 * from 161 m to 34 m with no asymptote, while the net figure settled. The API
 * does not return the totals at all, and this does not display them, because a
 * number on screen beside a real measurement reads as one.
 */
export default function MetricsPanel({ metrics }) {
  if (!metrics) return <div className="mini">Loading.</div>
  const w = metrics.whole_recording
  const i = metrics.in_play

  const fencer = (label, whole, scoped) => (
    <div key={label}>
      <Stat label={`${label} net displacement (whole)`}
            value={floored(whole.net_displacement_m)} />
      <Stat label={`${label} forward movement (in play)`}
            value={floored(scoped.net_forward_movement_m)} />
      <Stat label={`${label} closing share (in play)`}
            value={closing(scoped.closing_share_pct)} />
    </div>
  )

  return (
    <>
      <Stat label="confirmed touches" value={metrics.confirmed_touches.length} />
      <Stat label="in play" value={`${Math.round(100 * i.in_play_fraction)}%`} />
      <Stat label="mean distance (in play)"
            value={i.mean_distance_m != null ? `${i.mean_distance_m.toFixed(2)} m` : '-'} />
      <Stat label="mean distance (whole)" value={`${w.distance_m.mean.toFixed(2)} m`} />
      {fencer('F1', w.fencer_1, i.fencer_1)}
      {fencer('F2', w.fencer_2, i.fencer_2)}

      <div className="note">
        Scoping basis: {metrics.scoping_basis}
        <br />Movement derived from: {metrics.movement_basis.source}
        <br /><br />
        Net displacement is where a fencer finished relative to where they
        started, positive toward the opponent, and it is the reliable figure.
        Forward movement in play sums the same motion over playing time only, so
        it is not a displacement and can be larger: ground gained in a phrase is
        given back walking to the guard line. Closing share is the proportion of
        moving frames spent reducing the distance, accurate to a few points.
        Cumulative push and pull totals are not shown because they measure the
        smoothing, not the fencer.
        {w.data_quality_warnings && (
          <><br /><br /><b>Warning.</b> {w.data_quality_warnings.join(' ')}</>
        )}
      </div>
    </>
  )
}
