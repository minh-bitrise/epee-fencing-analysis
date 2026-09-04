import Disclosure from './Disclosure.jsx'

/**
 * How long reviewing this bout took, assisted against manual.
 *
 * WHY THIS PANEL EXISTS. The project's central claim is that confirming the
 * system's proposals costs less effort than labelling from scratch, and until
 * now nothing in it measured effort at all. A user study would measure it
 * better and is parked; this needs no participants and produces the figure as a
 * by-product of doing the work.
 *
 * WHY IT REFUSES TO SHOW A RATIO FROM ONE RUN EACH. It will show one, because
 * withholding it would be worse, but it labels it. A speed-up computed from a
 * single pass in each mode is an anecdote about one afternoon, and a number in a
 * box reads as a result unless it is told not to.
 */
const secs = (v) => (v == null ? '-' : `${v.toFixed(1)}s`)

function Mode({ label, stats }) {
  return (
    <div className="effort-mode">
      <h4>{label}</h4>
      {stats.runs === 0 ? (
        <div className="mini">no runs yet</div>
      ) : (
        <>
          <div className="effort-big">{secs(stats.mean_seconds_per_decision)}</div>
          <div className="mini">
            per decision, over {stats.total_decisions} decision
            {stats.total_decisions === 1 ? '' : 's'} in {stats.runs} run
            {stats.runs === 1 ? '' : 's'}
          </div>
        </>
      )}
    </div>
  )
}

// An empty mode rather than a crash. This panel is fed by a network response,
// and it sits in the same column as the review table: a response missing the
// shape it expects used to take the whole review screen down, which is the same
// failure the video play() guard exists for. A missing comparison is worth far
// less than the ability to keep reviewing.
const EMPTY = { runs: 0, total_decisions: 0, mean_seconds_per_decision: null }

export default function EffortPanel({ sessions }) {
  if (!sessions) return <div className="mini">Loading.</div>
  if (!sessions.assisted || !sessions.manual) {
    return <div className="mini">No timing recorded for this bout yet.</div>
  }

  return (
    <>
      <div className="effort">
        <Mode label="Assisted" stats={sessions.assisted || EMPTY} />
        <Mode label="Manual" stats={sessions.manual || EMPTY} />
      </div>

      {sessions.speedup != null && (
        <div className="effort-verdict">
          <b>{sessions.speedup}x</b> faster assisted
        </div>
      )}

      <div className="mini">{sessions.strength}.</div>
      <Disclosure label="What counts as a decision">
        One answered proposal in assisted mode, one entry created in manual
        mode. The two modes do not produce the same number of them, which is
        why the rate is per decision rather than per bout. Skipped items are
        excluded: they are not decisions, and counting them would improve the
        figure the more questions went unanswered.
      </Disclosure>
    </>
  )
}
