import Disclosure from './Disclosure.jsx'

/* How long reviewing this bout took, assisted against manual. WHY THIS PANEL EXISTS. */
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

// An empty mode rather than a crash.
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
