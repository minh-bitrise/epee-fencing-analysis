import { useEffect, useRef } from 'react'

/**
 * The proposals, with what each one rests on.
 *
 * Confidence and the separation that triggered the proposal are shown beside
 * every row because the user is being asked to adjudicate the system's guess,
 * and a guess presented without its basis is just an assertion. A low-confidence
 * proposal next to a large separation is a different thing to judge than a
 * high-confidence one, and the user cannot tell them apart otherwise.
 */
export default function TouchTable({ rows, selIdx, onSelect, onDecide, onDelete,
                                     scorerProposals }) {
  // Proposed scorers, keyed by the time they belong to. Matched on time rather
  // than id because the proposal is made against the CONFIRMED touch list, which
  // merges detector proposals and hand-added touches under different ids.
  const proposedFor = (at) => scorerProposals?.find(
    (p) => Math.abs(p.time_s - at) < 0.3)
  const selRef = useRef(null)

  // Keep the cursor visible during a keyboard walk. Without this the selection
  // moves off the bottom of the panel after a few decisions and the user is
  // reviewing a row they cannot see.
  useEffect(() => {
    selRef.current?.scrollIntoView({ block: 'nearest' })
  }, [selIdx])

  if (rows.length === 0) {
    return (
      <table><tbody>
        <tr><td colSpan={6} className="mini">No proposals for this bout.</td></tr>
      </tbody></table>
    )
  }

  return (
    <table>
      <thead>
        <tr>
          <th>time</th><th>conf</th><th>sep</th>
          <th>scorer</th><th>state</th><th></th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => {
          const sel = i === selIdx
          const ref = sel ? selRef : null
          if (r.kind === 'added') {
            return (
              <tr key={`added-${r.id}`} ref={ref}
                  className={`added${sel ? ' sel' : ''}`}
                  onClick={() => onSelect(i)}>
                <td>{r.at.toFixed(1)}s</td>
                <td className="mini">manual</td>
                <td>-</td>
                <td>{r.scorer}</td>
                <td>added</td>
                <td>
                  <button onClick={(e) => { e.stopPropagation(); onDelete(r.id) }}>
                    remove
                  </button>
                </td>
              </tr>
            )
          }
          return (
            <tr key={`proposed-${r.id}`} ref={ref}
                className={`${r.state}${sel ? ' sel' : ''}`}
                onClick={() => onSelect(i)}>
              <td>{r.at.toFixed(1)}s</td>
              <td>{r.confidence.toFixed(2)}</td>
              <td>{r.separation_m != null ? `${r.separation_m.toFixed(2)}m` : '-'}</td>
              <td>
                {r.scorer ?? (() => {
                  const p = proposedFor(r.at)
                  // Shown as a suggestion, visibly distinct from a decision the
                  // user has made. The lamp reading is evidence, not an answer.
                  return p && p.proposed !== 'unknown'
                    ? <span className="suggest" title={`green delta ${p.green_delta}, red delta ${p.red_delta}`}>
                        {p.proposed}?
                      </span>
                    : '-'
                })()}
              </td>
              <td>{r.state}</td>
              <td>
                <button onClick={(e) => { e.stopPropagation(); onDecide(r.id, 'confirmed') }}>
                  confirm
                </button>
                <button onClick={(e) => { e.stopPropagation(); onDecide(r.id, 'rejected') }}>
                  reject
                </button>
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
