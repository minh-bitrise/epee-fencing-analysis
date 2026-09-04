import { useState } from 'react'

/**
 * The side column as tabs rather than a stack of panels.
 *
 * WHY. Everything this application knows about a bout was previously shown at
 * once, in six bordered panels of equal visual weight: progress, metrics,
 * effort, profile, summary, and the tracking tools. Nothing was more important
 * than anything else, so the eye had nowhere to start, and the one thing the
 * user is actually doing, deciding about the proposal on screen, competed with
 * five things they are not.
 *
 * Tabs are the cheap fix and the right one here: the groups are genuinely
 * separate questions, asked at different times. What is in this bout, what were
 * the two fencers like, what did the model write, and the corrective tools that
 * are used rarely and never during a review pass.
 */
export default function Tabs({ tabs, initial }) {
  const names = Object.keys(tabs)
  const [active, setActive] = useState(initial || names[0])
  const Body = tabs[active]

  return (
    <div className="tabs">
      <div className="tabstrip" role="tablist">
        {names.map((n) => (
          <button key={n} role="tab" aria-selected={n === active}
                  className={n === active ? 'active' : undefined}
                  onClick={() => setActive(n)}>{n}</button>
        ))}
      </div>
      {/* Rendered rather than hidden, so an inactive tab does no work and holds
          no stale scroll position. The panels are cheap to mount; the fetches
          they depend on live above this component. */}
      <div className="tabbody" role="tabpanel">{typeof Body === 'function'
        ? <Body /> : Body}</div>
    </div>
  )
}
