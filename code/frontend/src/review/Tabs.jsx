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
  const body = tabs[active]

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
          no stale scroll position.

          THE FUNCTION IS CALLED, NOT MOUNTED AS A COMPONENT. Writing <Body />
          here made the active tab a new component TYPE on every render of the
          parent, because the caller builds its `tabs` object inline and those
          arrow functions have a fresh identity each time. React cannot know two
          different function objects are the same component, so it unmounted and
          remounted the entire tab body on every parent render.

          The parent re-renders on every `timeupdate` from the video, about four
          times a second while playing, and each remount re-ran the panels'
          effects. A single bout left playing issued about 150 requests for its
          profile. Calling the function instead keeps the elements it returns,
          whose types ARE stable, so the panels below stay mounted and fetch
          once. These bodies are render functions rather than components and
          must not use hooks. */}
      <div className="tabbody" role="tabpanel">
        {typeof body === 'function' ? body() : body}
      </div>
    </div>
  )
}
