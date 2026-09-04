import { useCallback, useEffect, useState } from 'react'
import { api, boutPath } from '../api.js'

/**
 * Loads everything the review screen shows for one bout, and reloads it after
 * any change.
 *
 * Touches and metrics are fetched together on purpose. Every annotation action
 * changes both: confirming a touch changes the review progress and also rescopes
 * every in-play metric, because the metrics are computed over the segments
 * between confirmed touches. Refreshing only the panel the user just touched
 * would leave the numbers beside it quietly describing the previous state.
 */
export function useBout(boutId) {
  const [data, setData] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState(null)
  const [metricsError, setMetricsError] = useState(null)
  const [loading, setLoading] = useState(false)

  const reload = useCallback(async () => {
    if (!boutId) return
    setLoading(true)
    // Fetched INDEPENDENTLY, not with Promise.all. Metrics can legitimately be
    // unavailable while touches are fine: a bout where the tracker never held
    // both fencers at once has no distance samples and so no statistics, which
    // happens on footage shot from behind the piste. Failing them together would
    // take the whole review interface down over a panel, leaving the user unable
    // to look at the touches or the video for a bout the system had processed
    // perfectly well. Found when the end-to-end test hit exactly that case.
    const [touches, m] = await Promise.allSettled([
      api(`${boutPath(boutId)}/touches`),
      api(`${boutPath(boutId)}/metrics`),
    ])

    if (touches.status === 'fulfilled') {
      setData(touches.value)
      setError(null)
    } else {
      setError(touches.reason.message)
    }
    setMetrics(m.status === 'fulfilled' ? m.value : null)
    setMetricsError(m.status === 'rejected' ? m.reason.message : null)
    setLoading(false)
  }, [boutId])

  useEffect(() => { reload() }, [reload])

  return { data, metrics, error, metricsError, loading, reload }
}

/**
 * Proposed and hand-added touches merged into one time-ordered list.
 *
 * The interface treats them as one sequence because the user reviews them as
 * one: the keyboard walk moves through everything on the timeline in order. They
 * keep a `kind` because the available actions differ. A proposal is confirmed or
 * rejected; a touch the user added themselves can only be removed, and offering
 * to "reject" it would be asking them to overrule themselves.
 */
export function mergeRows(data) {
  if (!data) return []
  return [
    ...data.proposed.map((p) => ({
      ...p, kind: 'proposed', at: p.corrected_time_s ?? p.time_s,
    })),
    ...data.added.map((a) => ({
      ...a, kind: 'added', at: a.time_s, state: 'added',
    })),
  ].sort((a, b) => a.at - b.at)
}
