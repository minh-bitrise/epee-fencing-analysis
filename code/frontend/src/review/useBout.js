import { useCallback, useEffect, useState } from 'react'
import { api, boutPath } from '../api.js'

/* Loads everything the review screen shows for one bout, and reloads it after any change. */
export function useBout(boutId) {
  const [data, setData] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState(null)
  const [metricsError, setMetricsError] = useState(null)
  const [loading, setLoading] = useState(false)

  const reload = useCallback(async () => {
    if (!boutId) return
    setLoading(true)
    // Fetched INDEPENDENTLY, not with Promise.all.
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

/* Proposed and hand-added touches merged into one time-ordered list. */
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
