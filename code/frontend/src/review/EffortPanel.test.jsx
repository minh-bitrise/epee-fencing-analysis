import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import EffortPanel from './EffortPanel.jsx'

/* The effort comparison. Its whole risk is that a ratio computed from one pass in each mode
   gets quoted as a result, so the tests are about the caveat being as visible as the number. */

const mode = (runs, decisions, rate) => ({
  runs, total_decisions: decisions, mean_seconds_per_decision: rate,
})

describe('EffortPanel', () => {
  it('says nothing has been timed rather than showing zeroes', () => {
    render(<EffortPanel sessions={{
      assisted: mode(0, 0, null), manual: mode(0, 0, null),
      speedup: null, strength: 'no comparison yet: nothing recorded in assisted or manual mode',
    }} />)
    expect(screen.getAllByText('no runs yet')).toHaveLength(2)
    expect(screen.getByText(/no comparison yet/)).toBeInTheDocument()
  })

  it('shows the rate per decision for each mode', () => {
    render(<EffortPanel sessions={{
      assisted: mode(1, 14, 4.2), manual: mode(1, 12, 19.8),
      speedup: 4.71, strength: 'one run in each mode: an illustration, not a measurement',
    }} />)
    expect(screen.getByText('4.2s')).toBeInTheDocument()
    expect(screen.getByText('19.8s')).toBeInTheDocument()
  })

  it('labels a one-run-each ratio as an illustration, beside the ratio', () => {
    /* The test that matters. A speed-up from a single pass in each mode is an anecdote about
       one afternoon, and a number in a box reads as a result unless it is told not to. */
    render(<EffortPanel sessions={{
      assisted: mode(1, 14, 4.2), manual: mode(1, 12, 19.8),
      speedup: 4.71, strength: 'one run in each mode: an illustration, not a measurement',
    }} />)
    expect(screen.getByText('4.71x')).toBeInTheDocument()
    expect(screen.getByText(/an illustration, not a measurement/))
      .toBeInTheDocument()
  })

  it('explains why the rate is per decision and not per bout', () => {
    // The two modes do not produce the same number of decisions, so a per-bout
    // figure would compare different amounts of work.
    render(<EffortPanel sessions={{
      assisted: mode(2, 28, 4.2), manual: mode(1, 12, 19.8),
      speedup: 4.71, strength: '2 assisted and 1 manual runs',
    }} />)
    expect(screen.getByText(/per decision rather than per bout/))
      .toBeInTheDocument()
  })

  it('survives a response missing the shape it expects', () => {
    // It sits in the same column as the review table. An unexpected payload used
    // to take the whole review screen down with it.
    render(<EffortPanel sessions={{ ok: true }} />)
    expect(screen.getByText(/No timing recorded/)).toBeInTheDocument()
  })

  it('shows a loading state rather than an empty panel', () => {
    render(<EffortPanel sessions={null} />)
    expect(screen.getByText('Loading.')).toBeInTheDocument()
  })
})
