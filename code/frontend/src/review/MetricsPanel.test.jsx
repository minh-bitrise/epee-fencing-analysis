import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import MetricsPanel from './MetricsPanel.jsx'

/**
 * The metrics panel is where this project's hard-won caveats live, and every one
 * of them is a rule about what NOT to show. Those rules were each paid for by a
 * measurement that went wrong, so they are asserted here rather than trusted to
 * survive the next edit.
 */

const metrics = (over = {}) => ({
  confirmed_touches: [{ time_s: 10 }],
  whole_recording: {
    distance_m: { mean: 2.5 },
    fencer_1: { net_displacement_m: 3.43 },
    fencer_2: { net_displacement_m: 0.43 },
    movement_basis: { source: 'raw per-frame positions' },
    ...over.whole_recording,
  },
  in_play: {
    in_play_fraction: 0.57,
    mean_distance_m: 2.2,
    fencer_1: { net_forward_movement_m: 7.6, closing_share_pct: 52.4 },
    fencer_2: { net_forward_movement_m: -0.2, closing_share_pct: 48.0 },
    ...over.in_play,
  },
  scoping_basis: 'confirmed touches',
  movement_basis: { source: 'raw per-frame positions' },
  ...over,
})

describe('MetricsPanel', () => {
  it('shows a direction only when the magnitude supports one', () => {
    // Fencer 2 measures +0.43 m from raw positions and -0.94 m from smoothed
    // endpoints: the two agree on the substance and disagree on the sign, so
    // printing either with a sign invents a direction the data cannot support.
    render(<MetricsPanel metrics={metrics()} />)
    expect(screen.getByText('+3.4 m')).toBeInTheDocument()
    expect(screen.getAllByText('no net change').length).toBeGreaterThan(0)
  })

  it('never displays cumulative push and pull totals', () => {
    // Re-measuring the same footage under smoothing windows from 1 to 121 frames
    // moved these from 161 m to 34 m with no asymptote. A number on screen beside
    // a real measurement reads as one, so they are absent by design.
    const { container } = render(<MetricsPanel metrics={metrics()} />)
    const text = container.textContent.toLowerCase()
    expect(text).not.toContain('advance')
    expect(text).not.toContain('retreat')
    expect(text).not.toContain('total distance')
  })

  it('says which derivation the movement figures came from', () => {
    // The position route and the differenced route disagreed by 3.5 m of net
    // displacement on the club clip, so a figure without its provenance is not
    // interpretable.
    render(<MetricsPanel metrics={metrics()} />)
    expect(screen.getByText(/raw per-frame positions/)).toBeInTheDocument()
  })

  it('says what the in-play figures were scoped by', () => {
    render(<MetricsPanel metrics={metrics()} />)
    // Matched on the full phrase: "confirmed touches" alone also appears as the
    // label of the touch-count row above.
    expect(screen.getByText(/Scoping basis: confirmed touches/))
      .toBeInTheDocument()
  })

  it('surfaces a data-quality warning rather than burying it', () => {
    const m = metrics({
      whole_recording: {
        distance_m: { mean: 2.5 },
        fencer_1: { net_displacement_m: 21.9 },
        fencer_2: { net_displacement_m: 0.4 },
        movement_basis: { source: 'raw per-frame positions' },
        data_quality_warnings: ['net_displacement_m is implausible here'],
      },
    })
    render(<MetricsPanel metrics={m} />)
    expect(screen.getByText(/implausible/)).toBeInTheDocument()
  })

  it('shows a dash rather than a zero when a distance is unavailable', () => {
    // A missing measurement and a measurement of zero are different claims, and
    // this project has already shipped one metric that conflated them.
    const m = metrics({ in_play: {
      in_play_fraction: 0,
      mean_distance_m: null,
      fencer_1: { net_forward_movement_m: 0, closing_share_pct: null },
      fencer_2: { net_forward_movement_m: 0, closing_share_pct: null },
    } })
    const { container } = render(<MetricsPanel metrics={m} />)
    expect(container.textContent).toContain('-')
  })

  it('renders nothing rather than crashing before metrics arrive', () => {
    render(<MetricsPanel metrics={null} />)
    expect(screen.getByText(/Loading/)).toBeInTheDocument()
  })
})
