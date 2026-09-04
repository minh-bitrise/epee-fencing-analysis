import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ScorePanel from './ScorePanel.jsx'

const score = (over = {}) => ({
  available: true,
  final: { fencer_1: 9, fencer_2: 9 },
  lead_changes: 3,
  time_leading_s: { 1: 87.0, 2: 38.0 },
  time_leading_pct: { 1: 69.6, 2: 30.4 },
  level_s: 55.0,
  unattributed: 0,
  timeline: [],
  ...over,
})

const zones = () => ({
  available: true, zones: ['their own third', 'the middle', 'the far third'],
  fencer_1: [2, 3, 0], fencer_2: [2, 3, 0],
})

describe('ScorePanel', () => {
  it('shows how long each fencer was ahead, not just the final score', () => {
    // A 9-9 that one fencer led for 70 per cent of and a 9-9 that was level
    // throughout are the same scoreline and different bouts.
    render(<ScorePanel score={score()} zones={zones()} />)
    expect(screen.getByText('70%')).toBeInTheDocument()
    expect(screen.getByText('30%')).toBeInTheDocument()
  })

  it('shows lead changes', () => {
    // Scoped to the stat row: "3" also appears in the zone counts, and a bare
    // text match would pass on the wrong element.
    const { container } = render(<ScorePanel score={score()} zones={zones()} />)
    const row = [...container.querySelectorAll('.stat')]
      .find((el) => el.textContent.startsWith('lead changes'))
    expect(row.querySelector('b').textContent).toBe('3')
  })

  it('flags touches with no scorer instead of scoring them as nil', () => {
    // An unattributed touch is an unknown event, not a nil-nil one, and the
    // scoreline beside it is incomplete by exactly that many touches.
    render(<ScorePanel score={score({ unattributed: 4 })} zones={zones()} />)
    expect(screen.getByText(/not counted/)).toBeInTheDocument()
    expect(screen.getByText('4')).toBeInTheDocument()
  })

  it('says the zones are measured from each fencer own end', () => {
    // Otherwise "far third" means opposite ends for the two rows.
    render(<ScorePanel score={score()} zones={zones()} />)
    expect(screen.getByText(/own end/)).toBeInTheDocument()
  })

  it('omits the zone table when nothing could be placed', () => {
    render(<ScorePanel score={score()}
                       zones={{ available: true, zones: [], fencer_1: [0, 0, 0],
                                fencer_2: [0, 0, 0] }} />)
    expect(screen.queryByText(/own end/)).toBeNull()
  })

  it('renders with no zones at all, since a swapped bout has none', () => {
    // The scoreline survives the tracking refusal; the zones do not, because
    // they need positions.
    render(<ScorePanel score={score()} />)
    expect(screen.getByText('9', { selector: '.s1' })).toBeInTheDocument()
  })

  it('explains itself rather than rendering an empty scoreline', () => {
    render(<ScorePanel score={{ available: false,
                                reason: 'no confirmed touches to build a scoreline from' }} />)
    expect(screen.getByText(/no confirmed touches/)).toBeInTheDocument()
  })
})
