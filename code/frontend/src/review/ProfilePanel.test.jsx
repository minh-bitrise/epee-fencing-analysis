import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ProfilePanel from './ProfilePanel.jsx'

/* The profile panel. Its risk is not that it breaks, it is that it draws a confident shape out
   of data that does not support one. */

const axis = (key, label, v1, s1, v2, s2, unit = '') => ({
  key, label, unit, explains: 'why this axis exists',
  fencer_1: { value: v1, score: s1 }, fencer_2: { value: v2, score: s2 },
})

const body = (over = {}) => ({
  available: true,
  scoping: 'active fencing only, resets between touches excluded',
  duration_s: 180, frames_used: 4000, mean_distance_m: 2.53,
  confirmed_touches: 14, confirmed_lunges: { 1: 0, 2: 30 },
  axes: [
    axis('territory_m', 'Territory', 2.26, 45, 2.78, 55, 'm'),
    axis('range_used_m', 'Ground used', 0.83, 34, 1.61, 66, 'm'),
    axis('scoring_share_pct', 'Scoring', 50.0, 50, 50.0, 50, '%'),
    axis('scoring_range_m', 'Scoring range', 2.42, 60, 1.58, 40, 'm'),
    axis('lunge_rate_per_min', 'Lunges', null, null, 10.0, null, '/min'),
    axis('longest_streak', 'Best run', 2, 50, 2, 50),
  ],
  note: 'Each axis scores one fencer against the other in this bout.',
  ...over,
})

const stub = (payload, ok = true) => {
  global.fetch = vi.fn(() => Promise.resolve({
    ok, status: ok ? 200 : 400, statusText: 'Bad Request',
    json: () => Promise.resolve(ok ? payload : { detail: payload }),
  }))
}

afterEach(() => { delete global.fetch })

describe('ProfilePanel', () => {
  it('draws one shape per fencer', async () => {
    stub(body())
    const { container } = render(
      <ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="0:0" />)
    await waitFor(() =>
      expect(container.querySelector('polygon.f1')).not.toBeNull())
    expect(container.querySelector('polygon.f2')).not.toBeNull()
  })

  it('shows the measured value beside every axis, not only the shape', async () => {
    // A radar with no numbers is a picture. The values are what makes it a
    // reading, and they are the part a marker can check against the data.
    stub(body())
    render(<ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="0:0" />)
    await waitFor(() => expect(screen.getByText('2.26 m')).toBeInTheDocument())
    expect(screen.getByText('1.61 m')).toBeInTheDocument()
  })

  it('carries the unit, so a share is not read as a distance', async () => {
    // "58.30" beside "1.67" invites reading both on the same scale.
    stub(body())
    render(<ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="0:0" />)
    await waitFor(() => expect(screen.getByText('10.0 /min')).toBeInTheDocument())
    // The percentage sign closes up against the number; every other unit does
    // not. Both fencers are at parity on the scoring axis in this fixture.
    expect(screen.getAllByText('50.0%').length).toBe(2)
  })

  it('names an unmeasured axis rather than showing it as a zero', async () => {
    /* The distinction the whole panel turns on. A fencer with no confirmed lunges has not been
       measured lunging rarely; they have not been measured. */
    stub(body())
    render(<ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="0:0" />)
    await waitFor(() => expect(screen.getByText('not measured')).toBeInTheDocument())
  })

  it('marks an unmeasured spoke so the shape does not read as a tie', async () => {
    /* An axis with no data is drawn at parity, because there is nowhere honest to put it. */
    stub(body())
    const { container } = render(
      <ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="0:0" />)
    await waitFor(() =>
      expect(container.querySelectorAll('line.spoke.absent').length).toBe(1))
    // The five measured axes are not marked, including the two genuinely tied.
    expect(container.querySelectorAll('line.spoke').length).toBe(6)
  })

  it('refuses to draw anything when the tracker swapped the fencers', async () => {
    // Clip 4 swapped 14 times. A radar drawn there describes the tracker and
    // looks exactly as convincing as a real one, so there must be no radar.
    stub({
      available: false, swaps: 14, left_share_pct: 26.9,
      reason: 'the tracker exchanged the two fencers 14 times on this bout',
    })
    const { container } = render(
      <ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="0:0" />)
    await waitFor(() =>
      expect(screen.getByText(/No profile for this bout/)).toBeInTheDocument())
    expect(container.querySelector('polygon.f1')).toBeNull()
    expect(screen.getByText(/exchanged the two fencers 14 times/))
      .toBeInTheDocument()
  })

  it('says how far the tracking drifted, since the count alone is abstract', async () => {
    stub({
      available: false, swaps: 14, left_share_pct: 26.9,
      reason: 'the tracker exchanged the two fencers 14 times on this bout',
    })
    render(<ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="0:0" />)
    await waitFor(() =>
      expect(screen.getByText(/26.9 per cent of\s+frames/)).toBeInTheDocument())
  })

  it('says the scoring axes are undrawn before any touch is confirmed', async () => {
    // Otherwise three parity spokes look like three measured ties.
    stub(body({ confirmed_touches: 0 }))
    render(<ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="0:0" />)
    await waitFor(() =>
      expect(screen.getByText(/No touches confirmed yet/)).toBeInTheDocument())
  })

  it('says what the profile rests on once touches exist', async () => {
    stub(body())
    render(<ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="0:0" />)
    await waitFor(() =>
      expect(screen.getByText(/14 confirmed touches/)).toBeInTheDocument())
  })

  it('states that the comparison is within the bout, not against other fencers',
     async () => {
    stub(body())
    render(<ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="0:0" />)
    await waitFor(() =>
      expect(screen.getByText(/against the other in this bout/))
        .toBeInTheDocument())
  })

  it('surfaces a failed request instead of spinning forever', async () => {
    stub('this bout was processed before the position columns existed', false)
    render(<ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="0:0" />)
    await waitFor(() =>
      expect(screen.getByText(/position columns existed/)).toBeInTheDocument())
  })

  it('refetches when a decision changes what the axes rest on', async () => {
    stub(body())
    const { rerender } = render(
      <ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="0:0" />)
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1))
    rerender(<ProfilePanel boutId="b" boutPath="/api/bouts/b" refreshKey="1:0" />)
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2))
  })
})
