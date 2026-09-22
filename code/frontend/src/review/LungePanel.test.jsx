import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import LungePanel from './LungePanel.jsx'

/* The lunge panel's job is to say what it does and does not know. */

const lunge = (t, slot = 1) => ({ id: `l${t}`, time_s: t, slot })

describe('LungePanel', () => {
  it('says scored or missed is unknown when there is nothing to derive it from', () => {
    // With no confirmed touches every lunge would be reported "missed", which is not the same
    // as unknown.
    render(<LungePanel boutId="b" lunges={[lunge(10)]} touchTimes={[]}
                       onChanged={vi.fn()} />)
    expect(screen.getByText(/scored or missed unknown/)).toBeInTheDocument()
  })

  it('derives scored from the confirmed touches when it can', () => {
    render(<LungePanel boutId="b" lunges={[lunge(10), lunge(50)]}
                       touchTimes={[10.4]} onChanged={vi.fn()} />)
    expect(screen.getByText(/1 scored, 1 missed/)).toBeInTheDocument()
  })

  it('counts a lunge as scored only just before the touch, not long after', () => {
    // A lunge that scored is one whose peak sits shortly BEFORE an awarded touch.
    render(<LungePanel boutId="b" lunges={[lunge(10)]} touchTimes={[14.0]}
                       onChanged={vi.fn()} />)
    expect(screen.getByText(/0 scored, 1 missed/)).toBeInTheDocument()
  })

  it('reports having none rather than rendering an empty list', () => {
    render(<LungePanel boutId="b" lunges={[]} touchTimes={[]}
                       onChanged={vi.fn()} />)
    expect(screen.getByText('none yet')).toBeInTheDocument()
  })

  it('offers to propose more for either fencer', () => {
    render(<LungePanel boutId="b" lunges={[lunge(10)]} touchTimes={[]}
                       onChanged={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Fencer 1' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Fencer 2' })).toBeInTheDocument()
  })
})

describe('lunge times read on the video clock', () => {
  it('writes a labelled lunge the way the player writes it', () => {
    // Reported from use: the player read 1:51 and this list read 111.75s, so
    // every item had to be converted in the head to be found in the video.
    render(<LungePanel boutId="b" lunges={[lunge(111.75)]} touchTimes={[]}
                       onChanged={() => {}} />)
    expect(screen.getByText(/1:51\.7/)).toBeInTheDocument()
    expect(screen.queryByText(/111\.75s/)).not.toBeInTheDocument()
  })
})
