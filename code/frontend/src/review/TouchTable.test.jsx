import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import TouchTable from './TouchTable.jsx'

/**
 * The touch table is where the user adjudicates the system's guesses, so its
 * duty is to show what each guess rests on and to keep the system's suggestions
 * visibly separate from the user's own decisions.
 */

const proposed = (over = {}) => ({
  id: 'p0', kind: 'proposed', at: 43.9, time_s: 43.9,
  confidence: 0.29, separation_m: 0.87, state: 'pending', scorer: null, ...over,
})

const added = (over = {}) => ({
  id: 'u0', kind: 'added', at: 90.0, time_s: 90.0,
  state: 'added', scorer: 'left', ...over,
})

const handlers = { onSelect: vi.fn(), onDecide: vi.fn(), onDelete: vi.fn() }

describe('TouchTable', () => {
  it('shows what a proposal rests on, not just that it exists', () => {
    // A guess presented without its basis is an assertion. A low-confidence
    // proposal beside a large separation is a different thing to judge.
    render(<TouchTable rows={[proposed()]} selIdx={0} {...handlers} />)
    expect(screen.getByText('0.29')).toBeInTheDocument()
    expect(screen.getByText('0.87m')).toBeInTheDocument()
  })

  it('offers confirm and reject for a proposal', () => {
    render(<TouchTable rows={[proposed()]} selIdx={0} {...handlers} />)
    expect(screen.getByRole('button', { name: 'confirm' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'reject' })).toBeInTheDocument()
  })

  it('offers only removal for a touch the user added themselves', () => {
    // Offering to "reject" a hand-added touch would be asking the user to
    // overrule themselves.
    render(<TouchTable rows={[added()]} selIdx={0} {...handlers} />)
    expect(screen.getByRole('button', { name: 'remove' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'confirm' })).toBeNull()
  })

  it('marks a proposed scorer as a suggestion, not a decision', () => {
    // The lamp reading is evidence. Showing it identically to a scorer the user
    // confirmed would let the system's guess be mistaken for their judgement.
    render(<TouchTable rows={[proposed()]} selIdx={0} {...handlers}
                       scorerProposals={[{ time_s: 43.9, proposed: 'left',
                                           green_delta: 12, red_delta: 400 }]} />)
    const cell = screen.getByText('left?')
    expect(cell).toBeInTheDocument()
    expect(cell.className).toContain('suggest')
  })

  it('does not attach a proposal to a different touch', () => {
    // Proposals are matched on time because they are made against the confirmed
    // list, which uses different ids. A loose match would attribute a scorer to
    // the wrong touch.
    render(<TouchTable rows={[proposed()]} selIdx={0} {...handlers}
                       scorerProposals={[{ time_s: 120.0, proposed: 'right',
                                           green_delta: 9, red_delta: 4 }]} />)
    expect(screen.queryByText('right?')).toBeNull()
  })

  it('shows nothing rather than a guess when the lamps were unreadable', () => {
    render(<TouchTable rows={[proposed()]} selIdx={0} {...handlers}
                       scorerProposals={[{ time_s: 43.9, proposed: 'unknown',
                                           green_delta: 0, red_delta: 0 }]} />)
    expect(screen.queryByText(/unknown\?/)).toBeNull()
  })

  it('says so plainly when a bout has no proposals', () => {
    render(<TouchTable rows={[]} selIdx={0} {...handlers} />)
    expect(screen.getByText(/No proposals for this bout/)).toBeInTheDocument()
  })

  it('marks the selected row so a keyboard walk is followable', () => {
    const { container } = render(
      <TouchTable rows={[proposed(), proposed({ id: 'p1', at: 60 })]}
                  selIdx={1} {...handlers} />)
    const selected = container.querySelectorAll('tr.sel')
    expect(selected.length).toBe(1)
  })
})
