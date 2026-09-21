import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import TouchTable from './TouchTable.jsx'

/* The touch table is where the user adjudicates the system's guesses, so its duty is to show
   what each guess rests on and to keep the system's suggestions visibly separate from the
   user's own decisions. */

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
    // The two figures are labelled inline now rather than sitting under column
    // headings, so they are matched inside the row's evidence line.
    const { container } = render(
      <TouchTable rows={[proposed()]} selIdx={0} {...handlers} />)
    const basis = container.querySelector('.basis').textContent
    expect(basis).toContain('0.29')
    expect(basis).toContain('0.87m')
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
    // Confirmed, because the lamp is only read for touches the user has
    // confirmed, so that is the only state a suggestion can appear in.
    render(<TouchTable rows={[proposed({ state: 'confirmed' })]} selIdx={0}
                       {...handlers}
                       scorerProposals={[{ time_s: 43.9, proposed: 'left',
                                           green_delta: 12, red_delta: 400 }]} />)
    const cell = screen.getByText(/lamp left/)
    expect(cell).toBeInTheDocument()
    expect(cell.className).toContain('lamp')
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
    const selected = container.querySelectorAll('.touch.sel')
    expect(selected.length).toBe(1)
  })
})

describe('recording who scored', () => {
  const onScorer = vi.fn()
  const confirmed = (over = {}) => ({
    id: 'p0', at: 19.0, kind: 'proposed', state: 'confirmed',
    confidence: 0.9, separation_m: 2.1, scorer: null, ...over,
  })

  beforeEach(() => onScorer.mockClear())

  it('lets a confirmed touch be attributed, which nothing did before', () => {
    // The lamp reading used to be shown in italics with nothing to press, so a
    // touch could never be attributed from the interface at all and the three
    // scoring figures on the profile could never be populated. Confirm and
    // reject answer whether the touch happened, not who scored it.
    render(<TouchTable rows={[confirmed()]} selIdx={0} onSelect={() => {}}
                       onDecide={() => {}} onScorer={onScorer} />)
    fireEvent.click(screen.getByTitle(/Fencer 2 scored/))
    expect(onScorer).toHaveBeenCalledWith('p0', 'right')
  })

  it('accepts the lamp suggestion on one press', () => {
    render(<TouchTable rows={[confirmed()]} selIdx={0} onSelect={() => {}}
                       onDecide={() => {}} onScorer={onScorer}
                       scorerProposals={[{ time_s: 19.0, proposed: 'left',
                                           green_delta: 9, red_delta: 1 }]} />)
    fireEvent.click(screen.getByText(/lamp left/))
    expect(onScorer).toHaveBeenCalledWith('p0', 'left')
  })

  it('clears an attribution when the same side is pressed again', () => {
    // Mis-clicking must be undoable, or a wrong attribution is permanent and
    // silently wrong, which is worse than an absent one.
    render(<TouchTable rows={[confirmed({ scorer: 'left' })]} selIdx={0}
                       onSelect={() => {}} onDecide={() => {}} onScorer={onScorer} />)
    fireEvent.click(screen.getByTitle(/Fencer 1 scored/))
    expect(onScorer).toHaveBeenCalledWith('p0', null)
  })

  it('offers nothing to attribute on a touch that was rejected', () => {
    render(<TouchTable rows={[confirmed({ state: 'rejected' })]} selIdx={0}
                       onSelect={() => {}} onDecide={() => {}} onScorer={onScorer} />)
    expect(screen.queryByTitle(/scored/)).not.toBeInTheDocument()
  })
})

describe('the lamp reading stays visible after the user answers', () => {
  const onScorer = vi.fn()
  const row = (over = {}) => ({
    id: 'p0', at: 19.0, kind: 'proposed', state: 'confirmed',
    confidence: 0.9, separation_m: 2.1, scorer: null, ...over,
  })
  const lamp = [{ time_s: 19.0, proposed: 'left', green_delta: 9, red_delta: 1 }]

  it('marks agreement', () => {
    render(<TouchTable rows={[row({ scorer: 'left' })]} selIdx={0} onSelect={() => {}}
                       onDecide={() => {}} onScorer={onScorer} scorerProposals={lamp} />)
    expect(screen.getByText(/lamp left/).className).toContain('agrees')
  })

  it('marks disagreement, which is the case worth finding', () => {
    // These rows are the evidence behind the attribution figures: where the
    // lamp reading and the person's judgement part company. Hiding the reading
    // once answered would hide exactly the cases worth looking at.
    render(<TouchTable rows={[row({ scorer: 'right' })]} selIdx={0} onSelect={() => {}}
                       onDecide={() => {}} onScorer={onScorer} scorerProposals={lamp} />)
    const chip = screen.getByText(/lamp left/)
    expect(chip.className).toContain('differs')
    expect(chip.title).toMatch(/you recorded right/)
  })

  it('cannot be pressed once the user has answered', () => {
    // Otherwise a stray click silently overwrites a judgement with a guess.
    render(<TouchTable rows={[row({ scorer: 'right' })]} selIdx={0} onSelect={() => {}}
                       onDecide={() => {}} onScorer={onScorer} scorerProposals={lamp} />)
    fireEvent.click(screen.getByText(/lamp left/))
    expect(onScorer).not.toHaveBeenCalled()
  })
})
