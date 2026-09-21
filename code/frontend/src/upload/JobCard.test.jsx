import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import JobCard from './JobCard.jsx'

/* The job card is the only window onto work that takes minutes, so its duty is to say what is
   happening in terms of the bout rather than the runner. */

const job = (over = {}) => ({
  job_id: 'abc123def456',
  filename: 'bout.mp4',
  state: 'running',
  stage: 'detect',
  stages_done: ['piste'],
  progress: { done: 50, total: 100, pct: 50 },
  video_info: { width: 1280, height: 720, duration_s: 180 },
  elapsed_s: 65,
  queue_position: null,
  error: null,
  ...over,
})

const noop = { onChanged: vi.fn(), onOpenBout: vi.fn() }

describe('JobCard', () => {
  it('names the stage in terms of the bout, not the runner', () => {
    // "detect" means nothing to someone who did not write the pipeline.
    render(<JobCard job={job()} {...noop} />)
    expect(screen.getByText(/Detecting and tracking fencers/)).toBeInTheDocument()
  })

  it('tells a queued job how many are ahead of it', () => {
    // Jobs run one at a time by design, so a waiting user is owed a position
    // rather than a spinner that looks identical to a stuck job.
    render(<JobCard job={job({ state: 'queued', queue_position: 2 })} {...noop} />)
    expect(screen.getByText(/2 jobs ahead/)).toBeInTheDocument()
  })

  it('distinguishes waiting to start from waiting behind others', () => {
    render(<JobCard job={job({ state: 'queued', queue_position: 0 })} {...noop} />)
    expect(screen.getByText(/Waiting to start/)).toBeInTheDocument()
  })

  it('offers the review handoff only once the bout exists', () => {
    const { rerender } = render(<JobCard job={job()} {...noop} />)
    expect(screen.queryByRole('button', { name: /Open in review/ })).toBeNull()
    rerender(<JobCard job={job({ state: 'done', bout_id: 'upload_x:bout_x' })}
                      {...noop} />)
    expect(screen.getByRole('button', { name: /Open in review/ })).toBeInTheDocument()
  })

  it('shows the pipeline output when a job fails', () => {
    // A failure here is almost always a Python traceback, and without it the
    // user is told only that something exited non-zero.
    render(<JobCard job={job({
      state: 'failed', error: 'the detection stage exited with code 1',
      log_tail: ['Traceback (most recent call last):', 'ValueError: no frames'],
    })} {...noop} />)
    expect(screen.getByText(/exited with code 1/)).toBeInTheDocument()
    expect(screen.getByText(/ValueError: no frames/)).toBeInTheDocument()
  })

  it('explains an interrupted job rather than showing it as failed', () => {
    // A restart is not a pipeline bug, and telling the user it was would send
    // them looking for one.
    render(<JobCard job={job({
      state: 'interrupted', error: 'the server stopped while this job was running',
    })} {...noop} />)
    expect(screen.getByText(/Interrupted by a server restart/)).toBeInTheDocument()
  })

  it('does not offer to cancel work that has already finished', () => {
    render(<JobCard job={job({ state: 'done', bout_id: 'b' })} {...noop} />)
    expect(screen.queryByRole('button', { name: 'Cancel' })).toBeNull()
  })

  it('does not offer to delete a job while it is still running', () => {
    // Deleting the source from under a running subprocess produces a failure
    // that looks like a pipeline bug.
    render(<JobCard job={job({ state: 'running' })} {...noop} />)
    expect(screen.queryByRole('button', { name: 'Delete' })).toBeNull()
  })

  it('reports the piste decision on a finished bout', () => {
    // Which region was used qualifies every figure the bout produced, so it is
    // shown rather than left in a log.
    render(<JobCard job={job({
      state: 'done', bout_id: 'b',
      piste: { polygon: [[0, 380], [1280, 380], [1280, 550], [0, 550]],
               decision: 'accepted as measured' },
    })} {...noop} />)
    expect(screen.getByText(/accepted as measured/)).toBeInTheDocument()
    expect(screen.getByText(/380 to 550/)).toBeInTheDocument()
  })

  it('says when a bout was processed with no piste region at all', () => {
    render(<JobCard job={job({
      state: 'done', bout_id: 'b',
      piste: { polygon: null, reason: 'no bystanders were detected' },
    })} {...noop} />)
    expect(screen.getByText(/without a piste region/)).toBeInTheDocument()
  })
})

describe('JobCard piste region', () => {
  it('does not crash when the region was carried over rather than measured', () => {
    // A reprocess records `polygon: true`, a flag meaning "there is one",
    // rather than the polygon itself. Indexing it threw and blanked the app.
    render(<JobCard job={{
      job_id: 'x', state: 'done', kind: 'processing', filename: 'rerun',
      stages_done: [], progress: { done: 1, total: 1, pct: 100 },
      piste: { needed: true, polygon: true, decision: 'carried over from the original run' },
    }} />)
    expect(screen.getByText(/carried over from the original run/)).toBeInTheDocument()
  })

  it('still prints the rows when it has a real polygon', () => {
    render(<JobCard job={{
      job_id: 'x', state: 'done', kind: 'processing', filename: 'f',
      stages_done: [], progress: { done: 1, total: 1, pct: 100 },
      piste: { needed: true, polygon: [[0, 120], [10, 120], [10, 400], [0, 400]],
               decision: 'measured' },
    }} />)
    expect(screen.getByText(/rows 120 to 400/)).toBeInTheDocument()
  })
})
