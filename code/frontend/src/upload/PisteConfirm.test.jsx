import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import PisteConfirm from './PisteConfirm.jsx'

/**
 * The piste confirmation step exists because of a specific past failure: a
 * polygon placed by eye on this project admitted the adjacent piste and raised
 * the count of physically impossible distance readings from 53 to 252, while the
 * headline coverage figure went up. So the user is asked to confirm a
 * MEASUREMENT, and these tests mostly assert that the measurement is shown to
 * them rather than hidden behind a picture.
 */

const job = (over = {}) => ({
  job_id: 'abc123def456',
  video_info: { width: 1280, height: 720 },
  piste: {
    needed: true,
    confident: true,
    polygon: [[0, 380], [1280, 380], [1280, 550], [0, 550]],
    frame_size: [1280, 720],
    reason: '600 of 600 sampled frames contain people besides the two fencers',
    measurements: {
      frames_sampled: 600, fencer_samples: 1200,
      fencer_feet_y_min: 393.3, fencer_feet_y_max: 527.0,
      other_groups: [{ feet_y: [207.2, 255.2], samples: 195 }],
    },
    ...over.piste,
  },
  ...over,
})

describe('PisteConfirm', () => {
  it('shows the measurement the region was derived from', () => {
    // The whole point of measuring rather than asking the user to draw. Without
    // the numbers they are confirming a picture, which is the same act as
    // drawing one by eye.
    render(<PisteConfirm job={job()} onDecided={vi.fn()} />)
    expect(screen.getByText(/600 sampled frames/)).toBeInTheDocument()
    expect(screen.getByText(/1200 fencer detections/)).toBeInTheDocument()
  })

  it('reports the group of people it excluded', () => {
    render(<PisteConfirm job={job()} onDecided={vi.fn()} />)
    expect(screen.getByText(/195 detections/)).toBeInTheDocument()
  })

  it('warns loudly when the measurement is not confident', () => {
    // The case that matters: a band the strip shares with other people at the
    // same apparent depth, which no horizontal region can separate.
    render(<PisteConfirm job={job({ piste: {
      needed: true, confident: false,
      polygon: [[0, 399], [1280, 399], [1280, 717], [0, 717]],
      frame_size: [1280, 720],
      reason: 'about 1.9 other people per frame still fall inside this band',
      measurements: {},
    } })} onDecided={vi.fn()} />)
    expect(screen.getByText(/Check this one/)).toBeInTheDocument()
    expect(screen.getByText(/1.9 other people per frame/)).toBeInTheDocument()
  })

  it('offers accepting the measurement and processing without a region', () => {
    render(<PisteConfirm job={job()} onDecided={vi.fn()} />)
    expect(screen.getByRole('button', { name: /Looks right/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /without a region/ }))
      .toBeInTheDocument()
  })

  it('says when skipping the region is safe and when it is not', () => {
    // Skipping is correct for footage containing only the two fencers and wrong
    // for a competition, and the difference is not obvious from the frame.
    render(<PisteConfirm job={job()} onDecided={vi.fn()} />)
    expect(screen.getByText(/only people on camera/)).toBeInTheDocument()
  })

  it('renders a frame for the user to judge against', () => {
    const { container } = render(<PisteConfirm job={job()} onDecided={vi.fn()} />)
    const img = container.querySelector('img')
    expect(img).not.toBeNull()
    expect(img.getAttribute('src')).toContain('/frame')
  })

  it('copes with a job whose measurement produced no polygon', () => {
    // Footage with no bystanders needs no region, and the step should not crash
    // trying to draw one.
    render(<PisteConfirm job={job({ piste: {
      needed: false, confident: true, polygon: null,
      reason: 'only 2 of 570 sampled frames contain anyone besides the fencers',
      measurements: {},
    } })} onDecided={vi.fn()} />)
    expect(screen.getByRole('button', { name: /without a region/ }))
      .toBeInTheDocument()
  })
})
