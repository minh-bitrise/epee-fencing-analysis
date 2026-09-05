import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import UploadView from './UploadView.jsx'

/**
 * The entry point. Everything else in the application depends on a bout having
 * been processed, so a failure here is a failure to start, and it is the one
 * screen a first-time user cannot route around.
 *
 * Two behaviours carry most of the risk and most of these tests. Polling has to
 * stop when nothing can change, or the page asks about a queue of finished jobs
 * for as long as it is left open; and an upload that fails has to say so,
 * because a silent failure is indistinguishable from a slow one and the user's
 * response to each is opposite.
 */

const job = (over = {}) => ({
  job_id: 'abc123def456', filename: 'bout.mp4', state: 'done', stage: 'detect',
  stages_done: ['piste', 'detect'], progress: { done: 100, total: 100, pct: 100 },
  video_info: { width: 1280, height: 720, duration_s: 180 },
  elapsed_s: 65, queue_position: null, error: null, ...over,
})

let uploads
vi.mock('../api.js', async (importOriginal) => {
  const actual = await importOriginal()
  return {
    ...actual,
    api: (url, opts) => globalThis.__api(url, opts),
    uploadVideo: (file, opts) => globalThis.__upload(file, opts),
  }
})

function stub({ jobs = [], uploadFails = null } = {}) {
  uploads = []
  globalThis.__api = vi.fn((url) => {
    if (url === '/api/jobs') return Promise.resolve({ jobs })
    if (url === '/api/storage') {
      return Promise.resolve({ total_mb: 512, groups: [], orphans: [] })
    }
    return Promise.resolve({ ok: true })
  })
  globalThis.__upload = vi.fn((file, opts) => {
    uploads.push({ file, opts })
    if (uploadFails) return Promise.reject(new Error(uploadFails))
    return Promise.resolve({ job_id: 'new1' })
  })
}

const drop = (el, file) =>
  fireEvent.drop(el, { dataTransfer: { files: file ? [file] : [] } })

const aFile = () => new File(['x'], 'bout.mp4', { type: 'video/mp4' })

beforeEach(() => { vi.useFakeTimers({ shouldAdvanceTime: true }) })
afterEach(() => {
  vi.useRealTimers()
  delete globalThis.__api
  delete globalThis.__upload
})

describe('UploadView', () => {
  it('says nothing has been processed rather than showing an empty list', async () => {
    stub()
    render(<UploadView onOpenBout={vi.fn()} />)
    await waitFor(() =>
      expect(screen.getByText(/Nothing processed yet/)).toBeInTheDocument())
  })

  it('lists a job once one exists', async () => {
    stub({ jobs: [job()] })
    render(<UploadView onOpenBout={vi.fn()} />)
    await waitFor(() => expect(screen.getByText('bout.mp4')).toBeInTheDocument())
  })

  it('uploads a dropped file', async () => {
    stub()
    const { container } = render(<UploadView onOpenBout={vi.fn()} />)
    await waitFor(() => expect(globalThis.__api).toHaveBeenCalled())
    drop(container.querySelector('.drop'), aFile())
    await waitFor(() => expect(uploads.length).toBe(1))
    expect(uploads[0].file.name).toBe('bout.mp4')
  })

  it('ignores a drop that carried no file', async () => {
    // Dragging a selection or a link onto the page produces a drop with an
    // empty file list, and uploading undefined fails deep in the request.
    stub()
    const { container } = render(<UploadView onOpenBout={vi.fn()} />)
    await waitFor(() => expect(globalThis.__api).toHaveBeenCalled())
    drop(container.querySelector('.drop'), null)
    await act(async () => {})
    expect(uploads.length).toBe(0)
  })

  it('carries the piste choice into the upload', async () => {
    // The checkbox decides whether the run stops to ask about the region. If it
    // did not reach the request the setting would look respected and be ignored.
    stub()
    const { container } = render(<UploadView onOpenBout={vi.fn()} />)
    await waitFor(() => expect(globalThis.__api).toHaveBeenCalled())
    fireEvent.click(screen.getByRole('checkbox'))
    drop(container.querySelector('.drop'), aFile())
    await waitFor(() => expect(uploads.length).toBe(1))
    expect(uploads[0].opts.confirmPiste).toBe(false)
  })

  it('shows an upload failure instead of silently doing nothing', async () => {
    /**
     * The failure this guards. A rejected upload left the page exactly as it
     * was, which is indistinguishable from a slow one, and the user's response
     * to each is opposite: wait, or try again.
     */
    stub({ uploadFails: 'that file is not a video the pipeline can read' })
    const { container } = render(<UploadView onOpenBout={vi.fn()} />)
    await waitFor(() => expect(globalThis.__api).toHaveBeenCalled())
    drop(container.querySelector('.drop'), aFile())
    await waitFor(() =>
      expect(screen.getByText(/not a video the pipeline can read/))
        .toBeInTheDocument())
  })

  it('reports progress while the file is going up', async () => {
    stub()
    globalThis.__upload = vi.fn((file, opts) => {
      opts.onProgress(42)
      return new Promise(() => {})    // never settles: the upload is in flight
    })
    const { container } = render(<UploadView onOpenBout={vi.fn()} />)
    await waitFor(() => expect(globalThis.__api).toHaveBeenCalled())
    drop(container.querySelector('.drop'), aFile())
    await waitFor(() => expect(screen.getByText(/42%/)).toBeInTheDocument())
  })

  it('polls quickly while a job is live', async () => {
    stub({ jobs: [job({ state: 'running' })] })
    render(<UploadView onOpenBout={vi.fn()} />)
    await waitFor(() => expect(globalThis.__api).toHaveBeenCalledTimes(1))
    await act(async () => { await vi.advanceTimersByTimeAsync(2100) })
    expect(globalThis.__api.mock.calls.filter((c) => c[0] === '/api/jobs').length)
      .toBeGreaterThan(1)
  })

  it('backs off once nothing can change', async () => {
    /**
     * Without this the page asks about a queue of finished jobs for as long as
     * it is left open. It does not stop entirely, because a job can be started
     * from another tab or a terminal and this page should notice.
     */
    stub({ jobs: [job({ state: 'done' })] })
    render(<UploadView onOpenBout={vi.fn()} />)
    await waitFor(() => expect(globalThis.__api).toHaveBeenCalledTimes(1))
    const calls = () =>
      globalThis.__api.mock.calls.filter((c) => c[0] === '/api/jobs').length
    await act(async () => { await vi.advanceTimersByTimeAsync(2100) })
    expect(calls()).toBe(1)
    await act(async () => { await vi.advanceTimersByTimeAsync(8100) })
    expect(calls()).toBe(2)
  })

  it('stops polling when the view goes away', async () => {
    // A timer that outlives the component keeps requesting against a server the
    // user has navigated away from, and sets state on an unmounted tree.
    stub({ jobs: [job({ state: 'running' })] })
    const { unmount } = render(<UploadView onOpenBout={vi.fn()} />)
    await waitFor(() => expect(globalThis.__api).toHaveBeenCalledTimes(1))
    unmount()
    const before = globalThis.__api.mock.calls.length
    await act(async () => { await vi.advanceTimersByTimeAsync(9000) })
    expect(globalThis.__api.mock.calls.length).toBe(before)
  })

  it('surfaces a failure to list the jobs at all', async () => {
    stub()
    globalThis.__api = vi.fn(() => Promise.reject(new Error('the server is down')))
    render(<UploadView onOpenBout={vi.fn()} />)
    await waitFor(() =>
      expect(screen.getByText(/the server is down/)).toBeInTheDocument())
  })

  it('explains the piste region without printing the explanation', async () => {
    // Folded behind a disclosure: it is a paragraph, and the upload screen is
    // three controls.
    stub()
    const { container } = render(<UploadView onOpenBout={vi.fn()} />)
    await waitFor(() => expect(globalThis.__api).toHaveBeenCalled())
    const d = container.querySelector('details.disclose')
    expect(d).not.toBeNull()
    expect(d.open).toBe(false)
    expect(d.textContent).toMatch(/98 per cent tracking coverage to 80/)
  })
})
