import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, boutPath, postJSON } from './api.js'

/* The wrapper exists mainly for the error path. */

afterEach(() => { delete global.fetch })

describe('api', () => {
  it('surfaces the server\'s explanation, not the status code', async () => {
    global.fetch = vi.fn(() => Promise.resolve({
      ok: false, status: 400, statusText: 'Bad Request',
      json: () => Promise.resolve({ detail: 'no touches confirmed yet' }),
    }))
    await expect(api('/api/x')).rejects.toThrow('no touches confirmed yet')
  })

  it('falls back to the status when there is no explanation', async () => {
    // A proxy error or a crash produces no JSON body, and "undefined" as an
    // error message tells the user nothing at all.
    global.fetch = vi.fn(() => Promise.resolve({
      ok: false, status: 502, statusText: 'Bad Gateway',
      json: () => Promise.reject(new Error('not json')),
    }))
    await expect(api('/api/x')).rejects.toThrow('502 Bad Gateway')
  })

  it('returns the parsed body on success', async () => {
    global.fetch = vi.fn(() => Promise.resolve({
      ok: true, json: () => Promise.resolve({ ok: true, n: 3 }),
    }))
    await expect(api('/api/x')).resolves.toEqual({ ok: true, n: 3 })
  })
})

describe('boutPath', () => {
  it('encodes a bout id so a colon cannot break the path', () => {
    // Every bout id contains a colon by construction, and uploaded ones are
    // built from a filename the user chose.
    expect(boutPath('results_current:fencing_clip2'))
      .toBe('/api/bouts/results_current%3Afencing_clip2')
  })

  it('encodes a slash, which would otherwise invent a route segment', () => {
    expect(boutPath('a/b')).toBe('/api/bouts/a%2Fb')
  })
})

describe('postJSON', () => {
  it('sets the content type the backend requires', () => {
    // FastAPI reads a JSON body only when told it is one; without this the
    // request arrives as an unparsed form and fails validation.
    const opts = postJSON({ state: 'confirmed' })
    expect(opts.method).toBe('POST')
    expect(opts.headers['Content-Type']).toBe('application/json')
    expect(JSON.parse(opts.body)).toEqual({ state: 'confirmed' })
  })
})
