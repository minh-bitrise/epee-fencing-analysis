// Every call to the backend goes through here.
//
// The wrapper exists mainly for the error path. FastAPI reports a problem as a
// JSON body with a `detail` field and a non-2xx status, and `fetch` treats a 400
// as a perfectly good response, so without this every call site would have to
// remember to check `ok` and dig the message out. Several of the backend's
// errors are ones the user genuinely needs to read: that an export would have
// been empty, that a piste decision arrived for a job that had moved on.

export async function api(path, options) {
  const res = await fetch(path, options)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export const postJSON = (body) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const del = { method: 'DELETE' }

export const boutPath = (boutId) => `/api/bouts/${encodeURIComponent(boutId)}`

/**
 * Upload a video, reporting progress as it goes.
 *
 * XMLHttpRequest rather than fetch, which is the whole reason this is not a
 * one-line call. `fetch` gives no upload progress at all: a request body is
 * either in flight or finished, with nothing in between. These files run to
 * hundreds of megabytes over a domestic connection, so a progress-free upload
 * would leave the user watching an inert page for minutes with no way to tell a
 * slow transfer from a hung one.
 */
export function uploadVideo(file, { confirmPiste = true, poseStride = 0,
                                    onProgress } = {}) {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('video', file)
    form.append('confirm_piste', String(confirmPiste))
    form.append('pose_stride', String(poseStride))

    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/jobs')
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((100 * e.loaded) / e.total))
      }
    })
    xhr.addEventListener('load', () => {
      let body = {}
      try { body = JSON.parse(xhr.responseText) } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body)
      else reject(new Error(body.detail || `upload failed (${xhr.status})`))
    })
    xhr.addEventListener('error', () =>
      reject(new Error('the upload failed before it reached the server')))
    xhr.addEventListener('abort', () => reject(new Error('upload cancelled')))
    xhr.send(form)
  })
}
