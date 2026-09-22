// Every call to the backend goes through here. The wrapper exists mainly for the error path.

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

/* Upload a video, reporting progress as it goes. */
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
