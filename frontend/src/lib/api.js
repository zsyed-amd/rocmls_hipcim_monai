// Thin client for the FastAPI backend. In dev, Vite proxies /api -> localhost:8600.

async function request(path, options) {
  const res = await fetch(path, options)
  if (!res.ok) {
    let detail = ''
    try {
      detail = (await res.json()).detail || ''
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(detail || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export function get(path) {
  return request(path)
}

export function post(path, body) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
}

// System
export const getSystemInfo = () => get('/api/system/info')

// Open a Server-Sent Events stream and dispatch named events to handlers.
// handlers: { [eventName]: (payload) => void, error?, open? }. Returns the
// EventSource so the caller can .close() it.
export function openSSE(path, handlers = {}) {
  const source = new EventSource(path)
  for (const [name, fn] of Object.entries(handlers)) {
    if (name === 'error') {
      source.onerror = fn
      continue
    }
    if (name === 'open') {
      source.onopen = fn
      continue
    }
    source.addEventListener(name, (e) => {
      let payload = e.data
      try {
        payload = JSON.parse(e.data)
      } catch {
        // leave as raw string (e.g. log lines)
      }
      fn(payload)
    })
  }
  return source
}
