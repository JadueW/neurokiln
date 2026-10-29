import { getToken } from './auth'

export async function fetchWithAuth(url) {
  const token = getToken()
  const response = await fetch(url, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (!response.ok) {
    let message = response.status === 401 ? '登录已过期，请重新登录' : `下载失败（HTTP ${response.status}）`
    try {
      const payload = await response.json()
      if (payload?.detail) message = payload.detail
    } catch {
      // response body is not JSON; keep the status message
    }
    throw new Error(message)
  }
  return response.blob()
}

export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

export async function downloadWithAuth(url, filename) {
  saveBlob(await fetchWithAuth(url), filename)
}
