import { reactive } from 'vue'

const STORAGE_KEY = 'neurokiln-auth'

function readSession() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

const saved = readSession()

export const authState = reactive({
  token: saved?.token || '',
  user: saved?.user || null,
  expiresAt: saved?.expiresAt || 0,
})

if (authState.token && authState.expiresAt && authState.expiresAt * 1000 <= Date.now()) {
  clearSession()
}

export function setSession(payload) {
  authState.token = payload.token
  authState.user = payload.user
  authState.expiresAt = payload.expires_at || 0
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    token: authState.token,
    user: authState.user,
    expiresAt: authState.expiresAt,
  }))
}

export function clearSession() {
  authState.token = ''
  authState.user = null
  authState.expiresAt = 0
  localStorage.removeItem(STORAGE_KEY)
}

export function isLoggedIn() {
  if (!authState.token) return false
  return !authState.expiresAt || authState.expiresAt * 1000 > Date.now()
}

export function getToken() {
  return isLoggedIn() ? authState.token : ''
}

export function isAdmin() {
  return authState.user?.role === 'admin'
}
