import i18n from '@/i18n'

export function translateError(code, detail, fallback = '') {
  if (code) {
    const key = `errors.${code}`
    if (i18n.global.te(key)) {
      return i18n.global.t(key, { detail: detail ?? '' })
    }
  }
  return detail || fallback
}

export function jobErrorMessage(job, fallback = '') {
  return translateError(job?.error_code, job?.error, fallback)
}
