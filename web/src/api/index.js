import axios from 'axios'
import { clearSession, getToken } from '@/utils/auth'
import { translateError } from '@/utils/errors'

const request = axios.create({ baseURL: '/api', timeout: 600000 })

request.interceptors.request.use((config) => {
  const token = getToken()
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

let redirecting = false

function redirectToLogin() {
  if (redirecting) return
  redirecting = true
  clearSession()
  const current = `${window.location.pathname}${window.location.search}`
  const suffix = current && !current.startsWith('/login')
    ? `?redirect=${encodeURIComponent(current)}`
    : ''
  window.location.replace(`/login${suffix}`)
}

request.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const detail = error.response?.data?.detail
    const raw = typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? detail.map((item) => item.msg).join('；')
        : ''
    const message = translateError(
      error.response?.data?.code,
      raw,
      translateError('request.network', '', `请求失败：${error.message}`),
    )
    const url = error.config?.url || ''
    const authAttempt = url.includes('/auth/login')
      || url.includes('/auth/bootstrap')
      || url.includes('/auth/status')
    if (error.response?.status === 401 && !authAttempt) redirectToLogin()
    else ElMessage.error(message)
    return Promise.reject(error)
  },
)

export const getHealth = () => request.get('/health')
export const getAuthStatus = () => request.get('/auth/status')
export const bootstrapAdmin = (data) => request.post('/auth/bootstrap', data)
export const login = (data) => request.post('/auth/login', data)
export const getCurrentUser = () => request.get('/auth/me')
export const getUsers = () => request.get('/auth/users')
export const createUser = (data) => request.post('/auth/users', data)
export const deleteUser = (id) => request.delete(`/auth/users/${id}`)

export const getDatasets = () => request.get('/datasets')
export const loadDataset = (datasetId, data = {}) => request.post(`/datasets/${datasetId}/load`, data)
export const importDataset = (formData) => request.post('/datasets/import', formData)
export const inspectDatasetImport = (formData) => request.post('/datasets/import/inspect', formData)
export const commitDatasetImport = (data) => request.post('/datasets/import/commit', data)
export const getPipelines = () => request.get('/pipelines')
export const getPipelineCatalog = () => request.get('/pipelines/catalog')
export const getPlugins = () => request.get('/plugins')
export const installBundledPlugin = (data) => request.post('/plugins/install/bundled', data)
export const installPluginZip = (formData) => request.post('/plugins/install', formData)
export const enablePlugin = (id) => request.post(`/plugins/${id}/enable`)
export const disablePlugin = (id) => request.post(`/plugins/${id}/disable`)
export const uninstallPlugin = (id) => request.delete(`/plugins/${id}`)
export const getCacheStats = () => request.get('/cache')
export const clearCache = () => request.delete('/cache')
export const getCustomPipelines = () => request.get('/pipelines/custom')
export const getCustomPipeline = (id) => request.get(`/pipelines/custom/${id}`)
export const createCustomPipeline = (data) => request.post('/pipelines/custom', data)
export const updateCustomPipeline = (id, data) => request.put(`/pipelines/custom/${id}`, data)
export const deleteCustomPipeline = (id) => request.delete(`/pipelines/custom/${id}`)
export const validatePipeline = (data) => request.post('/pipelines/validate', data)
export const getCustomPipelineSearchSpace = (id) => request.get(`/pipelines/custom/${id}/search-space`)
export const saveCustomPipelineSearchSpace = (id, dims) => request.put(`/pipelines/custom/${id}/search-space`, { dims })
export const getPipelineTemplates = () => request.get('/pipelines/templates')
export const installPipelineTemplate = (id, data = {}) => request.post(`/pipelines/templates/${id}/install`, data)
export const exportCustomPipeline = (id) => request.get(`/pipelines/custom/${id}/export`)
export const importCustomPipeline = (data) => request.post('/pipelines/custom/import', data)
export const getReproductionCatalog = () => request.get('/reproduction')
export const getCodex = () => request.get('/codex')
export const getProtocols = () => request.get('/protocols')
export const previewProtocol = (data) => request.post('/protocols/preview', data)
export const runArena = (data) => request.post('/arena', data)
export const runArenaDemo = (data) => request.post('/arena/demo', data)
export const runGoldenDemo = () => request.post('/demo/run')
export const submitArenaJob = (data) => request.post('/jobs/arena', data)
export const getJobs = (params = {}) => request.get('/jobs', { params })
export const getJob = (id) => request.get(`/jobs/${id}`)
export const cancelJob = (id) => request.post(`/jobs/${id}/cancel`)
export const getExperiments = (params = {}) => request.get('/experiments', { params })
export const getExperiment = (id) => request.get(`/experiments/${id}`)
export const deleteExperiment = (id) => request.delete(`/experiments/${id}`)
export const getExperimentStats = () => request.get('/experiments/stats')
export const getHPOTemplates = () => request.get('/hpo/templates')
export const submitHPOJob = (data) => request.post('/hpo/jobs', data)
export const promoteHPOResult = (data) => request.post('/hpo/promote', data)
export const submitExplanationJob = (data) => request.post('/explanations/jobs', data)
export const getExplanations = (params = {}) => request.get('/explanations', { params })
export const getExplanationAggregate = (params = {}) => request.get('/explanations/aggregate', { params })
export const getExplanation = (id) => request.get(`/explanations/${id}`)
export const deleteExplanation = (id) => request.delete(`/explanations/${id}`)
