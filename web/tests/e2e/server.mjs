import { spawn } from 'node:child_process'
import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '../../..')
const port = process.env.NEUROKILN_E2E_PORT || '8110'
const python = process.env.NEUROKILN_PYTHON
  || (existsSync(join(root, '.venv312/Scripts/python.exe'))
    ? join(root, '.venv312/Scripts/python.exe')
    : 'python')
const managedDataDir = !process.env.NEUROKILN_E2E_DATA
const dataDir = process.env.NEUROKILN_E2E_DATA
  || mkdtempSync(join(tmpdir(), 'neurokiln-e2e-'))

const child = spawn(
  python,
  ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', port],
  {
    cwd: join(root, 'server'),
    env: {
      ...process.env,
      NEUROKILN_DATA_DIR: dataDir,
      NEUROKILN_WEB_DIST: join(root, 'web', 'dist'),
      NEUROKILN_JOB_WORKERS: '1',
    },
    stdio: 'inherit',
  },
)

let stopping = false
function stop(signal = 'SIGTERM') {
  if (stopping) return
  stopping = true
  child.kill(signal)
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => stop(signal))
}

child.on('exit', (code) => {
  if (managedDataDir) rmSync(dataDir, { recursive: true, force: true })
  process.exit(code ?? 0)
})
