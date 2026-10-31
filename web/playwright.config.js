import { defineConfig, devices } from '@playwright/test'

const e2ePort = process.env.NEUROKILN_E2E_PORT || '8110'
const baseURL = process.env.NEUROKILN_E2E_BASE_URL || `http://127.0.0.1:${e2ePort}`

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 240_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list']],
  use: {
    baseURL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: process.env.PLAYWRIGHT_VIDEO || 'off',
  },
  projects: [
    {
      name: 'chromium',
      // Local machines without the downloaded Chromium can use PLAYWRIGHT_CHANNEL=chrome.
      use: { ...devices['Desktop Chrome'], channel: process.env.PLAYWRIGHT_CHANNEL || undefined },
    },
  ],
  webServer: process.env.NEUROKILN_E2E_BASE_URL
    ? undefined
    : {
        command: 'node tests/e2e/server.mjs',
        url: `${baseURL}/api/health`,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
        stdout: 'pipe',
        stderr: 'pipe',
      },
})
