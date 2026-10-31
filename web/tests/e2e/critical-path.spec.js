import { expect, test } from '@playwright/test'

const ADMIN = { username: 'e2eadmin', password: 'e2e-password-123' }

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('neurokiln.locale', 'en')
  })
})

async function bootstrapOrLogin(page) {
  await page.goto('/login')
  const setupHeading = page.getByRole('heading', { name: 'Create administrator' })
  const loginHeading = page.getByRole('heading', { name: 'Sign in', exact: true })
  await expect(setupHeading.or(loginHeading)).toBeVisible()
  // /api/auth/status decides which form to show after mount.
  await page.waitForLoadState('networkidle')

  if (await setupHeading.isVisible()) {
    await page.locator('input[autocomplete="username"]').fill(ADMIN.username)
    const passwords = page.locator('input[autocomplete="new-password"]')
    await passwords.nth(0).fill(ADMIN.password)
    await passwords.nth(1).fill(ADMIN.password)
    await page.getByRole('button', { name: 'Create and enter' }).click()
  } else {
    await page.locator('input[autocomplete="username"]').fill(ADMIN.username)
    await page.locator('input[autocomplete="current-password"]').fill(ADMIN.password)
    await page.getByRole('button', { name: 'Sign in', exact: true }).click()
  }
  await expect(page).toHaveURL(/\/dashboard/, { timeout: 30_000 })
}

test('bootstrap and golden demo reach the report', async ({ page }) => {
  await bootstrapOrLogin(page)

  await page.getByRole('button', { name: 'Run golden demo' }).first().click()
  await expect(page).toHaveURL(/\/report\?experiment=\d+/, { timeout: 180_000 })
  await expect(page.getByText(/Experiment #\d+ ·/).first()).toBeVisible()
})

test('XAI run, cross-fold aggregate and localized error', async ({ page }) => {
  await bootstrapOrLogin(page)

  await page.goto('/xai')
  await expect(page.getByRole('heading', { name: 'Explainability' })).toBeVisible()
  await page.getByRole('button', { name: 'Run analysis' }).click()
  await expect(page.getByText(/Decision score unit/)).toBeVisible({ timeout: 180_000 })
  await expect(page.getByText('Channel importance').first()).toBeVisible()

  const token = await page.evaluate(
    () => JSON.parse(localStorage.getItem('neurokiln-auth') || '{}').token,
  )
  expect(token).toBeTruthy()
  const headers = { Authorization: `Bearer ${token}` }
  const submitted = await page.request.post('/api/explanations/jobs', {
    headers,
    data: {
      dataset_id: 'synthetic',
      subject_ids: ['S01'],
      classes: 2,
      options: { n_trials_per_class: 40 },
      protocol: 'within_subject',
      folds: 5,
      repeats: 1,
      seeds: [0],
      pipeline: 'csp_lda',
      fold: 1,
      repeat: 0,
      window_ms: 250,
      max_blocks: 24,
      gradient_method: 'none',
      include_audit: true,
      include_sanity: false,
    },
  })
  expect(submitted.ok()).toBeTruthy()
  const jobId = (await submitted.json()).id
  await expect
    .poll(
      async () => {
        const response = await page.request.get(`/api/jobs/${jobId}`, { headers })
        return (await response.json()).status
      },
      { timeout: 180_000, intervals: [1000, 2000] },
    )
    .toBe('completed')

  await page.getByRole('button', { name: 'Cross-fold aggregate' }).click()
  const drawer = page.locator('.el-drawer').filter({ hasText: 'Cross-fold explanation aggregate' })
  await expect(drawer).toBeVisible()
  const explanationsCard = drawer.locator('.stat-card').filter({ hasText: 'Explanations' })
  const explanationCount = Number(await explanationsCard.locator('.stat-value').innerText())
  expect(explanationCount).toBeGreaterThanOrEqual(2)
  await expect(drawer.getByText('Robust channels (by occlusion magnitude)')).toBeVisible()

  await page.goto('/xai?explanation=999999')
  await expect(
    page.locator('.el-message').filter({ hasText: 'Explanation not found' }),
  ).toBeVisible()
})
