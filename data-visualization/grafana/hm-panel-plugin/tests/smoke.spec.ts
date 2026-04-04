import { expect, test } from '@grafana/plugin-e2e';

test.describe('Smoke test', () => {
  test('Grafana home page is accessible', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('Welcome to Grafana')).toBeVisible();
  });
});
