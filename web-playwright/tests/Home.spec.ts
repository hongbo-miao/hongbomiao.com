import { expect, test } from '@playwright/test';

test.describe('Home', () => {
  test('render Home', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.hm-home-container')).toContainText('Making magic happen');
  });

  test('receive GraphQL meQuery data', async ({ request }) => {
    const query = `
      query Me {
        me {
          name
          bio
        }
      }
    `;

    const response = await request.post('/graphql', {
      data: { query },
    });

    expect(response.status()).toBe(200);
    expect(await response.json()).toEqual({
      data: {
        me: {
          name: 'Hongbo Miao',
          bio: 'Making magic happen',
        },
      },
    });
  });
});
