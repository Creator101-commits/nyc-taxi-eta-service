import { expect, test } from "@playwright/test";

declare global {
  interface Window {
    __nyc?: { activeTag: string; recordCount: number; frames: number };
  }
}

test("map loads, taxis animate, and the console stays clean", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(String(err)));

  await page.goto("/");

  // __nyc is set inside the map load handler once zones, routes and the
  // TripsLayer are wired up.
  await page.waitForFunction(() => window.__nyc !== undefined, undefined, {
    timeout: 30_000,
  });

  const first = await page.evaluate(() => ({ ...window.__nyc! }));
  expect(first.recordCount).toBeGreaterThan(0);

  await page.waitForTimeout(2_000);
  const second = await page.evaluate(() => ({ ...window.__nyc! }));
  expect(second.frames).toBeGreaterThan(first.frames);

  await page.screenshot({
    path: "test-results/smoke.png",
    fullPage: true,
  });

  expect(errors).toEqual([]);
});
