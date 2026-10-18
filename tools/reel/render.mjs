// Render reel.html frame by frame with headless Chrome.
// Serve the repo root on :8766 first, then: node render.mjs [W] [H] [out_dir]
import puppeteer from 'puppeteer-core';
import { mkdirSync } from 'fs';
const [W = 1920, H = 1080, OUT = 'frames'] = process.argv.slice(2);
const FPS = 30, DUR = 0.8624 + 104 * 60 / 135;
mkdirSync(OUT, { recursive: true });
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true, args: ['--hide-scrollbars'] });
const page = await browser.newPage();
await page.setViewport({ width: +W, height: +H, deviceScaleFactor: 1 });
await page.goto(`http://localhost:8766/tools/reel/reel.html?w=${W}&h=${H}`, { waitUntil: 'networkidle0' });
await page.waitForFunction('window.READY === true');
const n = Math.ceil(DUR * FPS);
for (let i = 0; i < n; i++) {
  await page.evaluate(t => window.renderAt(t), i / FPS);
  await page.screenshot({ path: `${OUT}/${String(i).padStart(5, '0')}.jpg`, type: 'jpeg', quality: 92 });
  if (i % 120 === 0) console.log(`${i}/${n}`);
}
await browser.close();
console.log('done', n);
