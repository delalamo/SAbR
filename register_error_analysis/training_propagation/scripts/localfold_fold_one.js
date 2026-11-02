// Fold one Fv (H + L) on localfold.org in headless Chromium and time it.
// usage: node fold_one.js MODEL MSA RECYCLES H L outprefix   (MSA: none|search)
const { chromium } = require('playwright');
const [model, msa, rec, H, L, out] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-webgpu-adapter=swiftshader'] });
  const ctx = await b.newContext({ acceptDownloads: true, viewport: { width: 1400, height: 1000 } });
  const p = await ctx.newPage();
  p.on('console', m => { const t = m.text(); if (/error|warn|fail/i.test(t)) console.log('console:', t.slice(0, 200)); });
  await p.goto('https://localfold.org/', { waitUntil: 'networkidle', timeout: 120000 });
  const ta = p.locator('textarea'); await ta.first().fill(H);
  await p.click('#add-entity'); await p.waitForTimeout(500);
  await ta.nth(1).fill(L);
  await p.selectOption('#model-family', model); await p.waitForTimeout(500);
  await p.selectOption('#msa-mode', msa);
  if (await p.locator('#recycles').isVisible()) await p.selectOption('#recycles', rec);
  // the status box is the element whose text starts 'Ready.' before folding; remember it and read it afterwards
  await p.evaluate(() => { const e = [...document.querySelectorAll('*')].find(x => x.children.length === 0 && /^Ready\./.test((x.innerText || '').trim())); if (e) e.setAttribute('data-lfstatus', '1'); });
  const statusText = () => p.evaluate(() => { const e = document.querySelector('[data-lfstatus]'); const box = e ? (e.closest('div') || e) : null; return box ? box.innerText.replace(/\s+/g, ' ').slice(0, 200) : ''; });
  // the model may ask to accept terms first (AF3); accept any dialog button that says accept/continue
  const t0 = Date.now(); await p.click('#predict');
  let last = '', dl = null;
  p.on('download', d => { dl = d; });
  for (let i = 0; i < 360; i++) {
    await p.waitForTimeout(10000);
    const acc = p.locator('button:has-text("Accept"), button:has-text("Continue")');
    if (await acc.count() && await acc.first().isVisible()) { console.log('accepting terms dialog'); await acc.first().click(); }
    const s = await statusText();
    if (s !== last) { console.log(`[${((Date.now() - t0) / 1000).toFixed(0)} s] ${s}`); last = s; }
    if (i % 6 === 5) await p.screenshot({ path: `${out}_progress.png` });
    if (/pLDDT|Done|Folded|finished|complete/i.test(s)) break;
    if (/error|failed/i.test(s)) break;
  }
  console.log('wall', ((Date.now() - t0) / 1000).toFixed(0), 's');
  await p.screenshot({ path: `${out}.png` });
  // try to save the structure via the page's own PDB download
  const pdb = await p.evaluate(() => (window.__lastPdb || null));
  if (pdb) require('fs').writeFileSync(`${out}.pdb`, pdb);
  await b.close();
})();
