const { chromium } = require('playwright');
(async () => {
  const [html, png, w, h] = process.argv.slice(2);
  const b = await chromium.launch({args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--allow-file-access-from-files']});
  const p = await b.newPage({viewport:{width:+w,height:+h}, deviceScaleFactor:2});
  p.on('pageerror', e => console.log('pageerror', e.message));
  await p.goto('file://' + html);
  await p.waitForFunction(() => window.done === true, {timeout: 60000});
  await p.waitForTimeout(800);
  await p.screenshot({path: png});
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
