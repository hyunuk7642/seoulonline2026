// HTML 타임라인을 프레임 단위로 캡처해 mp4로 인코딩한다.
//   node render.mjs                 전체 렌더 (60fps, 1920x1080) → out/seoul-online-school-2026.mp4
//   node render.mjs --stills 1,2.5  지정한 시각(초)의 스틸만 frames/still_*.png 로 저장
//   node render.mjs --sheet         0.5초 간격 콘택트 시트(out/sheet_*.jpg) 생성
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import { mkdirSync, rmSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require('playwright')); }
catch { ({ chromium } = require('/opt/node-tools/node_modules/playwright')); }

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const URL = pathToFileURL(path.join(ROOT, 'src/index.html')).href + '?render=1';
const FRAMES = path.join(ROOT, 'frames');
const OUT = path.join(ROOT, 'out');
const FPS = 60, DUR = 40, TOTAL = FPS * DUR, WORKERS = 4;
const args = process.argv.slice(2);
const arg = k => { const i = args.indexOf(k); return i < 0 ? null : (args[i + 1] ?? true); };

async function open(browser) {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  page.on('pageerror', e => console.error('[page error]', e.message));
  await page.goto(URL);
  await page.waitForFunction('window.__ready === true', null, { timeout: 60000 });
  return page;
}

const launch = () => chromium.launch({ args: ['--font-render-hinting=none', '--disable-lcd-text'] });
const browser = await launch();
mkdirSync(FRAMES, { recursive: true }); mkdirSync(OUT, { recursive: true });

if (arg('--stills')) {
  const page = await open(browser);
  for (const t of String(arg('--stills')).split(',').map(Number)) {
    await page.evaluate(t => window.__seek(t), t);
    await page.screenshot({ path: path.join(FRAMES, `still_${t.toFixed(2).padStart(5, '0')}.png`) });
  }
  await browser.close();
} else if (arg('--sheet')) {
  const page = await open(browser);
  const step = Number(arg('--step')) || 0.5, from = Number(arg('--from')) || 0, to = Number(arg('--to')) || DUR;
  rmSync(path.join(FRAMES, 'sheet'), { recursive: true, force: true }); mkdirSync(path.join(FRAMES, 'sheet'), { recursive: true });
  let n = 0;
  for (let t = from; t < to - 1e-6; t += step, n++) {
    await page.evaluate(t => window.__seek(t), t);
    await page.screenshot({ path: path.join(FRAMES, 'sheet', `${String(n).padStart(4, '0')}.jpg`), type: 'jpeg', quality: 80 });
  }
  await browser.close();
  const tag = arg('--tag') || 'sheet';
  spawnSync('ffmpeg', ['-v', 'error', '-y', '-framerate', '1', '-i', path.join(FRAMES, 'sheet', '%04d.jpg'),
    '-vf', `scale=640:-1,drawtext=text='%{eif\\:${from}*100+n*${step * 100}\\:d}':x=8:y=8:fontsize=22:fontcolor=yellow:box=1:boxcolor=black@0.6,tile=4x4`,
    path.join(OUT, `${tag}_%d.jpg`)], { stdio: 'inherit' });
  console.log('sheets written');
} else {
  rmSync(FRAMES, { recursive: true, force: true }); mkdirSync(FRAMES, { recursive: true });
  const t0 = Date.now();
  await browser.close();
  // 워커마다 브라우저를 따로 띄운다 (캡처가 브라우저 프로세스에서 직렬화되기 때문)
  await Promise.all([...Array(WORKERS)].map(async (_, w) => {
    const br = await launch();
    const page = await open(br);
    const a = Math.floor(TOTAL * w / WORKERS), b = Math.floor(TOTAL * (w + 1) / WORKERS);
    for (let i = a; i < b; i++) {
      await page.evaluate(t => window.__seek(t), i / FPS);
      await page.screenshot({ path: path.join(FRAMES, `f_${String(i).padStart(5, '0')}.jpg`), type: 'jpeg', quality: 97 });
      if (w === 0 && (i - a) % 120 === 0) console.log(`  ${Math.round((i - a) / (b - a) * 100)}%  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    }
    await br.close();
  }));
  console.log(`captured ${TOTAL} frames in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  const audio = path.join(OUT, 'track.wav');
  const out = path.join(OUT, 'seoul-online-school-2026.mp4');
  const ff = ['-v', 'error', '-y', '-framerate', String(FPS), '-i', path.join(FRAMES, 'f_%05d.jpg')];
  if (existsSync(audio)) ff.push('-i', audio, '-c:a', 'aac', '-b:a', '192k', '-shortest');
  ff.push('-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
    '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-movflags', '+faststart', out);
  const r = spawnSync('ffmpeg', ff, { stdio: 'inherit' });
  if (r.status !== 0) process.exit(r.status ?? 1);
  console.log('wrote', out);
}
