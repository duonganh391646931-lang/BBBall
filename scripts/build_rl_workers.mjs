/**
 * Собирает RL-скрипты (rl_bridge, rl_datagen, rl_eval_game) в лёгкие автономные ESM-бандлы
 * через esbuild, чтобы запускать воркеры чистым `node --max-old-space-size=256` без накладных
 * расходов tsx (~60-80 МБ на воркер вместо ~320 МБ и старт за 30 мс).
 */
import fs from 'node:fs';
import path from 'node:path';
import { build } from 'esbuild';

const root = process.cwd();
const outDir = path.join(root, 'training', '.dist');
fs.mkdirSync(outDir, { recursive: true });

const entries = [
  ['scripts/rl_bridge.ts', 'rl_bridge.mjs'],
  ['scripts/rl_datagen.ts', 'rl_datagen.mjs'],
  ['scripts/rl_eval_game.ts', 'rl_eval_game.mjs'],
];

for (const [src, dst] of entries) {
  await build({
    entryPoints: [path.join(root, src)],
    outfile: path.join(outDir, dst),
    bundle: true,
    platform: 'node',
    format: 'esm',
    target: 'node22',
    logLevel: 'warning',
  });
}
console.log(`[build_rl_workers] Собрано ${entries.length} ESM-воркеров в ${outDir}`);
