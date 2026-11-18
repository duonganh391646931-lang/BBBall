import fs from 'node:fs';
import path from 'node:path';
import {
  CONT_ACTION_DIM,
  DISCRETE_ACTION_NAMES,
  HeadlessBasketballEnv,
  NUM_DISCRETE_ACTIONS,
  OBS_DIM,
} from '../src/game/rl';

function parseArg(name: string, fallback: string): string {
  const prefix = `--${name}=`;
  const hit = process.argv.find((a) => a.startsWith(prefix));
  return hit ? hit.slice(prefix.length) : fallback;
}

const targetSamples = Math.max(1000, Number(parseArg('samples', '250000')));
const outDir = path.resolve(process.cwd(), parseArg('out', 'training/data/bc_dataset'));
const frameSkip = Math.max(1, Number(parseArg('frameSkip', '4')));
const quarterDuration = Math.max(15, Number(parseArg('quarterDuration', '45')));
const seed = Number(parseArg('seed', '202601'));

fs.mkdirSync(outDir, { recursive: true });

const obsPath = path.join(outDir, 'obs.f32');
const masksPath = path.join(outDir, 'masks.u8');
const discPath = path.join(outDir, 'disc.u8');
const movePath = path.join(outDir, 'move.f32');
const rewardsPath = path.join(outDir, 'rewards.f32');
const metaPath = path.join(outDir, 'meta.json');

const obsFd = fs.openSync(obsPath, 'w');
const masksFd = fs.openSync(masksPath, 'w');
const discFd = fs.openSync(discPath, 'w');
const moveFd = fs.openSync(movePath, 'w');
const rewardsFd = fs.openSync(rewardsPath, 'w');

const env = new HeadlessBasketballEnv({
  controlledTeams: 'both',
  frameSkip,
  quarterDuration,
  randomizeRosters: true,
  randomizeTactics: true,
  seed,
});

const numAgents = env.numAgents; // 10
const preObsCopy = new Float32Array(numAgents * OBS_DIM);
const preMasksCopy = new Uint8Array(numAgents * NUM_DISCRETE_ACTIONS);
const stepDiscBuf = new Uint8Array(numAgents);
const stepMoveBuf = new Float32Array(numAgents * CONT_ACTION_DIM);
const actionHist = new Array<number>(NUM_DISCRETE_ACTIONS).fill(0);

let collected = 0;
let matchesCompleted = 0;
let totalScoreSum = 0;
const t0 = performance.now();

let cur = env.reset(seed);

while (collected < targetSamples) {
  preObsCopy.set(cur.obs);
  preMasksCopy.set(cur.masks);

  const next = env.stepTeacher();
  const remaining = targetSamples - collected;
  const takeAgents = Math.min(numAgents, remaining);

  for (let i = 0; i < takeAgents; i++) {
    const act = next.teacherActions[i];
    stepDiscBuf[i] = act.discrete;
    stepMoveBuf[i * 2] = act.moveX;
    stepMoveBuf[i * 2 + 1] = act.moveY;
    actionHist[act.discrete]++;
  }

  fs.writeSync(
    obsFd,
    Buffer.from(preObsCopy.buffer, preObsCopy.byteOffset, takeAgents * OBS_DIM * 4)
  );
  fs.writeSync(
    masksFd,
    Buffer.from(preMasksCopy.buffer, preMasksCopy.byteOffset, takeAgents * NUM_DISCRETE_ACTIONS)
  );
  fs.writeSync(discFd, Buffer.from(stepDiscBuf.buffer, stepDiscBuf.byteOffset, takeAgents));
  fs.writeSync(
    moveFd,
    Buffer.from(stepMoveBuf.buffer, stepMoveBuf.byteOffset, takeAgents * CONT_ACTION_DIM * 4)
  );
  fs.writeSync(
    rewardsFd,
    Buffer.from(next.rewards.buffer, next.rewards.byteOffset, takeAgents * 4)
  );

  collected += takeAgents;

  if (next.done) {
    matchesCompleted++;
    totalScoreSum += next.info.score[0] + next.info.score[1];
    cur = env.reset();
  } else {
    cur = next;
  }

  if (collected % 50000 === 0 || collected === targetSamples) {
    const elapsedSec = (performance.now() - t0) / 1000;
    const sps = Math.round(collected / Math.max(1e-3, elapsedSec));
    console.log(
      `[rl:datagen] ${collected.toLocaleString()} / ${targetSamples.toLocaleString()} примеров (${sps.toLocaleString()} агент-шагов/с, матчей: ${matchesCompleted})`
    );
  }
}

fs.closeSync(obsFd);
fs.closeSync(masksFd);
fs.closeSync(discFd);
fs.closeSync(moveFd);
fs.closeSync(rewardsFd);

const elapsedTotal = (performance.now() - t0) / 1000;
const distNamed: Record<string, number> = {};
DISCRETE_ACTION_NAMES.forEach((name, idx) => {
  distNamed[name] = actionHist[idx];
});

const meta = {
  formatVersion: 1,
  samples: collected,
  obsDim: OBS_DIM,
  numDiscreteActions: NUM_DISCRETE_ACTIONS,
  contActionDim: CONT_ACTION_DIM,
  frameSkip,
  quarterDuration,
  matchesCompleted,
  avgTotalMatchPoints: matchesCompleted > 0 ? Number((totalScoreSum / matchesCompleted).toFixed(2)) : 0,
  elapsedSec: Number(elapsedTotal.toFixed(2)),
  actionHistogram: distNamed,
  files: {
    obs: 'obs.f32',
    masks: 'masks.u8',
    disc: 'disc.u8',
    move: 'move.f32',
    rewards: 'rewards.f32',
  },
};

fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2), 'utf8');
console.log(`[rl:datagen] Готово! Датасет сохранён в ${outDir} за ${elapsedTotal.toFixed(2)}с.`);
