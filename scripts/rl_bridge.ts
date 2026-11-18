import fs from 'node:fs';
import readline from 'node:readline';
import {
  AgentAction,
  CONT_ACTION_DIM,
  ControlledTeamsMode,
  HeadlessBasketballEnv,
  NUM_DISCRETE_ACTIONS,
  NeuralPolicy,
  OBS_DIM,
} from '../src/game/rl';
import { MatchModifierId } from '../src/game/types';

interface InitCmd {
  cmd: 'init';
  numEnvs?: number;
  controlledTeams?: ControlledTeamsMode;
  frameSkip?: number;
  quarterDuration?: number;
  maxEpisodeSteps?: number;
  randomizeRosters?: boolean;
  randomizeTactics?: boolean;
  modifierProb?: number;
  iqPerceptionNoise?: boolean;
  modifiers?: MatchModifierId[];
  seed?: number;
}

interface ResetCmd {
  cmd: 'reset';
  seed?: number;
}

interface StepCmd {
  cmd: 'step';
  discB64: string;
  moveB64: string;
}

interface CurriculumCmd {
  cmd: 'set_curriculum';
  modifiers?: MatchModifierId[];
  quarterDuration?: number;
  randomizeTactics?: boolean;
  modifierProb?: number;
  iqPerceptionNoise?: boolean;
}

interface SetOpponentsCmd {
  cmd: 'set_opponents';
  /** Для каждой среды: null = встроенный эвристический ИИ, строка = путь к JSON весов NeuralPolicy */
  envPaths: (string | null)[];
}

interface CloseCmd {
  cmd: 'close';
}

type BridgeCommand = InitCmd | ResetCmd | StepCmd | CurriculumCmd | SetOpponentsCmd | CloseCmd;

const policyCache = new Map<string, { mtimeMs: number; policy: NeuralPolicy }>();

function loadCachedPolicy(filePath: string): NeuralPolicy {
  const stat = fs.statSync(filePath);
  const hit = policyCache.get(filePath);
  if (hit && hit.mtimeMs === stat.mtimeMs) {
    return hit.policy;
  }
  const compiled = new NeuralPolicy(JSON.parse(fs.readFileSync(filePath, 'utf8')));
  if (policyCache.size >= 14 && !policyCache.has(filePath)) {
    const oldestKey = policyCache.keys().next().value;
    if (oldestKey !== undefined) policyCache.delete(oldestKey);
  }
  policyCache.set(filePath, { mtimeMs: stat.mtimeMs, policy: compiled });
  return compiled;
}

function toBase64(view: ArrayBufferView): string {
  return Buffer.from(view.buffer, view.byteOffset, view.byteLength).toString('base64');
}

let envs: HeadlessBasketballEnv[] = [];
let numAgentsPerEnv = 10;
let totalAgents = 0;

let batchObs = new Float32Array(0);
let batchMasks = new Uint8Array(0);
let batchRewards = new Float32Array(0);
let batchDones = new Uint8Array(0);
let envActionsScratch: AgentAction[][] = [];

function writeJsonLine(payload: unknown): void {
  process.stdout.write(JSON.stringify(payload) + '\n');
}

function handleCommand(msg: BridgeCommand): void {
  if (msg.cmd === 'init') {
    const count = Math.max(1, msg.numEnvs ?? 8);
    const baseSeed = (msg.seed ?? 1000) >>> 0;
    envs = Array.from({ length: count }, (_, idx) => {
      return new HeadlessBasketballEnv({
        controlledTeams: msg.controlledTeams ?? 'both',
        frameSkip: msg.frameSkip ?? 4,
        quarterDuration: msg.quarterDuration ?? 45,
        maxEpisodeSteps: msg.maxEpisodeSteps ?? 4500,
        randomizeRosters: msg.randomizeRosters ?? true,
        randomizeTactics: msg.randomizeTactics ?? true,
        modifierProb: msg.modifierProb ?? 0,
        iqPerceptionNoise: msg.iqPerceptionNoise ?? false,
        modifiers: msg.modifiers ?? [],
        seed: baseSeed + idx * 9973,
      });
    });

    numAgentsPerEnv = envs[0].numAgents;
    totalAgents = count * numAgentsPerEnv;
    batchObs = new Float32Array(totalAgents * OBS_DIM);
    batchMasks = new Uint8Array(totalAgents * NUM_DISCRETE_ACTIONS);
    batchRewards = new Float32Array(totalAgents);
    batchDones = new Uint8Array(totalAgents);
    envActionsScratch = Array.from({ length: count }, () =>
      Array.from({ length: numAgentsPerEnv }, () => ({ discrete: 0, moveX: 0, moveY: 0 }))
    );

    writeJsonLine({
      ok: true,
      numEnvs: count,
      numAgentsPerEnv,
      totalAgents,
      obsDim: OBS_DIM,
      numDiscrete: NUM_DISCRETE_ACTIONS,
      contDim: CONT_ACTION_DIM,
    });
    return;
  }

  if (msg.cmd === 'reset') {
    const baseSeed = msg.seed;
    for (let e = 0; e < envs.length; e++) {
      const res = envs[e].reset(baseSeed !== undefined ? baseSeed + e * 9973 : undefined);
      batchObs.set(res.obs, e * numAgentsPerEnv * OBS_DIM);
      batchMasks.set(res.masks, e * numAgentsPerEnv * NUM_DISCRETE_ACTIONS);
    }
    writeJsonLine({
      obsB64: toBase64(batchObs),
      masksB64: toBase64(batchMasks),
    });
    return;
  }

  if (msg.cmd === 'step') {
    const discBuf = Buffer.from(msg.discB64, 'base64');
    const moveBuf = Buffer.from(msg.moveB64, 'base64');

    const episodes: Array<{
      envIdx: number;
      score: [number, number];
      winner: 0 | 1 | null;
      stepCount: number;
      violations: [number, number];
    }> = [];

    for (let e = 0; e < envs.length; e++) {
      const baseAgent = e * numAgentsPerEnv;
      const acts = envActionsScratch[e];
      for (let a = 0; a < numAgentsPerEnv; a++) {
        const g = baseAgent + a;
        acts[a].discrete = discBuf[g] ?? 0;
        const byteOff = g * 8;
        if (byteOff + 8 <= moveBuf.length) {
          acts[a].moveX = moveBuf.readFloatLE(byteOff);
          acts[a].moveY = moveBuf.readFloatLE(byteOff + 4);
        } else {
          acts[a].moveX = 0;
          acts[a].moveY = 0;
        }
      }

      const stepRes = envs[e].step(acts);
      batchRewards.set(stepRes.rewards, baseAgent);

      if (stepRes.done) {
        for (let a = 0; a < numAgentsPerEnv; a++) {
          batchDones[baseAgent + a] = 1;
        }
        episodes.push({
          envIdx: e,
          score: stepRes.info.score,
          winner: stepRes.info.winner,
          stepCount: stepRes.info.stepCount,
          violations: stepRes.info.violations,
        });
        // Auto-reset завершённого матча: возвращаем стартовые obs/masks нового эпизода
        const rst = envs[e].reset();
        batchObs.set(rst.obs, baseAgent * OBS_DIM);
        batchMasks.set(rst.masks, baseAgent * NUM_DISCRETE_ACTIONS);
      } else {
        for (let a = 0; a < numAgentsPerEnv; a++) {
          batchDones[baseAgent + a] = 0;
        }
        batchObs.set(stepRes.obs, baseAgent * OBS_DIM);
        batchMasks.set(stepRes.masks, baseAgent * NUM_DISCRETE_ACTIONS);
      }
    }

    writeJsonLine({
      obsB64: toBase64(batchObs),
      masksB64: toBase64(batchMasks),
      rewardsB64: toBase64(batchRewards),
      donesB64: toBase64(batchDones),
      episodes,
    });
    return;
  }

  if (msg.cmd === 'set_curriculum') {
    for (const env of envs) {
      env.setCurriculum({
        modifiers: msg.modifiers,
        quarterDuration: msg.quarterDuration,
        randomizeTactics: msg.randomizeTactics,
        modifierProb: msg.modifierProb,
        iqPerceptionNoise: msg.iqPerceptionNoise,
      });
    }
    writeJsonLine({ ok: true });
    return;
  }

  if (msg.cmd === 'set_opponents') {
    for (let e = 0; e < envs.length; e++) {
      const p = msg.envPaths[e] ?? null;
      envs[e].setOpponentPolicy(p ? loadCachedPolicy(p) : null);
    }
    writeJsonLine({ ok: true });
    return;
  }

  if (msg.cmd === 'close') {
    writeJsonLine({ ok: true });
    process.exit(0);
  }
}

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

rl.on('line', (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  try {
    const parsed = JSON.parse(trimmed) as BridgeCommand;
    handleCommand(parsed);
  } catch (err) {
    writeJsonLine({ error: String(err) });
  }
});
