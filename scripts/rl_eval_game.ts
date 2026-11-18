/**
 * Оценка нейросети РОВНО так, как она играет в игре: контроллер 'neural' движка sim.ts
 * (IQ-температура, инерционный шум восприятия, 15 Гц):
 *   1) против встроенного эвристического ИИ (--opponent=heuristic);
 *   2) против другой нейросетевой политики (--opponent=path/to/other.json), стороны меняются 50/50.
 *
 *   node training/.dist/rl_eval_game.mjs --weights=public/models/policy_latest.json --games=40 --seed=1
 * Печатает в stdout одну JSON-строку: {games, winRate, meanDiff, ci95, diffs}
 */
import fs from 'node:fs';
import path from 'node:path';
import { stepPlayerAI } from '../src/game/ai';
import { PlayerDecisionDriver, createMatch, stepMatch } from '../src/game/sim';
import { MatchModifierId, TeamTactic } from '../src/game/types';
import { applyAgentAction, computeActionMask } from '../src/game/rl/actions';
import { applyInertialIqPerceptionNoise, encodeObservation } from '../src/game/rl/encoder';
import { createSeededRng } from '../src/game/rl/env';
import {
  NeuralPolicy,
  computeIqTemperature,
  loadActiveNeuralPolicyFromJson,
} from '../src/game/rl/neuralPolicy';
import { AgentAction, NUM_DISCRETE_ACTIONS, OBS_DIM } from '../src/game/rl/schema';

function arg(name: string, fallback: string): string {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : fallback;
}

const ALL_TACTICS: readonly TeamTactic[] = ['balanced', 'blitz', 'perimeter', 'dirty'];
const ALL_MODIFIERS: readonly MatchModifierId[] = [
  'two_balls',
  'ice_court',
  'low_gravity',
  'paint_trampolines',
  'explosive_ball',
];

const weightsPath = path.resolve(process.cwd(), arg('weights', 'public/models/policy_latest.json'));
const games = Math.max(2, Number(arg('games', '40')));
const seed = Number(arg('seed', '1'));
const quarter = Number(arg('quarter', '45'));
const opponentArg = arg('opponent', 'heuristic');
const modProb = Math.max(0, Math.min(1, Number(arg('modProb', '0'))));

const mainJson = JSON.parse(fs.readFileSync(weightsPath, 'utf8'));
const policyA = loadActiveNeuralPolicyFromJson(mainJson);
const policyB: NeuralPolicy | null =
  opponentArg === 'heuristic'
    ? null
    : new NeuralPolicy(JSON.parse(fs.readFileSync(path.resolve(process.cwd(), opponentArg), 'utf8')));

const scratchObs = new Float32Array(OBS_DIM);
const scratchMask = new Uint8Array(NUM_DISCRETE_ACTIONS);
const scratchAct: AgentAction = { discrete: 0, moveX: 0, moveY: 0 };

const diffs: number[] = [];
const DT = 1 / 60;

for (let g = 0; g < games; g++) {
  const side = (g % 2) as 0 | 1;
  const rng = createSeededRng((seed * 1_000_003 + g * 7919) >>> 0);
  const tactics: [TeamTactic, TeamTactic] = [
    ALL_TACTICS[Math.floor(rng() * ALL_TACTICS.length)],
    ALL_TACTICS[Math.floor(rng() * ALL_TACTICS.length)],
  ];
  let modifiers: MatchModifierId[] = [];
  if (modProb > 0 && rng() < modProb) {
    modifiers = [ALL_MODIFIERS[Math.floor(rng() * ALL_MODIFIERS.length)]];
  }

  const state = createMatch({
    mode: 'exhibition',
    quarterDuration: quarter,
    modifiers,
    tactics,
    teamControllers:
      policyB === null
        ? side === 0
          ? ['neural', 'heuristic']
          : ['heuristic', 'neural']
        : ['external', 'external'],
    headless: true,
    rng,
  });

  const driver: PlayerDecisionDriver | undefined =
    policyB === null
      ? undefined
      : (st, p, dt, stepRng, callbacks) => {
          const pol = p.team === side ? policyA : policyB;
          if (!pol) {
            stepPlayerAI(st, p, dt, stepRng, callbacks);
            return;
          }
          if (p.decisionTimer > 1 / 15) p.decisionTimer = 1 / 15;
          p.decisionTimer -= dt;
          if (p.decisionTimer <= 0) {
            p.decisionTimer = 1 / 15;
            encodeObservation(st, p, scratchObs, 0);
            applyInertialIqPerceptionNoise(scratchObs, p, st, stepRng, 0);
            computeActionMask(st, p, scratchMask, 0);
            const iqTemp = computeIqTemperature(p.effectiveStats.iq);
            pol.act(scratchObs, scratchMask, false, stepRng, scratchAct, iqTemp);
            applyAgentAction(st, p, scratchAct, scratchMask, callbacks, 0, 1 / 15, stepRng);
          }
        };

  let guard = 0;
  while (!state.finished && guard < 60 * 60 * 30) {
    stepMatch(state, DT, rng, driver);
    guard++;
  }
  diffs.push(state.score[side] - state.score[1 - side]);
}

const n = diffs.length;
const mean = diffs.reduce((a, b) => a + b, 0) / n;
const sd = Math.sqrt(diffs.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, n - 1));
process.stdout.write(
  JSON.stringify({
    games: n,
    winRate: diffs.filter((d) => d > 0).length / n,
    meanDiff: mean,
    ci95: (1.96 * sd) / Math.sqrt(n),
    diffs,
  }) + '\n'
);
