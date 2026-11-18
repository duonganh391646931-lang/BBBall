import {
  AgentAction,
  HeadlessBasketballEnv,
  NUM_DISCRETE_ACTIONS,
  NeuralPolicy,
  OBS_DIM,
  createUntrainedPolicyWeights,
} from '../src/game/rl';

const stepsToRun = Number(process.argv[2] ?? '4000');

console.log('=== Бенчмарк безголовой среды БалбесБола (HeadlessBasketballEnv) ===');

// 1. Чистая скорость шага среды (как в воркерах rl_bridge.ts, когда нейросеть крутится на GPU RTX 5060)
const env = new HeadlessBasketballEnv({
  controlledTeams: 'both',
  frameSkip: 4,
  quarterDuration: 45,
  seed: 777,
});

const actions: AgentAction[] = Array.from({ length: env.numAgents }, (_, idx) => ({
  discrete: 0,
  moveX: idx < 5 ? 0.65 : -0.35,
  moveY: idx % 2 === 0 ? 0.25 : -0.25,
}));

let cur = env.reset(777);
let completedMatches = 0;

const tPure0 = performance.now();
for (let step = 0; step < stepsToRun; step++) {
  const next = env.step(actions);
  if (next.done) {
    completedMatches++;
    cur = env.reset();
  } else {
    cur = next;
  }
}
const pureSec = (performance.now() - tPure0) / 1000;
const pureEnvSps = stepsToRun / pureSec;
const purePhysicsTps = (stepsToRun * env.frameSkip) / pureSec;
const pureAgentSps = (stepsToRun * env.numAgents) / pureSec;

console.log(
  `[1] Чистая среда (1 CPU-воркер -> GPU):       ${Math.round(pureEnvSps).toLocaleString()} шаг/с | ` +
    `${Math.round(pureAgentSps).toLocaleString()} агент-шаг/с | ` +
    `${Math.round(purePhysicsTps).toLocaleString()} физ.тик/с (~${(purePhysicsTps / 60).toFixed(0)}x реалтайма)`
);
console.log(
  `    Оценка 8 CPU-воркеров (SubprocVecEnv):    ~${Math.round(pureAgentSps * 6.8).toLocaleString()} агент-шаг/с ` +
    `(завершено матчей: ${completedMatches})`
);

// 2. Замер браузерного zero-GC инференса NeuralPolicy (256x3 ResMLP на 10 игроков кадра 15 Гц)
const browserPolicy = new NeuralPolicy(createUntrainedPolicyWeights(256, 3, 2026));
const obsSlice = new Float32Array(OBS_DIM);
const maskSlice = new Uint8Array(NUM_DISCRETE_ACTIONS);
const inferFrames = 500;

const tInf0 = performance.now();
for (let f = 0; f < inferFrames; f++) {
  for (let a = 0; a < env.numAgents; a++) {
    obsSlice.set(cur.obs.subarray(a * OBS_DIM, (a + 1) * OBS_DIM));
    maskSlice.set(cur.masks.subarray(a * NUM_DISCRETE_ACTIONS, (a + 1) * NUM_DISCRETE_ACTIONS));
    browserPolicy.act(obsSlice, maskSlice, true, Math.random, actions[a]);
  }
}
const infMsPerDecisionTick = (performance.now() - tInf0) / inferFrames;
console.log(
  `[2] Браузерный инференс (256x3 ResMLP, 10 игр): ${infMsPerDecisionTick.toFixed(2)} мс на тик 15 Гц ` +
    `(${(infMsPerDecisionTick / 4).toFixed(2)} мс в среднем на кадр 60 FPS)`
);
