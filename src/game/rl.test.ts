import test from 'node:test';
import assert from 'node:assert/strict';
import { COURT_LEFT, COURT_W, HOOP_RIGHT_X, HOOP_Y, MID_X } from './physics';
import {
  adjudicatePersonalFoul,
  createMatch,
  giveBallToPlayer,
  launchBallShot,
  startInboundPlay,
  stepMatch,
} from './sim';
import {
  CONT_ACTION_DIM,
  DiscreteAction,
  HeadlessBasketballEnv,
  MatchRewardTracker,
  NUM_DISCRETE_ACTIONS,
  NeuralPolicy,
  OBS_DIM,
  applyInertialIqPerceptionNoise,
  computeActionMask,
  computeIqTemperature,
  createSeededRng,
  createUntrainedPolicyWeights,
  encodeObservation,
  getActiveNeuralPolicy,
} from './rl';

test('encodeObservation: выдаёт ровно 244 конечных признака (включая 2-й мяч и взгляд судьи) и строго инвариантен к зеркальному отражению команд 0 и 1', () => {
  const rng = createSeededRng(101);
  const state0 = createMatch({
    mode: 'exhibition',
    quarterDuration: 45,
    tactics: ['blitz', 'perimeter'],
    modifiers: ['two_balls', 'low_gravity'],
    headless: true,
    rng,
  });

  // Выравниваем статы и черты парных слотов обеих команд, чтобы проверить чистую геометрическую симметрию
  const team0Players = state0.players
    .filter((p) => p.team === 0)
    .sort((a, b) => a.slotIndex - b.slotIndex);
  const team1Players = state0.players
    .filter((p) => p.team === 1)
    .sort((a, b) => a.slotIndex - b.slotIndex);

  for (let i = 0; i < 5; i++) {
    const p0 = team0Players[i];
    const p1 = team1Players[i];
    p0.x = 320 + i * 110;
    p0.y = 180 + i * 55;
    p0.vx = 45 - i * 12;
    p0.vy = -20 + i * 9;
    p0.heading = 0.42;
    p0.refSuspicion = 2;

    p1.x = 410 + i * 95;
    p1.y = 210 + i * 48;
    p1.vx = -30 + i * 10;
    p1.vy = 15 - i * 6;
    p1.heading = 2.5;
    p1.refSuspicion = 1;
  }

  giveBallToPlayer(state0, state0.balls[0], team0Players[0], rng, false);
  if (state0.balls[1]) {
    state0.balls[1].holderId = null;
    state0.balls[1].x = 640;
    state0.balls[1].y = 210;
    state0.balls[1].z = 55;
    state0.balls[1].vx = 120;
    state0.balls[1].vy = -35;
    state0.balls[1].targetHoop = 1;
  }
  state0.score = [14, 9];
  state0.teamFouls = [2, 4];
  state0.possessionTeam = 0;
  state0.frontcourtEstablished = true;
  state0.shotClock = 16.5;
  state0.referee.x = 740;
  state0.referee.y = 260;
  state0.referee.gazeAngle = 0.65;

  // Создаём зеркальную копию состояния, где команда 1 находится в точности в зеркальной ситуации команды 0
  const state1 = structuredClone(state0);
  const mTeam0 = state1.players
    .filter((p) => p.team === 0)
    .sort((a, b) => a.slotIndex - b.slotIndex);
  const mTeam1 = state1.players
    .filter((p) => p.team === 1)
    .sort((a, b) => a.slotIndex - b.slotIndex);

  for (let i = 0; i < 5; i++) {
    const src0 = team0Players[i];
    const src1 = team1Players[i];
    const dst1 = mTeam1[i];
    const dst0 = mTeam0[i];

    // Команда 1 в state1 получает зеркальные координаты/характеристики команды 0 из state0
    dst1.role = src0.role;
    dst1.traitIds = [...src0.traitIds];
    dst1.effectiveStats = { ...src0.effectiveStats };
    dst1.energy = src0.energy;
    dst1.fouls = src0.fouls;
    dst1.refSuspicion = src0.refSuspicion;
    dst1.hotStreak = src0.hotStreak;
    dst1.x = COURT_W - src0.x;
    dst1.y = src0.y;
    dst1.z = src0.z;
    dst1.vx = -src0.vx;
    dst1.vy = src0.vy;
    dst1.vz = src0.vz;
    dst1.heading = Math.PI - src0.heading;

    // Команда 0 в state1 получает зеркальные координаты/характеристики команды 1 из state0
    dst0.role = src1.role;
    dst0.traitIds = [...src1.traitIds];
    dst0.effectiveStats = { ...src1.effectiveStats };
    dst0.energy = src1.energy;
    dst0.fouls = src1.fouls;
    dst0.refSuspicion = src1.refSuspicion;
    dst0.hotStreak = src1.hotStreak;
    dst0.x = COURT_W - src1.x;
    dst0.y = src1.y;
    dst0.z = src1.z;
    dst0.vx = -src1.vx;
    dst0.vy = src1.vy;
    dst0.vz = src1.vz;
    dst0.heading = Math.PI - src1.heading;
  }

  giveBallToPlayer(state1, state1.balls[0], mTeam1[0], rng, false);
  state1.balls[0].x = COURT_W - state0.balls[0].x;
  state1.balls[0].y = state0.balls[0].y;
  state1.balls[0].vx = -state0.balls[0].vx;
  if (state1.balls[1] && state0.balls[1]) {
    state1.balls[1].holderId = null;
    state1.balls[1].x = COURT_W - state0.balls[1].x;
    state1.balls[1].y = state0.balls[1].y;
    state1.balls[1].z = state0.balls[1].z;
    state1.balls[1].vx = -state0.balls[1].vx;
    state1.balls[1].vy = state0.balls[1].vy;
    state1.balls[1].targetHoop = 0;
  }
  state1.score = [9, 14];
  state1.teamFouls = [4, 2];
  state1.possessionTeam = 1;
  state1.frontcourtEstablished = true;
  state1.shotClock = 16.5;
  state1.backcourtTimer = state0.backcourtTimer;
  state1.tactics = [state0.tactics[1], state0.tactics[0]];
  state1.referee.x = COURT_W - state0.referee.x;
  state1.referee.y = state0.referee.y;
  state1.referee.gazeAngle = Math.PI - state0.referee.gazeAngle;

  const obs0 = encodeObservation(state0, team0Players[0]);
  const obs1 = encodeObservation(state1, mTeam1[0]);

  assert.equal(OBS_DIM, 244);
  assert.equal(obs0.length, OBS_DIM);
  assert.equal(obs1.length, OBS_DIM);
  assert.equal(obs0[103], 1, 'Слот второго мяча обязан быть активен при модификаторе two_balls');

  for (let d = 0; d < OBS_DIM; d++) {
    assert.ok(Number.isFinite(obs0[d]), `obs0[${d}] должно быть конечным числом`);
    assert.ok(Math.abs(obs0[d]) <= 2.0, `obs0[${d}]=${obs0[d]} выходит за допустимый масштаб`);
    assert.ok(
      Math.abs(obs0[d] - obs1[d]) < 1e-5,
      `Нарушена зеркальная симметрия в признаке #${d}: team0=${obs0[d]} vs team1=${obs1[d]}`
    );
  }
});

test('computeActionMask (23 действия): гарантирует >= 1 допустимого действия, позиционные стойки и блокирует задержку 5с', () => {
  const rng = createSeededRng(202);
  const state = createMatch({ mode: 'exhibition', headless: true, rng });

  assert.equal(NUM_DISCRETE_ACTIONS, 23);

  // Срочный ввод мяча (осталось <= 3.6с из 5с): NOOP_MOVE маскируется в 0, передачи открыты
  startInboundPlay(state, 0, COURT_LEFT - 18, HOOP_Y, 'Аут', 24, rng, true);
  assert.ok(state.inbound);
  state.inbound.timer = 2.5;
  const passer = state.players.find((p) => p.id === state.inbound?.passerId)!;
  const inboundMask = computeActionMask(state, passer);

  assert.equal(inboundMask.length, NUM_DISCRETE_ACTIONS);
  assert.equal(inboundMask[DiscreteAction.NOOP_MOVE], 0);
  assert.equal(inboundMask[DiscreteAction.SHOOT_STANDARD], 0);
  const passCount =
    inboundMask[DiscreteAction.PASS_MATE_0] +
    inboundMask[DiscreteAction.PASS_MATE_1] +
    inboundMask[DiscreteAction.PASS_MATE_2] +
    inboundMask[DiscreteAction.PASS_MATE_3];
  assert.ok(passCount >= 1, 'При вводе мяча должна быть доступна хотя бы одна передача');

  // В нападении возле чужого кольца с высоким прыжком доступны бросок и данк, а партнёру без мяча — заслон SCREEN_SET
  state.inbound = null;
  passer.x = HOOP_RIGHT_X - 55;
  passer.y = HOOP_Y;
  passer.z = 0;
  passer.actionCooldown = 0;
  passer.effectiveStats.jump = 85;
  giveBallToPlayer(state, state.balls[0], passer, rng, false);
  const frontMask = computeActionMask(state, passer);
  assert.equal(frontMask[DiscreteAction.SHOOT_STANDARD], 1);
  assert.equal(frontMask[DiscreteAction.DUNK], 1);

  const screener = state.players.find((p) => p.team === 0 && p.id !== passer.id)!;
  screener.x = HOOP_RIGHT_X - 180;
  screener.y = HOOP_Y + 30;
  const offBallMask = computeActionMask(state, screener);
  assert.equal(offBallMask[DiscreteAction.SCREEN_SET], 1);

  // Защитнику внутри полукруга под кольцом (dDefHoop < 72) TAKE_CHARGE запрещён правилами FIBA, а снаружи — доступен
  const defender = state.players.find((p) => p.team === 1)!;
  defender.x = HOOP_RIGHT_X - 30;
  defender.y = HOOP_Y;
  defender.effectiveStats.deviance = 70;
  defender.dirtyTrickCd = 0;
  const restrictedDefMask = computeActionMask(state, defender);
  assert.equal(restrictedDefMask[DiscreteAction.TAKE_CHARGE], 0);
  assert.equal(restrictedDefMask[DiscreteAction.DIRTY_TRICK], 1);

  passer.x = HOOP_RIGHT_X - 135;
  defender.x = HOOP_RIGHT_X - 105; // dDefHoop = 105 >= 72, dCarrier = 30 <= 82
  const defMask = computeActionMask(state, defender);
  assert.equal(defMask[DiscreteAction.TAKE_CHARGE], 1);
  assert.equal(defMask[DiscreteAction.DIRTY_TRICK], 1);

  // Удалённый на скамейку штрафников игрок имеет ровно одно действие NOOP_MOVE
  passer.penaltyTimer = 8.0;
  const benchMask = computeActionMask(state, passer);
  const totalBenchAllowed = benchMask.reduce((acc, v) => acc + v, 0);
  assert.equal(totalBenchAllowed, 1);
  assert.equal(benchMask[DiscreteAction.NOOP_MOVE], 1);
});

test('NeuralPolicy + IQ-температура + инерционный шум: соблюдают маску и не создают покадрового дребезга', () => {
  assert.ok(computeIqTemperature(95) < 0.35, 'Высокий IQ обязан снижать температуру до гроссмейстерской');
  assert.ok(computeIqTemperature(15) > 1.35, 'Низкий IQ обязан расширять температуру выбора балбеса');

  const rng = createSeededRng(777);
  const state = createMatch({ mode: 'exhibition', headless: true, rng });
  const lowIqPlayer = state.players[0];
  lowIqPlayer.effectiveStats.iq = 18;
  state.quarterTimeRemaining = 40.0;

  const baseObsA = encodeObservation(state, lowIqPlayer);
  const baseObsB = new Float32Array(baseObsA);
  applyInertialIqPerceptionNoise(baseObsA, lowIqPlayer, state, rng);
  state.quarterTimeRemaining = 39.85; // Внутри того же 0.75с окна восприятия
  applyInertialIqPerceptionNoise(baseObsB, lowIqPlayer, state, rng);
  assert.deepEqual(
    Array.from(baseObsA),
    Array.from(baseObsB),
    'Инерционный шум низкого IQ не должен прыгать между соседними кадрами внутри 0.75с интервала'
  );

  const weights = createUntrainedPolicyWeights(192, 2, 777);
  const policy = new NeuralPolicy(weights);

  const obs = new Float32Array(OBS_DIM).fill(0.15);
  const strictMask = new Uint8Array(NUM_DISCRETE_ACTIONS);
  strictMask[DiscreteAction.PASS_MATE_2] = 1; // Единственное разрешённое действие

  const detAct = policy.act(obs, strictMask, true);
  assert.equal(detAct.discrete, DiscreteAction.PASS_MATE_2);
  assert.ok(detAct.moveX >= -1 && detAct.moveX <= 1);
  assert.ok(detAct.moveY >= -1 && detAct.moveY <= 1);

  for (let i = 0; i < 25; i++) {
    const stochAct = policy.act(obs, strictMask, false, rng, undefined, computeIqTemperature(15));
    assert.equal(stochAct.discrete, DiscreteAction.PASS_MATE_2);
  }

  // Сериализация и повторная загрузка JSON не меняют выход сети
  const roundtrip = new NeuralPolicy(JSON.parse(JSON.stringify(weights)));
  const fullMask = new Uint8Array(NUM_DISCRETE_ACTIONS).fill(1);
  const a1 = policy.act(obs, fullMask, true);
  const a2 = roundtrip.act(obs, fullMask, true);
  assert.equal(a1.discrete, a2.discrete);
  assert.ok(Math.abs(a1.moveX - a2.moveX) < 1e-6);
  assert.ok(Math.abs(a1.moveY - a2.moveY) < 1e-6);
  assert.equal(CONT_ACTION_DIM, 2);
});

test('HeadlessBasketballEnv и встроенная обученная политика: штатно отыгрывают полный матч 5x5 с очками и мыслями ИИ', () => {
  const activePolicy = getActiveNeuralPolicy();
  assert.ok(activePolicy instanceof NeuralPolicy);

  const envA = new HeadlessBasketballEnv({
    controlledTeams: 'both',
    frameSkip: 4,
    quarterDuration: 15,
    seed: 12345,
  });
  const envB = new HeadlessBasketballEnv({
    controlledTeams: 'both',
    frameSkip: 4,
    quarterDuration: 15,
    seed: 12345,
  });

  let curA = envA.reset(12345);
  let curB = envB.reset(12345);
  assert.deepEqual(Array.from(curA.obs), Array.from(curB.obs));
  assert.deepEqual(Array.from(curA.masks), Array.from(curB.masks));

  // Проверяем совпадение траекторий и валидность записанных действий учителя (Behavior Cloning)
  for (let s = 0; s < 30; s++) {
    const maskSnapshot = new Uint8Array(curA.masks);
    const stepA = envA.stepTeacher();
    const stepB = envB.stepTeacher();

    for (let i = 0; i < envA.numAgents; i++) {
      const act = stepA.teacherActions[i];
      assert.equal(
        maskSnapshot[i * NUM_DISCRETE_ACTIONS + act.discrete],
        1,
        `Действие учителя ${act.discrete} для агента #${i} обязано быть разрешено маской`
      );
      assert.ok(act.moveX >= -1 && act.moveX <= 1);
      assert.ok(act.moveY >= -1 && act.moveY <= 1);
      assert.ok(Number.isFinite(stepA.rewards[i]));
    }
    assert.deepEqual(Array.from(stepA.rewards), Array.from(stepB.rewards));
    curA = stepA;
    curB = stepB;
  }

  // Прогоняем полный тихий матч ['neural', 'neural'] на встроенных весах нейросети
  const rng = createSeededRng(5060);
  const neuralMatch = createMatch({
    mode: 'exhibition',
    quarterDuration: 20,
    teamControllers: ['neural', 'neural'],
    headless: true,
    rng,
  });
  const tracker = new MatchRewardTracker();
  tracker.reset(neuralMatch);

  let ticks = 0;
  let sawRussianThought = false;
  while (!neuralMatch.finished && ticks < 12_000) {
    stepMatch(neuralMatch, 1 / 60, rng);
    if (ticks % 4 === 0) {
      const rew = tracker.computeStepRewards(neuralMatch);
      for (let i = 0; i < rew.length; i++) {
        assert.ok(Number.isFinite(rew[i]));
      }
      if (!sawRussianThought) {
        sawRussianThought = neuralMatch.players.some((p) => p.onCourt && p.aiThought.length > 2);
      }
    }
    ticks++;
  }
  assert.ok(neuralMatch.finished, 'Матч с нейросетевым контроллером должен штатно завершиться');
  assert.notEqual(neuralMatch.score[0], neuralMatch.score[1], 'Овертайм обязан выявить победителя');
  assert.ok(neuralMatch.score[0] + neuralMatch.score[1] >= 2, 'Обученные команды должны набирать очки');
  assert.ok(sawRussianThought, 'Русские мысли «Мозга ИИ» должны обновляться в ходе нейросетевого матча');
});

test('Целостность RL-сигналов: фол при броске и ввод из аута не начисляют фантомные подборы и лишние промахи с игры', () => {
  const rng = createSeededRng(31415);
  const state = createMatch({ mode: 'exhibition', headless: true, rng });
  const shooter = state.players.find((p) => p.team === 0)!;
  const defender = state.players.find((p) => p.team === 1)!;
  const ball = state.balls[0];

  // 1. Бросок с фолом (не попавший сразу): попытка с игры откатывается, wasShotAttempt сбрасывается
  shooter.x = HOOP_RIGHT_X - 90;
  shooter.y = HOOP_Y;
  defender.x = shooter.x + 18;
  defender.y = shooter.y;
  giveBallToPlayer(state, ball, shooter, rng, false);
  launchBallShot(state, ball, shooter, 1, 'normal', rng, 'standard');
  assert.equal(shooter.fgAttempted, 1);
  assert.equal(ball.wasShotAttempt, true);

  adjudicatePersonalFoul(state, defender, shooter, 'shooting_2', rng);
  assert.equal(shooter.fgAttempted, 0, 'Промах под фолом не должен идти в промахи с игры');
  assert.equal(ball.wasShotAttempt, false, 'Мяч на штрафном броске не должен нести старый флаг броска с игры');

  // 2. Ввод мяча из-за боковой после ушедшего в аут броска не должен давать подбор распасовщику или адресату
  state.freeThrow = null;
  ball.wasShotAttempt = true;
  ball.lastTouchTeam = 0;
  ball.lastShooterId = shooter.id;
  const totalReboundsBefore = state.players.reduce((s, p) => s + p.rebounds, 0);
  startInboundPlay(state, 1, MID_X, 50, 'Аут за боковую', 24, rng, false);
  assert.equal(ball.wasShotAttempt, false);
  const totalReboundsAfter = state.players.reduce((s, p) => s + p.rebounds, 0);
  assert.equal(totalReboundsAfter, totalReboundsBefore);

  // 3. Черта «Магнит» не искривляет траекторию передач в воздухе (действует только на ничейный мяч)
  state.inbound = null;
  const magnetDefender = state.players.find((p) => p.team === 1)!;
  magnetDefender.traitIds = ['magnet'];
  magnetDefender.x = MID_X;
  magnetDefender.y = HOOP_Y + 60;

  ball.holderId = null;
  ball.shotType = 'pass';
  ball.x = MID_X;
  ball.y = HOOP_Y;
  ball.z = 45;
  ball.vx = 420;
  ball.vy = 0;
  ball.vz = 20;

  stepMatch(state, 1 / 60, rng, () => {});
  assert.equal(ball.vy, 0, 'Магнит не должен притягивать чужой пас в полёте по оси Y');
});
