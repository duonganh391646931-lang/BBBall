import test from 'node:test';
import assert from 'node:assert/strict';
import {
  TRAIT_BY_ID,
  computeEffectiveStats,
  generateRandomPlayer,
  isLastHopeBuffActive,
  pickWeightedTrait,
} from './traits';
import {
  COURT_LEFT,
  COURT_RIGHT,
  HOOP_LEFT_X,
  HOOP_RIGHT_X,
  HOOP_Y,
  MID_X,
  RIM_Z,
  THREE_PT_RADIUS,
  computePlayerHorizontalInterceptRadius,
  computePlayerVerticalReach,
  computeRefereeDetectionChance,
} from './physics';
import {
  adjudicatePersonalFoul,
  callTeamTimeout,
  createMatch,
  giveBallToPlayer,
  launchBallShot,
  requestPlayerSubstitution,
  startInboundPlay,
  stepMatch,
  substitutePlayer,
  triggerCoachPlayCall,
  updatePlayerTraitsAndStatsInSandbox,
} from './sim';
import { chooseAdaptiveAiTactic, evaluatePassLaneRisk, stepPlayerAI } from './ai';

function createSeededRng(seed = 42): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

test('«Кривоногий и хромой» [Обычная] снижает все 9 характеристик ровно на 20%', () => {
  const trait = TRAIT_BY_ID['crooked_lame'];
  assert.equal(trait.rarity, 'common');
  const base = {
    speed: 50,
    shooting: 60,
    stamina: 70,
    steal: 80,
    jump: 50,
    teamwork: 50,
    dribble: 60,
    iq: 50,
    deviance: 40,
  };
  const eff = computeEffectiveStats(base, ['crooked_lame']);
  assert.equal(eff.speed, 40);
  assert.equal(eff.shooting, 48);
  assert.equal(eff.stamina, 56);
  assert.equal(eff.steal, 64);
  assert.equal(eff.jump, 40);
  assert.equal(eff.teamwork, 40);
  assert.equal(eff.dribble, 48);
  assert.equal(eff.iq, 40);
  assert.equal(eff.deviance, 32);
});

test('Обычные черты выпадают значительно чаще уникальных, а «Руки-загребуки» и «Эгоисто» присутствуют', () => {
  assert.equal(TRAIT_BY_ID['grabby_hands'].rarity, 'common');
  assert.equal(TRAIT_BY_ID['egoisto'].rarity, 'unique');
  assert.equal(TRAIT_BY_ID['crook'].rarity, 'rare');
  assert.equal(TRAIT_BY_ID['stealthy'].rarity, 'rare');

  const rng = createSeededRng(1337);
  let commonCount = 0;
  let rareCount = 0;
  let uniqueCount = 0;
  for (let i = 0; i < 2000; i++) {
    const t = pickWeightedTrait(rng, new Set());
    if (t.rarity === 'common') commonCount++;
    else if (t.rarity === 'rare') rareCount++;
    else uniqueCount++;
  }
  assert.ok(commonCount > uniqueCount * 5, `Common (${commonCount}) >> Unique (${uniqueCount})`);
  assert.ok(commonCount > rareCount * 2, `Common (${commonCount}) > Rare (${rareCount})`);
});

test('«Месси» [Уникальная] срабатывает с шансом 7% на своей половине (автогол в ближайшее кольцо) и 15% на чужой', () => {
  const rng = createSeededRng(99);
  const match = createMatch({ rng });
  const messiPlayer = match.players[0]; // team 0 (защищает ЛЕВОЕ кольцо 0, атакует ПРАВОЕ кольцо 1)
  updatePlayerTraitsAndStatsInSandbox(match, messiPlayer.id, ['messi']);

  // 1) Ставим Месси на СВОЮ половину (x = 320 < MID_X): при roll = 0.05 (< 7%) он бьёт в СВОЁ ближайшее кольцо 0!
  messiPlayer.x = 320;
  messiPlayer.y = HOOP_Y;
  giveBallToPlayer(match, match.balls[0], messiPlayer, () => 0.05, true);
  assert.equal(messiPlayer.aiAction, 'messi_windup');
  assert.equal(messiPlayer.windupTargetHoop, 0, 'На своей половине ближайшее кольцо — своё (0)');

  // При roll = 0.10 (> 7% на своей половине) инстинкт Месси НЕ срабатывает
  messiPlayer.windupTimer = 0;
  messiPlayer.aiAction = 'drive';
  giveBallToPlayer(match, match.balls[0], messiPlayer, () => 0.1, true);
  assert.notEqual(messiPlayer.aiAction, 'messi_windup');

  // 2) Ставим Месси на ЧУЖУЮ половину (x = 760 > MID_X): при roll = 0.12 (< 15%) он бьёт во вражеское кольцо 1!
  messiPlayer.x = 760;
  giveBallToPlayer(match, match.balls[0], messiPlayer, () => 0.12, true);
  assert.equal(messiPlayer.aiAction, 'messi_windup');
  assert.equal(messiPlayer.windupTargetHoop, 1, 'На чужой половине ближайшее кольцо — вражеское (1)');
});

test('«Кузнечик» прыгает высоко и перехватывает высоко летящий навес от «Месси» на пути полёта', () => {
  const rng = createSeededRng(2026);
  const match = createMatch({ rng });

  // Изолируем остальных игроков в стороне, чтобы проверить чистую дуэль Месси и Кузнечика
  for (const p of match.players) {
    p.x = 200;
    p.y = 95;
    p.traitIds = [];
  }

  const messi = match.players[0]; // team 0
  const grasshopper = match.players[5]; // team 1 (защищает правое кольцо 1)
  updatePlayerTraitsAndStatsInSandbox(match, messi.id, ['messi']);
  updatePlayerTraitsAndStatsInSandbox(match, grasshopper.id, ['grasshopper']);

  // Месси бьёт от центра поля по высокой дуге (peakZ = 232) в правое кольцо
  messi.x = MID_X + 20;
  messi.y = HOOP_Y;
  grasshopper.x = (messi.x + HOOP_RIGHT_X) * 0.5; // стоит ровно на пути полёта мяча!
  grasshopper.y = HOOP_Y;

  const ball = match.balls[0];
  launchBallShot(match, ball, messi, 1, 'messi', rng);

  let interceptedByGrasshopper = false;
  let maxGrasshopperZ = 0;

  for (let step = 0; step < 90; step++) {
    stepMatch(match, 1 / 60, rng);
    if (grasshopper.z > maxGrasshopperZ) maxGrasshopperZ = grasshopper.z;
    if (ball.holderId === grasshopper.id) {
      interceptedByGrasshopper = true;
      break;
    }
  }

  assert.ok(maxGrasshopperZ > 130, `Кузнечик подпрыгнул очень высоко: z=${maxGrasshopperZ}`);
  assert.equal(interceptedByGrasshopper, true, 'Кузнечик перехватил высокий удар Месси в полёте!');
  assert.ok(grasshopper.superJumpCd > 0, 'У Кузнечика запустился кулдаун пружин 3.5с');
});

test('«Руки-загребуки» [Обычная] всегда забирает стартовый мяч на вбрасывании', () => {
  const rng = createSeededRng(777);
  const match = createMatch({ rng });
  for (const p of match.players) {
    updatePlayerTraitsAndStatsInSandbox(match, p.id, []);
  }

  const grabber = match.players[7]; // игрок команды 1
  updatePlayerTraitsAndStatsInSandbox(match, grabber.id, ['grabby_hands']);

  for (let i = 0; i < 45; i++) {
    stepMatch(match, 1 / 60, rng);
    if (match.balls[0].holderId !== null) break;
  }

  assert.equal(match.balls[0].holderId, grabber.id);
});

test('Судья видит нарушение с 80% спереди и 50% за спиной, учитывает «Незаметного» и подозрительность', () => {
  const rng = createSeededRng(11);
  const p = generateRandomPlayer(0, 0, new Set(), rng);
  p.traitIds = [];
  p.refSuspicion = 0;

  const referee = {
    x: 500,
    y: 590,
    vx: 0,
    vy: 0,
    gazeAngle: -Math.PI / 2, // смотрит вверх (на площадку, y < 590)
    whistleTimer: 0,
    lastCallText: null,
    callsMade: 0,
    missedFouls: 0,
  };

  // Игрок перед глазами судьи (y = 400) -> 80%
  p.x = 500;
  p.y = 400;
  assert.ok(Math.abs(computeRefereeDetectionChance(referee, p).chance - 0.8) < 1e-6);

  // Судья отвернулся вниз (gazeAngle = +PI/2), игрок за его спиной -> 50%
  referee.gazeAngle = Math.PI / 2;
  assert.ok(Math.abs(computeRefereeDetectionChance(referee, p).chance - 0.5) < 1e-6);

  // Черта «Незаметный» спереди судьи: 80% - 45% = 35%
  referee.gazeAngle = -Math.PI / 2;
  p.traitIds = ['stealthy'];
  assert.ok(Math.abs(computeRefereeDetectionChance(referee, p).chance - 0.35) < 1e-6);

  // +2 единицы подозрительности судьи добавляют +20% (35% + 20% = 55%)
  p.refSuspicion = 2;
  assert.ok(Math.abs(computeRefereeDetectionChance(referee, p).chance - 0.55) < 1e-6);
});

test('Полноформатный матч 5 на 5 с модификатором «Два мяча» и «Гололёд» стабильно симулируется', () => {
  const rng = createSeededRng(555);
  const match = createMatch({
    quarterDuration: 12,
    modifiers: ['two_balls', 'ice_court'],
    rng,
  });

  assert.equal(match.players.length, 10);
  assert.equal(match.balls.length, 2);

  // Прокручиваем весь матч на высокой скорости до финальной сирены
  let steps = 0;
  while (!match.finished && steps < 6000) {
    stepMatch(match, 0.04, rng);
    steps++;
  }

  assert.equal(match.finished, true);
  assert.ok(match.quarter >= 4);
  assert.ok(match.summary !== null);
  assert.ok(match.score[0] + match.score[1] > 0, 'Команды ИИ набрали очки за матч');
});

test('Умный ИИ проверяет перекрытие линии паса и выбирает открытого адресата вместо защитника', () => {
  const rng = createSeededRng(101);
  const match = createMatch({ rng });
  for (const p of match.players) {
    updatePlayerTraitsAndStatsInSandbox(match, p.id, [], {
      speed: 65,
      shooting: 55,
      stamina: 80,
      steal: 75,
      jump: 60,
      teamwork: 92,
      dribble: 70,
      iq: 92,
      deviance: 5,
    });
    p.x = 180;
    p.y = 100;
  }

  const passer = match.players[0]; // PG команды 0
  const guardedMate = match.players[1]; // на прямой линии за спиной перехватчика
  const openMate = match.players[2]; // чистый коридор под кольцом
  const interceptor = match.players[5]; // защитник команды 1 прямо на линии паса к guardedMate

  passer.x = 650;
  passer.y = HOOP_Y;
  guardedMate.x = 920;
  guardedMate.y = HOOP_Y;
  interceptor.x = 785; // стоит ровно посередине линии (650,320)->(920,320)
  interceptor.y = HOOP_Y;

  openMate.x = 930;
  openMate.y = HOOP_Y + 135; // чистая линия передачи!

  const blockedRisk = evaluatePassLaneRisk(passer, guardedMate, [interceptor]);
  const clearRisk = evaluatePassLaneRisk(passer, openMate, [interceptor]);
  assert.ok(blockedRisk > 55, `Высокий штраф за перекрытую линию паса: ${blockedRisk}`);
  assert.equal(clearRisk, 0, 'Открытый коридор не имеет штрафа перехвата');

  giveBallToPlayer(match, match.balls[0], passer, rng, false);
  passer.decisionTimer = 0;
  stepMatch(match, 1 / 60, rng);

  assert.equal(match.balls[0].shotType, 'pass');
  assert.equal(match.balls[0].passReceiverId, openMate.id, 'Пасующий выбрал открытого партнёра!');
});

test('ИИ ставит заслон (Пик-н-ролл) в позиционной атаке и борется за отскок при броске', () => {
  const rng = createSeededRng(202);
  const match = createMatch({ rng });
  for (const p of match.players) {
    updatePlayerTraitsAndStatsInSandbox(match, p.id, [], {
      speed: 65,
      shooting: 60,
      stamina: 80,
      steal: 55,
      jump: 70,
      teamwork: 75,
      dribble: 70,
      iq: 78,
      deviance: 5,
    });
  }

  const pg = match.players[0]; // PG команды 0 с мячом на дуге
  const center = match.players[4]; // C команды 0 без мяча
  const onBallDef = match.players[5]; // опекун команды 1 перед PG

  pg.x = 760;
  pg.y = HOOP_Y;
  onBallDef.x = 815;
  onBallDef.y = HOOP_Y;
  center.x = 860;
  center.y = HOOP_Y + 70;

  giveBallToPlayer(match, match.balls[0], pg, rng, false);
  match.possessionTimer = 5.0; // уже позиционная атака (после фазы быстрого отрыва)
  pg.decisionTimer = 1.0; // ведёт мяч

  stepMatch(match, 1 / 60, rng);
  assert.equal(center.aiAction, 'screen_set', 'Центровой ставит заслон опекуну разыгрывающего');

  // Теперь запускаем бросок в кольцо и проверяем переход в борьбу за подбор (rebound_position) и страховку тыла (safety_back)
  const sf = match.players[2];
  launchBallShot(match, match.balls[0], sf, 1, 'normal', rng);
  stepMatch(match, 1 / 60, rng);

  assert.equal(center.aiAction, 'rebound_position', 'Центровой идёт на бокс-аут под отскок');
  assert.equal(pg.aiAction, 'safety_back', 'Умный разыгрывающий страхует тыл от быстрого отрыва');
});

test('ИИ совершает бросок под сирену в концовке четверти и бережётся при 4 фолах перед судьёй', () => {
  const rng = createSeededRng(303);
  const match = createMatch({ rng });
  for (const p of match.players) {
    updatePlayerTraitsAndStatsInSandbox(match, p.id, [], {
      speed: 65,
      shooting: 65,
      stamina: 80,
      steal: 55,
      jump: 60,
      teamwork: 60,
      dribble: 70,
      iq: 75,
      deviance: 72,
    });
  }

  // 1) Проверка броска под сирену четверти (< 2.4с)
  const shooter = match.players[0];
  shooter.x = 720;
  shooter.y = HOOP_Y;
  giveBallToPlayer(match, match.balls[0], shooter, rng, false);
  match.quarterTimeRemaining = 1.6;
  shooter.actionCooldown = 0;

  stepMatch(match, 1 / 60, rng);
  assert.equal(shooter.aiAction, 'shoot_windup', 'Игрок выбрасывает мяч под сирену четверти');

  // 2) Проверка дисциплины при 4 фолах на глазах у судьи
  const smartFoulTrouble = match.players[6]; // команда 1
  const nearbyEnemy = match.players[1];
  smartFoulTrouble.fouls = 4;
  smartFoulTrouble.dirtyTrickCd = 0;
  smartFoulTrouble.x = 590;
  smartFoulTrouble.y = 420;
  nearbyEnemy.x = 605;
  nearbyEnemy.y = 420;
  match.referee.x = 590;
  match.referee.y = 590;
  match.referee.gazeAngle = -Math.PI / 2; // судья смотрит прямо на smartFoulTrouble!

  for (let i = 0; i < 30; i++) {
    stepMatch(match, 1 / 60, () => 0.001); // минимальный roll спровоцировал бы фол без защиты от 5-го фола
  }
  assert.equal(
    smartFoulTrouble.dirtyPlays,
    0,
    'Умный игрок с 4 фолами воздерживается от грязного приёма прямо перед носом судьи'
  );

  // 3) Проверка адаптации тактики ИИ-тренером в концовке Q4 при отставании
  match.quarter = 4;
  match.quarterTimeRemaining = 15;
  match.score = [48, 41]; // команда 1 проигрывает -7
  const chosenTactic = chooseAdaptiveAiTactic(match, 1);
  assert.ok(
    chosenTactic === 'perimeter' || chosenTactic === 'blitz',
    `ИИ-тренер включает штурмовую тактику (${chosenTactic}) в концовке`
  );
});

test('Правило аута и 5 секунд на ввод мяча из-за линии работают и останавливают секундомер', () => {
  const rng = createSeededRng(404);
  const match = createMatch({ rng });
  for (const p of match.players) updatePlayerTraitsAndStatsInSandbox(match, p.id, []);

  // Мяч от команды 0 улетает за правую лицевую линию -> свисток «Аут!» и ввод команде 1
  const ball = match.balls[0];
  ball.holderId = null;
  ball.isTipoffBall = false;
  ball.lastTouchTeam = 0;
  ball.x = COURT_RIGHT - 2;
  ball.y = HOOP_Y;
  ball.z = 20;
  ball.vx = 250;
  ball.vy = 0;

  stepMatch(match, 1 / 60, rng);
  assert.ok(match.inbound !== null, 'Назначен ввод мяча из-за линии');
  assert.equal(match.inbound!.team, 1, 'Ввод достался противоположной команде 1');
  assert.ok(match.inbound!.x > COURT_RIGHT, 'Вводящий стоит строго за пределами лицевой линии');
  assert.equal(match.violationsCount[0], 1, 'Команде 0 записано нарушение (аут)');

  // Секундомер четверти стоит на паузе во время ввода из аута
  const timeBefore = match.quarterTimeRemaining;
  // Заблокируем возможность паса (положим всех партнёров в trippedTimer), чтобы проверить правило 5 секунд!
  for (const p of match.players) {
    if (p.team === 1 && p.id !== match.inbound!.passerId) {
      p.trippedTimer = 10;
    }
  }
  const readInbound = () => match.inbound;
  for (let i = 0; i < 110; i++) {
    stepMatch(match, 0.05, rng);
    if (readInbound()?.team === 0) break;
  }

  assert.equal(match.quarterTimeRemaining, timeBefore, 'Время четверти не шло во время мёртвого мяча');
  assert.equal(readInbound()?.team, 0, 'По истечении 5с свистнуто нарушение и мяч перешёл команде 0!');
});

test('Правило 8 секунд вывода мяча и Нарушение зоны (возврат на свою половину) фиксируются судьёй', () => {
  const rng = createSeededRng(505);
  const match = createMatch({ rng });
  for (const p of match.players) updatePlayerTraitsAndStatsInSandbox(match, p.id, []);

  const pg0 = match.players[0]; // команда 0 (тыловая зона x < MID_X)
  pg0.x = 260;
  pg0.y = HOOP_Y;
  giveBallToPlayer(match, match.balls[0], pg0, rng, false);
  assert.equal(match.frontcourtEstablished, false);

  // 1) Удерживаем игрока в тыловой зоне дольше 8 секунд (замораживаем кулдаун и позицию)
  match.backcourtTimer = 7.98;
  pg0.actionCooldown = 5;
  pg0.vx = 0;
  pg0.vy = 0;
  stepMatch(match, 0.04, rng);
  assert.equal(match.referee.lastCallText, 'НАРУШЕНИЕ 8 СЕКУНД!');
  assert.ok(match.inbound !== null && match.inbound.team === 1);

  // 2) Проверка Нарушения зоны (Over-and-Back): мяч уже в передовой зоне (frontcourtEstablished = true), и игрок заступает назад за MID_X
  match.inbound = null;
  pg0.x = MID_X + 35;
  pg0.y = HOOP_Y;
  giveBallToPlayer(match, match.balls[0], pg0, rng, false);
  assert.equal(match.frontcourtEstablished, true);

  // Игрок с мячом откатывается назад на свою половину (x < MID_X - 6)
  pg0.x = MID_X - 12;
  pg0.targetX = MID_X - 20;
  pg0.actionCooldown = 5;
  stepMatch(match, 1 / 60, rng);
  assert.equal(match.referee.lastCallText, 'НАРУШЕНИЕ ЗОНЫ!');
  const overBackInbound = () => match.inbound;
  assert.equal(overBackInbound()?.team, 1);
});

test('Правило 14 секунд на подборе в нападении после касания кольца работает по стандарту FIBA', () => {
  const rng = createSeededRng(606);
  const match = createMatch({ rng });
  for (const p of match.players) updatePlayerTraitsAndStatsInSandbox(match, p.id, []);

  const shooter = match.players[0]; // команда 0
  const rebounderOff = match.players[4]; // центровой команды 0
  const rebounderDef = match.players[9]; // центровой команды 1

  giveBallToPlayer(match, match.balls[0], shooter, rng, false);
  match.shotClock = 5.2; // осталось мало времени на атаку
  match.shotTouchedRim = true; // бросок ударился о дужку кольца!

  // Подбор в нападении сбрасывает счётчик с 5.2с ровно до 14 секунд (а не до 24!)
  giveBallToPlayer(match, match.balls[0], rebounderOff, rng, false);
  assert.equal(match.shotClock, 14);

  // А подбор в защите даёт полные 24 секунды нового владения
  match.shotClock = 6.0;
  match.shotTouchedRim = true;
  giveBallToPlayer(match, match.balls[0], rebounderDef, rng, false);
  assert.equal(match.shotClock, 24);
});

test('Нарушение 3 секунд в чужой краске и Пробежка (приземление с мячом) фиксируются судьёй', () => {
  const rng = createSeededRng(707);
  const match = createMatch({ rng });
  for (const p of match.players) updatePlayerTraitsAndStatsInSandbox(match, p.id, []);

  // 1) Правило 3 секунд в краске соперника
  const pg0 = match.players[0];
  const center0 = match.players[4];
  pg0.x = 760;
  pg0.y = HOOP_Y;
  giveBallToPlayer(match, match.balls[0], pg0, rng, false);
  assert.equal(match.frontcourtEstablished, true);

  // Ставим центрового команды 0 прямо в чужую краску (справа, x = COURT_RIGHT - 80) с уже набежавшими 2.98с
  center0.x = COURT_RIGHT - 80;
  center0.y = HOOP_Y;
  center0.paintTimer = 2.98;
  center0.effectiveStats.iq = 10; // низкий IQ — забыл выйти из краски
  pg0.actionCooldown = 5;

  stepMatch(match, 0.04, rng);
  assert.equal(match.referee.lastCallText, '3 СЕКУНДЫ В КРАСКЕ!');
  assert.ok(match.inbound !== null && match.inbound.team === 1);

  // 2) Пробежка: выпрыгнул с мячом без броска и приземлился с мячом в руках
  match.inbound = null;
  pg0.x = 720;
  pg0.y = HOOP_Y;
  giveBallToPlayer(match, match.balls[0], pg0, rng, false);
  pg0.z = 26;
  pg0.vz = -180;
  pg0.windupTimer = 0;
  pg0.actionCooldown = 5;

  const readTravelInbound = () => match.inbound;
  for (let i = 0; i < 15; i++) {
    stepMatch(match, 1 / 60, rng);
    if (readTravelInbound()) break;
  }
  assert.equal(match.referee.lastCallText, 'ПРОБЕЖКА!');
  assert.equal(readTravelInbound()?.team, 1);
});

test('Помеха попаданию (Goaltending) засчитывает очки при перехвате падающего в кольцо обычного броска', () => {
  const rng = createSeededRng(808);
  const match = createMatch({ rng });
  for (const p of match.players) {
    updatePlayerTraitsAndStatsInSandbox(match, p.id, []);
    p.x = 300;
    p.y = 120;
  }

  const shooter0 = match.players[0]; // команда 0 бьёт в правое кольцо 1 (двухочковый)
  const blocker1 = match.players[9]; // защитник команды 1 под самым кольцом

  const ball = match.balls[0];
  ball.holderId = null;
  ball.isTipoffBall = false;
  ball.lastTouchTeam = 0;
  ball.lastShooterId = shooter0.id;
  ball.shotType = 'normal';
  ball.targetHoop = 1;
  ball.shotOriginX = HOOP_RIGHT_X - 120; // двухочковая зона
  ball.shotOriginY = HOOP_Y;

  // Мяч уже прошёл вершину траектории и ПАДАЕТ ВНИЗ (vz = -95) прямо над цилиндром кольца (z = RIM_Z + 8)
  ball.x = HOOP_RIGHT_X - 10;
  ball.y = HOOP_Y;
  ball.z = RIM_Z + 8;
  ball.vx = 30;
  ball.vy = 0;
  ball.vz = -95;

  blocker1.x = HOOP_RIGHT_X - 10;
  blocker1.y = HOOP_Y;
  blocker1.z = 75;
  blocker1.vz = 0;

  stepMatch(match, 1 / 60, rng);
  assert.equal(match.score[0], 2, 'При помехе попаданию команде 0 автоматически засчитаны 2 очка!');
  assert.ok(
    match.referee.lastCallText?.includes('ПОМЕХА ПОПАДАНИЮ'),
    `Судья зафиксировал Goaltending: ${match.referee.lastCallText}`
  );
});

test('Фол в нападении не даёт штрафных, а фол на броске и 5-й командный фол назначают серию штрафных бросков', () => {
  const rng = createSeededRng(909);
  const match = createMatch({ rng });
  for (const p of match.players) {
    updatePlayerTraitsAndStatsInSandbox(match, p.id, [], {
      speed: 65,
      shooting: 95,
      stamina: 80,
      steal: 60,
      jump: 60,
      teamwork: 65,
      dribble: 70,
      iq: 70,
      deviance: 20,
    });
  }

  const attacker0 = match.players[0]; // команда 0
  const defender1 = match.players[5]; // команда 1

  // 1) Фол в нападении (даже если у команды 0 уже 5 командных фолов) НЕ даёт штрафных бросков, а передаёт мяч вводом сбоку!
  match.possessionTeam = 0;
  match.teamFouls[0] = 5;
  adjudicatePersonalFoul(match, attacker0, defender1, 'offensive', rng);
  assert.equal(match.freeThrow, null, 'За фол в нападении штрафные НЕ пробиваются');
  assert.ok(match.inbound !== null && match.inbound.team === 1, 'Владение перешло команде 1 вводом из-за боковой');

  const readFt = () => match.freeThrow;
  const readFtInbound = () => match.inbound;

  // 2) Фол на трёхочковом броске назначает ровно 3 живых штрафных броска с линии!
  match.inbound = null;
  match.possessionTeam = 0;
  adjudicatePersonalFoul(match, defender1, attacker0, 'shooting_3', rng);
  const activeFt = readFt();
  assert.ok(activeFt !== null, 'Назначена серия штрафных');
  assert.equal(activeFt.totalShots, 3);
  assert.equal(activeFt.shotsRemaining, 3);

  // Прокручиваем исполнение всех 3 штрафных (с rng = () => 0.1 снайпер забивает все 3!)
  let ticks = 0;
  while (readFt() !== null && ticks < 400) {
    stepMatch(match, 0.03, () => 0.1);
    ticks++;
  }
  assert.equal(readFt(), null, 'Серия из 3 штрафных завершена');
  assert.equal(match.score[0], 3, 'Все 3 штрафных броска принесли по 1 очку');
  assert.equal(attacker0.freeThrowsMade, 3);
  assert.equal(readFtInbound()?.team, 1, 'После последнего забитого штрафного команда 1 вводит из-за лицевой');
});

test('Классический матч 5×5 с одним мячом и полным сводом правил проходит все 4 четверти без зависаний', () => {
  const rng = createSeededRng(2027);
  const match = createMatch({
    quarterDuration: 14,
    modifiers: [],
    rng,
  });

  let steps = 0;
  while (!match.finished && steps < 8000) {
    stepMatch(match, 0.035, rng);
    steps++;
  }

  assert.equal(match.finished, true, 'Матч корректно дошёл до финальной сирены');
  assert.ok(match.quarter >= 4);
  assert.ok(match.score[0] + match.score[1] > 0, 'Команды забивают очки с соблюдением всех правил');
});

test('После забитого мяча ввод из-за лицевой не дарит перехват под своим кольцом, а соперники отходят в защиту', () => {
  const rng = createSeededRng(3031);
  const match = createMatch({ rng });
  match.tactics = ['balanced', 'balanced'];

  for (const p of match.players) {
    updatePlayerTraitsAndStatsInSandbox(match, p.id, [], {
      speed: 65,
      shooting: 65,
      stamina: 85,
      steal: 75,
      jump: 65,
      teamwork: 65,
      dribble: 65,
      iq: 55,
      deviance: 10,
    });
  }

  // Имитируем ситуацию сразу после гола команды 1 в кольцо команды 0 (слева):
  // все 5 игроков команды 1 столпились прямо под кольцом команды 0 в краске!
  for (let i = 5; i < 10; i++) {
    const opp = match.players[i];
    opp.x = HOOP_LEFT_X + 18 + (i - 5) * 10;
    opp.y = HOOP_Y + ((i % 2 === 0 ? 1 : -1) * 18);
  }

  startInboundPlay(
    match,
    0,
    COURT_LEFT,
    HOOP_Y,
    'после попадания',
    24,
    rng,
    true
  );

  const scoreBefore = match.score[1];
  let ticks = 0;
  // Прокручиваем 2.4 секунды симуляции ввода и выхода со своей половины
  while (ticks < 144) {
    stepMatch(match, 1 / 60, rng);
    ticks++;
  }

  assert.equal(match.inbound, null, 'Ввод из-за лицевой успешно разыгран');
  assert.equal(
    match.score[1],
    scoreBefore,
    'Забившая команда не смогла мгновенно перехватить ввод под кольцом и нафармить очки'
  );
  assert.equal(
    match.possessionTeam,
    0,
    'Вводившая мяч команда уверенно сохранила владение после ввода из-за лицевой'
  );

  // Проверяем, что защищающаяся команда 1 отошла из чужой краски к центру площадки
  const avgDefX =
    match.players
      .filter((p) => p.team === 1)
      .reduce((acc, p) => acc + p.x, 0) / 5;
  assert.ok(
    avgDefX > 360,
    `Защитники после гола отступили из чужой краски к своей половине (средний X=${avgDefX.toFixed(1)})`
  );
});

test('Игрок с мячом и защитник из разных команд не застревают лоб в лоб на одной линии к кольцу', () => {
  const rng = createSeededRng(4042);
  const match = createMatch({ rng });
  match.inbound = null;

  for (const p of match.players) {
    updatePlayerTraitsAndStatsInSandbox(match, p.id, [], {
      speed: 68,
      shooting: 60,
      stamina: 90,
      steal: 30, // умеренный перехват, чтобы проверить именно обводку/движение без мгновенного перехвата
      jump: 60,
      teamwork: 55,
      dribble: 72,
      iq: 65,
      deviance: 5,
    });
  }

  // Разводим остальных игроков по флангам, а носителя мяча и защитника ставим строго лоб в лоб на линии HOOP_Y!
  const carrier = match.players[0]; // PG команды 0 (атакует вправо к HOOP_RIGHT_X)
  const defender = match.players[5]; // PG команды 1 (защищает правое кольцо)

  const startX = 610;
  const startY = HOOP_Y;
  carrier.x = startX;
  carrier.y = startY;
  carrier.vx = 0;
  carrier.vy = 0;

  defender.x = startX + 25; // прямо перед грудью носителя мяча строго на линии к кольцу!
  defender.y = startY;
  defender.vx = 0;
  defender.vy = 0;

  // Остальные партнёры и защитники на периметре, чтобы проверить чистую дуэль 1-в-1
  const spots0 = [
    [540, 120],
    [540, 560],
    [820, 115],
    [820, 565],
  ];
  const spots1 = [
    [575, 135],
    [575, 545],
    [855, 135],
    [855, 545],
  ];
  for (let idx = 1; idx <= 4; idx++) {
    match.players[idx].x = spots0[idx - 1][0];
    match.players[idx].y = spots0[idx - 1][1];
    match.players[5 + idx].x = spots1[idx - 1][0];
    match.players[5 + idx].y = spots1[idx - 1][1];
  }

  giveBallToPlayer(match, match.balls[0], carrier, rng, false);

  // Симулируем 2.2 секунды игры (132 кадра)
  for (let i = 0; i < 132; i++) {
    stepMatch(match, 1 / 60, rng);
  }

  const carrierMovedDist = Math.hypot(carrier.x - startX, carrier.y - startY);
  const ballHolderChanged = match.balls[0].holderId !== carrier.id;

  assert.ok(
    carrierMovedDist > 55 || ballHolderChanged || match.inbound !== null,
    `Балбесы не застряли лоб в лоб: смещение носителя=${carrierMovedDist.toFixed(1)}px, пас/бросок=${ballHolderChanged}`
  );
});

test('Скамейка запасных (по 3 игрока на команду): ручные замены, восстановление сил и выход на 5-м фоле', () => {
  const rng = createSeededRng(5051);
  const match = createMatch({ rng });

  assert.equal(match.players.length, 10, 'На площадке ровно 10 стартовых игроков (5×5)');
  assert.equal(match.bench.length, 6, 'На скамейках ровно 6 запасных (по 3 на команду)');
  assert.equal(match.bench.filter((b) => b.team === 0).length, 3);
  assert.equal(match.bench.filter((b) => b.team === 1).length, 3);
  assert.ok(match.bench.every((b) => b.startedOnBench && !b.onCourt));

  // Запасной восстанавливает выносливость, сидя на скамейке (шагами по 0.05с с учётом clamp dt)
  const sub0 = match.bench[0];
  sub0.energy = 55;
  for (let i = 0; i < 10; i++) {
    stepMatch(match, 0.05, rng);
  }
  assert.ok(sub0.energy > 58, `Запасной восстановил силы на скамейке (${sub0.energy.toFixed(1)}%)`);

  // Прямая замена игрока стартовой пятёрки на запасного
  const starter0 = match.players[0];
  const entered = requestPlayerSubstitution(match, sub0.id, rng);
  assert.ok(entered !== null && entered.id === sub0.id, 'Запасной вышел на площадку');
  assert.equal(sub0.onCourt, true);
  assert.equal(sub0.subbedInOnce, true);
  assert.ok(match.players.some((p) => p.id === sub0.id));
  assert.ok(match.bench.some((b) => b.id === starter0.id && !b.onCourt));

  // При 5-м персональном фоле удалённого игрока меняет реальный запасной со скамейки команды 1
  const offender1 = match.players.find((p) => p.team === 1)!;
  const benchIdsTeam1 = new Set(match.bench.filter((b) => b.team === 1).map((b) => b.id));
  offender1.fouls = 4;
  adjudicatePersonalFoul(match, offender1, sub0, 'reach_in', rng);
  assert.equal(offender1.disqualified, true, 'Нарушитель с 5 фолами дисквалифицирован');
  assert.ok(
    match.bench.some((b) => b.id === offender1.id),
    'Удалённый игрок отправлен на командную скамейку'
  );
  assert.ok(
    match.players.some((p) => benchIdsTeam1.has(p.id) && p.onCourt),
    'На паркет вместо удалённого вышел настоящий запасной со скамейки команды 1'
  );
});

test('Уникальная черта «Последняя надежда» даёт огромные баффы только запасному при выходе на поле', () => {
  const rng = createSeededRng(6062);
  const match = createMatch({ rng });

  const traitDef = TRAIT_BY_ID['last_hope'];
  assert.ok(traitDef, 'Черта last_hope зарегистрирована в каталоге');
  assert.equal(traitDef.rarity, 'unique');
  assert.equal(traitDef.name, 'Последняя надежда');

  const base = {
    speed: 60,
    shooting: 60,
    stamina: 60,
    steal: 60,
    jump: 60,
    teamwork: 60,
    dribble: 60,
    iq: 60,
    deviance: 20,
  };

  // 1) Игрок стартовой пятёрки с «Последней надеждой» НЕ получает бафф запасного!
  const starter = match.players[0];
  updatePlayerTraitsAndStatsInSandbox(match, starter.id, ['last_hope'], base);
  assert.equal(starter.startedOnBench, false);
  assert.equal(isLastHopeBuffActive(starter), false);
  assert.equal(starter.effectiveStats.shooting, 60);

  // 2) Запасной игрок с «Последней надеждой»: пока сидит на скамейке — бафф ждёт выхода, а при выходе на паркет взлетает!
  const benchJoker = match.bench.find((b) => b.team === 0)!;
  updatePlayerTraitsAndStatsInSandbox(match, benchJoker.id, ['last_hope'], base);
  assert.equal(benchJoker.startedOnBench, true);
  assert.equal(isLastHopeBuffActive(benchJoker), false, 'На скамейке бафф ещё не активен');
  assert.equal(benchJoker.effectiveStats.shooting, 60);

  substitutePlayer(match, match.players[1].id, benchJoker.id, 'Проверка джокера', rng);
  assert.equal(benchJoker.onCourt, true);
  assert.equal(benchJoker.subbedInOnce, true);
  assert.equal(isLastHopeBuffActive(benchJoker), true, 'При выходе со скамейки бафф активировался!');
  assert.ok(
    benchJoker.effectiveStats.speed >= 84 &&
      benchJoker.effectiveStats.shooting >= 84 &&
      benchJoker.effectiveStats.stamina >= 87 &&
      benchJoker.effectiveStats.steal >= 84 &&
      benchJoker.effectiveStats.jump >= 84 &&
      benchJoker.effectiveStats.dribble >= 84 &&
      benchJoker.effectiveStats.iq >= 84,
    `Все характеристики запасного получили огромный бафф: speed=${benchJoker.effectiveStats.speed}, shooting=${benchJoker.effectiveStats.shooting}`
  );
});

test('Игрок не крутится между двумя направлениями равной длины, когда перед ним по прямой бежит противник', () => {
  const rng = createSeededRng(7073);
  const match = createMatch({ rng });
  match.inbound = null;

  for (const p of match.players) {
    updatePlayerTraitsAndStatsInSandbox(match, p.id, [], {
      speed: 70,
      shooting: 60,
      stamina: 95,
      steal: 20,
      jump: 60,
      teamwork: 20,
      dribble: 78,
      iq: 70,
      deviance: 0,
    });
  }

  // Убираем остальных игроков к дальним углам, чтобы создать идеальную симметрию двух путей обхода (сверху и снизу от HOOP_Y)!
  for (let idx = 1; idx <= 4; idx++) {
    match.players[idx].x = 220;
    match.players[idx].y = idx % 2 === 0 ? 95 : 525;
    match.players[5 + idx].x = 260;
    match.players[5 + idx].y = idx % 2 === 0 ? 110 : 510;
  }

  const runner = match.players[0]; // команда 0, бежит строго по центральной оси Y = HOOP_Y вправо!
  const blocker = match.players[5]; // противник бежит прямо перед ним по той же оси Y = HOOP_Y!

  runner.x = 480;
  runner.y = HOOP_Y;
  runner.vx = 140;
  runner.vy = 0;
  runner.heading = 0;

  blocker.x = 518;
  blocker.y = HOOP_Y;
  blocker.vx = 95;
  blocker.vy = 0;
  blocker.heading = 0;

  giveBallToPlayer(match, match.balls[0], runner, rng, false);

  let verticalSignFlips = 0;
  let lastSign = 0;
  let maxSingleFrameAngleJump = 0;
  let prevHeading = runner.heading;

  // Симулируем 75 кадров (1.25 секунды), пока противник маячит прямо перед бегущим игроком на симметричной линии
  for (let frame = 0; frame < 75; frame++) {
    // Удерживаем противника строго на симметричной центральной линии впереди первые 25 кадров (самый провокационный случай равных путей!)
    if (frame < 25 && match.balls[0].holderId === runner.id) {
      blocker.x = runner.x + 36;
      blocker.y = HOOP_Y;
      blocker.vy = 0;
    }

    stepMatch(match, 1 / 60, rng);

    let dAng = Math.abs(runner.heading - prevHeading) % (Math.PI * 2);
    if (dAng > Math.PI) dAng = Math.PI * 2 - dAng;
    if (dAng > maxSingleFrameAngleJump) {
      maxSingleFrameAngleJump = dAng;
    }
    prevHeading = runner.heading;

    const sinH = Math.sin(runner.heading);
    if (Math.abs(sinH) > 0.16) {
      const sign = sinH > 0 ? 1 : -1;
      if (lastSign !== 0 && sign !== lastSign) {
        verticalSignFlips++;
      }
      lastSign = sign;
    }
  }

  assert.ok(
    verticalSignFlips <= 1,
    `Никакого бешеного кручения между верхним и нижним обходом (переключений знака угла=${verticalSignFlips})`
  );
  assert.ok(
    maxSingleFrameAngleJump < 0.3,
    `Поворот корпуса плавный на каждом кадре (макс. скачок угла за кадр=${maxSingleFrameAngleJump.toFixed(3)} рад)`
  );
  assert.ok(
    runner.x > 545,
    `Игрок уверенно продвинулся вперёд в выбранную сторону обхода (x=${runner.x.toFixed(1)}, y=${runner.y.toFixed(1)})`
  );
});

test('Ложный замах (Pump Fake): умный стрелок подкидывает в воздух глупого защитника и бьёт открытым', () => {
  const match = createMatch({ rng: () => 0.1 });
  match.inbound = null;
  match.frontcourtEstablished = true;

  for (const p of match.players) {
    p.x = 260;
    p.y = 120 + p.slotIndex * 60;
    updatePlayerTraitsAndStatsInSandbox(match, p.id, [], {
      speed: 65,
      shooting: 82,
      stamina: 90,
      steal: 20,
      jump: 60,
      teamwork: 20, // без открытых пасов рядом, чтобы проверить дуэль бросок/фейк
      dribble: 50,
      iq: 88,
      deviance: 5,
    });
  }

  const shooter = match.players[0]; // Команда 0, у дуги перед правым кольцом
  const eagerDef = match.players[5]; // Команда 1, низкий IQ — легко ведётся на фейк!
  updatePlayerTraitsAndStatsInSandbox(match, eagerDef.id, ['panicker'], {
    speed: 65,
    shooting: 45,
    stamina: 90,
    steal: 20,
    jump: 70,
    teamwork: 45,
    dribble: 45,
    iq: 18,
    deviance: 5,
  });

  shooter.x = HOOP_RIGHT_X - 215;
  shooter.y = HOOP_Y;
  shooter.decisionTimer = 0;
  shooter.actionCooldown = 0;
  shooter.crossoverCd = 3.0; // отключаем кроссовер, чтобы проверить именно ложный замах

  eagerDef.x = shooter.x + 34;
  eagerDef.y = HOOP_Y;
  eagerDef.z = 0;

  giveBallToPlayer(match, match.balls[0], shooter, () => 0.2, false);
  shooter.actionCooldown = 0;
  shooter.decisionTimer = 0;

  // Первый шаг с rng=0.15 запускает onPumpFake и подбрасывает глупого защитника!
  stepMatch(match, 0.04, () => 0.15);

  assert.equal(shooter.aiAction, 'pump_fake', 'Стрелок выполнил ложный замах вместо броска в руки');
  assert.ok(
    (eagerDef.bittenByFakeTimer ?? 0) > 0 && eagerDef.z > 0,
    `Защитник с низким IQ купился на фейк и взлетел в воздух (z=${eagerDef.z.toFixed(1)})`
  );
});

test('Комбинации «Пик-н-поп», «Дай и беги» и защитная смена на заслоне (Switch Screen)', () => {
  const rng = createSeededRng(8084);
  const match = createMatch({ rng });
  match.inbound = null;
  match.frontcourtEstablished = true;
  match.possessionTimer = 6.0;

  const carrier = match.players[0]; // PG команды 0
  const stretchBig = match.players[3]; // PF команды 0 (хороший бросок + высокий IQ)
  const onBallDef = match.players[5]; // PG команды 1
  const screenDef = match.players[8]; // PF команды 1 (умный защитник — должен сделать смену на заслоне!)
  const rimProtector = match.players[9]; // C команды 1 (стоит под кольцом, забивая краску)

  updatePlayerTraitsAndStatsInSandbox(match, stretchBig.id, [], {
    speed: 68,
    shooting: 80,
    stamina: 88,
    steal: 50,
    jump: 66,
    teamwork: 75,
    dribble: 60,
    iq: 78,
    deviance: 10,
  });
  updatePlayerTraitsAndStatsInSandbox(match, screenDef.id, [], {
    speed: 70,
    shooting: 55,
    stamina: 88,
    steal: 68,
    jump: 64,
    teamwork: 70,
    dribble: 55,
    iq: 76,
    deviance: 10,
  });

  // 1) Смена на заслоне у защиты, пока stretchBig держит заслон рядом с мячом
  carrier.x = 760;
  carrier.y = HOOP_Y - 40;
  giveBallToPlayer(match, match.balls[0], carrier, rng, false);

  onBallDef.x = 742;
  onBallDef.y = HOOP_Y - 40;
  stretchBig.x = 752;
  stretchBig.y = HOOP_Y - 16;
  stretchBig.aiAction = 'screen_set';

  screenDef.x = 796;
  screenDef.y = HOOP_Y - 16;
  rimProtector.x = HOOP_RIGHT_X - 52;
  rimProtector.y = HOOP_Y;

  const noopCallbacks = {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  };
  stepPlayerAI(match, screenDef, 0.05, rng, noopCallbacks);
  assert.equal(screenDef.aiAction, 'help_defense', 'Защитник на заслоне переключился встречать мяч');

  // 2) Когда носитель проскочил заслон, а чужая краска занята центровым — бросающий большой делает «Пик-н-поп» на дугу!
  carrier.x = 790;
  onBallDef.x = 745;
  stepPlayerAI(match, stretchBig, 0.05, rng, noopCallbacks);
  assert.equal(
    stretchBig.aiAction,
    'pick_and_pop',
    'При забитой краске бросающий большой выбрал отскок на трёшку (Пик-н-поп)!'
  );

  // 3) Комбинация «Дай и беги» (Give-and-Go) после паса с периметра умным командным игроком
  updatePlayerTraitsAndStatsInSandbox(match, carrier.id, [], {
    speed: 76,
    shooting: 70,
    stamina: 90,
    steal: 60,
    jump: 65,
    teamwork: 82,
    dribble: 78,
    iq: 80,
    deviance: 5,
  });
  carrier.x = 720;
  carrier.y = HOOP_Y - 90;
  carrier.decisionTimer = 0;
  carrier.actionCooldown = 0;
  carrier.crossoverCd = 4;
  stretchBig.x = 810;
  stretchBig.y = HOOP_Y + 135;
  // Уводим защитников от линии паса, чтобы пас точно прошёл, а rng=0.2 (<0.58) включил рывок give_and_go!
  for (let i = 5; i < 10; i++) {
    match.players[i].x = 640;
    match.players[i].y = 90 + (i - 5) * 45;
  }
  stepMatch(match, 0.04, () => 0.2);
  assert.equal(carrier.aiAction, 'give_and_go', 'Отдав пас на периметре, умный игрок рванул под ответку (Дай и беги)!');
});

test('Кроссовер «Голеностополом», смешной финт неуклюжего балбеса себе в ногу и физический Бокс-аут', () => {
  const match = createMatch({ rng: () => 0.08 });
  match.inbound = null;
  match.frontcourtEstablished = true;

  const handler = match.players[0];
  const clumsyDef = match.players[5];

  for (const p of match.players) {
    p.x = 260;
    p.y = 110 + p.slotIndex * 55;
  }

  // 1) Мастер дриблинга укладывает неуклюжего защитника на паркет кроссовером без фола!
  updatePlayerTraitsAndStatsInSandbox(match, handler.id, ['showboat'], {
    speed: 88,
    shooting: 70,
    stamina: 92,
    steal: 50,
    jump: 68,
    teamwork: 50,
    dribble: 94,
    iq: 78,
    deviance: 5,
  });
  updatePlayerTraitsAndStatsInSandbox(match, clumsyDef.id, [], {
    speed: 48,
    shooting: 40,
    stamina: 80,
    steal: 22,
    jump: 45,
    teamwork: 45,
    dribble: 40,
    iq: 22,
    deviance: 5,
  });

  handler.x = 690;
  handler.y = HOOP_Y;
  handler.actionCooldown = 0;
  handler.crossoverCd = 0;
  clumsyDef.x = 724;
  clumsyDef.y = HOOP_Y;
  giveBallToPlayer(match, match.balls[0], handler, () => 0.5, false);
  handler.actionCooldown = 0;

  stepMatch(match, 0.04, () => 0.08);
  assert.ok(
    clumsyDef.trippedTimer > 0 && clumsyDef.fouls === 0 && handler.fouls === 0,
    'Защитник упал от чистого кроссовера («сломал лодыжки») без персонального фола!'
  );

  // 2) Глупый балбес с дырявыми руками пытается финтить и выбивает мяч себе в кроссовок!
  clumsyDef.trippedTimer = 0;
  clumsyDef.x = 724;
  clumsyDef.y = HOOP_Y;
  updatePlayerTraitsAndStatsInSandbox(match, handler.id, ['butterfingers'], {
    speed: 55,
    shooting: 45,
    stamina: 85,
    steal: 30,
    jump: 50,
    teamwork: 40,
    dribble: 18,
    iq: 24,
    deviance: 10,
  });
  handler.x = 692;
  handler.y = HOOP_Y;
  handler.crossoverCd = 0;
  handler.actionCooldown = 0;
  giveBallToPlayer(match, match.balls[0], handler, () => 0.5, false);
  handler.actionCooldown = 0;

  stepMatch(match, 0.04, () => 0.1);
  assert.ok(
    match.balls[0].holderId !== handler.id && handler.aiThought.includes('кроссовок'),
    'Неуклюжий балбес с низким IQ потерял мяч, стукнув им себе в кроссовок!'
  );

  // 3) Настоящий Бокс-аут: умный центровой встаёт строго между соперником и кольцом во время полёта броска!
  const centerDef = match.players[9]; // C команды 1 (защищает правое кольцо HOOP_RIGHT_X)
  const crasherAtk = match.players[4]; // C команды 0 (идёт на подбор под правым кольцом)
  updatePlayerTraitsAndStatsInSandbox(match, centerDef.id, [], {
    speed: 65,
    shooting: 55,
    stamina: 90,
    steal: 60,
    jump: 78,
    teamwork: 70,
    dribble: 50,
    iq: 82,
    deviance: 10,
  });
  crasherAtk.x = HOOP_RIGHT_X - 100;
  crasherAtk.y = HOOP_Y - 40;
  centerDef.x = HOOP_RIGHT_X - 70;
  centerDef.y = HOOP_Y + 10;

  const b = match.balls[0];
  b.holderId = null;
  b.shotType = 'normal';
  b.targetHoop = 1;
  b.x = 780;
  b.y = HOOP_Y;
  b.z = 130;

  stepPlayerAI(match, centerDef, 0.05, () => 0.5, {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  });

  assert.equal(centerDef.aiAction, 'rebound_position');
  const distTargetToHoop = Math.hypot(centerDef.targetX - HOOP_RIGHT_X, centerDef.targetY - HOOP_Y);
  const distCrasherToHoop = Math.hypot(crasherAtk.x - HOOP_RIGHT_X, crasherAtk.y - HOOP_Y);
  assert.ok(
    distTargetToHoop < distCrasherToHoop && centerDef.targetY < HOOP_Y,
    `Умный центровой оттирает спиной соперника между ним и кольцом (distTarget=${distTargetToHoop.toFixed(1)} < distCrasher=${distCrasherToHoop.toFixed(1)})`
  );
});

test('Правило зоны (FIBA 30.1.1): пас в воздухе над центральной линией со спорного не даёт ложное нарушение зоны', () => {
  const rng = createSeededRng(9091);
  const match = createMatch({ rng });
  match.inbound = null;
  match.possessionTeam = 0;
  match.frontcourtEstablished = false;
  match.possessionTimer = 0.2;

  const receiver = match.players[1]; // Команда 0 стоит чуть в тыловой зоне у центрального круга
  receiver.x = 542;
  receiver.y = HOOP_Y - 40;

  const ball = match.balls[0];
  ball.holderId = null;
  ball.isTipoffBall = false;
  ball.shotType = 'pass';
  ball.lastTouchTeam = 0;
  ball.x = 564; // Мяч на 4px за центральной линией в воздухе!
  ball.y = HOOP_Y - 30;
  ball.z = 22;
  ball.vx = -120;
  ball.vy = -40;

  stepMatch(match, 1 / 60, rng);
  assert.equal(
    match.frontcourtEstablished,
    false,
    'Мяч в воздухе над чужой половиной без контроля игрока ещё не устанавливает статус передовой зоны'
  );

  giveBallToPlayer(match, ball, receiver, rng, false);
  stepMatch(match, 1 / 60, rng);
  assert.equal(
    match.inbound,
    null,
    'Приём такого паса у центральной линии в тыловой зоне легален и не вызывает свисток о нарушении зоны!'
  );
});

test('Горячая рука (hotStreak), полный протокол (FG, 3PT, Подборы) и Карта бросков (shotChart)', () => {
  const rng = createSeededRng(10101);
  const match = createMatch({ rng });
  match.inbound = null;

  // Убираем защитников в дальний угол, чтобы проверить чистую баллистику броска и подбор
  for (const p of match.players) {
    p.x = 180;
    p.y = 95 + p.slotIndex * 35;
  }

  const sniper = match.players[1]; // Команда 0, атакует правое кольцо (x = 1090)
  const rebounder = match.players[9]; // Центровой команды 1 под кольцом
  updatePlayerTraitsAndStatsInSandbox(match, sniper.id, ['sniper'], {
    speed: 75,
    shooting: 98,
    stamina: 95,
    steal: 50,
    jump: 70,
    teamwork: 65,
    dribble: 75,
    iq: 85,
    deviance: 5,
  });

  // Серия из 3 точных трёхочковых бросков подряд -> hotStreak = 3 («В ОГНЕ!»)
  for (let i = 0; i < 3; i++) {
    match.inbound = null;
    sniper.x = HOOP_RIGHT_X - 255; // За трёхочковой дугой (255 > 228)
    sniper.y = HOOP_Y;
    const ball = match.balls[0];
    giveBallToPlayer(match, ball, sniper, () => 0.01, false);
    launchBallShot(match, ball, sniper, 1, 'normal', () => 0.01);
    assert.equal(ball.isGuaranteedSwish, true);

    for (let step = 0; step < 120 && !match.inbound; step++) {
      stepMatch(match, 1 / 60, () => 0.5);
    }
  }

  assert.equal(sniper.fgAttempted, 3);
  assert.equal(sniper.fgMade, 3);
  assert.equal(sniper.threeAttempted, 3);
  assert.equal(sniper.threeMade, 3);
  assert.equal(sniper.points, 9);
  assert.equal(sniper.hotStreak, 3, 'После 3 попаданий подряд активировалась Горячая рука (hotStreak = 3)!');
  assert.equal(match.shotChart.length, 3);
  assert.ok(match.shotChart.every((s) => s.made && s.isThree && s.team === 0));

  // Теперь гарантированный промах со средней дистанции и подбор защитника под кольцом
  match.inbound = null;
  sniper.x = HOOP_RIGHT_X - 150;
  sniper.y = HOOP_Y;
  rebounder.x = HOOP_RIGHT_X - 24;
  rebounder.y = HOOP_Y;
  const ball = match.balls[0];
  giveBallToPlayer(match, ball, sniper, () => 0.99, false);
  launchBallShot(match, ball, sniper, 1, 'normal', () => 0.99);
  assert.equal(ball.isGuaranteedSwish, false);

  for (let step = 0; step < 150 && match.balls[0].holderId !== rebounder.id; step++) {
    rebounder.x = match.balls[0].x;
    rebounder.y = match.balls[0].y;
    stepMatch(match, 1 / 60, () => 0.5);
  }

  assert.equal(sniper.fgAttempted, 4);
  assert.equal(sniper.fgMade, 3);
  assert.equal(sniper.hotStreak, -1, 'Промах сбрасывает серию попаданий до -1');
  assert.ok(rebounder.rebounds >= 1, `Подбор записан в протокол центровому (rebounds=${rebounder.rebounds})`);
  assert.equal(match.shotChart.length, 4, 'Промах также записан на карту бросков');
  assert.equal(match.shotChart[3].made, false);
  assert.equal(match.shotChart[3].isThree, false);
});

test('Тренерский тайм-аут и живые комбинации с планшета (Изоляция / Алли-уп / Снайпер)', () => {
  const rng = createSeededRng(20202);
  const match = createMatch({ rng });
  match.inbound = null;
  match.frontcourtEstablished = true;

  // Имитируем усталость нашей пятёрки и горячий рывок соперника 9:0
  for (const p of match.players) {
    if (p.team === 0) {
      p.energy = 45;
      p.hotStreak = -3;
    } else {
      p.hotStreak = 4;
    }
  }
  match.scoringRun = { team: 1, points: 9 };
  assert.equal(match.timeoutsRemaining[0], 2);

  const okTimeout = callTeamTimeout(match, 0, rng);
  assert.equal(okTimeout, true);
  assert.equal(match.timeoutsRemaining[0], 1, 'Списан 1 тренерский тайм-аут первой половины');
  assert.equal(match.scoringRun.points, 0, 'Рывок соперника сбит тайм-аутом');
  assert.ok(
    match.players.filter((p) => p.team === 0).every((p) => p.energy >= 67 && p.hotStreak >= 0),
    'Игроки команды восстановили силы (+22%) и сбросили мандраж промахов'
  );
  assert.ok(
    match.players.filter((p) => p.team === 1).every((p) => p.hotStreak <= 1),
    'Горячая рука соперников остужена тренерским тайм-аутом!'
  );

  // Заказ комбинации «Навес Алли-уп» с планшета тренера
  const carrier = match.players.find((p) => p.team === 0 && p.slotIndex === 0)!;
  const flyer = match.players.find((p) => p.team === 0 && p.slotIndex === 4)!;
  updatePlayerTraitsAndStatsInSandbox(match, flyer.id, ['sixth_sense_alley'], {
    speed: 78,
    shooting: 65,
    stamina: 92,
    steal: 50,
    jump: 90,
    teamwork: 75,
    dribble: 60,
    iq: 78,
    deviance: 5,
  });

  carrier.x = 760;
  carrier.y = HOOP_Y;
  flyer.x = 860;
  flyer.y = HOOP_Y - 70;
  giveBallToPlayer(match, match.balls[0], carrier, rng, false);

  const okPlay = triggerCoachPlayCall(match, 0, 'alley_oop');
  assert.equal(okPlay, true);
  assert.equal(match.activePlayCall?.play, 'alley_oop');
  assert.equal(
    triggerCoachPlayCall(match, 0, 'iso_star'),
    false,
    'Повторный заказ комбинации заблокирован кулдауном планшета'
  );

  stepPlayerAI(match, flyer, 0.05, rng, {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  });
  assert.equal(
    flyer.aiAction,
    'roll_to_rim',
    'По команде «Алли-уп» прыгучий партнёр немедленно врывается под кольцо под навес!'
  );
});

test('Новые черты: «Актёр Оскара» (фол в атаке vs технарь за симуляцию), «Король Алли-упов» и «Король щита»', () => {
  const rng = createSeededRng(30303);
  const match = createMatch({ rng });
  match.inbound = null;
  match.possessionTeam = 0;

  // Проверяем физические бонусы новых черт
  const alleyKing = match.players[2];
  const baseReach = computePlayerVerticalReach({ ...alleyKing, traitIds: [] });
  const alleyReach = computePlayerVerticalReach({ ...alleyKing, traitIds: ['sixth_sense_alley'] });
  assert.equal(alleyReach - baseReach, 12, 'Король Алли-упов получает +12 к вертикальному дотягиванию');

  const boardKing = match.players[4];
  const baseHoriz = computePlayerHorizontalInterceptRadius({ ...boardKing, traitIds: [] });
  const boardHoriz = computePlayerHorizontalInterceptRadius({ ...boardKing, traitIds: ['brick_magnet_board'] });
  assert.equal(boardHoriz - baseHoriz, 8, 'Король щита получает +8px к радиусу подбора');

  // Проверяем «Актёра Оскара» (flopper):
  // 1) Умный свежий актёр продаёт фол в нападении при столкновении с несущимся соперником!
  for (const p of match.players) {
    p.x = 220;
    p.y = 95 + p.slotIndex * 38;
    p.vx = 0;
    p.vy = 0;
  }
  const driver = match.players[0]; // Команда 0 в атаке
  const actor = match.players[5]; // Команда 1 в защите с чертой flopper
  updatePlayerTraitsAndStatsInSandbox(match, actor.id, ['flopper'], {
    speed: 70,
    shooting: 60,
    stamina: 90,
    steal: 60,
    jump: 60,
    teamwork: 70,
    dribble: 60,
    iq: 95,
    deviance: 25,
  });

  giveBallToPlayer(match, match.balls[0], driver, () => 0.5, false);
  match.frontcourtEstablished = true;
  driver.x = 700;
  driver.y = HOOP_Y;
  driver.targetX = HOOP_RIGHT_X;
  driver.targetY = HOOP_Y;
  driver.decisionTimer = 5;
  driver.actionCooldown = 5;
  driver.vx = 155;
  driver.vy = 0;
  actor.x = 714;
  actor.y = HOOP_Y;
  actor.targetX = 700;
  actor.targetY = HOOP_Y;
  actor.decisionTimer = 5;
  actor.stealCooldown = 5;
  actor.flopCd = 0;
  actor.refSuspicion = 0;

  // rng() = 0.12 (< 0.34 триггер флопа и < sellProb ~0.70 -> судья верит и свистит фол в атаке!)
  stepMatch(match, 1 / 60, () => 0.12);
  assert.equal(driver.fouls, 1, 'Судья поверил «Актёру Оскара» и выписал фол в нападении атакующему!');
  const inbAfterFlop = match.inbound as { team: 0 | 1 } | null;
  assert.equal(inbAfterFlop?.team, 1, 'Владение перешло команде актёра');

  // 2) Когда актёр уже примелькался судье (refSuspicion = 4), судья раскусывает флоп и даёт ТЕХНИЧЕСКИЙ штрафной!
  match.inbound = null;
  match.freeThrow = null;
  match.possessionTeam = 0;
  match.frontcourtEstablished = true;
  actor.trippedTimer = 0;
  actor.flopCd = 0;
  actor.stealCooldown = 5;
  actor.decisionTimer = 5;
  actor.refSuspicion = 4; // sellProb падает до ~0.146
  giveBallToPlayer(match, match.balls[0], driver, () => 0.5, false);
  driver.x = 700;
  driver.y = HOOP_Y;
  driver.targetX = HOOP_RIGHT_X;
  driver.targetY = HOOP_Y;
  driver.decisionTimer = 5;
  driver.actionCooldown = 5;
  driver.vx = 155;
  driver.vy = 0;
  actor.x = 714;
  actor.y = HOOP_Y;

  // rng() = 0.28 (< 0.34 триггер, но > sellProb 0.146 и < sellProb + 0.36 -> ТЕХНАРЬ!)
  stepMatch(match, 1 / 60, () => 0.28);
  const ftAfterFlop = match.freeThrow as { totalShots: number; shootingTeam: 0 | 1 } | null;
  assert.ok(
    ftAfterFlop !== null && ftAfterFlop.totalShots === 1 && ftAfterFlop.shootingTeam === 0,
    'Раскусив повторную симуляцию, судья наказал актёра техническим фолом (1 штрафной + владение)!'
  );
});

test('Добивание сверху в прыжке на чужом щите (Putback Dunk) и быстрый Первый пас в отрыв (Outlet Pass)', () => {
  const rng = createSeededRng(40404);
  const match = createMatch({ rng });
  match.inbound = null;
  match.possessionTeam = 0;
  match.frontcourtEstablished = true;

  for (const p of match.players) {
    p.x = 240;
    p.y = 100 + p.slotIndex * 40;
  }

  // 1) Добивание сверху (Putback Dunk): прыгучий центровой снимает промах партнёра в воздухе возле кольца и мгновенно вколачивает сверху!
  const putbackBig = match.players[4]; // C команды 0 (атакует правое кольцо HOOP_RIGHT_X)
  updatePlayerTraitsAndStatsInSandbox(match, putbackBig.id, ['gorilla_dunk'], {
    speed: 68,
    shooting: 65,
    stamina: 90,
    steal: 50,
    jump: 88,
    teamwork: 60,
    dribble: 55,
    iq: 75,
    deviance: 10,
  });
  putbackBig.x = HOOP_RIGHT_X - 48;
  putbackBig.y = HOOP_Y + 12;
  putbackBig.z = 26;

  const ball = match.balls[0];
  ball.holderId = null;
  ball.wasShotAttempt = true;
  ball.shotChartRecorded = false;
  ball.lastTouchTeam = 0;
  ball.lastShooterId = match.players[0].id;
  ball.shotOriginX = HOOP_RIGHT_X - 240;
  ball.shotOriginY = HOOP_Y;
  ball.x = HOOP_RIGHT_X - 42;
  ball.y = HOOP_Y + 10;
  ball.z = 46;

  giveBallToPlayer(match, ball, putbackBig, () => 0.2, true);
  assert.equal(putbackBig.rebounds, 1, 'Подбор в нападении записан центровому');
  assert.equal(ball.shotType, 'dunk', 'Центровой мгновенно отправил мяч в кольцо добиванием сверху!');
  assert.ok(putbackBig.aiThought.includes('ДОБИВАНИЕ'), `Мысль ИИ отражает добивание: ${putbackBig.aiThought}`);

  // 2) Первый пас в отрыв (Outlet Pass): центровой команды 1 после подбора под правым щитом сразу кидает вперёд бегущему защитнику!
  const defCenter = match.players[9]; // C команды 1 (защищает правое кольцо, атакует левое HOOP_LEFT_X)
  const sprinterGuard = match.players[5]; // PG команды 1 в отрыве ближе к левому кольцу
  updatePlayerTraitsAndStatsInSandbox(match, defCenter.id, ['pass_telepath'], {
    speed: 60,
    shooting: 55,
    stamina: 88,
    steal: 55,
    jump: 80,
    teamwork: 84,
    dribble: 55,
    iq: 82,
    deviance: 5,
  });

  for (let i = 0; i < 5; i++) {
    match.players[i].x = 980;
    match.players[i].y = 110 + i * 32; // соперники остались глубоко справа и не перекрывают линию паса по низу!
  }
  for (let i = 6; i < 9; i++) {
    match.players[i].x = 920;
    match.players[i].y = 140 + (i - 6) * 40;
  }

  defCenter.x = HOOP_RIGHT_X - 70;
  defCenter.y = HOOP_Y + 110;
  sprinterGuard.x = MID_X - 60;
  sprinterGuard.y = HOOP_Y + 120;
  sprinterGuard.aiAction = 'fast_break';

  giveBallToPlayer(match, ball, defCenter, () => 0.4, false);
  match.possessionTimer = 0.6;
  defCenter.decisionTimer = 0;
  defCenter.actionCooldown = 0;

  let passedToId: string | null = null;
  stepPlayerAI(match, defCenter, 0.05, () => 0.3, {
    onShoot: () => {},
    onPass: (_p, _b, rec) => {
      passedToId = rec.id;
    },
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  });

  assert.equal(passedToId, sprinterGuard.id, 'Центровой выдал быстрый первый пас в отрыв убегающему разыгрывающему!');
  assert.ok(defCenter.aiThought.includes('Первый пас в отрыв'), `Мысль ИИ: ${defCenter.aiThought}`);
});

test('Степ-бэк трёшка, Евростеп в краске и связка «Скидка из краски + Смещение по дуге»', () => {
  const match = createMatch({ rng: () => 0.2 });
  match.inbound = null;
  match.frontcourtEstablished = true;
  match.possessionTeam = 0;

  for (const p of match.players) {
    p.x = 640;
    p.y = 100 + p.slotIndex * 38;
  }

  const star = match.players[0]; // PG команды 0
  const sniperWing = match.players[1]; // SG команды 0
  const defOnBall = match.players[5]; // Защитник команды 1

  // 1) Степ-бэк за трёхочковую дугу при плотной опеке у линии (dist ~210 -> отпрыгивает за THREE_PT_RADIUS = 228!)
  updatePlayerTraitsAndStatsInSandbox(match, star.id, [], {
    speed: 82,
    shooting: 86,
    stamina: 92,
    steal: 55,
    jump: 70,
    teamwork: 55,
    dribble: 84,
    iq: 82,
    deviance: 5,
  });
  star.x = HOOP_RIGHT_X - 208;
  star.y = HOOP_Y;
  star.stepBackCd = 0;
  star.eurostepCd = 0;
  star.actionCooldown = 0;
  defOnBall.x = star.x + 34;
  defOnBall.y = HOOP_Y;

  giveBallToPlayer(match, match.balls[0], star, () => 0.3, false);
  star.actionCooldown = 0;
  stepMatch(match, 1 / 60, () => 0.25);

  const distAfterStepBack = Math.hypot(star.x - HOOP_RIGHT_X, star.y - HOOP_Y);
  assert.equal(star.aiAction, 'step_back', 'Снайпер с дриблингом исполнил Степ-бэк!');
  assert.ok(
    distAfterStepBack > THREE_PT_RADIUS + 8 && star.windupTimer > 0,
    `Степ-бэк разорвал дистанцию за трёхочковую дугу (dist=${distAfterStepBack.toFixed(1)}px > ${THREE_PT_RADIUS})`
  );

  // 2) Евростеп в проходе против защитника, вставшего на пути в краске!
  star.windupTimer = 0;
  star.windupBallId = null;
  star.windupTargetHoop = null;
  giveBallToPlayer(match, match.balls[0], star, () => 0.3, false);
  star.x = HOOP_RIGHT_X - 135;
  star.y = HOOP_Y;
  star.eurostepCd = 0;
  star.actionCooldown = 0;
  defOnBall.x = HOOP_RIGHT_X - 96;
  defOnBall.y = HOOP_Y + 6;
  defOnBall.aiAction = 'take_charge';
  defOnBall.chargePlantTimer = 0.4;

  stepMatch(match, 1 / 60, () => 0.25);
  assert.equal(star.aiAction, 'eurostep', 'В проходе слэшер ушёл от тарана защитника Евростепом!');
  assert.equal(star.fouls, 0, 'Евростеп позволил обойти подставившегося защитника без фола в атаке!');

  // 3) Смещение снайпера по дуге (relocate_arc) и скидка из краски (Drive-and-Kick) при стягивании 2 защитников!
  updatePlayerTraitsAndStatsInSandbox(match, sniperWing.id, ['parking_sniper'], {
    speed: 74,
    shooting: 88,
    stamina: 90,
    steal: 50,
    jump: 62,
    teamwork: 75,
    dribble: 65,
    iq: 80,
    deviance: 5,
  });
  giveBallToPlayer(match, match.balls[0], star, () => 0.3, false);
  star.x = HOOP_RIGHT_X - 115;
  star.y = HOOP_Y;
  star.decisionTimer = 0;
  star.actionCooldown = 0;
  star.eurostepCd = 5;
  star.crossoverCd = 5;

  // Два защитника сжимают проход в краске, освобождая дугу для sniperWing
  match.players[5].x = star.x + 28;
  match.players[5].y = HOOP_Y - 12;
  match.players[6].x = star.x + 26;
  match.players[6].y = HOOP_Y + 16;
  for (let k = 7; k < 10; k++) {
    match.players[k].x = 660;
    match.players[k].y = 480;
  }

  sniperWing.x = HOOP_RIGHT_X - 245;
  sniperWing.y = HOOP_Y - 135;

  stepPlayerAI(match, sniperWing, 0.05, () => 0.3, {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  });
  assert.equal(
    sniperWing.aiAction,
    'relocate_arc',
    'Свободный снайпер смещается по трёхочковой дуге под скидку из краски!'
  );

  let kickedOutTo: string | null = null;
  stepPlayerAI(match, star, 0.05, () => 0.3, {
    onShoot: () => {},
    onPass: (_p, _b, rec) => {
      kickedOutTo = rec.id;
    },
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  });
  assert.equal(kickedOutTo, sniperWing.id, 'Игрок в краске под двойной опекой выдал скидку на дугу (Drive-and-Kick)!');
  assert.ok(star.aiThought.includes('скинул на дугу'), `Мысль ИИ: ${star.aiThought}`);
});

test('Дисциплина опорной ноги при остановленном ведении vs Двойное ведение у глупого балбеса', () => {
  const match = createMatch({ rng: () => 0.1 });
  match.inbound = null;
  match.frontcourtEstablished = true;
  match.possessionTeam = 0;

  for (const p of match.players) {
    p.x = 650;
    p.y = 110 + p.slotIndex * 45;
    p.vx = 0;
    p.vy = 0;
  }

  const smartHandler = match.players[0];
  const helperWing = match.players[1];
  updatePlayerTraitsAndStatsInSandbox(match, smartHandler.id, [], {
    speed: 75,
    shooting: 45,
    stamina: 90,
    steal: 50,
    jump: 60,
    teamwork: 75,
    dribble: 72,
    iq: 80,
    deviance: 5,
  });

  giveBallToPlayer(match, match.balls[0], smartHandler, () => 0.3, false);
  smartHandler.x = 760;
  smartHandler.y = HOOP_Y;
  smartHandler.hasStoppedDribble = true;
  smartHandler.decisionTimer = 0.8; // пока ждёт открывания партнёра

  // 1) Умный игрок без ведения держит опорную ногу на месте, а партнёр бежит открываться под скидку!
  stepPlayerAI(match, smartHandler, 0.05, () => 0.5, {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  });
  assert.equal(smartHandler.targetX, smartHandler.x, 'Умный игрок без ведения не срывает опорную ногу');
  assert.equal(smartHandler.targetY, smartHandler.y);

  updatePlayerTraitsAndStatsInSandbox(match, helperWing.id, [], {
    speed: 78,
    shooting: 65,
    stamina: 90,
    steal: 50,
    jump: 60,
    teamwork: 78,
    dribble: 70,
    iq: 76,
    deviance: 5,
  });
  helperWing.x = 710;
  helperWing.y = HOOP_Y - 55;

  stepPlayerAI(match, helperWing, 0.05, () => 0.5, {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  });
  assert.ok(
    helperWing.aiThought.includes('убрал ведение'),
    `Партнёр без мяча сразу бежит выручать игрока без ведения: ${helperWing.aiThought}`
  );

  // 2) Глупый балбес (IQ = 18) с остановленным ведением забывается, разгоняется с мячом и получает ДВОЙНОЕ ВЕДЕНИЕ!
  updatePlayerTraitsAndStatsInSandbox(match, smartHandler.id, [], {
    speed: 82,
    shooting: 30,
    stamina: 90,
    steal: 30,
    jump: 50,
    teamwork: 15,
    dribble: 50,
    iq: 18,
    deviance: 10,
  });
  for (let i = 1; i < 5; i++) {
    match.players[i].x = 640;
    match.players[i].y = 95 + i * 35;
    match.players[4 + i].x = 648; // плотно закрываем всех партнёров, чтобы не отдал пас
    match.players[4 + i].y = 95 + i * 35;
  }
  match.players[9].x = 620;
  match.players[9].y = 500;

  giveBallToPlayer(match, match.balls[0], smartHandler, () => 0.3, false);
  smartHandler.x = 740;
  smartHandler.y = HOOP_Y;
  smartHandler.hasStoppedDribble = true;
  smartHandler.decisionTimer = 2.0;

  for (let step = 0; step < 30 && !match.inbound; step++) {
    stepMatch(match, 1 / 60, () => 0.1);
  }
  const violationInb = match.inbound as { team: 0 | 1; reason: string } | null;
  assert.ok(
    violationInb !== null && violationInb.team === 1,
    'Судья зафиксировал двойное ведение у забывчивого глупого балбеса и передал мяч сопернику!'
  );
});

test('Продвинутая защита ИИ: подстановка под фол в атаке (Take Charge), заслон ПОВЕРХ/ПОД и Клоузаут', () => {
  const rng = createSeededRng(50505);
  const match = createMatch({ rng });
  match.inbound = null;
  match.frontcourtEstablished = true;
  match.possessionTeam = 0;

  for (const p of match.players) {
    p.x = 640;
    p.y = 100 + p.slotIndex * 38;
    p.vx = 0;
    p.vy = 0;
  }

  const recklessDriver = match.players[0]; // Команда 0 несётся на таран в правое кольцо
  const smartDefender = match.players[5]; // Команда 1 (высокий IQ, БЕЗ черты flopper!)
  const screener = match.players[3]; // PF команды 0 ставит заслон

  updatePlayerTraitsAndStatsInSandbox(match, smartDefender.id, [], {
    speed: 74,
    shooting: 60,
    stamina: 92,
    steal: 70,
    jump: 64,
    teamwork: 78,
    dribble: 60,
    iq: 86,
    deviance: 8,
  });
  updatePlayerTraitsAndStatsInSandbox(match, recklessDriver.id, ['gorilla_dunk'], {
    speed: 86,
    shooting: 42,
    stamina: 90,
    steal: 40,
    jump: 85,
    teamwork: 30,
    dribble: 48, // не делает евростеп — прёт напролом!
    iq: 34,
    deviance: 20,
  });

  // 1) Умный защитник выбирает проход ПОД заслоном против небросающего данкера и ПОВЕРХ заслона против снайпера!
  giveBallToPlayer(match, match.balls[0], recklessDriver, () => 0.4, false);
  recklessDriver.x = HOOP_RIGHT_X - 245;
  recklessDriver.y = HOOP_Y;
  screener.x = recklessDriver.x + 18;
  screener.y = HOOP_Y - 22;
  screener.aiAction = 'screen_set';
  smartDefender.x = recklessDriver.x + 34;
  smartDefender.y = HOOP_Y;

  const noopCb = {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  };
  stepPlayerAI(match, smartDefender, 0.05, () => 0.4, noopCb);
  const underTargetX = smartDefender.targetX;
  assert.ok(
    smartDefender.aiThought.includes('под заслон'),
    `Против небросающего данкера защитник ныряет под заслон: ${smartDefender.aiThought}`
  );

  // Теперь делаем носителя смертоносным снайпером — защитник обязан лезть ПОВЕРХ заслона ближе к мячу!
  updatePlayerTraitsAndStatsInSandbox(match, recklessDriver.id, ['parking_sniper'], {
    speed: 78,
    shooting: 90,
    stamina: 90,
    steal: 45,
    jump: 65,
    teamwork: 50,
    dribble: 75,
    iq: 75,
    deviance: 10,
  });
  stepPlayerAI(match, smartDefender, 0.05, () => 0.4, noopCb);
  assert.ok(
    smartDefender.aiThought.includes('поверх заслона') && smartDefender.targetX < underTargetX,
    `Против снайпера защитник пролазит поверх заслона плотнее к мячу: ${smartDefender.aiThought}`
  );

  // 2) Грамотный Клоузаут (closeout), когда защитник бежит с дистанции 115px на открытого стрелка!
  screener.aiAction = 'cut_open';
  screener.x = 660;
  smartDefender.x = recklessDriver.x + 115;
  smartDefender.y = HOOP_Y;
  stepPlayerAI(match, smartDefender, 0.05, () => 0.4, noopCb);
  assert.equal(smartDefender.aiAction, 'closeout', 'На дистанции восстановления защитник включает режим Клоузаута!');

  // 3) Подстановка под фол в атаке (take_charge): защитник заранее встал неподвижно за полукругом (distHoop=88px), а таранщик врезается в грудь!
  updatePlayerTraitsAndStatsInSandbox(match, recklessDriver.id, ['gorilla_dunk'], {
    speed: 88,
    shooting: 42,
    stamina: 90,
    steal: 40,
    jump: 85,
    teamwork: 20,
    dribble: 45,
    iq: 30,
    deviance: 20,
  });
  giveBallToPlayer(match, match.balls[0], recklessDriver, () => 0.4, false);
  recklessDriver.x = HOOP_RIGHT_X - 118;
  recklessDriver.y = HOOP_Y;
  recklessDriver.vx = 165;
  recklessDriver.vy = 0;
  recklessDriver.targetX = HOOP_RIGHT_X;
  recklessDriver.targetY = HOOP_Y;
  recklessDriver.decisionTimer = 5;
  recklessDriver.actionCooldown = 5;

  smartDefender.x = HOOP_RIGHT_X - 96; // 96px от кольца (вне полукруга!), прямо по курсу тарана!
  smartDefender.y = HOOP_Y;
  smartDefender.vx = 0;
  smartDefender.vy = 0;

  // Судья смотрит прямо на эпизод
  match.referee.x = HOOP_RIGHT_X - 110;
  match.referee.y = HOOP_Y - 95;
  match.referee.gazeAngle = Math.PI / 2;

  stepMatch(match, 1 / 60, () => 0.15);
  assert.equal(
    recklessDriver.fouls,
    1,
    'Умный защитник без симуляции поймал таранщика на чистый фол в нападении (Take Charge)!'
  );
  const chargeInb = match.inbound as { team: 0 | 1 } | null;
  assert.equal(chargeInb?.team, 1, 'Мяч перешёл команде грамотно подставившегося защитника');
});

test('ИИ 3-го поколения: Флоутер-слеза через руки «Шлагбаума» и Крюк центрового из лоу-поста', () => {
  const match = createMatch({ rng: createSeededRng(70101) });
  match.inbound = null;
  match.frontcourtEstablished = true;
  match.possessionTeam = 0;

  for (const p of match.players) {
    p.x = 640;
    p.y = 110 + p.slotIndex * 38;
    p.vx = 0;
    p.vy = 0;
  }

  const craftyGuard = match.players[0]; // PG команды 0
  const postCenter = match.players[4]; // C команды 0
  const giantBlocker = match.players[9]; // C команды 1 с «Шлагбаумом»

  updatePlayerTraitsAndStatsInSandbox(match, craftyGuard.id, ['short_spark'], {
    speed: 86,
    shooting: 76,
    stamina: 90,
    steal: 60,
    jump: 58,
    teamwork: 65,
    dribble: 84,
    iq: 80,
    deviance: 8,
  });
  updatePlayerTraitsAndStatsInSandbox(match, giantBlocker.id, ['barrier_blocker'], {
    speed: 55,
    shooting: 50,
    stamina: 85,
    steal: 60,
    jump: 80,
    teamwork: 60,
    dribble: 40,
    iq: 74,
    deviance: 10,
  });

  // 1) Малыш-разыгрывающий врывается на 128px от кольца против «Шлагбаума» (стоящего глубже у щита на 64px) и запускает высокий ФЛОУТЕР!
  giveBallToPlayer(match, match.balls[0], craftyGuard, () => 0.3, false);
  craftyGuard.x = HOOP_RIGHT_X - 128;
  craftyGuard.y = HOOP_Y;
  craftyGuard.actionCooldown = 0;
  craftyGuard.eurostepCd = 5;
  giantBlocker.x = HOOP_RIGHT_X - 64;
  giantBlocker.y = HOOP_Y;

  stepMatch(match, 1 / 60, () => 0.2);
  assert.equal(craftyGuard.aiAction, 'floater', 'Разыгрывающий выбрал мягкий флоутер против гиганта у кольца!');
  assert.equal(craftyGuard.windupShotStyle, 'floater');

  // Доводим замах до выпуска мяча и убеждаемся, что вершина траектории флоутера выше потолка перехвата «Шлагбаума» (145)!
  for (let i = 0; i < 12 && match.balls[0].holderId === craftyGuard.id; i++) {
    stepMatch(match, 1 / 60, () => 0.1);
  }
  assert.equal(match.balls[0].shotStyle, 'floater', 'Мяч выпущен в стиле высокого флоутера');
  let maxFloaterZ = match.balls[0].z;
  for (let i = 0; i < 45 && match.balls[0].vz > 0; i++) {
    stepMatch(match, 1 / 60, () => 0.5);
    if (match.balls[0].z > maxFloaterZ) maxFloaterZ = match.balls[0].z;
  }
  assert.ok(
    maxFloaterZ >= 190,
    `Флоутер взлетел крутой слезой выше рук центрового (maxZ=${maxFloaterZ.toFixed(1)} >= 190)`
  );

  // 2) Центровой без мяча занимает позицию в лоу-посте (post_up), а получив мяч против низкого защитника — забивает КРЮК ИЗ ПОСТА!
  match.inbound = null;
  match.frontcourtEstablished = true;
  match.possessionTeam = 0;
  giveBallToPlayer(match, match.balls[0], craftyGuard, () => 0.4, false);
  craftyGuard.x = HOOP_RIGHT_X - 270;
  craftyGuard.y = HOOP_Y - 60;
  postCenter.x = HOOP_RIGHT_X - 120;
  postCenter.y = HOOP_Y + 80;
  postCenter.paintTimer = 0;
  updatePlayerTraitsAndStatsInSandbox(match, postCenter.id, ['lanky_beanpole'], {
    speed: 56,
    shooting: 60,
    stamina: 85,
    steal: 50,
    jump: 62, // ниже порога данка 70, чтобы исполнил классический крюк!
    teamwork: 66,
    dribble: 48,
    iq: 74,
    deviance: 8,
  });

  stepPlayerAI(match, postCenter, 0.05, () => 0.4, {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  });
  assert.equal(postCenter.aiAction, 'post_up', 'Большой занял глубокую позицию спиной к кольцу в лоу-посте!');

  // Вручаем мяч центровому на усах против маленького PG и проверяем бросок крюком
  giveBallToPlayer(match, match.balls[0], postCenter, () => 0.4, false);
  postCenter.x = HOOP_RIGHT_X - 78;
  postCenter.y = HOOP_Y + 62;
  postCenter.decisionTimer = 0;
  postCenter.actionCooldown = 0;
  giantBlocker.x = 640; // убираем второго защитника
  const tinyDef = match.players[5]; // PG соперника
  tinyDef.x = postCenter.x + 28;
  tinyDef.y = postCenter.y - 10;

  let chosenHookStyle: string | undefined;
  stepPlayerAI(match, postCenter, 0.05, () => 0.4, {
    onShoot: (_pl, _b, _h, _dunk, _panic, style) => {
      chosenHookStyle = style;
    },
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  });
  assert.equal(chosenHookStyle, 'hook', 'Центровой в лоу-посте завершил атаку классическим крюком (hook)!');
  launchBallShot(match, match.balls[0], postCenter, 1, 'normal', () => 0.05, 'hook');
  assert.equal(match.balls[0].shotStyle, 'hook');
});

test('Передача из рук в руки с заслоном (DHO) и Заслон без мяча под снайпера (Pin-Down)', () => {
  const match = createMatch({ rng: createSeededRng(70202) });
  match.inbound = null;
  match.frontcourtEstablished = true;
  match.possessionTeam = 0;

  for (const p of match.players) {
    p.x = 640;
    p.y = 100 + p.slotIndex * 42;
    p.vx = 0;
    p.vy = 0;
  }

  const pointGuard = match.players[0]; // PG
  const wingSniper = match.players[1]; // SG снайпер
  const teamBig = match.players[4]; // C большой

  updatePlayerTraitsAndStatsInSandbox(match, teamBig.id, [], {
    speed: 60,
    shooting: 48, // небросающий командный центровой
    stamina: 88,
    steal: 50,
    jump: 72,
    teamwork: 82,
    dribble: 52,
    iq: 76,
    deviance: 8,
  });
  updatePlayerTraitsAndStatsInSandbox(match, wingSniper.id, ['parking_sniper'], {
    speed: 76,
    shooting: 90,
    stamina: 90,
    steal: 50,
    jump: 60,
    teamwork: 70,
    dribble: 70,
    iq: 80,
    deviance: 5,
  });

  // 1) Dribble Handoff (DHO): центровой с мячом на дуге (dist ~225px) вручает мяч пробегающему рядом снайперу и встаёт в заслон!
  giveBallToPlayer(match, match.balls[0], teamBig, () => 0.3, false);
  teamBig.x = HOOP_RIGHT_X - 225;
  teamBig.y = HOOP_Y - 40;
  teamBig.actionCooldown = 0;
  teamBig.handoffCd = 0;
  wingSniper.x = teamBig.x - 18;
  wingSniper.y = teamBig.y - 32; // в 36px от центрового!

  stepMatch(match, 1 / 60, () => 0.3);
  assert.equal(match.balls[0].holderId, wingSniper.id, 'Центровой передал мяч из рук в руки снайперу!');
  assert.equal(teamBig.aiAction, 'screen_set', 'После передачи из рук в руки большой мгновенно встал в заслон!');
  assert.ok((teamBig.handoffHoldTimer ?? 0) > 0.9, 'Таймер удержания заслона DHO активирован');

  // 2) Заслон без мяча под закрытого снайпера (offball_screen) и выбегание снайпера на трёшку (Pin-Down Curl)!
  teamBig.handoffHoldTimer = 0;
  giveBallToPlayer(match, match.balls[0], pointGuard, () => 0.3, false);
  pointGuard.x = HOOP_RIGHT_X - 295;
  pointGuard.y = HOOP_Y + 95; // PG свободен с мячом (рядом нет опекуна для он-болл заслона)

  wingSniper.x = HOOP_RIGHT_X - 210;
  wingSniper.y = HOOP_Y - 145;
  const glueDefender = match.players[6]; // защитник наглухо держит нашего снайпера без мяча!
  glueDefender.x = wingSniper.x + 18;
  glueDefender.y = wingSniper.y + 12;

  teamBig.x = HOOP_RIGHT_X - 185;
  teamBig.y = HOOP_Y - 95;

  const noopCb = {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  };
  stepPlayerAI(match, teamBig, 0.05, () => 0.3, noopCb);
  assert.equal(
    teamBig.aiAction,
    'offball_screen',
    `Свободный большой пошёл ставить заслон без мяча под снайпера: ${teamBig.aiThought}`
  );

  stepPlayerAI(match, wingSniper, 0.05, () => 0.3, noopCb);
  assert.equal(wingSniper.aiAction, 'relocate_arc', 'Снайпер огибает заслон без мяча и выбегает на чистую дугу!');
  assert.ok(wingSniper.aiThought.includes('из-под заслона без мяча'));
});

test('Охота за мисматчем, Блок-шот вдогонку (Chasedown) и Ротация защиты X-Out', () => {
  const match = createMatch({ rng: createSeededRng(70303) });
  match.inbound = null;
  match.frontcourtEstablished = true;
  match.possessionTeam = 0;

  for (const p of match.players) {
    p.x = 640;
    p.y = 105 + p.slotIndex * 40;
    p.vx = 0;
    p.vy = 0;
  }

  // 1) Охота за мисматчем: умный PG видит своего мощного центрового под кольцом против маленького «Короткого замыкания»!
  const smartPG = match.players[0];
  const wingCovered = match.players[1];
  const dominantCenter = match.players[4];
  const tinyMismatchDef = match.players[5]; // PG соперника с short_spark кроет нашего центрового!
  const normalWingDef = match.players[6];

  updatePlayerTraitsAndStatsInSandbox(match, smartPG.id, [], {
    speed: 78,
    shooting: 52,
    stamina: 90,
    steal: 55,
    jump: 60,
    teamwork: 84,
    dribble: 78,
    iq: 86,
    deviance: 5,
  });
  updatePlayerTraitsAndStatsInSandbox(match, dominantCenter.id, ['gorilla_dunk'], {
    speed: 68,
    shooting: 65,
    stamina: 88,
    steal: 50,
    jump: 68, // ниже 74, чтобы сработал чистый пас на мисматч, а не Алли-уп
    teamwork: 65,
    dribble: 55,
    iq: 72,
    deviance: 10,
  });
  updatePlayerTraitsAndStatsInSandbox(match, tinyMismatchDef.id, ['short_spark'], {
    speed: 88,
    shooting: 70,
    stamina: 88,
    steal: 60,
    jump: 52,
    teamwork: 60,
    dribble: 80,
    iq: 65,
    deviance: 10,
  });

  giveBallToPlayer(match, match.balls[0], smartPG, () => 0.4, false);
  smartPG.x = HOOP_RIGHT_X - 285;
  smartPG.y = HOOP_Y;
  smartPG.decisionTimer = 0;
  smartPG.actionCooldown = 0;

  dominantCenter.x = HOOP_RIGHT_X - 96;
  dominantCenter.y = HOOP_Y + 58;
  tinyMismatchDef.x = dominantCenter.x + 32; // позади центрового, линия паса чиста!
  tinyMismatchDef.y = dominantCenter.y + 8;

  wingCovered.x = HOOP_RIGHT_X - 240;
  wingCovered.y = HOOP_Y - 140;
  normalWingDef.x = wingCovered.x + 22;
  normalWingDef.y = wingCovered.y;

  let mismatchTargetId: string | null = null;
  stepPlayerAI(match, smartPG, 0.05, () => 0.4, {
    onShoot: () => {},
    onPass: (_p, _b, rec) => {
      mismatchTargetId = rec.id;
    },
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  });
  assert.equal(mismatchTargetId, dominantCenter.id, 'Умный плеймейкер распознал мисматч и загрузил мяч большому против малыша!');
  assert.ok(smartPG.aiThought.includes('мисматч'), `Мысль плеймейкера: ${smartPG.aiThought}`);

  // 2) Блок-шот вдогонку со спины (chasedown_block): атлетичный защитник догоняет убежавшего к кольцу игрока с мячом!
  const loneDriver = match.players[0];
  const chasedownWing = match.players[7]; // SF команды 1
  updatePlayerTraitsAndStatsInSandbox(match, chasedownWing.id, ['grasshopper'], {
    speed: 84,
    shooting: 60,
    stamina: 92,
    steal: 65,
    jump: 88,
    teamwork: 70,
    dribble: 65,
    iq: 78,
    deviance: 8,
  });
  for (let i = 5; i < 10; i++) {
    match.players[i].x = 580; // вся защита осталась позади в центре поля
    match.players[i].y = 120 + (i - 5) * 65;
  }
  giveBallToPlayer(match, match.balls[0], loneDriver, () => 0.4, false);
  loneDriver.x = HOOP_RIGHT_X - 115;
  loneDriver.y = HOOP_Y;
  loneDriver.windupTimer = 0.14;
  chasedownWing.x = HOOP_RIGHT_X - 172; // бежит за спиной в 57px!
  chasedownWing.y = HOOP_Y + 8;

  let jumpedForChasedown = false;
  stepPlayerAI(match, chasedownWing, 0.05, () => 0.4, {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {
      jumpedForChasedown = true;
    },
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  });
  assert.equal(chasedownWing.aiAction, 'chasedown_block', 'Атлетичный защитник включил погоню на блок-шот вдогонку!');
  assert.equal(jumpedForChasedown, true, 'При замахе атакующего догоняющий защитник вылетел на блок-шот со спины!');

  // 3) Ротация защиты «Помоги помогающему» (X-Out): партнёр ушёл в help_defense, и защитник слабой стороны летит на открытого снайпера!
  loneDriver.windupTimer = 0;
  loneDriver.x = HOOP_RIGHT_X - 95;
  loneDriver.y = HOOP_Y;
  const openCornerSniper = match.players[1];
  updatePlayerTraitsAndStatsInSandbox(match, openCornerSniper.id, ['parking_sniper'], {
    speed: 75,
    shooting: 90,
    stamina: 90,
    steal: 50,
    jump: 60,
    teamwork: 70,
    dribble: 65,
    iq: 75,
    deviance: 5,
  });
  openCornerSniper.x = HOOP_RIGHT_X - 235;
  openCornerSniper.y = HOOP_Y - 155; // совершенно открытый снайпер на дуге!

  match.players[5].x = loneDriver.x - 22;
  match.players[5].y = HOOP_Y;
  match.players[5].aiAction = 'guard_carrier';

  match.players[6].x = HOOP_RIGHT_X - 110;
  match.players[6].y = HOOP_Y - 25;
  match.players[6].aiAction = 'help_defense'; // бросил своего снайпера ради подстраховки краски!

  const smartWeakSideDef = match.players[8]; // PF команды 1 с высоким IQ ротируется на дугу!
  updatePlayerTraitsAndStatsInSandbox(match, smartWeakSideDef.id, [], {
    speed: 76,
    shooting: 60,
    stamina: 90,
    steal: 68,
    jump: 68,
    teamwork: 82,
    dribble: 58,
    iq: 84,
    deviance: 6,
  });
  smartWeakSideDef.x = HOOP_RIGHT_X - 175;
  smartWeakSideDef.y = HOOP_Y - 70;

  stepPlayerAI(match, smartWeakSideDef, 0.05, () => 0.4, {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
  });
  assert.equal(smartWeakSideDef.aiAction, 'closeout', 'Умный защитник слабой стороны побежал закрывать оставленного снайпера!');
  assert.ok(smartWeakSideDef.aiThought.includes('X-Out'), `Мысль защитника: ${smartWeakSideDef.aiThought}`);
});

test('Концовка Q4 (Тактический фол vs Защита без фола), Спасение мяча из аута и Ложный замах в пустоту', () => {
  const match = createMatch({ rng: createSeededRng(70404) });
  match.inbound = null;
  match.frontcourtEstablished = true;

  // 1) Тактический фол в концовке 4-й четверти: команда 1 уступает -3 за 9с до сирены (счётчик 24с больше остатка четверти!)
  match.quarter = 4;
  match.quarterTimeRemaining = 9.0;
  match.shotClock = 19.0;
  match.score = [64, 61];
  match.teamFouls = [2, 4]; // 5-й фол команды 1 остановит секундомер и отправит бить штрафные!

  for (const p of match.players) {
    p.x = 620;
    p.y = 120 + p.slotIndex * 38;
    p.vx = 0;
    p.vy = 0;
  }

  const clockMilker = match.players[0]; // Команда 0 ведёт +3 и бережно убивает время
  const clutchDefender = match.players[5]; // Умный защитник команды 1 идёт на тактический фол!
  updatePlayerTraitsAndStatsInSandbox(match, clockMilker.id, [], {
    speed: 78,
    shooting: 70,
    stamina: 90,
    steal: 60,
    jump: 60,
    teamwork: 78,
    dribble: 80,
    iq: 80,
    deviance: 5,
  });
  updatePlayerTraitsAndStatsInSandbox(match, clutchDefender.id, [], {
    speed: 80,
    shooting: 65,
    stamina: 90,
    steal: 65,
    jump: 60,
    teamwork: 80,
    dribble: 65,
    iq: 82,
    deviance: 10,
  });

  giveBallToPlayer(match, match.balls[0], clockMilker, () => 0.4, false);
  match.possessionTimer = 2.5;
  clockMilker.x = 760;
  clockMilker.y = HOOP_Y;
  clutchDefender.x = 786; // в 26px от носителя мяча
  clutchDefender.y = HOOP_Y;

  stepMatch(match, 1 / 60, () => 0.4);
  assert.equal(clutchDefender.fouls, 1, 'Уступающая команда совершила умышленный тактический фол ради остановки времени!');
  assert.ok(match.freeThrow !== null, 'Судья остановил время и назначил пробитие штрафных по бонусу фолов');

  // 2) Обратная ситуация: команда 1 ВЕДЁТ +2 за 8с до конца Q4 — умный защитник защищается с поднятыми руками БЕЗ попытки рискованного перехвата!
  match.freeThrow = null;
  match.inbound = null;
  match.score = [61, 63]; // Команда 1 ведёт +2!
  match.quarterTimeRemaining = 8.0;
  giveBallToPlayer(match, match.balls[0], clockMilker, () => 0.4, false);
  clockMilker.x = 820;
  clockMilker.y = HOOP_Y;
  clutchDefender.x = 846;
  clutchDefender.y = HOOP_Y;
  clutchDefender.stealCooldown = 0;

  let attemptedRiskySteal = false;
  stepPlayerAI(match, clutchDefender, 0.05, () => 0.4, {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {
      attemptedRiskySteal = true;
    },
    onAttemptDirtyTrick: () => {},
  });
  assert.equal(attemptedRiskySteal, false, 'Лидируя +2 в концовке, умный защитник не лезет фолить руками!');
  assert.ok(clutchDefender.aiThought.includes('без фола в концовке'));

  // 3) Спасение уходящего в аут мяча в прыжке: умный игрок спасает партнёру, а глупый балбес дарит мяч сопернику!
  const saveMatch = createMatch({ rng: createSeededRng(70405) });
  saveMatch.inbound = null;
  saveMatch.frontcourtEstablished = true;
  saveMatch.possessionTeam = 0;
  for (const p of saveMatch.players) {
    p.x = 760;
    p.y = 260 + (p.slotIndex % 3) * 45;
    p.vx = 0;
    p.vy = 0;
  }

  const smartHustler = saveMatch.players[0]; // Команда 0, высокий IQ
  const openAlly = saveMatch.players[1]; // Партнёр команды 0
  updatePlayerTraitsAndStatsInSandbox(saveMatch, smartHustler.id, [], {
    speed: 82,
    shooting: 60,
    stamina: 90,
    steal: 70,
    jump: 70,
    teamwork: 80,
    dribble: 70,
    iq: 78,
    deviance: 5,
  });
  openAlly.x = 920;
  openAlly.y = HOOP_Y;
  smartHustler.x = COURT_RIGHT - 18;
  smartHustler.y = HOOP_Y;

  const looseBall = saveMatch.balls[0];
  looseBall.holderId = null;
  looseBall.isTipoffBall = false;
  looseBall.isGuaranteedSwish = false;
  looseBall.shotType = 'none';
  looseBall.lastTouchTeam = 0;
  looseBall.x = COURT_RIGHT - 2;
  looseBall.y = HOOP_Y;
  looseBall.z = 18;
  looseBall.vx = 240; // летит прямо за правую лицевую в аут!
  looseBall.vy = 0;
  looseBall.vz = 0;

  stepMatch(saveMatch, 1 / 60, () => 0.3);
  assert.equal(saveMatch.inbound, null, 'Умный игрок в прыжке не дал мячу уйти в аут!');
  assert.equal(looseBall.shotType, 'pass', 'Мяч спасён из аута передачей назад в площадку');
  assert.equal(looseBall.passReceiverId, openAlly.id, 'Умный игрок спас мяч точно своему открытому партнёру!');

  // Теперь рядом с боковой линией оказывается глупый балбес (IQ = 16) — спасает мяч прямо чужому игроку!
  const goofyHustler = saveMatch.players[2];
  const waitingOpponent = saveMatch.players[5];
  updatePlayerTraitsAndStatsInSandbox(saveMatch, goofyHustler.id, [], {
    speed: 75,
    shooting: 40,
    stamina: 85,
    steal: 50,
    jump: 65,
    teamwork: 30,
    dribble: 40,
    iq: 16,
    deviance: 10,
  });
  smartHustler.x = 640; // убираем умного игрока подальше
  goofyHustler.x = COURT_LEFT + 18;
  goofyHustler.y = HOOP_Y;
  goofyHustler.saveOobCd = 0;
  goofyHustler.trippedTimer = 0;
  waitingOpponent.x = COURT_LEFT + 140;
  waitingOpponent.y = HOOP_Y;

  looseBall.holderId = null;
  looseBall.shotType = 'none';
  looseBall.x = COURT_LEFT + 2;
  looseBall.y = HOOP_Y;
  looseBall.z = 16;
  looseBall.vx = -240;
  looseBall.vy = 0;
  looseBall.vz = 0;

  stepMatch(saveMatch, 1 / 60, () => 0.3);
  assert.equal(saveMatch.inbound, null, 'Глупый балбес тоже героически вытащил мяч из аута...');
  assert.equal(
    looseBall.passReceiverId,
    waitingOpponent.id,
    '...но с перепугу швырнул спасённый мяч прямо в руки сопернику!'
  );

  // 4) Комедия глупого балбеса (IQ <= 25): качает воздух ложным замахом, когда в радиусе 150px вообще никого нет!
  const airFaker = saveMatch.players[3];
  updatePlayerTraitsAndStatsInSandbox(saveMatch, airFaker.id, [], {
    speed: 70,
    shooting: 38,
    stamina: 85,
    steal: 40,
    jump: 55,
    teamwork: 10,
    dribble: 50,
    iq: 18,
    deviance: 10,
  });
  for (let i = 5; i < 10; i++) {
    saveMatch.players[i].x = 420; // все защитники далеко на другой половине!
    saveMatch.players[i].y = 140 + (i - 5) * 50;
  }
  giveBallToPlayer(saveMatch, looseBall, airFaker, () => 0.3, false);
  airFaker.x = HOOP_RIGHT_X - 230;
  airFaker.y = HOOP_Y;
  airFaker.decisionTimer = 0;
  airFaker.actionCooldown = 0;
  airFaker.pumpFakeCd = 0;

  let didGhostFake = false;
  stepPlayerAI(saveMatch, airFaker, 0.05, () => 0.1, {
    onShoot: () => {},
    onPass: () => {},
    onJump: () => {},
    onAttemptSteal: () => {},
    onAttemptDirtyTrick: () => {},
    onGhostPumpFake: () => {
      didGhostFake = true;
    },
  });
  assert.equal(didGhostFake, true, 'Глупый балбес исполнил грозный ложный замах против пустого воздуха!');
});
