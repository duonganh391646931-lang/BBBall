import {
  CENTER_CIRCLE_RADIUS,
  COURT_BOTTOM,
  COURT_LEFT,
  COURT_RIGHT,
  COURT_TOP,
  HOOP_Y,
  MID_X,
  RIM_Z,
  THREE_PT_RADIUS,
  computePlayerHorizontalInterceptRadius,
  computeRefereeDetectionChance,
  getAttackingHoop,
  getDefendingHoop,
  getHoopPosition,
  isInAttackingPaint,
  isInFrontcourt,
  projectPointOntoSegment,
} from './physics';
import { BallState, MatchState, PlayerEntity, ShotStyle, TeamTactic } from './types';

export interface AIDecisionCallbacks {
  onShoot: (
    player: PlayerEntity,
    ball: BallState,
    targetHoop: 0 | 1,
    isDunk: boolean,
    isPanic: boolean,
    shotStyle?: ShotStyle
  ) => void;
  onPass: (
    player: PlayerEntity,
    ball: BallState,
    receiver: PlayerEntity,
    isAlleyOop?: boolean,
    isSafeHighLob?: boolean
  ) => void;
  onJump: (player: PlayerEntity, useSuperJump: boolean, reason: string) => void;
  onAttemptSteal: (defender: PlayerEntity, carrier: PlayerEntity, ball: BallState) => void;
  onAttemptDirtyTrick: (
    offender: PlayerEntity,
    kind: 'trip' | 'kick_ball' | 'shove',
    victim?: PlayerEntity,
    ball?: BallState
  ) => void;
  onPumpFake?: (shooter: PlayerEntity, defender: PlayerEntity) => void;
  onGhostPumpFake?: (shooter: PlayerEntity) => void;
  onCrossoverMove?: (carrier: PlayerEntity, defender: PlayerEntity, ball: BallState) => void;
  onStepBackMove?: (
    carrier: PlayerEntity,
    defender: PlayerEntity,
    ball: BallState,
    targetHoop: 0 | 1
  ) => void;
  onEurostepMove?: (
    carrier: PlayerEntity,
    defender: PlayerEntity,
    ball: BallState,
    targetHoop: 0 | 1
  ) => void;
  onHandoffMove?: (giver: PlayerEntity, receiver: PlayerEntity, ball: BallState) => void;
  onIntentionalFoul?: (defender: PlayerEntity, carrier: PlayerEntity) => void;
}

function clampCourtX(x: number): number {
  return Math.max(COURT_LEFT + 24, Math.min(COURT_RIGHT - 24, x));
}

function clampCourtY(y: number): number {
  return Math.max(COURT_TOP + 24, Math.min(COURT_BOTTOM - 24, y));
}

function clampFrontcourtX(team: 0 | 1, x: number, frontcourtEstablished: boolean): number {
  const cx = clampCourtX(x);
  if (!frontcourtEstablished) return cx;
  return team === 0 ? Math.max(MID_X + 24, cx) : Math.min(MID_X - 24, cx);
}

export function wrapAngle(rad: number): number {
  let a = rad % (Math.PI * 2);
  if (a > Math.PI) a -= Math.PI * 2;
  if (a < -Math.PI) a += Math.PI * 2;
  return a;
}

/**
 * Выбирает и фиксирует сторону обхода препятствия (-1 = вверх по паркету, +1 = вниз по паркету)
 * с временным гистерезисом, чтобы игрок никогда не метался между двумя путями равной длины.
 */
export function getCommittedBypassSide(player: PlayerEntity): -1 | 1 {
  if (player.y < COURT_TOP + 48) {
    player.dodgeSide = 1;
    player.dodgeCommitTimer = Math.max(player.dodgeCommitTimer, 0.85);
    return 1;
  }
  if (player.y > COURT_BOTTOM - 48) {
    player.dodgeSide = -1;
    player.dodgeCommitTimer = Math.max(player.dodgeCommitTimer, 0.85);
    return -1;
  }
  if (player.dodgeCommitTimer > 0) {
    return player.dodgeSide;
  }

  const topRoom = player.y - COURT_TOP;
  const botRoom = COURT_BOTTOM - player.y;
  if (topRoom > botRoom + 28) {
    player.dodgeSide = -1;
  } else if (botRoom > topRoom + 28) {
    player.dodgeSide = 1;
  }
  // При почти равном пространстве сверху и снизу сохраняем прежний player.dodgeSide без перескока!
  player.dodgeCommitTimer = 0.95;
  return player.dodgeSide;
}

/**
 * Чистый геометрический риск перехвата на линии паса (без скидки на низкий IQ).
 */
export function computeRawPassLaneDanger(
  passer: PlayerEntity,
  receiver: PlayerEntity,
  opponents: PlayerEntity[]
): number {
  const dx = receiver.x - passer.x;
  const dy = receiver.y - passer.y;
  const passDist = Math.hypot(dx, dy);
  if (passDist < 18) return 0;

  let rawRisk = 0;
  for (const opp of opponents) {
    if (opp.penaltyTimer > 0 || opp.trippedTimer > 0) continue;
    const tRaw = ((opp.x - passer.x) * dx + (opp.y - passer.y) * dy) / (passDist * passDist);
    // Учитываем всех защитников между пасующим и принимающим (включая стоящих прямо перед руками!)
    if (tRaw < 0.04 || tRaw > 0.98) continue;

    const projX = passer.x + tRaw * dx;
    const projY = passer.y + tRaw * dy;
    const perpDist = Math.hypot(opp.x - projX, opp.y - projY);
    const stealReach =
      36 + (opp.effectiveStats.steal / 100) * 20 + (opp.traitIds.includes('pickpocket') ? 10 : 0);

    if (perpDist < stealReach) {
      const closeness = 1 - perpDist / stealReach;
      const midWeight = tRaw > 0.14 && tRaw < 0.88 ? 1.3 : 0.95;
      rawRisk += closeness * 76 * midWeight;
    }
  }
  return rawRisk;
}

/**
 * Оценивает риск перехвата передачи на линии паса между пасующим и адресатом.
 * Возвращает штраф (0 = коридор чистый, до ~115 = защитник стоит прямо в линии паса).
 */
export function evaluatePassLaneRisk(
  passer: PlayerEntity,
  receiver: PlayerEntity,
  opponents: PlayerEntity[]
): number {
  const passDist = Math.hypot(receiver.x - passer.x, receiver.y - passer.y);
  const rawRisk = computeRawPassLaneDanger(passer, receiver, opponents);

  // Насколько хорошо пасующий читает линию перехвата (даже глупый игрок видит защитника в упор!)
  let visionFactor = 0.35 + (passer.effectiveStats.iq / 100) * 0.85;
  if (passer.traitIds.includes('pass_telepath')) visionFactor = 1.3;
  if (passer.traitIds.includes('sleepy_fly')) visionFactor *= 0.78;
  if (passer.traitIds.includes('panicker')) visionFactor *= 0.68;
  if (passDist < 220) {
    visionFactor = Math.max(0.65, visionFactor);
  }

  return Math.min(125, rawRisk * visionFactor);
}

/**
 * Выбирает адаптивную тактику для ИИ-тренера команды соперника по счёту, времени и ростеру.
 */
export function chooseAdaptiveAiTactic(state: MatchState, team: 0 | 1): TeamTactic {
  const roster = state.players.filter((p) => p.team === team && p.penaltyTimer <= 0);
  const oppTeam: 0 | 1 = team === 0 ? 1 : 0;
  const scoreDiff = state.score[team] - state.score[oppTeam];
  const lateGame = state.quarter >= 4 && state.quarterTimeRemaining < 24;

  // В концовке при отставании включаем агрессивный прессинг или трёхочковый штурм
  if (lateGame && scoreDiff <= -5) {
    const avgThree =
      roster.reduce((s, p) => s + p.effectiveStats.shooting, 0) / Math.max(1, roster.length);
    return avgThree >= 62 ? 'perimeter' : 'blitz';
  }
  if (lateGame && scoreDiff < 0) {
    return 'blitz';
  }

  const snipers = roster.filter(
    (p) => p.traitIds.includes('parking_sniper') || p.effectiveStats.shooting >= 76
  ).length;
  if (snipers >= 2 && scoreDiff <= 4) {
    return 'perimeter';
  }

  const avgDeviance =
    roster.reduce((s, p) => s + p.effectiveStats.deviance, 0) / Math.max(1, roster.length);
  if (avgDeviance >= 56 && state.teamFouls[team] <= 2) {
    return 'dirty';
  }

  const oppRoster = state.players.filter((p) => p.team === oppTeam && p.penaltyTimer <= 0);
  const oppHasDeadlyShooters =
    oppRoster.filter(
      (p) => p.traitIds.includes('messi') || p.traitIds.includes('parking_sniper')
    ).length >= 2;

  const avgStealSpeed =
    roster.reduce((s, p) => s + p.effectiveStats.steal + p.effectiveStats.speed, 0) /
    Math.max(1, roster.length * 2);
  if ((avgStealSpeed >= 68 && scoreDiff < 0) || (oppHasDeadlyShooters && avgStealSpeed >= 58)) {
    return 'blitz';
  }

  return 'balanced';
}

/**
 * Возвращает базовую позиционную точку в атаке для пятёрки (PG, SG, SF, PF, C)
 */
export function getOffensiveSpot(
  player: PlayerEntity,
  tactic: TeamTactic
): { x: number; y: number } {
  const atkHoopIdx = getAttackingHoop(player.team);
  const hoop = getHoopPosition(atkHoopIdx);
  const dirFromHoop = player.team === 0 ? -1 : 1; // от чужого кольца к центру

  if (player.traitIds.includes('parking_sniper') || tactic === 'perimeter') {
    // Расстановка 5-out по периметру трёхочковой дуги
    const angles = [-1.02, -0.5, 0, 0.5, 1.02];
    const ang = angles[player.slotIndex % 5];
    const dist = THREE_PT_RADIUS + 34;
    return {
      x: clampCourtX(hoop.x + dirFromHoop * Math.cos(ang) * dist),
      y: clampCourtY(hoop.y + Math.sin(ang) * dist * 0.92),
    };
  }

  // Классическая расстановка 5x5 (4-out-1-in / Horns): PG в топе, SG/SF по крыльям, PF на усах, C в данкер-споте
  switch (player.role) {
    case 'PG':
      return {
        x: clampCourtX(hoop.x + dirFromHoop * (THREE_PT_RADIUS + 40)),
        y: HOOP_Y,
      };
    case 'SG':
      return {
        x: clampCourtX(hoop.x + dirFromHoop * (THREE_PT_RADIUS - 6)),
        y: clampCourtY(HOOP_Y - 180),
      };
    case 'SF':
      return {
        x: clampCourtX(hoop.x + dirFromHoop * (THREE_PT_RADIUS - 6)),
        y: clampCourtY(HOOP_Y + 180),
      };
    case 'PF':
      return {
        x: clampCourtX(hoop.x + dirFromHoop * 148),
        y: clampCourtY(HOOP_Y - 102),
      };
    case 'C':
    default:
      // Данкер-спот сразу за линией трёхсекундной зоны (чтобы не ловить нарушение 3 секунд стоя!)
      return {
        x: clampCourtX(hoop.x + dirFromHoop * 78),
        y: clampCourtY(HOOP_Y + 92),
      };
  }
}

/**
 * Умный выбор траектории ведения мяча (обход защитников, использование заслона, выход из ловушек).
 */
function chooseSmartDriveTarget(
  state: MatchState,
  player: PlayerEntity,
  teammates: PlayerEntity[],
  opponents: PlayerEntity[]
): { x: number; y: number; thought: string } {
  const atkHoopIdx = getAttackingHoop(player.team);
  const atkHoop = getHoopPosition(atkHoopIdx);
  const dirBack = player.team === 0 ? -1 : 1;
  const distToHoop = Math.hypot(player.x - atkHoop.x, player.y - atkHoop.y);
  const oppTeam: 0 | 1 = player.team === 0 ? 1 : 0;
  const needThreeLate =
    state.quarter >= 4 &&
    state.quarterTimeRemaining < 18 &&
    state.score[player.team] - state.score[oppTeam] <= -3 &&
    player.effectiveStats.iq >= 48;

  // Снайпер с парковки или нужда в трёшке в концовке — ищем чистую точку за дугой
  const wantsPerimeter =
    player.traitIds.includes('parking_sniper') ||
    (needThreeLate && player.effectiveStats.shooting >= 55);

  const desiredRadius = wantsPerimeter
    ? THREE_PT_RADIUS + 32
    : player.traitIds.includes('gorilla_dunk') || player.effectiveStats.jump >= 74
    ? 68
    : player.effectiveStats.shooting >= 74
    ? 185
    : 92;

  const candidates: Array<{ x: number; y: number; bonus: number; label: string }> = [];

  // Кандидаты атак по центру и обоим флангам/полуфлангам
  const angles = wantsPerimeter ? [-0.88, -0.44, 0, 0.44, 0.88] : [-0.72, -0.35, 0, 0.35, 0.72];
  for (const ang of angles) {
    const cx = clampCourtX(atkHoop.x + dirBack * Math.cos(ang) * desiredRadius);
    const cy = clampCourtY(atkHoop.y + Math.sin(ang) * desiredRadius * 0.95);
    candidates.push({
      x: cx,
      y: cy,
      bonus: Math.abs(ang) < 0.1 ? 8 : 4,
      label: wantsPerimeter ? 'Выходит на дугу под трёшку' : 'Врывается в свободный коридор',
    });
  }

  // Если рядом стоит союзник с заслоном (screen_set) — добавляем маршрут впритирку к заслону (Пик-н-ролл!)
  const activeScreener = teammates.find((m) => m.aiAction === 'screen_set');
  if (activeScreener && Math.hypot(activeScreener.x - player.x, activeScreener.y - player.y) < 190) {
    const screenSide =
      Math.abs(activeScreener.y - HOOP_Y) < 22
        ? getCommittedBypassSide(player)
        : activeScreener.y < HOOP_Y
        ? 1
        : -1;
    const overTopY = activeScreener.y + screenSide * 28;
    const pastScreenX = clampCourtX(activeScreener.x - dirBack * 55);
    candidates.push({
      x: pastScreenX,
      y: clampCourtY(overTopY),
      bonus: 36 + (player.effectiveStats.iq / 100) * 18,
      label: `Уходит под заслон ${activeScreener.name.split(' ')[0]}!`,
    });
  }

  // Если мяч ещё в тыловой зоне — срочно пересекаем центральную линию (правило 8 секунд!)
  if (!isInFrontcourt(player.team, player.x)) {
    const crossX = player.team === 0 ? MID_X + 75 : MID_X - 75;
    for (const laneY of [player.y, HOOP_Y - 110, HOOP_Y, HOOP_Y + 110]) {
      candidates.push({
        x: clampCourtX(crossX),
        y: clampCourtY(laneY),
        bonus: state.backcourtTimer >= 3.5 ? 65 : 32,
        label:
          state.backcourtTimer >= 4.0
            ? 'Срочно выводит из зоны (правило 8с)!'
            : 'Переводит мяч через центр',
      });
    }
  }

  // Если игрок с мячом застоялся в чужой краске — выходим за линию 3-секундной зоны!
  if (player.paintTimer >= 1.8 && isInAttackingPaint(player)) {
    const paintSide =
      Math.abs(player.y - HOOP_Y) < 24
        ? getCommittedBypassSide(player)
        : player.y < HOOP_Y
        ? -1
        : 1;
    const exitY = HOOP_Y + paintSide * 96;
    candidates.push({
      x: clampFrontcourtX(player.team, player.x, state.frontcourtEstablished),
      y: clampCourtY(exitY),
      bonus: 58,
      label: 'Выходит из краски (правило 3с)!',
    });
  }

  // Локальный кроссовер и уход в сторону от защитника, стоящего прямо перед носом
  let closeDefs = 0;
  let nearestFrontDef: PlayerEntity | null = null;
  let nearestFrontDist = 999;
  const toHoopAng = Math.atan2(atkHoop.y - player.y, atkHoop.x - player.x);

  for (const opp of opponents) {
    if (opp.trippedTimer > 0) continue;
    const d = Math.hypot(opp.x - player.x, opp.y - player.y);
    if (d < 62) closeDefs++;
    if (d < 92 && d < nearestFrontDist) {
      const angToOpp = Math.atan2(opp.y - player.y, opp.x - player.x);
      const diff = Math.abs(wrapAngle(angToOpp - toHoopAng));
      if (diff < 1.25) {
        nearestFrontDist = d;
        nearestFrontDef = opp;
      }
    }
  }

  const committedSide = getCommittedBypassSide(player);

  if (nearestFrontDef) {
    const baseAng = Math.atan2(nearestFrontDef.y - player.y, nearestFrontDef.x - player.x);

    for (const sign of [-1, 1] as const) {
      const crossAng = baseAng + sign * 1.16; // ~66 градусов в обход бедра защитника!
      const rawCx = player.x + Math.cos(crossAng) * 132;
      const rawCy = player.y + Math.sin(crossAng) * 132;
      const matchesCommitted =
        (committedSide < 0 && rawCy <= player.y) || (committedSide > 0 && rawCy > player.y);
      candidates.push({
        x: clampFrontcourtX(player.team, rawCx, state.frontcourtEstablished),
        y: clampCourtY(rawCy),
        bonus: (nearestFrontDist < 46 ? 54 : 36) + (matchesCommitted ? 36 : 0),
        label: 'Резкий кроссовер в обход защитника!',
      });
    }

    // Если зажали очень плотно — диагональный отскок с ведением в зафиксированную сторону (без перескока у HOOP_Y!)
    if (nearestFrontDist < 38 || closeDefs >= 2) {
      const escapeY = player.y + committedSide * 115;
      const retreatX = player.x + dirBack * 68;
      candidates.push({
        x: clampFrontcourtX(player.team, retreatX, state.frontcourtEstablished),
        y: clampCourtY(escapeY),
        bonus: 46,
        label: 'Уходит с ведением из-под зажима!',
      });
    }
  }

  if (closeDefs >= 2 && distToHoop < 220 && !player.traitIds.includes('gorilla_dunk')) {
    candidates.push({
      x: clampFrontcourtX(
        player.team,
        atkHoop.x + dirBack * (THREE_PT_RADIUS + 26),
        state.frontcourtEstablished
      ),
      y: clampCourtY(HOOP_Y + committedSide * 90),
      bonus: 34,
      label: 'Степ-бэк из-под двойной опеки!',
    });
  }

  // Оцениваем каждый кандидат с учётом открытости коридора, инерции курса и гистерезиса цели!
  let best = candidates[0];
  let bestScore = -9999;
  const iqNorm = player.effectiveStats.iq / 100;
  const speedNow = Math.hypot(player.vx, player.vy);
  const refHeading = speedNow > 22 ? Math.atan2(player.vy, player.vx) : player.heading;
  const hasPrevCommit =
    player.driveCommitX !== undefined && player.driveCommitY !== undefined;

  for (const c of candidates) {
    const travelDist = Math.hypot(c.x - player.x, c.y - player.y);
    let score = c.bonus - travelDist * 0.08;

    // Бонус сохранения текущего курса: если два направления вокруг защитника равны по длине, балбес уверенно держит выбранную дугу!
    if (travelDist > 16) {
      const candAng = Math.atan2(c.y - player.y, c.x - player.x);
      score += Math.cos(wrapAngle(candAng - refHeading)) * 22;
    }
    if (
      hasPrevCommit &&
      Math.hypot(c.x - player.driveCommitX!, c.y - player.driveCommitY!) < 46
    ) {
      score += 18;
    }

    // Насколько чиста сама точка и первая половина пути до неё от защитников
    for (const opp of opponents) {
      if (opp.trippedTimer > 0) continue;
      const dSpot = Math.hypot(opp.x - c.x, opp.y - c.y);
      if (dSpot < 95) {
        score -= (95 - dSpot) * (0.45 + iqNorm * 0.45);
      }
      if (travelDist > 24) {
        const proj = projectPointOntoSegment(opp.x, opp.y, player.x, player.y, c.x, c.y);
        if (proj.perpDist < 42 && proj.t < 0.75) {
          score -= (42 - proj.perpDist) * (0.6 + iqNorm * 0.55);
        }
      }
    }

    // Не сбиваемся в кучу с собственными партнёрами (кроме заслоняющего)
    for (const mate of teammates) {
      if (mate.aiAction === 'screen_set') continue;
      const dMate = Math.hypot(mate.x - c.x, mate.y - c.y);
      if (dMate < 68) {
        score -= (68 - dMate) * 0.35;
      }
    }

    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }

  // Короткая фиксация выбранного ориентира (0.26с), чтобы исключить микро-колебания между соседними кадрами
  let chosenX = clampFrontcourtX(player.team, best.x, state.frontcourtEstablished);
  let chosenY = clampCourtY(best.y);
  if (
    (player.driveLockTimer ?? 0) > 0 &&
    hasPrevCommit &&
    Math.hypot(player.driveCommitX! - player.x, player.driveCommitY! - player.y) > 26
  ) {
    const prevBlocked = opponents.some((opp) => {
      if (opp.trippedTimer > 0) return false;
      const proj = projectPointOntoSegment(
        opp.x,
        opp.y,
        player.x,
        player.y,
        player.driveCommitX!,
        player.driveCommitY!
      );
      return proj.perpDist < 24 && proj.t < 0.65;
    });
    if (!prevBlocked && Math.hypot(chosenX - player.driveCommitX!, chosenY - player.driveCommitY!) > 55) {
      chosenX = clampFrontcourtX(player.team, player.driveCommitX!, state.frontcourtEstablished);
      chosenY = clampCourtY(player.driveCommitY!);
    } else {
      player.driveCommitX = chosenX;
      player.driveCommitY = chosenY;
    }
  } else {
    player.driveCommitX = chosenX;
    player.driveCommitY = chosenY;
    player.driveLockTimer = 0.26;
  }

  // Черта «Суетливый краб» добавляет плавный зигзаг при ведении
  const crabOffset = player.traitIds.includes('crab_scuttler')
    ? Math.sin(state.quarterTimeRemaining * 5.2 + player.slotIndex) * 58
    : 0;

  return {
    x: chosenX,
    y: clampCourtY(chosenY + crabOffset),
    thought: best.label,
  };
}

/**
 * Динамическое распределение персональной опеки без перекрёстной беготни через всю площадку.
 */
function findAssignedOpponent(
  defender: PlayerEntity,
  defTeammates: PlayerEntity[],
  opponents: PlayerEntity[],
  carrierId: string | null
): PlayerEntity | null {
  const defHoop = getHoopPosition(getDefendingHoop(defender.team));

  // Список полевых защитников, которые не заняты страховкой кольца и не сидят на носителе мяча
  const offBallOpps = opponents
    .filter((o) => o.id !== carrierId && o.penaltyTimer <= 0)
    .sort((a, b) => {
      const da = Math.hypot(a.x - defHoop.x, a.y - defHoop.y);
      const db = Math.hypot(b.x - defHoop.x, b.y - defHoop.y);
      return da - db;
    });

  if (offBallOpps.length === 0) {
    return opponents[0] ?? null;
  }

  const offBallDefs = [defender, ...defTeammates]
    .filter(
      (d) =>
        d.penaltyTimer <= 0 &&
        d.aiAction !== 'guard_carrier' &&
        d.aiAction !== 'closeout' &&
        d.aiAction !== 'take_charge' &&
        d.aiAction !== 'rim_protect' &&
        !d.schizoBreadMode
    )
    .sort((a, b) => a.slotIndex - b.slotIndex);

  // Жадно сопоставляем каждого опасного атакующего ближайшему свободному защитнику (с гистерезисом markTargetId!)
  const usedDefIds = new Set<string>();
  for (const opp of offBallOpps) {
    let closestDef: PlayerEntity | null = null;
    let closestDist = 99999;
    for (const d of offBallDefs) {
      if (usedDefIds.has(d.id)) continue;
      const stickyBonus = d.markTargetId === opp.id ? -34 : 0;
      const dist = Math.hypot(d.x - opp.x, d.y - opp.y) + stickyBonus;
      if (dist < closestDist) {
        closestDist = dist;
        closestDef = d;
      }
    }
    if (closestDef) {
      usedDefIds.add(closestDef.id);
      if (closestDef.id === defender.id) {
        defender.markTargetId = opp.id;
        return opp;
      }
    }
  }

  // Запасной вариант — ближайший соперник без мяча (с учётом липкости прошлой цели)
  let fallback = offBallOpps[0];
  let bestD =
    Math.hypot(defender.x - fallback.x, defender.y - fallback.y) +
    (defender.markTargetId === fallback.id ? -34 : 0);
  for (const opp of offBallOpps) {
    const d =
      Math.hypot(defender.x - opp.x, defender.y - opp.y) +
      (defender.markTargetId === opp.id ? -34 : 0);
    if (d < bestD) {
      bestD = d;
      fallback = opp;
    }
  }
  defender.markTargetId = fallback.id;
  return fallback;
}

export function stepPlayerAI(
  state: MatchState,
  player: PlayerEntity,
  dt: number,
  rng: () => number,
  callbacks: AIDecisionCallbacks
): void {
  if (player.cutTimer > 0) {
    player.cutTimer = Math.max(0, player.cutTimer - dt);
  }
  if ((player.pumpFakeTimer ?? 0) > 0) {
    player.pumpFakeTimer = Math.max(0, (player.pumpFakeTimer ?? 0) - dt);
  }
  if ((player.pumpFakeCd ?? 0) > 0) {
    player.pumpFakeCd = Math.max(0, (player.pumpFakeCd ?? 0) - dt);
  }
  if ((player.bittenByFakeTimer ?? 0) > 0) {
    player.bittenByFakeTimer = Math.max(0, (player.bittenByFakeTimer ?? 0) - dt);
  }
  if ((player.crossoverCd ?? 0) > 0) {
    player.crossoverCd = Math.max(0, (player.crossoverCd ?? 0) - dt);
  }
  if ((player.stepBackCd ?? 0) > 0) {
    player.stepBackCd = Math.max(0, (player.stepBackCd ?? 0) - dt);
  }
  if ((player.eurostepCd ?? 0) > 0) {
    player.eurostepCd = Math.max(0, (player.eurostepCd ?? 0) - dt);
  }
  if ((player.chargePlantTimer ?? 0) > 0) {
    player.chargePlantTimer = Math.max(0, (player.chargePlantTimer ?? 0) - dt);
  }
  if ((player.ballWatchTimer ?? 0) > 0) {
    player.ballWatchTimer = Math.max(0, (player.ballWatchTimer ?? 0) - dt);
  }
  if ((player.handoffHoldTimer ?? 0) > 0) {
    player.handoffHoldTimer = Math.max(0, (player.handoffHoldTimer ?? 0) - dt);
  }
  if ((player.handoffCd ?? 0) > 0) {
    player.handoffCd = Math.max(0, (player.handoffCd ?? 0) - dt);
  }

  // На скамейке штрафников или сбит с ног — не управляется
  if (player.penaltyTimer > 0) {
    player.aiAction = 'penalty_bench';
    player.aiThought = `Штрафная скамья (${player.penaltyTimer.toFixed(0)}с)`;
    player.screenTargetId = null;
    return;
  }
  if (player.trippedTimer > 0) {
    player.aiAction = 'tripped_down';
    player.aiThought = 'Ай, подножка!';
    player.screenTargetId = null;
    return;
  }

  // Во время исполнения штрафных бросков позициями управляет судейский протокол в sim.ts
  if (state.freeThrow) {
    player.screenTargetId = null;
    return;
  }

  const activePlayers = state.players.filter((p) => p.penaltyTimer <= 0);
  const teammates = activePlayers.filter((p) => p.team === player.team && p.id !== player.id);
  const opponents = activePlayers.filter((p) => p.team !== player.team);
  const teamTactic = state.tactics[player.team];

  const heldBall = state.balls.find((b) => b.holderId === player.id) ?? null;

  // ==================== 0. ВВОД МЯЧА ИЗ-ЗА БОКОВОЙ / ЛИЦЕВОЙ ЛИНИИ (ПРАВИЛО 5 СЕКУНД) ====================
  if (state.inbound) {
    const inb = state.inbound;
    player.screenTargetId = null;
    const inbInBackcourt = !isInFrontcourt(inb.team, inb.x);

    // 0а) Игрок, вводящий мяч из-за пределов площадки (обязан отдать чистый пас за 5 секунд!)
    if (player.id === inb.passerId && heldBall) {
      player.aiAction = 'inbound_pass';

      // По правилам FIBA/NBA после забитого мяча (isAfterScore) вводящий имеет право двигаться вдоль лицевой линии, выбирая чистый коридор!
      if (inb.isAfterScore) {
        const topDefs = opponents.filter((o) => o.y < HOOP_Y && Math.abs(o.x - inb.x) < 260).length;
        const botDefs = opponents.filter((o) => o.y >= HOOP_Y && Math.abs(o.x - inb.x) < 260).length;
        const desiredInbY = topDefs <= botDefs ? HOOP_Y - 88 : HOOP_Y + 88;
        inb.y += Math.sign(desiredInbY - inb.y) * Math.min(Math.abs(desiredInbY - inb.y), 135 * dt);
        inb.y = Math.max(COURT_TOP + 54, Math.min(COURT_BOTTOM - 54, inb.y));
      }

      player.targetX = inb.x;
      player.targetY = inb.y;
      player.aiThought = `Ищет открытого под ввод (${inb.timer.toFixed(1)}с)...`;

      // Даём 0.58с на открывание партнёров и отход соперников в защиту, не швыряя мяч в руки чужим под кольцом!
      if (inb.timer <= 4.42 && player.actionCooldown <= 0) {
        let bestMate: PlayerEntity | null = null;
        let bestScore = -9999;
        let bestRawDanger = 999;

        for (const mate of teammates) {
          if (mate.trippedTimer > 0) continue;
          const dInb = Math.hypot(mate.x - inb.x, mate.y - inb.y);
          let nearestDef = 999;
          for (const opp of opponents) {
            if (opp.trippedTimer > 0) continue;
            const d = Math.hypot(opp.x - mate.x, opp.y - mate.y);
            if (d < nearestDef) nearestDef = d;
          }
          const rawDanger = computeRawPassLaneDanger(player, mate, opponents);
          const guardBonus = mate.role === 'PG' ? 18 : mate.role === 'SG' ? 12 : 0;
          const distSuitability = dInb >= 70 && dInb <= 320 ? 34 : dInb < 490 ? 14 : -30;
          const score =
            distSuitability +
            guardBonus +
            Math.min(135, nearestDef) * 0.65 -
            rawDanger * 1.55;

          if (score > bestScore) {
            bestScore = score;
            bestMate = mate;
            bestRawDanger = rawDanger;
          }
        }

        const cleanWindow = bestRawDanger < 20 && bestScore > 22;
        const urgentWindow = inb.timer <= 2.2 && bestRawDanger < 34 && bestScore > 8;
        const mustReleaseNow = inb.timer <= 1.1;

        if (bestMate && (cleanWindow || urgentWindow || mustReleaseNow)) {
          const useSafeHighLob = bestRawDanger >= 16;
          player.aiThought = useSafeHighLob
            ? `Навешивает через руки на ${bestMate.name.split(' ')[0]}!`
            : `Вбрасывает на открытого ${bestMate.name.split(' ')[0]}!`;
          callbacks.onPass(player, heldBall, bestMate, false, useSafeHighLob);
        }
      }
      return;
    }

    // 0б) Партнёры открываются на паркете под приём мяча из-за линии (растягивая оборону от своего щита!)
    if (player.team === inb.team) {
      player.aiAction = 'inbound_wait';
      player.aiThought = 'Открывается под ввод мяча!';
      const inDirX = inb.x < COURT_LEFT ? 1 : inb.x > COURT_RIGHT ? -1 : player.team === 0 ? 1 : -1;
      const inDirY = inb.y < COURT_TOP ? 1 : inb.y > COURT_BOTTOM ? -1 : 0;
      const spreadIdx = player.slotIndex % 4;

      if (Math.abs(inDirY) > 0) {
        // Ввод с боковой линии
        const offsetsX = [-135, -35, 75, 180];
        const depthsY = [95, 165, 120, 215];
        player.targetX = clampCourtX(inb.x + offsetsX[spreadIdx]);
        player.targetY = clampCourtY(
          (inb.y < COURT_TOP ? COURT_TOP : COURT_BOTTOM) + inDirY * depthsY[spreadIdx]
        );
      } else {
        // Ввод с лицевой линии: PG и SG открываются широкими углами у краёв, а SF/PF/C уходят к центру площадки!
        const baseEdgeX = inb.x < COURT_LEFT ? COURT_LEFT : COURT_RIGHT;
        if (player.role === 'PG') {
          const cleanerY = inb.y <= HOOP_Y ? HOOP_Y - 118 : HOOP_Y + 118;
          player.targetX = clampCourtX(baseEdgeX + inDirX * 122);
          player.targetY = clampCourtY(cleanerY);
        } else if (player.role === 'SG') {
          const oppositeY = inb.y <= HOOP_Y ? HOOP_Y + 132 : HOOP_Y - 132;
          player.targetX = clampCourtX(baseEdgeX + inDirX * 195);
          player.targetY = clampCourtY(oppositeY);
        } else {
          const upcourtDepths = [295, 380, 455];
          const upcourtYs = [HOOP_Y, HOOP_Y - 125, HOOP_Y + 125];
          const idx = player.slotIndex % 3;
          player.targetX = clampCourtX(baseEdgeX + inDirX * upcourtDepths[idx]);
          player.targetY = clampCourtY(upcourtYs[idx]);
        }
      }
      return;
    }

    // 0в) Защищающаяся команда при вводе мяча соперником:
    // Если соперник вводит мяч со СВОЕЙ половины (после нашего гола или аута у их лицевой),
    // без тактики «Прессинг» команда организованно откатывается в позиционную защиту на свою половину!
    const defHoop = getHoopPosition(getDefendingHoop(player.team));
    const towardDefHalf = player.team === 0 ? -1 : 1;

    if (inbInBackcourt && teamTactic !== 'blitz') {
      player.aiAction = 'mark_man';
      player.aiThought = 'Возвращается в защиту на свою половину';
      const retreatSpots: Record<PlayerEntity['role'], { x: number; y: number }> = {
        PG: { x: MID_X + towardDefHalf * 62, y: HOOP_Y },
        SG: { x: MID_X + towardDefHalf * 145, y: HOOP_Y - 140 },
        SF: { x: MID_X + towardDefHalf * 145, y: HOOP_Y + 140 },
        PF: { x: MID_X + towardDefHalf * 235, y: HOOP_Y - 82 },
        C: { x: defHoop.x - towardDefHalf * 92, y: HOOP_Y + 24 },
      };
      const sp = retreatSpots[player.role];
      player.targetX = clampCourtX(sp.x);
      player.targetY = clampCourtY(sp.y);
      return;
    }

    // Даже при активном «Прессинге» (blitz) на чужой половине прессингуют только PG и SG (на дистанции >= 96px от вводящего), а большие страхуют тыл!
    if (inbInBackcourt && teamTactic === 'blitz' && player.role !== 'PG' && player.role !== 'SG') {
      player.aiAction = 'mark_man';
      player.aiThought = 'Страхует центр при высоком прессинге';
      player.targetX = clampCourtX(
        player.role === 'C' ? defHoop.x - towardDefHalf * 105 : MID_X + towardDefHalf * 85
      );
      player.targetY = clampCourtY(
        player.role === 'SF' ? HOOP_Y - 115 : player.role === 'PF' ? HOOP_Y + 115 : HOOP_Y
      );
      return;
    }

    const assigned = findAssignedOpponent(player, teammates, opponents, inb.passerId);
    let tx = assigned
      ? assigned.x + (defHoop.x - assigned.x) * 0.28
      : defHoop.x + (player.team === 0 ? 120 : -120);
    let ty = assigned ? assigned.y + (defHoop.y - assigned.y) * 0.28 : HOOP_Y;
    const minClearance = inbInBackcourt ? 98 : 76;
    const dToInb = Math.hypot(tx - inb.x, ty - inb.y);
    if (dToInb < minClearance && dToInb > 1e-2) {
      tx = inb.x + ((tx - inb.x) / dToInb) * minClearance;
      ty = inb.y + ((ty - inb.y) / dToInb) * minClearance;
    }
    player.aiAction = 'mark_man';
    player.aiThought = 'Кроет приём ввода из аута';
    player.targetX = clampCourtX(tx);
    player.targetY = clampCourtY(ty);
    return;
  }

  // Обработка «Биполярного гения» (только в живой игре вне ввода из аута)
  if (player.traitIds.includes('schizo_tactician')) {
    player.schizoTimer -= dt;
    if (player.schizoTimer <= 0) {
      player.schizoTimer = 9 + rng() * 3;
      player.schizoBreadMode = !player.schizoBreadMode;
    }
    if (player.schizoBreadMode && !heldBall) {
      player.aiAction = 'bread_mode';
      player.aiThought = 'Режим хлебушка: считает ворон...';
      player.targetX = player.team === 0 ? COURT_LEFT + 45 : COURT_RIGHT - 45;
      player.targetY = player.slotIndex % 2 === 0 ? COURT_TOP + 38 : COURT_BOTTOM - 38;
      player.screenTargetId = null;
      return;
    }
  }

  // Если он готовится к броску (замах)
  if (player.windupTimer > 0 && heldBall) {
    player.screenTargetId = null;
    return;
  }

  // ==================== 1. ПЕРЕХВАТ ВЕРХОВЫХ БРОСКОВ («КУЗНЕЧИК» / «ШЛАГБАУМ») И ПАСОВ ====================
  if (!heldBall) {
    for (const ball of state.balls) {
      if (
        ball.holderId === null &&
        (ball.shotType === 'messi' || ball.shotType === 'normal' || ball.shotType === 'pass')
      ) {
        const defHoopIdx = getDefendingHoop(player.team);
        const isThreatToOurHoop = ball.targetHoop === defHoopIdx;
        const isEnemyPass = ball.shotType === 'pass' && ball.lastTouchTeam !== player.team;

        // 1а) Высотный перехват бросков Кузнечиком или Шлагбаумом
        if (isThreatToOurHoop || (isEnemyPass && player.traitIds.includes('grasshopper'))) {
          let destX = ball.x + ball.vx * 0.55;
          let destY = ball.y + ball.vy * 0.55;
          if (ball.targetHoop !== null) {
            const h = getHoopPosition(ball.targetHoop);
            destX = h.x;
            destY = h.y;
          }
          const proj = projectPointOntoSegment(player.x, player.y, ball.x, ball.y, destX, destY);
          const canHighIntercept =
            player.traitIds.includes('grasshopper') ||
            (player.traitIds.includes('barrier_blocker') &&
              ball.z < 145 &&
              ball.shotStyle !== 'floater');

          // Умный защитник не трогает обычный бросок на снижении прямо над цилиндром кольца (Помеха попаданию / Goaltending!)
          const distBallToHoop = Math.hypot(ball.x - destX, ball.y - destY);
          const wouldBeGoaltending =
            ball.shotType === 'normal' &&
            ball.vz < -18 &&
            ball.z >= RIM_Z - 3 &&
            distBallToHoop <= 88 &&
            player.effectiveStats.iq >= 38;

          if (canHighIntercept && !wouldBeGoaltending && proj.perpDist < 195) {
            player.aiAction = 'intercept_air';
            player.screenTargetId = null;
            player.aiThought =
              ball.shotType === 'messi' ? 'На пути удара Месси!' : 'Перехват в воздухе!';
            player.targetX = clampCourtX(proj.projX);
            player.targetY = clampCourtY(proj.projY);

            const horizDistToBall = Math.hypot(player.x - ball.x, player.y - ball.y);
            const grabR = computePlayerHorizontalInterceptRadius(player);
            if (
              player.z === 0 &&
              horizDistToBall <= grabR + Math.hypot(ball.vx, ball.vy) * 0.24 &&
              ball.z > 55
            ) {
              const useSuper = player.traitIds.includes('grasshopper') && player.superJumpCd <= 0;
              callbacks.onJump(player, useSuper, 'Перехват дуги!');
            }
            return;
          }
        }

        // 1б) Чтение и перехват вражеского паса защитником на линии передачи!
        if (isEnemyPass) {
          const destX = ball.x + ball.vx * 0.45;
          const destY = ball.y + ball.vy * 0.45;
          const proj = projectPointOntoSegment(player.x, player.y, ball.x, ball.y, destX, destY);
          const readRange =
            48 +
            (player.effectiveStats.iq / 100) * 32 +
            (player.effectiveStats.steal / 100) * 24 +
            (player.traitIds.includes('pickpocket') ? 22 : 0);

          if (proj.perpDist < readRange) {
            player.aiAction = 'intercept_air';
            player.screenTargetId = null;
            player.aiThought = 'Читает и перехватывает пас!';
            player.targetX = clampCourtX(proj.projX);
            player.targetY = clampCourtY(proj.projY);
            if (
              player.z === 0 &&
              Math.hypot(player.x - ball.x, player.y - ball.y) < 48 &&
              ball.z > 42 &&
              ball.z < 125
            ) {
              callbacks.onJump(player, false, 'Перехват паса!');
            }
            return;
          }
        }
      }
    }
  }

  // ==================== 2. АДРЕСАТ ВСТРЕЧАЕТ ЛЕТЯЩИЙ СОЮЗНЫЙ ПАС ====================
  if (!heldBall) {
    const incomingPass = state.balls.find(
      (b) =>
        b.holderId === null &&
        b.shotType === 'pass' &&
        b.lastTouchTeam === player.team &&
        b.passReceiverId === player.id
    );
    if (incomingPass) {
      player.aiAction = 'receive_pass';
      player.screenTargetId = null;
      player.aiThought = 'Открылся и ловит передачу!';
      player.targetX = clampCourtX(incomingPass.x + incomingPass.vx * 0.24);
      player.targetY = clampCourtY(incomingPass.y + incomingPass.vy * 0.24);
      if (
        player.z === 0 &&
        Math.hypot(player.x - incomingPass.x, player.y - incomingPass.y) < 46 &&
        incomingPass.z > 48
      ) {
        callbacks.onJump(player, false, 'Ловит навес!');
      }
      return;
    }
  }

  // ==================== 3. ГРЯЗНАЯ ИГРА С УЧЁТОМ ФОЛ-ТРАБЛА И ВЗГЛЯДА СУДЬИ ====================
  if (!heldBall && player.dirtyTrickCd <= 0 && player.energy >= 14 && player.z === 0) {
    const baseDeviance =
      player.effectiveStats.deviance + (teamTactic === 'dirty' ? 22 : 0);

    // Дисциплина ИИ: при 4 фолах (на грани удаления!) или при удержании шаткого счёта в концовке умный игрок не фолит грязно
    const refCheck = computeRefereeDetectionChance(state.referee, player);
    const oppTeamIdx: 0 | 1 = player.team === 0 ? 1 : 0;
    const myLead = state.score[player.team] - state.score[oppTeamIdx];
    const protectingLateLead =
      state.quarter >= 4 &&
      state.quarterTimeRemaining <= 14 &&
      myLead >= 1 &&
      myLead <= 4 &&
      player.effectiveStats.iq >= 52 &&
      baseDeviance < 85;
    const foulTroubleCaution =
      protectingLateLead ||
      (player.fouls >= 4 &&
        player.effectiveStats.iq >= 45 &&
        baseDeviance < 88 &&
        refCheck.inFrontOfRef);

    if (baseDeviance >= 32 && !foulTroubleCaution) {
      const stealthMultiplier =
        !refCheck.inFrontOfRef && player.effectiveStats.iq > 52 ? 1.65 : 0.92;
      const foulPenaltyMod = player.fouls >= 3 && player.effectiveStats.iq > 55 ? 0.55 : 1.0;
      const attemptProb =
        Math.pow(Math.max(0, baseDeviance - 20) / 80, 2.0) *
        0.32 *
        stealthMultiplier *
        foulPenaltyMod *
        dt;

      if (rng() < attemptProb) {
        const kickableBall = state.balls.find(
          (b) => b.holderId === null && b.z < 34 && Math.hypot(b.x - player.x, b.y - player.y) < 38
        );
        if (kickableBall && rng() < 0.55) {
          callbacks.onAttemptDirtyTrick(player, 'kick_ball', undefined, kickableBall);
          return;
        }

        const nearbyVictim = opponents.find(
          (opp) =>
            opp.trippedTimer <= 0 &&
            Math.hypot(opp.x - player.x, opp.y - player.y) < 40
        );
        if (nearbyVictim) {
          const oppHasBall = state.balls.some((b) => b.holderId === nearbyVictim.id);
          // При 4 фолах избегаем грубой подножки, если можно ограничиться мелким толчком
          const doFlagrantTrip =
            player.fouls < 4 && (oppHasBall || baseDeviance > 62 || rng() < 0.45);
          callbacks.onAttemptDirtyTrick(
            player,
            doFlagrantTrip ? 'trip' : 'shove',
            nearbyVictim
          );
          return;
        }
      }
    }
  }

  // ==================== 4. ИГРОК ВЛАДЕЕТ МЯЧОМ ====================
  if (heldBall) {
    player.screenTargetId = null;
    const atkHoopIdx = getAttackingHoop(player.team);
    const atkHoop = getHoopPosition(atkHoopIdx);
    const distToHoop = Math.hypot(player.x - atkHoop.x, player.y - atkHoop.y);
    const oppTeam: 0 | 1 = player.team === 0 ? 1 : 0;
    const scoreDiff = state.score[player.team] - state.score[oppTeam];

    // Короткая фаза показа мяча при ложном замахе (пока защитник взлетает мимо!)
    if ((player.pumpFakeTimer ?? 0) > 0) {
      player.aiAction = 'pump_fake';
      player.targetX = player.x;
      player.targetY = player.y;
      return;
    }

    let nearestDefDist = 9999;
    let closeDefenders = 0;
    let nearestGroundedDef: PlayerEntity | null = null;
    for (const opp of opponents) {
      if (opp.trippedTimer > 0) continue;
      const d = Math.hypot(opp.x - player.x, opp.y - player.y);
      if (d < nearestDefDist) nearestDefDist = d;
      // Защитник, который купился на ложный замах и улетел в воздух, больше не мешает чистому броску или проходу!
      const bitAndAirborne = (opp.bittenByFakeTimer ?? 0) > 0 || opp.z > 20;
      if (d < 75 && !bitAndAirborne) {
        closeDefenders++;
        if (!nearestGroundedDef || d < Math.hypot(nearestGroundedDef.x - player.x, nearestGroundedDef.y - player.y)) {
          nearestGroundedDef = opp;
        }
      }
    }

    // Ситуативный бросок под сирену (конец четверти или истечение 24 секунд!)
    const isBuzzerBeaterWindow =
      (state.quarterTimeRemaining <= 2.4 && distToHoop < 560) ||
      (state.shotClock <= 2.8 && distToHoop < 540);

    if (isBuzzerBeaterWindow && player.effectiveStats.iq >= 34 && player.actionCooldown <= 0) {
      player.aiThought =
        state.quarterTimeRemaining <= 2.4 ? 'Бросок под сирену четверти!' : 'Бросок на флажке 24с!';
      callbacks.onShoot(player, heldBall, atkHoopIdx, false, false);
      return;
    }

    const defHoop = getHoopPosition(getDefendingHoop(player.team));
    const distToOwnHoop = Math.hypot(player.x - defHoop.x, player.y - defHoop.y);

    // Черта «Паникёр»: при плотном прессинге мгновенно паникует, но НЕ отдаёт слепой пас врагу под своим кольцом!
    if (player.traitIds.includes('panicker') && nearestDefDist < 54 && player.actionCooldown <= 0) {
      if (teammates.length > 0 && !player.traitIds.includes('egoisto')) {
        const safePanicMate = teammates
          .filter((m) => m.trippedTimer <= 0 && (!state.frontcourtEstablished || isInFrontcourt(player.team, m.x)))
          .sort(
            (a, b) =>
              computeRawPassLaneDanger(player, a, opponents) -
              computeRawPassLaneDanger(player, b, opponents)
          )[0];
        if (
          safePanicMate &&
          (distToOwnHoop > 260 || computeRawPassLaneDanger(player, safePanicMate, opponents) < 26) &&
          rng() < 0.65
        ) {
          player.aiThought = 'А-А-А! ПАНИКА! СКИДКА!';
          callbacks.onPass(player, heldBall, safePanicMate, false, distToOwnHoop < 260);
          return;
        }
      }
      if (distToOwnHoop < 340) {
        // Под своим щитом паникёр убегает к свободной бровке вместо броска через всё поле или паса чужому!
        player.aiAction = 'drive';
        player.aiThought = 'ПАНИКА! УБЕГАЮ ОТ СВОЕГО КОЛЬЦА!';
        player.targetX = clampFrontcourtX(
          player.team,
          player.x + (player.team === 0 ? 155 : -155),
          state.frontcourtEstablished
        );
        player.targetY = clampCourtY(player.y < HOOP_Y ? COURT_TOP + 42 : COURT_BOTTOM - 42);
        return;
      }
      player.aiThought = 'ПАНИКА! ШВЫРЯЮ!';
      callbacks.onShoot(player, heldBall, atkHoopIdx, false, true);
      return;
    }

    // Черта «Данк-Горилла» или мощный прыгун в проходе под кольцом: слэм-данк!
    const canDunk =
      (player.traitIds.includes('gorilla_dunk') && distToHoop < 145) ||
      (player.effectiveStats.jump >= 70 && distToHoop < 96 && closeDefenders <= 1);

    if (canDunk && player.actionCooldown <= 0) {
      player.aiThought = 'СЛЭМ-ДАНК СВЕРХУ!';
      callbacks.onShoot(player, heldBall, atkHoopIdx, true, false);
      return;
    }

    // 4а-0) Евростеп в проходе: техничный слэшер на ведении обходит подставившегося защитника в краске без фола в нападении!
    if (
      callbacks.onEurostepMove &&
      isInFrontcourt(player.team, player.x) &&
      !player.hasStoppedDribble &&
      (player.eurostepCd ?? 0) <= 0 &&
      player.actionCooldown <= 0 &&
      distToHoop >= 76 &&
      distToHoop <= 172 &&
      nearestGroundedDef
    ) {
      const dFront = Math.hypot(nearestGroundedDef.x - player.x, nearestGroundedDef.y - player.y);
      const angToHoop = Math.atan2(atkHoop.y - player.y, atkHoop.x - player.x);
      const angToDef = Math.atan2(nearestGroundedDef.y - player.y, nearestGroundedDef.x - player.x);
      const blockingLane = dFront < 54 && Math.abs(wrapAngle(angToDef - angToHoop)) < 0.98;
      const canEurostep =
        (player.effectiveStats.dribble >= 64 && player.effectiveStats.iq >= 52) ||
        player.traitIds.includes('short_spark') ||
        player.traitIds.includes('crab_scuttler');

      if (blockingLane && canEurostep) {
        callbacks.onEurostepMove(player, nearestGroundedDef, heldBall, atkHoopIdx);
        return;
      }
    }

    const protectingClockLead =
      state.quarter >= 4 &&
      state.quarterTimeRemaining <= 14 &&
      scoreDiff >= 1 &&
      scoreDiff <= 6 &&
      state.shotClock > state.quarterTimeRemaining + 1.0 &&
      player.effectiveStats.iq >= 48 &&
      !player.traitIds.includes('showboat') &&
      !player.traitIds.includes('egoisto');

    // 4а-1) Степ-бэк на трёшку: техничный снайпер отпрыгивает назад за дугу из-под плотной опеки и бьёт открытым!
    if (
      callbacks.onStepBackMove &&
      !protectingClockLead &&
      isInFrontcourt(player.team, player.x) &&
      !player.hasStoppedDribble &&
      (player.stepBackCd ?? 0) <= 0 &&
      player.actionCooldown <= 0 &&
      distToHoop >= 182 &&
      distToHoop <= 258 &&
      state.shotClock > 3.2 &&
      nearestGroundedDef
    ) {
      const dFront = Math.hypot(nearestGroundedDef.x - player.x, nearestGroundedDef.y - player.y);
      const canStepBack =
        dFront < 48 &&
        ((player.effectiveStats.shooting >= 74 && player.effectiveStats.dribble >= 66) ||
          player.traitIds.includes('parking_sniper') ||
          (player.traitIds.includes('egoisto') && player.effectiveStats.shooting >= 66));

      if (canStepBack && rng() < 0.56) {
        callbacks.onStepBackMove(player, nearestGroundedDef, heldBall, atkHoopIdx);
        return;
      }
    }

    // 4а-2) Кроссовер «Голеностополом» 1-в-1 или смешная потеря неуклюжего балбеса себе в ногу!
    if (
      callbacks.onCrossoverMove &&
      !protectingClockLead &&
      isInFrontcourt(player.team, player.x) &&
      !player.hasStoppedDribble &&
      (player.crossoverCd ?? 0) <= 0 &&
      player.actionCooldown <= 0 &&
      nearestGroundedDef
    ) {
      const dFront = Math.hypot(nearestGroundedDef.x - player.x, nearestGroundedDef.y - player.y);
      const angToHoop = Math.atan2(atkHoop.y - player.y, atkHoop.x - player.x);
      const angToDef = Math.atan2(nearestGroundedDef.y - player.y, nearestGroundedDef.x - player.x);
      const defInPath = dFront < 46 && Math.abs(wrapAngle(angToDef - angToHoop)) < 1.15;
      const skilledDribbler =
        player.effectiveStats.dribble >= 62 ||
        player.traitIds.includes('crab_scuttler') ||
        player.traitIds.includes('short_spark') ||
        player.traitIds.includes('showboat');
      const clumsyShowoff =
        (player.effectiveStats.dribble <= 28 || player.traitIds.includes('butterfingers')) &&
        player.effectiveStats.iq <= 40 &&
        rng() < 0.35;

      if (defInPath && (skilledDribbler || clumsyShowoff)) {
        callbacks.onCrossoverMove(player, nearestGroundedDef, heldBall);
        if (heldBall.holderId !== player.id) return;
      }
    }

    // 4а-3) Передача из рук в руки с мгновенным заслоном (Dribble Handoff / DHO):
    // командный большой за дугой или на усах вручает мяч пробегающему рядом защитнику/снайперу и тут же ставит стену его опекуну!
    if (
      callbacks.onHandoffMove &&
      isInFrontcourt(player.team, player.x) &&
      (player.handoffCd ?? 0) <= 0 &&
      player.actionCooldown <= 0 &&
      (player.role === 'C' || player.role === 'PF') &&
      player.effectiveStats.shooting < 66 &&
      player.effectiveStats.teamwork >= 38 &&
      !player.traitIds.includes('egoisto') &&
      !player.traitIds.includes('ball_hog') &&
      distToHoop >= 142 &&
      distToHoop <= 345
    ) {
      const handoffMate = teammates.find(
        (m) =>
          m.trippedTimer <= 0 &&
          !m.schizoBreadMode &&
          isInFrontcourt(player.team, m.x) &&
          (m.role === 'PG' ||
            m.role === 'SG' ||
            m.effectiveStats.shooting >= 68 ||
            m.traitIds.includes('parking_sniper')) &&
          Math.hypot(m.x - player.x, m.y - player.y) < 62 &&
          computeRawPassLaneDanger(player, m, opponents) < 20
      );
      if (handoffMate) {
        callbacks.onHandoffMove(player, handoffMate, heldBall);
        return;
      }
    }

    // 4а-4) Высокий Флоутер-слеза (Floater) в проходе через руки гиганта-центрового / «Шлагбаума»!
    const rimProtectorAhead = opponents.find((opp) => {
      if (opp.trippedTimer > 0) return false;
      const isGiantBlocker =
        opp.role === 'C' ||
        opp.traitIds.includes('barrier_blocker') ||
        opp.traitIds.includes('lanky_beanpole') ||
        opp.traitIds.includes('grasshopper');
      if (!isGiantBlocker) return false;
      const dOpp = Math.hypot(opp.x - player.x, opp.y - player.y);
      const oppDistHoop = Math.hypot(opp.x - atkHoop.x, opp.y - atkHoop.y);
      const angToHoop = Math.atan2(atkHoop.y - player.y, atkHoop.x - player.x);
      const angToOpp = Math.atan2(opp.y - player.y, opp.x - player.x);
      return (
        dOpp < 88 &&
        oppDistHoop < distToHoop - 10 &&
        Math.abs(wrapAngle(angToOpp - angToHoop)) < 0.9
      );
    });
    const canShootFloater =
      !canDunk &&
      !player.traitIds.includes('parking_sniper') &&
      distToHoop >= 72 &&
      distToHoop <= 156 &&
      closeDefenders <= 1 &&
      Boolean(rimProtectorAhead) &&
      ((player.effectiveStats.dribble >= 56 &&
        player.effectiveStats.shooting >= 56 &&
        player.effectiveStats.iq >= 50) ||
        player.traitIds.includes('short_spark'));

    if (canShootFloater && player.actionCooldown <= 0 && !player.hasStoppedDribble) {
      player.aiAction = 'floater';
      player.aiThought = `Мягкий флоутер-слеза через руки ${rimProtectorAhead!.name.split(' ')[0]}!`;
      callbacks.onShoot(player, heldBall, atkHoopIdx, false, false, 'floater');
      return;
    }

    // Таймер принятия решений (ускоряется при плотном зажиме, остановленном ведении или первом пасе в отрыв!)
    const deadDribble = player.hasStoppedDribble;
    const isPinnedByDefender = nearestDefDist < 42 || player.closelyGuardedTimer >= 0.75;
    const isOutletWindow =
      heldBall.lastPasserId === null &&
      state.possessionTimer < 2.5 &&
      distToOwnHoop < 255 &&
      (player.role === 'C' ||
        player.role === 'PF' ||
        player.effectiveStats.iq >= 52 ||
        player.traitIds.includes('pass_telepath'));
    player.decisionTimer -= deadDribble
      ? dt * 2.5
      : isPinnedByDefender || isOutletWindow
      ? dt * 1.75
      : dt;
    if (player.decisionTimer <= 0 && player.actionCooldown <= 0) {
      const iqDelay = 0.26 - (player.effectiveStats.iq / 100) * 0.15;
      player.decisionTimer = player.traitIds.includes('sleepy_fly') ? iqDelay + 0.22 : iqDelay;

      // Комедия глупого балбеса: оказавшись в полном одиночестве (в радиусе 150px ни души!), грозно качает пустой воздух ложным замахом!
      if (
        callbacks.onGhostPumpFake &&
        !player.hasStoppedDribble &&
        !isOutletWindow &&
        player.effectiveStats.iq <= 25 &&
        nearestDefDist > 150 &&
        isInFrontcourt(player.team, player.x) &&
        distToHoop < 395 &&
        state.shotClock > 5.0 &&
        (player.pumpFakeCd ?? 0) <= 0 &&
        rng() < 0.28
      ) {
        callbacks.onGhostPumpFake(player);
        return;
      }

      // Оценка ПАСА (с защитой от потерь под своим щитом, первым пасом в отрыв и скидкой из краски на дугу!)
      let bestPassScore = -999;
      let bestReceiver: PlayerEntity | null = null;
      let bestIsAlleyOop = false;
      let bestUseSafeLob = false;
      let bestIsOutletPass = false;
      let bestIsDriveKick = false;
      let bestIsMismatch = false;
      let bestIsPostEntry = false;

      if (!player.traitIds.includes('egoisto') && player.effectiveStats.teamwork > 0) {
        for (const mate of teammates) {
          if (mate.trippedTimer > 0 || mate.schizoBreadMode) continue;
          // Правило зоны (Over-and-Back): если мяч уже в передовой зоне, НЕЛЬЗЯ пасовать назад за центральную линию!
          if (state.frontcourtEstablished && !isInFrontcourt(player.team, mate.x)) {
            continue;
          }
          // Правило 8 секунд: если истекает время вывода из тыла, не пасуем назад в тыл!
          if (
            !state.frontcourtEstablished &&
            state.backcourtTimer >= 4.2 &&
            !isInFrontcourt(player.team, mate.x)
          ) {
            continue;
          }

          const mateDistToHoop = Math.hypot(mate.x - atkHoop.x, mate.y - atkHoop.y);
          let mateClosestDef = 999;
          let matePrimaryDef: PlayerEntity | null = null;
          let mateDefsNear = 0;
          for (const opp of opponents) {
            if (opp.trippedTimer > 0) continue;
            const d = Math.hypot(opp.x - mate.x, opp.y - mate.y);
            if (d < 68) mateDefsNear++;
            if (d < mateClosestDef) {
              mateClosestDef = d;
              matePrimaryDef = opp;
            }
          }

          // Опасаемся пасовать союзному «Месси» на своей половине поля, если хватает мозгов
          const mateIsOwnHalfMessi =
            mate.traitIds.includes('messi') &&
            (mate.team === 0 ? mate.x < MID_X : mate.x > MID_X);
          const messiDangerPenalty =
            mateIsOwnHalfMessi && player.effectiveStats.iq > 66 ? -72 : 0;

          // Под СВОИМ кольцом категорически избегаем передач сквозь руки соперников!
          const rawDanger = computeRawPassLaneDanger(player, mate, opponents);
          const laneRiskWeight = distToOwnHoop < 250 ? 1.75 : 1.12;
          const laneRiskPenalty =
            Math.max(evaluatePassLaneRisk(player, mate, opponents), distToOwnHoop < 250 ? rawDanger : 0) *
            laneRiskWeight;

          // Бонус за врывание под кольцо (пик-н-ролл, бэкдор-кат, дай-и-беги или быстрый отрыв!), позицию в лоу-посте и смещение по дуге!
          const isCutting =
            mate.aiAction === 'roll_to_rim' ||
            mate.aiAction === 'backdoor_cut' ||
            mate.aiAction === 'give_and_go' ||
            mate.aiAction === 'fast_break';
          const cutBonus =
            mate.aiAction === 'give_and_go' && mateDistToHoop < distToHoop + 35
              ? 30
              : isCutting && mateDistToHoop < distToHoop
              ? 26
              : (mate.aiAction === 'pick_and_pop' || mate.aiAction === 'relocate_arc') &&
                mateClosestDef > 54
              ? 26
              : 0;

          const isPostEntryCandidate =
            mate.aiAction === 'post_up' &&
            mateDefsNear <= 1 &&
            mateClosestDef > 30 &&
            rawDanger < 24 &&
            player.effectiveStats.iq >= 44;
          const postEntryBonus = isPostEntryCandidate ? 20 : 0;

          // Охота за мисматчем (Mismatch Hunting): быстрый малыш против медленного центрового на дуге ИЛИ большой против малыша под кольцом!
          const speedPerimeterMismatch =
            Boolean(matePrimaryDef) &&
            mateClosestDef < 68 &&
            mateDefsNear === 1 &&
            mateDistToHoop >= 165 &&
            (mate.role === 'PG' ||
              mate.role === 'SG' ||
              mate.effectiveStats.speed >= 76 ||
              mate.traitIds.includes('turbo_hamster') ||
              mate.traitIds.includes('short_spark')) &&
            (matePrimaryDef!.role === 'C' || matePrimaryDef!.traitIds.includes('lanky_beanpole')) &&
            matePrimaryDef!.effectiveStats.speed <= mate.effectiveStats.speed - 14;
          const sizePostMismatch =
            Boolean(matePrimaryDef) &&
            mateClosestDef < 68 &&
            mateDefsNear === 1 &&
            mateDistToHoop <= 148 &&
            (mate.role === 'C' ||
              mate.role === 'PF' ||
              mate.traitIds.includes('gorilla_dunk') ||
              mate.traitIds.includes('lanky_beanpole')) &&
            (matePrimaryDef!.role === 'PG' || matePrimaryDef!.traitIds.includes('short_spark')) &&
            !matePrimaryDef!.traitIds.includes('barrier_blocker');
          const isMismatchCandidate =
            player.effectiveStats.iq >= 50 &&
            rawDanger < 26 &&
            (speedPerimeterMismatch || sizePostMismatch);
          const mismatchPassBonus = isMismatchCandidate ? 26 : 0;

          // Умный первый пас в отрыв (Outlet Pass) сразу после подбора на своём щите!
          const isOutletCandidate =
            isOutletWindow &&
            (mate.role === 'PG' ||
              mate.role === 'SG' ||
              mate.role === 'SF' ||
              mate.aiAction === 'fast_break') &&
            distToHoop - mateDistToHoop > 140 &&
            rawDanger < 22;
          const outletPassBonus = isOutletCandidate ? 32 : 0;

          // Скидка из краски или из поста на открытую дугу (Drive-and-Kick / Inside-Out), когда защита стянулась вдвоём!
          const isDriveKickCandidate =
            distToHoop < 180 &&
            closeDefenders >= 2 &&
            player.effectiveStats.iq >= 46 &&
            mateDistToHoop >= THREE_PT_RADIUS - 14 &&
            mateDistToHoop <= 365 &&
            mateClosestDef >= 54 &&
            mate.effectiveStats.shooting >= 56 &&
            rawDanger < 26;
          const driveKickBonus = isDriveKickCandidate ? 32 : 0;

          // Срочный поиск открытого партнёра с опорной ноги, когда ведение уже остановлено!
          const deadDribbleBonus =
            deadDribble && player.effectiveStats.iq >= 34 && rawDanger < 28 && mateClosestDef > 40
              ? 30
              : 0;

          // Проверка на Алли-уп навес под кольцо для прыгучего партнёра или «Короля Алли-упов»
          const teamPlayCall =
            state.activePlayCall?.team === player.team ? state.activePlayCall.play : null;
          const alleyDistCap = mate.traitIds.includes('sixth_sense_alley') ? 168 : 130;
          const canMateAlleyOop =
            mateDistToHoop < alleyDistCap &&
            mateClosestDef > 38 &&
            rawDanger < 30 &&
            (mate.traitIds.includes('gorilla_dunk') ||
              mate.traitIds.includes('grasshopper') ||
              mate.traitIds.includes('sixth_sense_alley') ||
              mate.effectiveStats.jump >= 74) &&
            (player.effectiveStats.teamwork >= 54 ||
              player.traitIds.includes('pass_telepath') ||
              player.traitIds.includes('sixth_sense_alley') ||
              teamPlayCall === 'alley_oop');
          const alleyOopBonus = canMateAlleyOop
            ? teamPlayCall === 'alley_oop'
              ? 48
              : mate.traitIds.includes('sixth_sense_alley')
              ? 32
              : 24
            : 0;

          // Бонус за «Горячую руку» партнёра (3+ попаданий подряд!) и за тренерские комбинации с планшета!
          const hotMateBonus = mate.hotStreak >= 3 ? 22 : 0;
          let playCallPassBonus = 0;
          if (teamPlayCall === 'feed_sniper') {
            const isBestSniper = !teammates.some(
              (other) =>
                other.id !== mate.id &&
                other.effectiveStats.shooting > mate.effectiveStats.shooting + 4
            );
            if (isBestSniper && mateDistToHoop >= THREE_PT_RADIUS - 18) {
              playCallPassBonus = 40;
            }
          } else if (teamPlayCall === 'iso_star') {
            const myIsoRating = player.effectiveStats.dribble + player.effectiveStats.shooting;
            const mateIsoRating = mate.effectiveStats.dribble + mate.effectiveStats.shooting;
            if (mateIsoRating > myIsoRating + 14) {
              playCallPassBonus = 34;
            } else {
              playCallPassBonus = -18; // лидер сам решает 1-в-1!
            }
          } else if (
            teamPlayCall === 'pick_and_roll' &&
            (mate.aiAction === 'roll_to_rim' || mate.aiAction === 'pick_and_pop')
          ) {
            playCallPassBonus = 28;
          }

          // Разгрузка из-под плотной опеки (чтобы два игрока никогда не застревали друг перед другом с мячом!)
          const pressureEscapeBonus =
            isPinnedByDefender && rawDanger < 26 && mateClosestDef > 48
              ? 24 + Math.min(28, player.closelyGuardedTimer * 16)
              : closeDefenders >= 2 && mateClosestDef > 70
              ? 28
              : 0;

          // Срочный вывод мяча вперёд через центр по правилу 8 секунд!
          const advance8sBonus =
            !state.frontcourtEstablished &&
            state.backcourtTimer >= 3.2 &&
            isInFrontcourt(player.team, mate.x)
              ? 36
              : 0;

          // В концовке при отставании в 3+ очков ценим открытых снайперов на дуге
          const lateThreeBonus =
            state.quarter >= 4 &&
            state.quarterTimeRemaining < 18 &&
            scoreDiff <= -3 &&
            mateDistToHoop >= THREE_PT_RADIUS - 10 &&
            mate.effectiveStats.shooting >= 62
              ? 22
              : 0;

          const progressGain = Math.max(isPinnedByDefender ? -6 : -24, (distToHoop - mateDistToHoop) * 0.23);
          const openness = Math.min(140, mateClosestDef) * 0.46;
          const shooterQuality = mateDistToHoop < 290 ? mate.effectiveStats.shooting * 0.14 : 0;
          const teamworkBias = (player.effectiveStats.teamwork - 45) * 0.62;
          const kindBonus = player.traitIds.includes('kind_soul') ? 26 : 0;
          const telepathBonus = player.traitIds.includes('pass_telepath') ? 24 : 0;
          const hogPenalty =
            player.traitIds.includes('ball_hog') && player.closelyGuardedTimer < 2.2 ? -28 : 0;

          const score =
            progressGain +
            openness +
            shooterQuality +
            teamworkBias +
            kindBonus +
            telepathBonus +
            cutBonus +
            postEntryBonus +
            mismatchPassBonus +
            outletPassBonus +
            driveKickBonus +
            deadDribbleBonus +
            alleyOopBonus +
            hotMateBonus +
            playCallPassBonus +
            pressureEscapeBonus +
            advance8sBonus +
            lateThreeBonus +
            hogPenalty +
            messiDangerPenalty -
            laneRiskPenalty;

          if (score > bestPassScore) {
            bestPassScore = score;
            bestReceiver = mate;
            bestIsAlleyOop = canMateAlleyOop;
            bestUseSafeLob = !canMateAlleyOop && distToOwnHoop < 250 && rawDanger >= 14;
            bestIsOutletPass = isOutletCandidate;
            bestIsDriveKick = isDriveKickCandidate;
            bestIsMismatch = isMismatchCandidate;
            bestIsPostEntry = isPostEntryCandidate;
          }
        }
      }

      // Оценка БРОСКА (включая классический крюк большого из лоу-поста и высокий флоутер!)
      const isParkingSniper = player.traitIds.includes('parking_sniper');
      const canPostSkyhook =
        !isParkingSniper &&
        !canDunk &&
        distToHoop >= 44 &&
        distToHoop <= 136 &&
        closeDefenders <= 1 &&
        (player.role === 'C' ||
          player.role === 'PF' ||
          player.traitIds.includes('lanky_beanpole'));
      const hookShotBonus = canPostSkyhook
        ? nearestGroundedDef &&
          (nearestGroundedDef.role === 'PG' || nearestGroundedDef.traitIds.includes('short_spark'))
          ? 24
          : 16
        : 0;
      const floaterShotBonus = canShootFloater ? 18 : 0;

      let rangeSuitability = 0;
      if (isParkingSniper) {
        rangeSuitability =
          distToHoop >= 240 && distToHoop <= 415 ? 50 : distToHoop < 185 ? -40 : 12;
      } else {
        rangeSuitability =
          distToHoop < 155 ? 56 : distToHoop < 285 ? 35 : distToHoop < 380 ? 14 : -35;
      }

      // Бонус трёхочкового в концовке при отставании
      if (
        state.quarter >= 4 &&
        state.quarterTimeRemaining < 18 &&
        scoreDiff <= -3 &&
        distToHoop >= THREE_PT_RADIUS &&
        distToHoop <= 390
      ) {
        rangeSuitability += 26;
      }

      // Контроль времени при победном счёте в последние секунды матча
      const clockMilkRestraint =
        state.quarter >= 4 &&
        state.quarterTimeRemaining < 11 &&
        scoreDiff > 0 &&
        state.shotClock > state.quarterTimeRemaining + 1.5 &&
        player.effectiveStats.iq >= 62 &&
        !player.traitIds.includes('egoisto') &&
        distToHoop > 110
          ? -38
          : 0;

      const contestPenalty = player.traitIds.includes('egoisto')
        ? closeDefenders * 6
        : canPostSkyhook || canShootFloater
        ? closeDefenders * 9
        : closeDefenders * 19;
      const shotClockUrgency = state.shotClock < 4.5 ? 85 : state.shotClock < 7.5 ? 28 : 0;
      const egoBonus = player.traitIds.includes('egoisto') ? 28 : 0;
      const hotShooterBonus =
        player.hotStreak >= 3
          ? 14
          : player.hotStreak <= -3 &&
            player.effectiveStats.iq >= 52 &&
            !player.traitIds.includes('egoisto') &&
            !player.traitIds.includes('ball_hog')
          ? -16
          : 0;
      const bankShotBonus =
        player.traitIds.includes('brick_magnet_board') &&
        distToHoop >= 85 &&
        distToHoop <= 205 &&
        Math.abs(player.y - HOOP_Y) >= 42
          ? 12
          : 0;
      const shootScore =
        rangeSuitability +
        player.effectiveStats.shooting * 0.45 -
        contestPenalty +
        shotClockUrgency +
        egoBonus +
        hotShooterBonus +
        bankShotBonus +
        hookShotBonus +
        floaterShotBonus +
        clockMilkRestraint;

      const passThreshold =
        deadDribble && player.effectiveStats.iq >= 34
          ? 20
          : isPinnedByDefender || isOutletWindow
          ? 32
          : 52;
      if (bestReceiver && bestPassScore > passThreshold && bestPassScore > shootScore + 2) {
        const firstName = bestReceiver.name.split(' ')[0];
        player.aiThought = bestIsAlleyOop
          ? `Алли-уп навес на ${firstName}!`
          : bestIsOutletPass
          ? `Первый пас в отрыв на ${firstName}!`
          : bestIsDriveKick
          ? `Стянул защиту в краску и скинул на дугу ${firstName}!`
          : bestIsMismatch
          ? `Наказывает мисматч — грузит мяч на ${firstName}!`
          : bestIsPostEntry
          ? `Загружает мяч в лоу-пост на ${firstName}!`
          : deadDribble
          ? `Убрал ведение — отдаёт с опорной ноги на ${firstName}!`
          : bestReceiver.hotStreak >= 3
          ? `Кормит горячую руку ${firstName} (В ОГНЕ)!`
          : bestReceiver.aiAction === 'give_and_go'
          ? `Возвращает стенку на ${firstName} (Дай и беги)!`
          : bestReceiver.aiAction === 'pick_and_pop' || bestReceiver.aiAction === 'relocate_arc'
          ? `Скидывает на открытую дугу ${firstName}!`
          : bestReceiver.aiAction === 'fast_break'
          ? `Длинный пас в отрыв на ${firstName}!`
          : bestReceiver.aiAction === 'roll_to_rim' || bestReceiver.aiAction === 'backdoor_cut'
          ? `Тонкий пас на вход ${firstName}!`
          : isPinnedByDefender
          ? `Скидывает из-под прессинга на ${firstName}!`
          : `Пас открытому ${firstName}!`;
        callbacks.onPass(player, heldBall, bestReceiver, bestIsAlleyOop, bestUseSafeLob);
        return;
      }

      // 4б) Ложный замах (Pump Fake): вместо тяжёлого броска через руки стоящего в упор защитника умный игрок убирает его на фейке!
      if (
        callbacks.onPumpFake &&
        nearestGroundedDef &&
        Math.hypot(nearestGroundedDef.x - player.x, nearestGroundedDef.y - player.y) < 52 &&
        distToHoop < 385 &&
        state.shotClock > 4.2 &&
        (player.pumpFakeCd ?? 0) <= 0 &&
        (player.effectiveStats.iq >= 46 || player.traitIds.includes('showboat')) &&
        !player.traitIds.includes('egoisto')
      ) {
        const fakeChance =
          0.48 +
          (player.effectiveStats.iq / 100) * 0.34 +
          (player.traitIds.includes('showboat') ? 0.18 : 0);
        if (rng() < fakeChance) {
          callbacks.onPumpFake(player, nearestGroundedDef);
          return;
        }
      }

      const shootThreshold = deadDribble && distToHoop < 330 ? 44 : 58;
      if (shootScore > shootThreshold && distToHoop < 465) {
        const selectedStyle: ShotStyle = canShootFloater
          ? 'floater'
          : canPostSkyhook
          ? 'hook'
          : 'standard';
        player.aiThought =
          selectedStyle === 'floater'
            ? 'Мягкий флоутер высокой дугой через руки большого!'
            : selectedStyle === 'hook'
            ? 'Закручивает классический крюк из поста!'
            : player.hotStreak >= 3
            ? 'Бросает на кураже (В ОГНЕ)!'
            : player.traitIds.includes('egoisto')
            ? 'Эгоисто бьёт с отклонением!'
            : bankShotBonus > 0
            ? 'Целится с отскоком от щита!'
            : deadDribble
            ? 'Бросает с опорной ноги без ведения!'
            : distToHoop >= THREE_PT_RADIUS
            ? 'Открытый трёхочковый!'
            : 'Бросок с удобной дистанции!';
        callbacks.onShoot(player, heldBall, atkHoopIdx, false, false, selectedStyle);
        return;
      }

      // Комедия глупого балбеса: с крошечным IQ и ужасным броском внезапно возомнил себя снайпером!
      if (
        player.effectiveStats.iq <= 26 &&
        player.effectiveStats.shooting <= 40 &&
        distToHoop >= 210 &&
        distToHoop <= 385 &&
        !player.traitIds.includes('kind_soul') &&
        rng() < 0.16
      ) {
        player.aiThought = 'Возомнил себя суперзвездой — швыряет кирпич!';
        callbacks.onShoot(player, heldBall, atkHoopIdx, false, false, 'standard');
        return;
      }
    }

    // Дисциплина опорной ноги при остановленном ведении (hasStoppedDribble):
    // Умный игрок держит опорную ногу и ждёт открывания партнёров, а глупый балбес может забыться и схватить двойное ведение!
    if (player.hasStoppedDribble) {
      if (player.effectiveStats.iq >= 34 || rng() >= 0.25) {
        player.aiAction = 'drive';
        player.targetX = player.x;
        player.targetY = player.y;
        player.aiThought =
          player.effectiveStats.iq >= 34
            ? 'Убрал ведение — крутится на опорной и ищет пас!'
            : 'Завис без ведения и чешет репу...';
        return;
      }
      // Игрок с IQ < 34 забыл, что уже останавливал дриблинг, и побежал снова!
      player.aiAction = 'drive';
      const clumsyDrive = chooseSmartDriveTarget(state, player, teammates, opponents);
      player.targetX = clumsyDrive.x;
      player.targetY = clumsyDrive.y;
      player.aiThought = 'Забыл, что ведение остановлено, и снова стучит мячом!';
      return;
    }

    // Сохраняем активный манёвр евростепа до завершения фазы шага (для иммунитета к фолу в атаке)
    if (player.actionCooldown > 0 && player.aiAction === 'eurostep') {
      return;
    }

    // Умный дриблинг по свободному коридору или под заслон (с сохранением метки кроссовера на время рывка)
    if (player.aiAction !== 'ankle_breaker' || (player.crossoverCd ?? 0) <= 4.45) {
      player.aiAction = 'drive';
    }
    const driveChoice = chooseSmartDriveTarget(state, player, teammates, opponents);
    player.targetX = driveChoice.x;
    player.targetY = driveChoice.y;
    if (player.aiAction === 'drive') {
      player.aiThought = driveChoice.thought;
    }
    return;
  }

  // ==================== 5. ИГРОК БЕЗ МЯЧА ====================
  // Черта «Стадо баранов»: забывает про тактику и бежит толпой за мячиком!
  if (player.traitIds.includes('sheep_herd') && state.balls.length > 0) {
    const b = state.balls[player.slotIndex % state.balls.length];
    player.aiAction = 'chase_ball';
    player.screenTargetId = null;
    player.aiThought = 'Бежит за мячиком толпой!';
    player.targetX = clampCourtX(b.x);
    player.targetY = clampCourtY(b.y);
    return;
  }

  // 5а) Если в воздухе летит бросок в кольцо — борьба за позицию на подборе (бокс-аут) и страховка тыла!
  const shotInFlight = state.balls.find(
    (b) =>
      b.holderId === null &&
      b.targetHoop !== null &&
      (b.shotType === 'normal' || b.shotType === 'panic' || b.shotType === 'messi')
  );
  if (shotInFlight && shotInFlight.targetHoop !== null) {
    const targetHoopPos = getHoopPosition(shotInFlight.targetHoop);

    // Пижон или самоуверенный Эгоисто после дальнего броска может отвернуться праздновать ещё пока мяч летит!
    if (
      shotInFlight.lastShooterId === player.id &&
      (player.traitIds.includes('showboat') ||
        (player.traitIds.includes('egoisto') && player.effectiveStats.iq < 50)) &&
      Math.hypot(
        shotInFlight.shotOriginX - targetHoopPos.x,
        shotInFlight.shotOriginY - targetHoopPos.y
      ) >= THREE_PT_RADIUS - 12
    ) {
      player.aiAction = 'celebrate';
      player.screenTargetId = null;
      player.aiThought = 'Отвернулся праздновать ещё до попадания!';
      player.targetX = player.x;
      player.targetY = player.y;
      player.heading = player.team === 0 ? Math.PI : 0;
      return;
    }

    if (
      player.role === 'PG' &&
      player.effectiveStats.iq >= 45 &&
      !player.traitIds.includes('magnet')
    ) {
      player.aiAction = 'safety_back';
      player.screenTargetId = null;
      player.aiThought = 'Страхует тыл от быстрого отрыва';
      player.targetX = clampCourtX(MID_X + (player.team === 0 ? -95 : 95));
      player.targetY = HOOP_Y;
      return;
    }

    const isRebounder =
      player.role === 'C' ||
      player.role === 'PF' ||
      player.role === 'SF' ||
      player.effectiveStats.jump >= 68 ||
      player.traitIds.includes('magnet') ||
      player.traitIds.includes('brick_magnet_board');

    if (isRebounder) {
      player.aiAction = 'rebound_position';
      player.screenTargetId = null;

      // Настоящий физический Бокс-аут: умный игрок находит ближайшего соперника у кольца и встаёт спиной между ним и щитом!
      const crashOpps = opponents
        .filter((o) => Math.hypot(o.x - targetHoopPos.x, o.y - targetHoopPos.y) < 175)
        .sort(
          (a, b) =>
            Math.hypot(a.x - player.x, a.y - player.y) -
            Math.hypot(b.x - player.x, b.y - player.y)
        );
      const boxTarget =
        player.effectiveStats.iq >= 44
          ? crashOpps.find(
              (o) =>
                !teammates.some(
                  (m) =>
                    m.aiAction === 'rebound_position' &&
                    Math.hypot(m.x - o.x, m.y - o.y) < Math.hypot(player.x - o.x, player.y - o.y) - 10
                )
            ) ?? null
          : null;

      if (boxTarget) {
        const vxH = boxTarget.x - targetHoopPos.x;
        const vyH = boxTarget.y - targetHoopPos.y;
        const dH = Math.max(1, Math.hypot(vxH, vyH));
        const sealDist = Math.max(42, Math.min(76, dH * 0.68));
        player.targetX = clampCourtX(targetHoopPos.x + (vxH / dH) * sealDist);
        player.targetY = clampCourtY(targetHoopPos.y + (vyH / dH) * sealDist);
        player.aiThought = `Оттирает спиной ${boxTarget.name.split(' ')[0]} (бокс-аут)!`;
      } else {
        player.aiThought = 'Занимает позицию под отскок!';
        const rebAngles = [-0.95, 0, 0.95, -0.45, 0.45];
        const ang = rebAngles[player.slotIndex % 5];
        const dirToCenter = shotInFlight.targetHoop === 0 ? 1 : -1;
        player.targetX = clampCourtX(targetHoopPos.x + dirToCenter * Math.cos(ang) * 56);
        player.targetY = clampCourtY(targetHoopPos.y + Math.sin(ang) * 64);
      }

      // Если это чужое кольцо (подбор в нападении!) и промахнувшийся мяч снижается возле дужки — прыгучий большой вылетает на добивание сверху!
      if (
        shotInFlight.lastTouchTeam === player.team &&
        !shotInFlight.isGuaranteedSwish &&
        shotInFlight.lastShooterId !== player.id &&
        player.z === 0 &&
        Math.hypot(player.x - targetHoopPos.x, player.y - targetHoopPos.y) < 88 &&
        Math.hypot(shotInFlight.x - targetHoopPos.x, shotInFlight.y - targetHoopPos.y) < 92 &&
        shotInFlight.z >= 36 &&
        shotInFlight.z <= 125 &&
        shotInFlight.vz < 28 &&
        (player.effectiveStats.jump >= 68 ||
          player.traitIds.includes('gorilla_dunk') ||
          player.traitIds.includes('brick_magnet_board') ||
          player.traitIds.includes('sixth_sense_alley'))
      ) {
        callbacks.onJump(player, false, 'Вылетает на добивание в прыжке!');
      }
      return;
    }
  }

  // 5б) Ничейные мячи на полу / вбрасывании (передачи в воздухе обрабатываются в блоках 1б и 2!)
  const looseBalls = state.balls.filter(
    (b) =>
      b.holderId === null &&
      b.targetHoop === null &&
      b.shotType !== 'pass'
  );
  if (looseBalls.length > 0) {
    let closestLoose = looseBalls[0];
    let bestD = Math.hypot(player.x - closestLoose.x, player.y - closestLoose.y);
    for (const lb of looseBalls) {
      const d = Math.hypot(player.x - lb.x, player.y - lb.y);
      if (d < bestD) {
        bestD = d;
        closestLoose = lb;
      }
    }

    // Правило центрального круга на стартовом спорном: прыгают только Центровые (или Руки-загребуки), остальные ждут за кругом!
    if (
      closestLoose.isTipoffBall &&
      player.role !== 'C' &&
      !player.traitIds.includes('grabby_hands')
    ) {
      player.aiAction = 'tipoff';
      player.screenTargetId = null;
      player.aiThought = 'Ждёт отскока за центральным кругом';
      const ringAngles = [-2.2, -1.25, 1.25, 2.2];
      const baseAng = ringAngles[player.slotIndex % 4] + (player.team === 0 ? Math.PI : 0) * 0.15;
      const sideSign = player.team === 0 ? -1 : 1;
      player.targetX = clampCourtX(MID_X + sideSign * (CENTER_CIRCLE_RADIUS + 26));
      player.targetY = clampCourtY(HOOP_Y + Math.sin(baseAng) * 115);
      return;
    }

    const closerMates = teammates.filter(
      (m) => Math.hypot(m.x - closestLoose.x, m.y - closestLoose.y) < bestD - 15
    ).length;

    if (
      closerMates < 2 ||
      (closestLoose.isTipoffBall && player.traitIds.includes('grabby_hands')) ||
      player.traitIds.includes('magnet')
    ) {
      player.aiAction = 'chase_ball';
      player.screenTargetId = null;
      player.aiThought = closestLoose.isTipoffBall ? 'Прыжок за спорным!' : 'Ныряет за ничейным мячом!';
      player.targetX = clampCourtX(closestLoose.x + closestLoose.vx * 0.24);
      player.targetY = clampCourtY(closestLoose.y + closestLoose.vy * 0.24);
      if (player.z === 0 && bestD < 46 && closestLoose.z > 40 && closestLoose.z < 140) {
        callbacks.onJump(player, false, 'Подбор в прыжке!');
      }
      return;
    }
  }

  const alliedCarrierBall = state.balls.find((b) => {
    const holder = activePlayers.find((p) => p.id === b.holderId);
    return holder && holder.team === player.team;
  });
  const enemyCarrierBall = state.balls.find((b) => {
    const holder = activePlayers.find((p) => p.id === b.holderId);
    return holder && holder.team !== player.team;
  });
  const enemyPassBall = state.balls.find(
    (b) => b.holderId === null && b.shotType === 'pass' && b.lastTouchTeam !== null && b.lastTouchTeam !== player.team
  );
  const isEnemyTeamPossession =
    Boolean(enemyCarrierBall) ||
    Boolean(enemyPassBall) ||
    (state.possessionTeam !== null && state.possessionTeam !== player.team && !alliedCarrierBall);

  // ==================== 6. КОМАНДНАЯ ЗАЩИТА (СМЕНА, ПОДСТРАХОВКА, ПЕРЕКРЫТИЕ ПАСА) ====================
  if (isEnemyTeamPossession) {
    player.screenTargetId = null;
    const carrier = enemyCarrierBall
      ? opponents.find((o) => o.id === enemyCarrierBall.holderId)
      : undefined;
    const defHoop = getHoopPosition(getDefendingHoop(player.team));
    const oppTeamForDef: 0 | 1 = player.team === 0 ? 1 : 0;
    const defScoreDiff = state.score[player.team] - state.score[oppTeamForDef];
    const shouldIntentionalFoul =
      Boolean(callbacks.onIntentionalFoul) &&
      Boolean(enemyCarrierBall) &&
      Boolean(carrier) &&
      state.quarter >= 4 &&
      state.quarterTimeRemaining <= 16 &&
      state.quarterTimeRemaining > 1.2 &&
      defScoreDiff <= -1 &&
      defScoreDiff >= -6 &&
      state.shotClock > state.quarterTimeRemaining + 0.8 &&
      state.possessionTimer >= 1.8 &&
      player.effectiveStats.iq >= 48 &&
      player.fouls < 4;

    if (shouldIntentionalFoul && carrier) {
      const distToCarrierFoul = Math.hypot(carrier.x - player.x, carrier.y - player.y);
      const closerAllDefs = teammates.filter(
        (m) => Math.hypot(carrier.x - m.x, carrier.y - m.y) < distToCarrierFoul - 10
      ).length;
      if (closerAllDefs <= 1) {
        player.aiAction = 'intentional_foul';
        player.targetX = clampCourtX(carrier.x);
        player.targetY = clampCourtY(carrier.y);
        player.aiThought = 'Бежит на тактический фол ради остановки времени!';
        if (distToCarrierFoul < 42 && !state.inbound && !state.freeThrow && state.balls.length === 1) {
          callbacks.onIntentionalFoul!(player, carrier);
        }
        return;
      }
    }

    // 6а) Центровой или «Шлагбаум» страхует краску и выходит на блок-шот (или летит на блок вдогонку, если проспал отрыв!)
    if (player.traitIds.includes('barrier_blocker') || player.role === 'C') {
      const carrierDistHoop = carrier
        ? Math.hypot(carrier.x - defHoop.x, carrier.y - defHoop.y)
        : 999;
      const myDistHoop = Math.hypot(player.x - defHoop.x, player.y - defHoop.y);
      const dCarrierC = carrier ? Math.hypot(carrier.x - player.x, carrier.y - player.y) : 999;
      if (
        carrier &&
        carrierDistHoop < 195 &&
        myDistHoop > carrierDistHoop + 12 &&
        dCarrierC < 112
      ) {
        player.aiAction = 'chasedown_block';
        player.aiThought = 'Летит со спины на блок-шот вдогонку!';
        player.targetX = clampCourtX(carrier.x + (defHoop.x - carrier.x) * 0.22);
        player.targetY = clampCourtY(carrier.y + (defHoop.y - carrier.y) * 0.22);
        if (carrier.windupTimer > 0 && dCarrierC < 96 && player.z === 0) {
          callbacks.onJump(player, player.traitIds.includes('grasshopper'), 'Блок вдогонку!');
        }
        return;
      }
      // Если соперник ворвался прямо под щит — встречаем его плотно!
      if (carrier && carrierDistHoop < 135) {
        player.aiAction = 'help_defense';
        player.aiThought = 'Встречает проход под кольцом!';
        player.targetX = clampCourtX(carrier.x + (defHoop.x - carrier.x) * 0.32);
        player.targetY = clampCourtY(carrier.y + (defHoop.y - carrier.y) * 0.32);
      } else {
        const guardX = defHoop.x + (player.team === 0 ? 78 : -78);
        const guardY = carrier ? HOOP_Y + (carrier.y - HOOP_Y) * 0.38 : HOOP_Y;
        player.aiAction = 'rim_protect';
        player.aiThought = 'Страхует трёхсекундную зону';
        player.targetX = clampCourtX(guardX);
        player.targetY = clampCourtY(guardY);
      }

      if (
        carrier &&
        carrier.windupTimer > 0 &&
        Math.hypot(carrier.x - player.x, carrier.y - player.y) < 98 &&
        player.z === 0
      ) {
        callbacks.onJump(player, player.traitIds.includes('grasshopper'), 'Блок-шот!');
      }
      return;
    }

    if (carrier) {
      const distToCarrier = Math.hypot(carrier.x - player.x, carrier.y - player.y);
      const carrierDistHoopForDef = Math.hypot(carrier.x - defHoop.x, carrier.y - defHoop.y);
      const myDistDefHoop = Math.hypot(player.x - defHoop.x, player.y - defHoop.y);
      const carrierSpeed = Math.hypot(carrier.vx, carrier.vy);

      // 6б-chasedown) Атлетичный блок-шот вдогонку со спины (Chasedown Block) при выходе соперника один на один с кольцом!
      const noDefenderInFront = !teammates.some((m) => {
        const mDistHoop = Math.hypot(m.x - defHoop.x, m.y - defHoop.y);
        return mDistHoop < carrierDistHoopForDef && Math.hypot(m.x - carrier.x, m.y - carrier.y) < 76;
      });
      const canChasedown =
        player.effectiveStats.jump >= 72 ||
        player.traitIds.includes('grasshopper') ||
        player.traitIds.includes('turbo_hamster');
      if (
        canChasedown &&
        noDefenderInFront &&
        carrierDistHoopForDef < 195 &&
        myDistDefHoop > carrierDistHoopForDef + 10 &&
        distToCarrier < 108
      ) {
        player.aiAction = 'chasedown_block';
        player.targetX = clampCourtX(carrier.x + (defHoop.x - carrier.x) * 0.22);
        player.targetY = clampCourtY(carrier.y + (defHoop.y - carrier.y) * 0.22);
        player.aiThought = 'Разгоняется со спины на блок-шот вдогонку!';
        if (carrier.windupTimer > 0 && distToCarrier < 86 && player.z === 0) {
          callbacks.onJump(player, player.traitIds.includes('grasshopper'), 'Блок-шот вдогонку!');
        }
        return;
      }

      // 6б-0) Подстановка под фол в атаке (Take Charge): умный защитник без фол-трабла заранее занимает точку на пути тарана вне полукруга под кольцом!
      if (
        (player.chargePlantTimer ?? 0) > 0 &&
        distToCarrier < 88 &&
        carrier.windupTimer <= 0 &&
        player.z === 0
      ) {
        player.aiAction = 'take_charge';
        player.targetX = player.x;
        player.targetY = player.y;
        player.aiThought = 'Встал как вкопанный под фол в атаке!';
        return;
      }
      if (
        player.effectiveStats.iq >= 66 &&
        player.fouls < 4 &&
        player.z === 0 &&
        carrierSpeed >= 95 &&
        carrierDistHoopForDef >= 88 &&
        carrierDistHoopForDef <= 230 &&
        carrier.aiAction !== 'eurostep' &&
        myDistDefHoop >= 56 &&
        myDistDefHoop <= 138 &&
        distToCarrier <= 84
      ) {
        const projCharge = projectPointOntoSegment(
          player.x,
          player.y,
          carrier.x,
          carrier.y,
          defHoop.x,
          defHoop.y
        );
        if (projCharge.perpDist < 28 && projCharge.t > 0.06 && projCharge.t < 0.86) {
          player.aiAction = 'take_charge';
          player.chargePlantTimer = 0.48;
          player.targetX = player.x;
          player.targetY = player.y;
          player.aiThought = 'Встал как вкопанный под фол в атаке!';
          return;
        }
      }

      const alreadyOnBallMargin =
        player.aiAction === 'guard_carrier' || player.aiAction === 'closeout' ? 22 : 12;
      const closerDefenders = teammates.filter(
        (m) =>
          m.role !== 'C' &&
          Math.hypot(carrier.x - m.x, carrier.y - m.y) < distToCarrier - alreadyOnBallMargin
      ).length;

      // 6б) Первичный защитник на мяче (с выбором прохода поверх/под заслоном и грамотным клоузаутом на дугу!)
      const allowDoubleTeam =
        teamTactic === 'blitz' && closerDefenders === 1 && distToCarrier < 165;

      if (closerDefenders === 0 || player.traitIds.includes('pickpocket') || allowDoubleTeam) {
        const leadCarrierX = carrier.x + carrier.vx * 0.04;
        const leadCarrierY = carrier.y + carrier.vy * 0.04;
        const dxHoop = defHoop.x - leadCarrierX;
        const dyHoop = defHoop.y - leadCarrierY;
        const dHoop = Math.max(1, Math.hypot(dxHoop, dyHoop));

        const nearbyScreenerOnMe = opponents.find(
          (o) => o.aiAction === 'screen_set' && Math.hypot(o.x - carrier.x, o.y - carrier.y) < 78
        );
        const disciplinedNoFoulLead =
          state.quarter >= 4 &&
          state.quarterTimeRemaining <= 14 &&
          defScoreDiff >= 1 &&
          defScoreDiff <= 4 &&
          player.effectiveStats.iq >= 54 &&
          !player.traitIds.includes('pickpocket');

        let stanceDist = Math.min(
          dHoop * 0.65,
          carrier.traitIds.includes('parking_sniper') ? 28 : teamTactic === 'blitz' ? 29 : 34
        );
        let lateralShade =
          allowDoubleTeam && closerDefenders === 1 ? (player.slotIndex % 2 === 0 ? 18 : -18) : 0;

        if (nearbyScreenerOnMe && player.effectiveStats.iq >= 52 && !allowDoubleTeam) {
          const deadlyShooter =
            carrier.effectiveStats.shooting >= 70 ||
            carrier.traitIds.includes('parking_sniper') ||
            carrier.hotStreak >= 3;
          const nonShooterSlasher =
            carrier.effectiveStats.shooting <= 56 || carrier.traitIds.includes('gorilla_dunk');
          if (deadlyShooter) {
            // Пролазим ПОВЕРХ заслона (Over), чтобы не дать открытый трёхочковый!
            player.aiAction = 'guard_carrier';
            stanceDist = Math.min(dHoop * 0.55, 20);
            const awayFromScreenY = Math.sign(carrier.y - nearbyScreenerOnMe.y || 1) * 14;
            lateralShade += awayFromScreenY;
            player.aiThought = 'Пролазит поверх заслона против снайпера!';
          } else if (nonShooterSlasher) {
            // Ныряем ПОД заслон (Under), отрезая проход и данк!
            player.aiAction = 'guard_carrier';
            stanceDist = Math.min(dHoop * 0.72, 54);
            player.aiThought = 'Ныряет под заслон — страхует проход!';
          } else {
            player.aiAction = 'guard_carrier';
            player.aiThought = 'Борется через заслон на мяче';
          }
        } else if (!allowDoubleTeam && distToCarrier >= 72 && distToCarrier <= 185 && dHoop < 385) {
          // Клоузаут на дугу: умный защитник гасит скорость у дуги, а паникёр летит сломя голову!
          player.aiAction = 'closeout';
          if (player.effectiveStats.iq >= 56 && !player.traitIds.includes('panicker')) {
            stanceDist = Math.min(dHoop * 0.65, 34);
            player.aiThought = 'Грамотный клоузаут на дугу без фола!';
          } else {
            stanceDist = Math.min(dHoop * 0.45, 14);
            player.aiThought = 'Летит накрывать бросок сломя голову!';
          }
        } else {
          player.aiAction = 'guard_carrier';
          player.aiThought = disciplinedNoFoulLead
            ? 'Защищается с поднятыми руками без фола в концовке!'
            : allowDoubleTeam
            ? 'Зажимает в дабл-тим!'
            : 'Плотно держит игрока с мячом';
        }

        player.targetX = clampCourtX(leadCarrierX + (dxHoop / dHoop) * stanceDist);
        player.targetY = clampCourtY(leadCarrierY + (dyHoop / dHoop) * stanceDist + lateralShade);

        if (
          enemyCarrierBall &&
          distToCarrier < 40 &&
          player.stealCooldown <= 0 &&
          !disciplinedNoFoulLead
        ) {
          callbacks.onAttemptSteal(player, carrier, enemyCarrierBall);
        }
        if (carrier.windupTimer > 0 && distToCarrier < 70 && player.z === 0) {
          callbacks.onJump(player, false, 'Накрывает бросок!');
        }
        return;
      }

      // 6в) Подстраховка (Help Defense), если носитель мяча обыграл опекуна и режет угол к кольцу!
      const carrierDistHoop = Math.hypot(carrier.x - defHoop.x, carrier.y - defHoop.y);
      const anyMateInFront = teammates.some((m) => {
        const mDistHoop = Math.hypot(m.x - defHoop.x, m.y - defHoop.y);
        return mDistHoop < carrierDistHoop && Math.hypot(m.x - carrier.x, m.y - carrier.y) < 70;
      });
      if (carrierDistHoop < 210 && !anyMateInFront && closerDefenders === 1) {
        player.aiAction = 'help_defense';
        player.aiThought = 'Смещается на подстраховку в краску!';
        player.targetX = clampCourtX(carrier.x + (defHoop.x - carrier.x) * 0.35);
        player.targetY = clampCourtY(carrier.y + (defHoop.y - carrier.y) * 0.35);
        if (enemyCarrierBall && distToCarrier < 38 && player.stealCooldown <= 0) {
          callbacks.onAttemptSteal(player, carrier, enemyCarrierBall);
        }
        return;
      }
      // 6в-bis) Переключение на заслоне (Switch Screen): если наш подопечный ставит заслон носителю мяча, умный защитник выходит встречать мяч!
      const activeEnemyScreener = opponents.find(
        (o) =>
          o.aiAction === 'screen_set' &&
          Math.hypot(o.x - carrier.x, o.y - carrier.y) < 76
      );
      if (
        activeEnemyScreener &&
        player.effectiveStats.iq >= 50 &&
        Math.hypot(player.x - activeEnemyScreener.x, player.y - activeEnemyScreener.y) < 74
      ) {
        player.aiAction = 'help_defense';
        player.aiThought = 'Смена на заслоне — выходит на мяч!';
        player.targetX = clampCourtX(carrier.x + (defHoop.x - carrier.x) * 0.24);
        player.targetY = clampCourtY(carrier.y + (defHoop.y - carrier.y) * 0.24);
        if (enemyCarrierBall && distToCarrier < 38 && player.stealCooldown <= 0) {
          callbacks.onAttemptSteal(player, carrier, enemyCarrierBall);
        }
        return;
      }
    }

    // 6г-0) Защитная ротация «Помоги помогающему» (X-Out Rotation):
    // когда партнёр бросил своего снайпера на дуге ради подстраховки в краске (help_defense), умный защитник слабой стороны летит закрывать открытого стрелка!
    const mateHelpingPaint = teammates.some((m) => m.aiAction === 'help_defense');
    if (mateHelpingPaint && carrier && player.effectiveStats.iq >= 54 && !player.traitIds.includes('sleepy_fly')) {
      const openDeadlyShooter = opponents.find((o) => {
        if (o.id === carrier.id || o.trippedTimer > 0) return false;
        const isDeadly =
          o.effectiveStats.shooting >= 68 ||
          o.traitIds.includes('parking_sniper') ||
          o.hotStreak >= 3;
        if (!isDeadly) return false;
        const oDistHoop = Math.hypot(o.x - defHoop.x, o.y - defHoop.y);
        if (oDistHoop < THREE_PT_RADIUS - 28 || oDistHoop > 395) return false;
        const anyTeammateClose = teammates.some(
          (m) => m.aiAction !== 'help_defense' && Math.hypot(m.x - o.x, m.y - o.y) < 84
        );
        return !anyTeammateClose;
      });

      if (openDeadlyShooter) {
        const myDistOpen = Math.hypot(player.x - openDeadlyShooter.x, player.y - openDeadlyShooter.y);
        const closerRotatorExists = teammates.some(
          (m) =>
            m.id !== player.id &&
            m.role !== 'C' &&
            m.aiAction !== 'help_defense' &&
            m.aiAction !== 'guard_carrier' &&
            Math.hypot(m.x - openDeadlyShooter.x, m.y - openDeadlyShooter.y) < myDistOpen - 12
        );
        if (!closerRotatorExists) {
          player.aiAction = 'closeout';
          player.markTargetId = openDeadlyShooter.id;
          player.targetX = clampCourtX(openDeadlyShooter.x + (defHoop.x - openDeadlyShooter.x) * 0.16);
          player.targetY = clampCourtY(openDeadlyShooter.y + (defHoop.y - openDeadlyShooter.y) * 0.16);
          player.aiThought = `Ротация X-Out — закрывает снайпера ${openDeadlyShooter.name.split(' ')[0]}!`;
          return;
        }
      }
    }

    // 6г) Умная позиционная опека («Мяч — Я — Игрок», страховка слабой стороны и смешное залипание глупых балбесов на мяч!)
    const assignedOpp = findAssignedOpponent(player, teammates, opponents, carrier?.id ?? null);
    if (assignedOpp) {
      player.aiAction = 'mark_man';
      const firstName = assignedOpp.name.split(' ')[0];

      // Балбесы с низким IQ или «Сонная муха» периодически засматриваются на мяч и теряют своего игрока за спиной!
      if (
        carrier &&
        (player.effectiveStats.iq <= 34 || player.traitIds.includes('sleepy_fly')) &&
        (player.ballWatchTimer ?? 0) <= 0 &&
        rng() < 0.09 * dt * 10
      ) {
        player.ballWatchTimer = 1.35;
      }
      if (carrier && (player.ballWatchTimer ?? 0) > 0) {
        player.targetX = clampCourtX(player.x * 0.55 + carrier.x * 0.45);
        player.targetY = clampCourtY(player.y * 0.55 + carrier.y * 0.45);
        player.aiThought = `Засмотрелся на мячик и упустил ${firstName}!`;
        return;
      }

      const denyWeight =
        0.18 + (player.effectiveStats.steal / 100) * 0.16 + (player.effectiveStats.iq / 100) * 0.1;

      if (carrier) {
        const oppDistCarrier = Math.hypot(assignedOpp.x - carrier.x, assignedOpp.y - carrier.y);
        const oppIsSniper = assignedOpp.traitIds.includes('parking_sniper');
        const oppIsOnFire = assignedOpp.hotStreak >= 3 && player.effectiveStats.iq >= 50;
        // На слабой стороне (далеко от мяча) умный защитник глубже смещается к краске («помоги помогающему»), а против снайпера или игрока «В ОГНЕ» играет впритирку!
        const sagFactor = oppIsOnFire
          ? 0.09
          : oppIsSniper
          ? 0.12
          : oppDistCarrier > 255 && player.effectiveStats.iq >= 54
          ? 0.38
          : 0.22;
        const towardHoopX = assignedOpp.x + (defHoop.x - assignedOpp.x) * sagFactor;
        const towardHoopY = assignedOpp.y + (defHoop.y - assignedOpp.y) * sagFactor;
        const passLineX = assignedOpp.x + (carrier.x - assignedOpp.x) * denyWeight;
        const passLineY = assignedOpp.y + (carrier.y - assignedOpp.y) * denyWeight;
        player.targetX = clampCourtX(towardHoopX * 0.58 + passLineX * 0.42);
        player.targetY = clampCourtY(towardHoopY * 0.58 + passLineY * 0.42);
        player.aiThought = oppIsOnFire
          ? `Не даёт дышать горячему ${firstName}!`
          : oppIsSniper
          ? `Впритирку пасёт снайпера ${firstName}`
          : sagFactor > 0.3
          ? `Страхует краску и держит ${firstName}`
          : `Держит ${firstName} и кроет пас`;
      } else {
        let tx = assignedOpp.x + (defHoop.x - assignedOpp.x) * 0.25;
        const ty = assignedOpp.y + (defHoop.y - assignedOpp.y) * 0.25;
        if (!state.frontcourtEstablished && teamTactic !== 'blitz') {
          tx = player.team === 0 ? Math.min(tx, MID_X - 65) : Math.max(tx, MID_X + 65);
        }
        player.targetX = clampCourtX(tx);
        player.targetY = clampCourtY(ty);
        player.aiThought = `Держит ${firstName}`;
      }
      return;
    }
  }

  // ==================== 7. ПОЗИЦИОННАЯ АТАКА 5×5 (БЫСТРЫЙ ОТРЫВ, ЗАСЛОНЫ, БЭКДОР-КАТЫ, СПЕЙСИНГ) ====================
  if (alliedCarrierBall || !enemyCarrierBall) {
    const carrier = alliedCarrierBall
      ? teammates.find((m) => m.id === alliedCarrierBall.holderId) ?? null
      : null;
    const atkHoopIdx = getAttackingHoop(player.team);
    const atkHoop = getHoopPosition(atkHoopIdx);
    const dirFromHoop = player.team === 0 ? -1 : 1;

    // 7.0) Правило 3 секунд в атаке: умный игрок вовремя делает шаг из краски, чтобы судья не свистнул нарушение!
    if (
      player.paintTimer >= 1.9 &&
      isInAttackingPaint(player) &&
      player.effectiveStats.iq >= 34
    ) {
      player.aiAction = 'cut_open';
      player.cutTimer = 0;
      player.screenTargetId = null;
      player.aiThought = 'Выходит из краски (правило 3с)!';
      player.targetX = clampFrontcourtX(
        player.team,
        atkHoop.x + dirFromHoop * 92,
        state.frontcourtEstablished
      );
      const exitSide =
        Math.abs(player.y - HOOP_Y) < 24
          ? getCommittedBypassSide(player)
          : player.y < HOOP_Y
          ? -1
          : 1;
      player.targetY = clampCourtY(HOOP_Y + exitSide * 98);
      return;
    }

    // 7.05) Удержание заслона сразу после передачи мяча из рук в руки (DHO -> Screen)!
    if ((player.handoffHoldTimer ?? 0) > 0) {
      player.aiAction = 'screen_set';
      player.targetX = player.x;
      player.targetY = player.y;
      player.aiThought = 'Держит заслон после передачи из рук в руки!';
      return;
    }

    // 7.1) Если нашего партнёра с мячом зажали под плотной опекой ИЛИ он уже остановил ведение — ближайший защитник/винг открывается рядом под скидку!
    if (
      carrier &&
      (carrier.closelyGuardedTimer >= 0.75 || carrier.hasStoppedDribble) &&
      (player.role === 'PG' || player.role === 'SG' || player.role === 'SF')
    ) {
      const myDist = Math.hypot(player.x - carrier.x, player.y - carrier.y);
      const closerHelper = teammates.some(
        (m) =>
          m.id !== carrier.id &&
          (m.role === 'PG' || m.role === 'SG' || m.role === 'SF') &&
          Math.hypot(m.x - carrier.x, m.y - carrier.y) < myDist - 12
      );
      if (!closerHelper) {
        player.aiAction = 'cut_open';
        player.screenTargetId = null;
        player.aiThought = carrier.hasStoppedDribble
          ? 'Бежит открыться — партнёр убрал ведение!'
          : 'Бежит на выручку под скидку!';
        const helpSign =
          Math.abs(carrier.y - HOOP_Y) < 24
            ? getCommittedBypassSide(player)
            : carrier.y < HOOP_Y
            ? 1
            : -1;
        const helpSideY = carrier.y + helpSign * 125;
        const helpX = carrier.x + (player.team === 0 ? 48 : -48);
        player.targetX = clampFrontcourtX(player.team, helpX, state.frontcourtEstablished);
        player.targetY = clampCourtY(helpSideY);
        return;
      }
    }

    // 7.1б) Выполнение живой тренерской комбинации с планшета (activePlayCall)!
    const myCoachCall =
      carrier && state.activePlayCall?.team === player.team ? state.activePlayCall.play : null;
    if (carrier && myCoachCall === 'iso_star') {
      // Освобождаем центральный коридор для изоляции лидера — растягиваемся в углы и к бровкам!
      player.aiAction = 'cut_open';
      player.screenTargetId = null;
      const cornerY = player.slotIndex % 2 === 0 ? COURT_TOP + 36 : COURT_BOTTOM - 36;
      const wingDepth = player.slotIndex < 2 ? 48 : THREE_PT_RADIUS + 18;
      player.targetX = clampFrontcourtX(
        player.team,
        atkHoop.x + dirFromHoop * wingDepth,
        state.frontcourtEstablished
      );
      player.targetY = clampCourtY(cornerY);
      player.aiThought = `Уходит в угол — чистит изоляцию для ${carrier.name.split(' ')[0]}!`;
      return;
    }
    if (carrier && myCoachCall === 'alley_oop') {
      const myLobScore =
        player.effectiveStats.jump +
        (player.traitIds.includes('sixth_sense_alley') ? 35 : 0) +
        (player.traitIds.includes('gorilla_dunk') ? 30 : 0) +
        (player.traitIds.includes('grasshopper') ? 30 : 0);
      const betterFlyerExists = teammates.some((m) => {
        if (m.id === carrier.id || m.id === player.id) return false;
        const mScore =
          m.effectiveStats.jump +
          (m.traitIds.includes('sixth_sense_alley') ? 35 : 0) +
          (m.traitIds.includes('gorilla_dunk') ? 30 : 0) +
          (m.traitIds.includes('grasshopper') ? 30 : 0);
        return mScore > myLobScore;
      });
      if (!betterFlyerExists) {
        player.aiAction = 'roll_to_rim';
        player.cutTimer = Math.max(player.cutTimer, 1.2);
        player.screenTargetId = null;
        player.targetX = clampFrontcourtX(
          player.team,
          atkHoop.x + dirFromHoop * 56,
          state.frontcourtEstablished
        );
        player.targetY = clampCourtY(HOOP_Y + (player.slotIndex % 2 === 0 ? -24 : 24));
        player.aiThought = 'Врывается под парашют Алли-уп по сигналу тренера!';
        return;
      }
    }
    if (carrier && myCoachCall === 'feed_sniper') {
      const bestTeamSniper = !teammates.some(
        (m) =>
          m.id !== carrier.id &&
          m.id !== player.id &&
          m.effectiveStats.shooting > player.effectiveStats.shooting + 2
      );
      if (bestTeamSniper) {
        const sniperAng = player.y <= HOOP_Y ? -0.48 : 0.48;
        const sniperDist = THREE_PT_RADIUS + 28;
        player.popSpotX = clampFrontcourtX(
          player.team,
          atkHoop.x + dirFromHoop * Math.cos(sniperAng) * sniperDist,
          state.frontcourtEstablished
        );
        player.popSpotY = clampCourtY(HOOP_Y + Math.sin(sniperAng) * sniperDist * 0.94);
        player.aiAction = 'pick_and_pop';
        player.cutTimer = Math.max(player.cutTimer, 1.3);
        player.screenTargetId = null;
        player.targetX = player.popSpotX;
        player.targetY = player.popSpotY;
        player.aiThought = 'Выходит на чистую дугу по комбинации «На снайпера»!';
        return;
      }
    }

    // 7а) Продолжение начатого манёвра: рывок под кольцо (Roll / Backdoor / Дай-и-беги) или отскок на трёшку (Пик-н-поп)!
    if (
      player.cutTimer > 0 &&
      (player.aiAction === 'roll_to_rim' ||
        (player.aiAction === 'backdoor_cut' && !player.traitIds.includes('parking_sniper')) ||
        player.aiAction === 'give_and_go' ||
        player.aiAction === 'pick_and_pop')
    ) {
      player.screenTargetId = null;
      if (player.aiAction === 'pick_and_pop' && player.popSpotX !== undefined && player.popSpotY !== undefined) {
        player.targetX = clampFrontcourtX(player.team, player.popSpotX, state.frontcourtEstablished);
        player.targetY = clampCourtY(player.popSpotY);
        player.aiThought = 'Отскочил на дугу под трёшку (Пик-н-поп)!';
        return;
      }
      player.targetX = clampFrontcourtX(
        player.team,
        atkHoop.x + dirFromHoop * 58,
        state.frontcourtEstablished
      );
      player.targetY = clampCourtY(HOOP_Y + (player.slotIndex % 2 === 0 ? -36 : 36));
      player.aiThought =
        player.aiAction === 'give_and_go'
          ? 'Врывается под ответную стенку (Дай и беги)!'
          : player.aiAction === 'roll_to_rim'
          ? 'Врывается под кольцо после заслона!'
          : 'Резкий рывок за спину под щит!';
      return;
    }

    // 7б) Быстрый отрыв (Fast Break) в первые 3.6с владения, пока мяч ещё в тылу!
    const carrierInBackcourt = carrier
      ? player.team === 0
        ? carrier.x < MID_X + 55
        : carrier.x > MID_X - 55
      : false;
    const isFastRunner =
      player.role === 'SG' ||
      player.role === 'SF' ||
      player.traitIds.includes('turbo_hamster') ||
      player.effectiveStats.speed >= 68;

    if (
      carrier &&
      state.possessionTimer < 3.6 &&
      carrierInBackcourt &&
      isFastRunner &&
      !player.traitIds.includes('parking_sniper')
    ) {
      player.aiAction = 'fast_break';
      player.screenTargetId = null;
      player.aiThought = 'Летит вперёд в быстрый отрыв!';
      const laneOffset = player.role === 'SG' ? -115 : player.role === 'SF' ? 115 : 0;
      player.targetX = clampCourtX(atkHoop.x + dirFromHoop * 105);
      player.targetY = clampCourtY(HOOP_Y + laneOffset);
      return;
    }

    // 7в) Пик-н-ролл ИЛИ Пик-н-поп: ближайший к мячу большой (C / PF / Дылда) ставит заслон опекуну разыгрывающего!
    const forceCoachScreen = myCoachCall === 'pick_and_roll';
    const canPlayerScreen = (p: PlayerEntity) =>
      (p.role === 'C' || p.role === 'PF' || p.traitIds.includes('lanky_beanpole')) &&
      (teamTactic !== 'perimeter' || forceCoachScreen) &&
      (p.effectiveStats.teamwork >= 36 || forceCoachScreen) &&
      !p.traitIds.includes('parking_sniper') &&
      !p.schizoBreadMode &&
      p.trippedTimer <= 0;

    if (carrier && canPlayerScreen(player)) {
      const myDistToCarrier = Math.hypot(carrier.x - player.x, carrier.y - player.y);
      const closerScreenerExists = teammates.some(
        (m) =>
          m.id !== carrier.id &&
          canPlayerScreen(m) &&
          (m.aiAction === 'screen_set' ||
            Math.hypot(carrier.x - m.x, carrier.y - m.y) < myDistToCarrier - 10)
      );

      if (!closerScreenerExists) {
        const carrierDistHoop = Math.hypot(carrier.x - atkHoop.x, carrier.y - atkHoop.y);
        const defSearchRadius = forceCoachScreen ? 185 : 88;
        // Ищем защитника, который плотно сидит на нашем игроке с мячом на периметре
        const onBallDef = opponents.find(
          (o) =>
            o.trippedTimer <= 0 &&
            Math.hypot(o.x - carrier.x, o.y - carrier.y) < defSearchRadius
        );

        if (onBallDef && carrierDistHoop > 155 && carrierDistHoop < 475) {
          const carrierCloserThanDef =
            carrierDistHoop < Math.hypot(onBallDef.x - atkHoop.x, onBallDef.y - atkHoop.y) - 10;

          // Если мы поставили заслон и партнёр проскочил: выбираем между провалом под щит (Пик-н-ролл) и отскоком на дугу (Пик-н-поп)!
          if (player.aiAction === 'screen_set' && myDistToCarrier < 66 && carrierCloserThanDef) {
            const paintClogged = opponents.some(
              (o) => Math.hypot(o.x - atkHoop.x, o.y - atkHoop.y) < 108
            );
            const canStretchPop =
              player.effectiveStats.shooting >= 64 &&
              player.effectiveStats.iq >= 46 &&
              paintClogged;

            player.screenTargetId = null;
            if (canStretchPop) {
              const popAng = player.y <= HOOP_Y ? -0.52 : 0.52;
              const popDist = THREE_PT_RADIUS + 26;
              player.popSpotX = clampFrontcourtX(
                player.team,
                atkHoop.x + dirFromHoop * Math.cos(popAng) * popDist,
                state.frontcourtEstablished
              );
              player.popSpotY = clampCourtY(HOOP_Y + Math.sin(popAng) * popDist * 0.94);
              player.aiAction = 'pick_and_pop';
              player.cutTimer = 1.95;
              player.aiThought = 'Отскакивает на трёшку (Пик-н-поп)!';
              player.targetX = player.popSpotX;
              player.targetY = player.popSpotY;
            } else {
              player.aiAction = 'roll_to_rim';
              player.cutTimer = 1.85;
              player.aiThought = 'Проваливается под кольцо (Пик-н-ролл)!';
              player.targetX = clampCourtX(atkHoop.x + dirFromHoop * 58);
              player.targetY = clampCourtY(HOOP_Y);
            }
            return;
          }

          // Иначе встаём в жёсткий заслон сбоку от защитника на мяче (с защитой от дрожания на HOOP_Y)
          const sideSign =
            Math.abs(onBallDef.y - HOOP_Y) < 22
              ? getCommittedBypassSide(player)
              : onBallDef.y < HOOP_Y
              ? 1
              : -1;
          const screenX = clampCourtX(onBallDef.x + dirFromHoop * 10);
          const screenY = clampCourtY(onBallDef.y + sideSign * 24);
          player.aiAction = 'screen_set';
          player.screenTargetId = onBallDef.id;
          player.aiThought = `Ставит заслон для ${carrier.name.split(' ')[0]}!`;
          player.targetX = screenX;
          player.targetY = screenY;
          return;
        }

        // Комедия глупого большого (IQ <= 23): перепутал команды и встал в заслон прямо перед собственным разыгрывающим!
        if (
          player.effectiveStats.iq <= 23 &&
          isInFrontcourt(player.team, carrier.x) &&
          myDistToCarrier < 95 &&
          rng() < 0.04
        ) {
          player.aiAction = 'screen_set';
          player.screenTargetId = carrier.id;
          player.handoffHoldTimer = 0.75;
          player.targetX = clampFrontcourtX(
            player.team,
            carrier.x - dirFromHoop * 26,
            state.frontcourtEstablished
          );
          player.targetY = clampCourtY(carrier.y);
          player.aiThought = 'Ой! Перепутал и поставил заслон своему партнёру!';
          return;
        }
      }

      // 7в-bis) Заслон без мяча под снайпера (Pin-Down / Off-Ball Screen):
      // свободный большой идёт ставить заслон опекуну нашего закрытого снайпера на крыле/в углу!
      if (
        isInFrontcourt(player.team, carrier.x) &&
        player.effectiveStats.iq >= 46 &&
        !teammates.some((m) => m.aiAction === 'offball_screen')
      ) {
        let targetSniper: PlayerEntity | null = null;
        let sniperDefender: PlayerEntity | null = null;
        for (const m of teammates) {
          if (m.id === carrier.id || m.id === player.id || m.trippedTimer > 0) continue;
          const isDeadlySniper =
            m.effectiveStats.shooting >= 74 ||
            m.traitIds.includes('parking_sniper') ||
            m.hotStreak >= 3;
          if (!isDeadlySniper) continue;
          const mDistHoop = Math.hypot(m.x - atkHoop.x, m.y - atkHoop.y);
          if (mDistHoop < 150) continue;
          const tightDef = opponents.find(
            (o) => o.trippedTimer <= 0 && Math.hypot(o.x - m.x, o.y - m.y) < 52
          );
          if (tightDef) {
            targetSniper = m;
            sniperDefender = tightDef;
            break;
          }
        }

        if (targetSniper && sniperDefender) {
          const pinSide = targetSniper.y <= HOOP_Y ? 1 : -1;
          player.aiAction = 'offball_screen';
          player.screenTargetId = sniperDefender.id;
          player.targetX = clampFrontcourtX(
            player.team,
            sniperDefender.x + dirFromHoop * 12,
            state.frontcourtEstablished
          );
          player.targetY = clampCourtY(sniperDefender.y + pinSide * 20);
          player.aiThought = `Ставит заслон без мяча под снайпера ${targetSniper.name.split(' ')[0]}!`;
          return;
        }
      }
    }

    // 7г) Бэкдор-кат (рывок за спину), если защитник слишком плотно переигрывает линию паса!
    if (
      carrier &&
      (player.role === 'PG' || player.role === 'SG' || player.role === 'SF') &&
      !player.traitIds.includes('parking_sniper') &&
      player.effectiveStats.iq >= 48
    ) {
      const myDistHoop = Math.hypot(player.x - atkHoop.x, player.y - atkHoop.y);
      const tightOverplayingDef = opponents.find((o) => {
        if (o.trippedTimer > 0) return false;
        const dMe = Math.hypot(o.x - player.x, o.y - player.y);
        const oDistHoop = Math.hypot(o.x - atkHoop.x, o.y - atkHoop.y);
        return dMe < 46 && oDistHoop > myDistHoop - 8;
      });

      if (tightOverplayingDef && myDistHoop > 155) {
        player.aiAction = 'backdoor_cut';
        player.cutTimer = 1.65;
        player.screenTargetId = null;
        player.aiThought = 'Рывок за спину защитнику под кольцо!';
        player.targetX = clampCourtX(atkHoop.x + dirFromHoop * 62);
        const cutSign =
          Math.abs(player.y - HOOP_Y) < 22
            ? getCommittedBypassSide(player)
            : player.y < HOOP_Y
            ? -1
            : 1;
        player.targetY = clampCourtY(HOOP_Y + cutSign * 38);
        return;
      }
    }

    // 7д-pin) Выбегание снайпера из-под заслона без мяча (Pin-Down Curl) на чистую трёхочковую дугу!
    const nearbyOffBallScreener =
      carrier &&
      (player.effectiveStats.shooting >= 70 ||
        player.traitIds.includes('parking_sniper') ||
        player.hotStreak >= 3)
        ? teammates.find(
            (m) =>
              m.aiAction === 'offball_screen' &&
              Math.hypot(m.x - player.x, m.y - player.y) < 110
          ) ?? null
        : null;
    if (carrier && nearbyOffBallScreener) {
      const curlAng = nearbyOffBallScreener.y <= HOOP_Y ? -0.56 : 0.56;
      const curlR = THREE_PT_RADIUS + 26;
      player.aiAction = 'relocate_arc';
      player.screenTargetId = null;
      player.targetX = clampFrontcourtX(
        player.team,
        atkHoop.x + dirFromHoop * Math.cos(curlAng) * curlR,
        state.frontcourtEstablished
      );
      player.targetY = clampCourtY(atkHoop.y + Math.sin(curlAng) * curlR * 0.93);
      player.aiThought = `Выбегает из-под заслона без мяча на трёшку!`;
      return;
    }

    // 7д-0) Смещение снайпера по трёхочковой дуге (Relocation), когда партнёр прорвался в краску и стянул защиту!
    if (
      carrier &&
      isInFrontcourt(player.team, carrier.x) &&
      Math.hypot(carrier.x - atkHoop.x, carrier.y - atkHoop.y) < 190 &&
      player.effectiveStats.iq >= 46 &&
      (player.effectiveStats.shooting >= 62 ||
        player.traitIds.includes('parking_sniper') ||
        player.role === 'SG' ||
        player.role === 'PG')
    ) {
      let myNearestDef = 999;
      for (const opp of opponents) {
        if (opp.trippedTimer > 0) continue;
        const d = Math.hypot(opp.x - player.x, opp.y - player.y);
        if (d < myNearestDef) myNearestDef = d;
      }
      if (myNearestDef > 56) {
        const arcAngles = [-0.95, -0.56, -0.18, 0.18, 0.56, 0.95];
        const arcR = THREE_PT_RADIUS + 26;
        let bestRelocX = player.x;
        let bestRelocY = player.y;
        let bestRelocScore = -9999;

        for (const ang of arcAngles) {
          const rx = clampFrontcourtX(
            player.team,
            atkHoop.x + dirFromHoop * Math.cos(ang) * arcR,
            state.frontcourtEstablished
          );
          const ry = clampCourtY(atkHoop.y + Math.sin(ang) * arcR * 0.93);
          const moveDist = Math.hypot(rx - player.x, ry - player.y);
          if (moveDist > 165) continue;

          const virtualReceiver = { ...player, x: rx, y: ry };
          const laneDanger = computeRawPassLaneDanger(carrier, virtualReceiver, opponents);
          let minOppD = 999;
          for (const opp of opponents) {
            if (opp.trippedTimer > 0) continue;
            const d = Math.hypot(opp.x - rx, opp.y - ry);
            if (d < minOppD) minOppD = d;
          }
          let minMateD = 999;
          for (const mate of teammates) {
            if (mate.id === carrier.id || mate.id === player.id) continue;
            const d = Math.hypot(mate.x - rx, mate.y - ry);
            if (d < minMateD) minMateD = d;
          }
          const relocScore =
            Math.min(130, minOppD) * 0.65 -
            laneDanger * 1.4 -
            moveDist * 0.16 -
            (minMateD < 88 ? (88 - minMateD) * 0.8 : 0);
          if (relocScore > bestRelocScore) {
            bestRelocScore = relocScore;
            bestRelocX = rx;
            bestRelocY = ry;
          }
        }

        if (bestRelocScore > 12) {
          player.aiAction = 'relocate_arc';
          player.screenTargetId = null;
          player.aiThought = 'Смещается по дуге под скидку из краски!';
          player.targetX = bestRelocX;
          player.targetY = bestRelocY;
          return;
        }
      }
    }

    // 7д-post) Занятие глубокой позиции в лоу-посте на «усах» (Post-Up) центровым или мощным форвардом без нарушения 3 секунд!
    if (
      carrier &&
      isInFrontcourt(player.team, carrier.x) &&
      teamTactic !== 'perimeter' &&
      (player.role === 'C' || (player.role === 'PF' && player.effectiveStats.shooting < 64)) &&
      !player.traitIds.includes('parking_sniper') &&
      player.paintTimer < 1.3 &&
      state.shotClock > 6.0 &&
      !teammates.some((m) => m.aiAction === 'post_up')
    ) {
      player.aiAction = 'post_up';
      player.screenTargetId = null;
      const blockSign = carrier.y <= HOOP_Y ? -1 : 1;
      player.targetX = clampFrontcourtX(
        player.team,
        atkHoop.x + dirFromHoop * 78,
        state.frontcourtEstablished
      );
      player.targetY = clampCourtY(HOOP_Y + blockSign * 88);
      player.aiThought = 'Просит мяч спиной к кольцу на усах (лоу-пост)!';
      return;
    }

    // 7д) Умное позиционное открывание со спейсингом (анти-толпа)
    player.screenTargetId = null;
    const spot = getOffensiveSpot(player, teamTactic);
    const cutWave = Math.sin(state.quarterTimeRemaining * 1.8 + player.slotIndex * 1.7);
    let targetX = spot.x + cutWave * 24;
    let targetY = spot.y + Math.cos(state.quarterTimeRemaining * 1.5 + player.slotIndex) * 28;

    // Отталкивание от соседних партнёров без мяча, чтобы сохранять широту атаки (100px спейсинг)
    for (const mate of teammates) {
      if (mate.id === carrier?.id || mate.aiAction === 'screen_set' || mate.aiAction === 'offball_screen') continue;
      const dx = targetX - mate.x;
      const dy = targetY - mate.y;
      const d = Math.hypot(dx, dy);
      if (d > 1 && d < 96) {
        const push = (96 - d) * 0.55;
        targetX += (dx / d) * push;
        targetY += (dy / d) * push;
      }
    }

    // Если у игрока черта «Месси» и неплохой IQ, он старается открываться строго за центральной линией!
    if (player.traitIds.includes('messi') && player.effectiveStats.iq > 45) {
      targetX = player.team === 0 ? Math.max(MID_X + 68, targetX) : Math.min(MID_X - 68, targetX);
    }

    const gluedByDefender = opponents.some(
      (o) => o.trippedTimer <= 0 && Math.hypot(o.x - player.x, o.y - player.y) < 36
    );
    player.aiAction = 'cut_open';
    player.aiThought =
      player.effectiveStats.iq <= 25 && gluedByDefender
        ? 'Машет руками и просит пас, хотя наглухо закрыт!'
        : player.traitIds.includes('parking_sniper')
        ? 'Ждёт скидку на трёхочковой дуге'
        : 'Держит спейсинг и открывается под пас';
    player.targetX = clampFrontcourtX(player.team, targetX, state.frontcourtEstablished);
    player.targetY = clampCourtY(targetY);
  }
}
