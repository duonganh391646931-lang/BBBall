import { evaluatePassLaneRisk } from '../ai';
import {
  COURT_H,
  COURT_W,
  HOOP_Y,
  MID_X,
  THREE_PT_RADIUS,
  computeRefereeDetectionChance,
  getAttackingHoop,
  getDefendingHoop,
  getHoopPosition,
  isInAttackingPaint,
} from '../physics';
import { MATCH_MODIFIERS, TRAIT_CATALOG, isLastHopeBuffActive } from '../traits';
import { BallState, CoachPlayCall, MatchState, PlayerEntity, PositionRole, TeamTactic } from '../types';
import { OBS_DIM } from './schema';

const ROLES_INDEX: Record<PositionRole, number> = {
  PG: 0,
  SG: 1,
  SF: 2,
  PF: 3,
  C: 4,
};

const TACTIC_CODE: Record<TeamTactic, number> = {
  balanced: 0,
  blitz: 1 / 3,
  perimeter: 2 / 3,
  dirty: 1,
};

const PLAY_CALL_CODE: Record<CoachPlayCall, number> = {
  iso_star: 0.25,
  pick_and_roll: 0.5,
  alley_oop: 0.75,
  feed_sniper: 1.0,
};

const TRAIT_INDEX_MAP: ReadonlyMap<string, number> = new Map(
  TRAIT_CATALOG.map((t, idx) => [t.id, idx])
);

const MODIFIER_IDS = MATCH_MODIFIERS.map((m) => m.id);

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Переводит мировую X-координату в каноническую систему команды:
 * своё кольцо всегда слева (x = 96), чужое кольцо всегда справа (x = 1084).
 */
export function toCanonicalX(team: 0 | 1, worldX: number): number {
  return team === 0 ? worldX : COURT_W - worldX;
}

export function toCanonicalVx(team: 0 | 1, worldVx: number): number {
  return team === 0 ? worldVx : -worldVx;
}

export function fromCanonicalDx(team: 0 | 1, canonicalDx: number): number {
  return team === 0 ? canonicalDx : -canonicalDx;
}

/**
 * Возвращает до 4 партнёров по команде на паркете в детерминированном порядке slotIndex.
 * Индекс партнёра [0..3] строго совпадает с действиями PASS_MATE_0..3 и ALLEY_OOP_MATE_0..3.
 */
export function getOrderedTeammates(state: MatchState, player: PlayerEntity): PlayerEntity[] {
  const mates: PlayerEntity[] = [];
  for (let i = 0; i < state.players.length; i++) {
    const p = state.players[i];
    if (p.team === player.team && p.id !== player.id) {
      mates.push(p);
    }
  }
  mates.sort((a, b) => a.slotIndex - b.slotIndex);
  return mates;
}

/**
 * Возвращает соперников на паркете, отсортированных по расстоянию до игрока (ближайший — [0]).
 */
export function getOrderedOpponents(state: MatchState, player: PlayerEntity): PlayerEntity[] {
  const opps: PlayerEntity[] = [];
  for (let i = 0; i < state.players.length; i++) {
    const p = state.players[i];
    if (p.team !== player.team) {
      opps.push(p);
    }
  }
  opps.sort(
    (a, b) =>
      Math.hypot(a.x - player.x, a.y - player.y) - Math.hypot(b.x - player.x, b.y - player.y)
  );
  return opps;
}

/**
 * Инерционный шум восприятия для персонажей с низким статом «ИИ» (обновляется раз в ~0.75с,
 * не создавая покадрового дребезга направления движения).
 */
interface IqNoiseEntry {
  nextRefreshKey: number;
  timerBias: number;
  lineBias: number;
  refBias: number;
}

const iqNoiseCache = new Map<string, IqNoiseEntry>();

export function applyInertialIqPerceptionNoise(
  obs: Float32Array,
  player: PlayerEntity,
  state: MatchState,
  rng: () => number,
  offset = 0
): void {
  const iq = player.effectiveStats.iq;
  if (iq >= 82) return;

  // Масштаб ошибки восприятия: 0 при IQ >= 82, до ~0.22 при IQ = 10
  const strength = clamp((82 - iq) / 290, 0, 0.25);
  const bucket = Math.floor(state.quarter * 1000 - state.quarterTimeRemaining / 0.75);
  let entry = iqNoiseCache.get(player.id);
  if (!entry || entry.nextRefreshKey !== bucket) {
    entry = {
      nextRefreshKey: bucket,
      timerBias: (rng() * 2 - 1) * strength,
      lineBias: (rng() * 2 - 1) * strength * 0.65,
      refBias: (rng() * 2 - 1) * strength * 1.15,
    };
    iqNoiseCache.set(player.id, entry);
  }

  // Слегка искажаем восприятие таймеров 24с / 8с / 3с, края площадки и взгляда судьи
  obs[offset + 2] = clamp(obs[offset + 2] + entry.timerBias, 0, 1); // 24с
  obs[offset + 3] = clamp(obs[offset + 3] + entry.timerBias, 0, 1); // 8с
  obs[offset + 15] = clamp(obs[offset + 15] + entry.refBias, -1, 1); // косинус взгляда судьи
  obs[offset + 17] = clamp(obs[offset + 17] + entry.refBias * 0.8, 0, 1); // оценка шанса свистка
  obs[offset + 18] = clamp(obs[offset + 18] + entry.lineBias, -1.2, 1.2); // восприятие X/аута
  obs[offset + 19] = clamp(obs[offset + 19] + entry.lineBias, -1.2, 1.2); // восприятие Y/аута
  obs[offset + 32] = clamp(obs[offset + 32] + entry.timerBias, 0, 1.2); // таймер 3с в краске
}

function selectOrderedBalls(state: MatchState, player: PlayerEntity): [BallState | null, BallState | null] {
  if (state.balls.length === 0) return [null, null];
  if (state.balls.length === 1) return [state.balls[0], null];

  const b0 = state.balls[0];
  const b1 = state.balls[1];
  if (b1.holderId === player.id || b1.passReceiverId === player.id) {
    return [b1, b0];
  }
  if (b0.holderId === player.id || b0.passReceiverId === player.id) {
    return [b0, b1];
  }
  const d0 = Math.hypot(b0.x - player.x, b0.y - player.y);
  const d1 = Math.hypot(b1.x - player.x, b1.y - player.y);
  return d1 < d0 ? [b1, b0] : [b0, b1];
}

/**
 * Кодирует состояние матча с точки зрения конкретного игрока в вектор длины OBS_DIM (244 float32).
 * Все координаты зеркалируются для команды 1, поэтому одна общая условная нейросеть
 * симметрично управляет всеми 10 игроками обеих команд.
 */
export function encodeObservation(
  state: MatchState,
  player: PlayerEntity,
  out?: Float32Array,
  outOffset = 0
): Float32Array {
  const buf = out ?? new Float32Array(OBS_DIM);
  let k = outOffset;

  const myTeam = player.team;
  const oppTeam: 0 | 1 = myTeam === 0 ? 1 : 0;
  const atkHoopIdx = getAttackingHoop(myTeam);
  const defHoopIdx = getDefendingHoop(myTeam);
  const atkHoop = getHoopPosition(atkHoopIdx);
  const defHoop = getHoopPosition(defHoopIdx);

  const selfCanonX = toCanonicalX(myTeam, player.x);
  const distToAtkHoop = Math.hypot(player.x - atkHoop.x, player.y - atkHoop.y);
  const [primaryBall, secondaryBall] = selectOrderedBalls(state, player);
  const heldBall = primaryBall && primaryBall.holderId === player.id ? primaryBall : null;

  // Скалярное произведение направления взгляда судьи на вектор (Судья -> Игрок): +1 = смотрит в упор, -1 = спиной
  const refDx = player.x - state.referee.x;
  const refDy = player.y - state.referee.y;
  const refDist = Math.max(1e-3, Math.hypot(refDx, refDy));
  const gazeCos =
    (Math.cos(state.referee.gazeAngle) * refDx + Math.sin(state.referee.gazeAngle) * refDy) /
    refDist;
  const refSuspicion = clamp((player.refSuspicion ?? 0) / 5, 0, 1);

  // 1. Глобальный контекст, часы и Судья (18 float: 0..17)
  buf[k++] = Math.min(state.quarter, 5) / 5;
  buf[k++] = clamp(state.quarterTimeRemaining / Math.max(1, state.quarterDuration), 0, 1);
  buf[k++] = clamp(state.shotClock / 24, 0, 1);
  buf[k++] = clamp(state.backcourtTimer / 8, 0, 1);
  buf[k++] = clamp((state.score[myTeam] - state.score[oppTeam]) / 30, -1, 1);
  buf[k++] = state.possessionTeam === myTeam ? 1 : 0;
  buf[k++] = state.possessionTeam === oppTeam ? 1 : 0;
  buf[k++] = state.frontcourtEstablished && state.possessionTeam === myTeam ? 1 : 0;
  buf[k++] = Math.min(state.teamFouls[myTeam], 5) / 5;
  buf[k++] = Math.min(state.teamFouls[oppTeam], 5) / 5;
  buf[k++] = state.inbound && state.inbound.team === myTeam ? 1 : 0;
  buf[k++] = state.inbound && state.inbound.team === oppTeam ? 1 : 0;
  buf[k++] = state.inbound ? clamp(state.inbound.timer / 5, 0, 1) : 0;
  buf[k++] = (toCanonicalX(myTeam, state.referee.x) - MID_X) / (COURT_W * 0.5);
  buf[k++] = (state.referee.y - HOOP_Y) / (COURT_H * 0.5);
  buf[k++] = clamp(gazeCos, -1, 1);
  buf[k++] = refSuspicion;
  buf[k++] = computeRefereeDetectionChance(state.referee, player).chance;

  // 2. Блок самосознания игрока (69 float: 18..86)
  // 2а. Кинематика (10: 18..27)
  buf[k++] = clamp((selfCanonX - MID_X) / (COURT_W * 0.5), -1.2, 1.2);
  buf[k++] = clamp((player.y - HOOP_Y) / (COURT_H * 0.5), -1.2, 1.2);
  buf[k++] = clamp(player.z / 250, 0, 1.5);
  buf[k++] = clamp(toCanonicalVx(myTeam, player.vx) / 300, -1.5, 1.5);
  buf[k++] = clamp(player.vy / 300, -1.5, 1.5);
  buf[k++] = clamp(player.vz / 500, -1.5, 1.5);
  const cosH = Math.cos(player.heading);
  buf[k++] = myTeam === 0 ? cosH : -cosH;
  buf[k++] = Math.sin(player.heading);
  buf[k++] = clamp(distToAtkHoop / 1100, 0, 1.2);
  buf[k++] = distToAtkHoop >= THREE_PT_RADIUS ? 1 : 0;

  // 2б. Игровые статусы, таймеры и бафф «Последняя надежда» (14: 28..41)
  buf[k++] = heldBall ? 1 : 0;
  buf[k++] = player.hasStoppedDribble ? 1 : 0;
  buf[k++] = state.inbound?.passerId === player.id ? 1 : 0;
  buf[k++] = isInAttackingPaint(player) ? 1 : 0;
  buf[k++] = clamp(player.paintTimer / 3, 0, 1.2);
  buf[k++] = clamp(player.closelyGuardedTimer / 5, 0, 1.2);
  buf[k++] = clamp(player.energy / 100, 0, 1);
  buf[k++] = clamp(player.fouls / 5, 0, 1);
  buf[k++] = clamp(player.hotStreak / 4, -1, 1);
  buf[k++] = player.stealCooldown <= 0 ? 1 : 0;
  buf[k++] =
    player.traitIds.includes('grasshopper') && player.superJumpCd <= 0 && player.energy >= 15
      ? 1
      : 0;
  buf[k++] = (player.stepBackCd ?? 0) <= 0 || (player.eurostepCd ?? 0) <= 0 ? 1 : 0;
  buf[k++] = (player.crossoverCd ?? 0) <= 0 || (player.pumpFakeCd ?? 0) <= 0 ? 1 : 0;
  buf[k++] = isLastHopeBuffActive(player) ? 1 : 0;

  // 2в. Позиция one-hot (5: 42..46)
  const rIdx = ROLES_INDEX[player.role] ?? 0;
  for (let r = 0; r < 5; r++) {
    buf[k++] = r === rIdx ? 1 : 0;
  }

  // 2г. Все 9 эффективных характеристик (9: 47..55)
  const st = player.effectiveStats;
  buf[k++] = st.speed / 115;
  buf[k++] = st.shooting / 115;
  buf[k++] = st.stamina / 115;
  buf[k++] = st.steal / 115;
  buf[k++] = st.jump / 115;
  buf[k++] = st.teamwork / 115;
  buf[k++] = st.dribble / 115;
  buf[k++] = st.iq / 115;
  buf[k++] = st.deviance / 100;

  // 2д. Полный каталог всех 31 черт multi-hot (31: 56..86)
  const traitStart = k;
  for (let t = 0; t < 31; t++) {
    buf[k++] = 0;
  }
  for (let i = 0; i < player.traitIds.length; i++) {
    const tIdx = TRAIT_INDEX_MAP.get(player.traitIds[i]);
    if (tIdx !== undefined && tIdx < 31) {
      buf[traitStart + tIdx] = 1;
    }
  }

  // 3. Блок мячей: приоритетный мяч (16) + второй мяч для «Два мяча» (8) = 24 float (87..110)
  if (primaryBall) {
    const bCanonX = toCanonicalX(myTeam, primaryBall.x);
    buf[k++] = clamp((bCanonX - selfCanonX) / COURT_W, -1.2, 1.2);
    buf[k++] = clamp((primaryBall.y - player.y) / COURT_H, -1.2, 1.2);
    buf[k++] = clamp((bCanonX - MID_X) / (COURT_W * 0.5), -1.2, 1.2);
    buf[k++] = clamp((primaryBall.y - HOOP_Y) / (COURT_H * 0.5), -1.2, 1.2);
    buf[k++] = clamp(primaryBall.z / 250, 0, 1.5);
    buf[k++] = clamp(toCanonicalVx(myTeam, primaryBall.vx) / 600, -1.5, 1.5);
    buf[k++] = clamp(primaryBall.vy / 600, -1.5, 1.5);
    buf[k++] = clamp(primaryBall.vz / 600, -1.5, 1.5);
    buf[k++] = primaryBall.holderId === null && primaryBall.shotType === 'none' ? 1 : 0;
    buf[k++] = primaryBall.shotType === 'pass' ? 1 : 0;
    buf[k++] = primaryBall.holderId === null && primaryBall.targetHoop !== null ? 1 : 0;
    buf[k++] = primaryBall.passReceiverId === player.id ? 1 : 0;
    buf[k++] = primaryBall.targetHoop === atkHoopIdx ? 1 : 0;
    buf[k++] = primaryBall.targetHoop === defHoopIdx ? 1 : 0;
    buf[k++] = state.shotTouchedRim ? 1 : 0;
    buf[k++] = primaryBall.isTipoffBall ? 1 : 0;
  } else {
    for (let b = 0; b < 16; b++) buf[k++] = 0;
  }

  if (secondaryBall) {
    const b2CanonX = toCanonicalX(myTeam, secondaryBall.x);
    buf[k++] = 1;
    buf[k++] = clamp((b2CanonX - selfCanonX) / COURT_W, -1.2, 1.2);
    buf[k++] = clamp((secondaryBall.y - player.y) / COURT_H, -1.2, 1.2);
    buf[k++] = clamp(secondaryBall.z / 250, 0, 1.5);
    buf[k++] = clamp(toCanonicalVx(myTeam, secondaryBall.vx) / 600, -1.5, 1.5);
    buf[k++] = clamp(secondaryBall.vy / 600, -1.5, 1.5);
    buf[k++] = secondaryBall.targetHoop === atkHoopIdx ? 1 : 0;
    buf[k++] = secondaryBall.targetHoop === defHoopIdx ? 1 : 0;
  } else {
    for (let b = 0; b < 8; b++) buf[k++] = 0;
  }

  // 4. Четыре партнёра по команде в порядке slotIndex (4 * 14 = 56 float: 111..166)
  const mates = getOrderedTeammates(state, player);
  const opps = getOrderedOpponents(state, player);
  const activeOpps = opps.filter((o) => o.penaltyTimer <= 0 && o.trippedTimer <= 0);

  for (let mIdx = 0; mIdx < 4; mIdx++) {
    const m = mates[mIdx];
    if (!m) {
      for (let f = 0; f < 14; f++) buf[k++] = 0;
      continue;
    }
    const mCanonX = toCanonicalX(myTeam, m.x);
    const mHasBall = state.balls.some((b) => b.holderId === m.id);
    const mDistAtk = Math.hypot(m.x - atkHoop.x, m.y - atkHoop.y);
    let minDefD = 300;
    for (let j = 0; j < activeOpps.length; j++) {
      const d = Math.hypot(activeOpps[j].x - m.x, activeOpps[j].y - m.y);
      if (d < minDefD) minDefD = d;
    }
    const laneDanger = evaluatePassLaneRisk(player, m, activeOpps);

    buf[k++] = m.penaltyTimer <= 0 && m.trippedTimer <= 0 && !m.schizoBreadMode ? 1 : 0;
    buf[k++] = clamp((mCanonX - selfCanonX) / COURT_W, -1.2, 1.2);
    buf[k++] = clamp((m.y - player.y) / COURT_H, -1.2, 1.2);
    buf[k++] = clamp((mCanonX - MID_X) / (COURT_W * 0.5), -1.2, 1.2);
    buf[k++] = clamp((m.y - HOOP_Y) / (COURT_H * 0.5), -1.2, 1.2);
    buf[k++] = clamp(m.z / 250, 0, 1.5);
    buf[k++] = clamp(toCanonicalVx(myTeam, m.vx) / 300, -1.5, 1.5);
    buf[k++] = clamp(m.vy / 300, -1.5, 1.5);
    buf[k++] = mHasBall ? 1 : 0;
    buf[k++] = clamp(mDistAtk / 1100, 0, 1.2);
    buf[k++] = clamp(minDefD / 300, 0, 1);
    buf[k++] = clamp(laneDanger / 60, 0, 1.2);
    buf[k++] = m.effectiveStats.shooting / 115;
    buf[k++] = m.effectiveStats.jump / 115;
  }

  // 5. Пять соперников в порядке близости к игроку (5 * 14 = 70 float: 167..236)
  for (let oIdx = 0; oIdx < 5; oIdx++) {
    const o = opps[oIdx];
    if (!o) {
      for (let f = 0; f < 14; f++) buf[k++] = 0;
      continue;
    }
    const oCanonX = toCanonicalX(myTeam, o.x);
    const oHasBall = state.balls.some((b) => b.holderId === o.id);
    const oDistDef = Math.hypot(o.x - defHoop.x, o.y - defHoop.y);

    buf[k++] = o.penaltyTimer <= 0 && o.trippedTimer <= 0 ? 1 : 0;
    buf[k++] = clamp((oCanonX - selfCanonX) / COURT_W, -1.2, 1.2);
    buf[k++] = clamp((o.y - player.y) / COURT_H, -1.2, 1.2);
    buf[k++] = clamp((oCanonX - MID_X) / (COURT_W * 0.5), -1.2, 1.2);
    buf[k++] = clamp((o.y - HOOP_Y) / (COURT_H * 0.5), -1.2, 1.2);
    buf[k++] = clamp(o.z / 250, 0, 1.5);
    buf[k++] = clamp(toCanonicalVx(myTeam, o.vx) / 300, -1.5, 1.5);
    buf[k++] = clamp(o.vy / 300, -1.5, 1.5);
    buf[k++] = oHasBall ? 1 : 0;
    buf[k++] = o.windupTimer > 0 ? 1 : 0;
    buf[k++] = (o.bittenByFakeTimer ?? 0) > 0 || o.z > 20 ? 1 : 0;
    buf[k++] = clamp(oDistDef / 1100, 0, 1.2);
    buf[k++] = o.effectiveStats.shooting / 115;
    buf[k++] = o.effectiveStats.steal / 115;
  }

  // 6. Тактика тренера, комбинация планшета [1..4] и все 5 модификаторов физики (7 float: 237..243)
  buf[k++] = TACTIC_CODE[state.tactics[myTeam]] ?? 0;
  buf[k++] =
    state.activePlayCall && state.activePlayCall.team === myTeam
      ? PLAY_CALL_CODE[state.activePlayCall.play] ?? 0
      : 0;
  for (let m = 0; m < 5; m++) {
    const modId = MODIFIER_IDS[m];
    buf[k++] = modId && state.modifiers.includes(modId) ? 1 : 0;
  }

  return buf;
}
