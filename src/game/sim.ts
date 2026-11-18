import {
  chooseAdaptiveAiTactic,
  getCommittedBypassSide,
  stepPlayerAI,
  wrapAngle,
} from './ai';
import {
  CENTER_CIRCLE_RADIUS,
  COURT_BOTTOM,
  COURT_LEFT,
  COURT_RIGHT,
  COURT_TOP,
  HOOP_LEFT_X,
  HOOP_RIGHT_X,
  HOOP_Y,
  MID_X,
  PENALTY_BENCH_POS,
  RIM_RADIUS,
  RIM_Z,
  THREE_PT_RADIUS,
  TRAMPOLINE_PADS,
  computeArcVelocity,
  computeJumpVelocity,
  computePlayerHorizontalInterceptRadius,
  computePlayerVerticalReach,
  computeRefereeDetectionChance,
  getAttackingHoop,
  getDefendingHoop,
  getEffectiveGravity,
  getFreeThrowLinePos,
  getHoopPosition,
  getNearestHoop,
  getTeamBenchSeatPos,
  isInAttackingPaint,
  isInFrontcourt,
  isOnOwnHalf,
  isPointOutOfBounds,
  isThreePointShot,
} from './physics';
import { sound } from './sound';
import {
  CLUB_PRESETS,
  computeBodyScale,
  computeEffectiveStats,
  createClumsyReservePlayer,
  generateRandomPlayer,
  getHomeFormationCoords,
  getPlayerOverall,
  isLastHopeBuffActive,
} from './traits';
import {
  BallState,
  BallTrailPoint,
  CoachPlayCall,
  MatchModifierId,
  MatchState,
  MatchSummarySnapshot,
  PlayerEntity,
  PlayerStats,
  PooledParticle,
  ShotStyle,
  TacticalCallout,
  TeamTactic,
} from './types';

const PARTICLE_POOL_SIZE = 160;
const CALLOUT_POOL_SIZE = 18;
const TRAIL_LENGTH = 16;

function createEmptyTrail(): BallTrailPoint[] {
  return Array.from({ length: TRAIL_LENGTH }, () => ({
    x: MID_X,
    y: HOOP_Y,
    z: 0,
    alpha: 0,
    isMessi: false,
  }));
}

export function createTipoffBall(id: number, offsetY = 0): BallState {
  return {
    id,
    x: MID_X,
    y: HOOP_Y + offsetY,
    z: 95,
    vx: 0,
    vy: 0,
    vz: 165,
    holderId: null,
    lastTouchTeam: null,
    lastShooterId: null,
    lastPasserId: null,
    passReceiverId: null,
    shotType: 'tipoff',
    shotStyle: 'standard',
    targetHoop: null,
    shotOriginX: MID_X,
    shotOriginY: HOOP_Y + offsetY,
    isGuaranteedSwish: false,
    isTipoffBall: true,
    wasShotAttempt: false,
    shotChartRecorded: false,
    rimBounceCount: 0,
    spinAngle: 0,
    trail: createEmptyTrail(),
    trailHead: 0,
  };
}

export interface CreateMatchOptions {
  mode?: 'exhibition' | 'tournament';
  teamNames?: [string, string];
  teamColors?: [string, string];
  rosterA?: PlayerEntity[];
  rosterB?: PlayerEntity[];
  quarterDuration?: number;
  modifiers?: MatchModifierId[];
  tactics?: [TeamTactic, TeamTactic];
  rng?: () => number;
}

function rotateAngleToward(current: number, target: number, maxDelta: number): number {
  const diff = wrapAngle(target - current);
  if (Math.abs(diff) <= maxDelta) return wrapAngle(target);
  return wrapAngle(current + Math.sign(diff) * maxDelta);
}

export function findAnyPlayerById(state: MatchState, id: string | null): PlayerEntity | null {
  if (!id) return null;
  return (
    state.players.find((p) => p.id === id) ??
    state.bench.find((p) => p.id === id) ??
    null
  );
}

export function resetPlayerForMatch(
  p: PlayerEntity,
  team: 0 | 1,
  slotIndex: number,
  isBench = false
): PlayerEntity {
  const home = isBench
    ? getTeamBenchSeatPos(team, slotIndex, 3)
    : getHomeFormationCoords(team, slotIndex);
  return {
    ...p,
    team,
    slotIndex,
    effectiveStats: computeEffectiveStats(p.baseStats, p.traitIds, false),
    bodyScale: computeBodyScale(p.traitIds),
    x: home.x,
    y: home.y,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    heading: isBench ? Math.PI / 2 : team === 0 ? 0 : Math.PI,
    energy: 100,
    aiAction: isBench ? 'team_bench' : 'tipoff',
    aiThought: isBench ? 'Готов ворваться со скамейки!' : 'Ждёт вбрасывания',
    targetX: home.x,
    targetY: home.y,
    decisionTimer: 0.1,
    actionCooldown: 0,
    stealCooldown: 0,
    pickupLockout: 0,
    windupTimer: 0,
    windupTargetHoop: null,
    windupBallId: null,
    windupShotStyle: 'standard',
    schizoBreadMode: false,
    cutTimer: 0,
    screenTargetId: null,
    superJumpCd: 0,
    dirtyTrickCd: 1.5,
    refSuspicion: 0,
    trippedTimer: 0,
    penaltyTimer: 0,
    fouls: 0,
    isReserveSub: false,
    onCourt: !isBench,
    startedOnBench: isBench,
    subbedInOnce: false,
    lastHopeAnnounced: false,
    benchSlot: isBench ? slotIndex : 0,
    disqualified: false,
    paintTimer: 0,
    closelyGuardedTimer: 0,
    airborneWithBall: false,
    hasStoppedDribble: false,
    dribbleLiveTimer: 0,
    dodgeSide: slotIndex % 2 === 0 ? 1 : -1,
    dodgeCommitTimer: 0,
    driveLockTimer: 0,
    markTargetId: null,
    pumpFakeTimer: 0,
    pumpFakeCd: 0,
    bittenByFakeTimer: 0,
    crossoverCd: 0,
    stepBackCd: 0,
    eurostepCd: 0,
    chargePlantTimer: 0,
    ballWatchTimer: 0,
    handoffHoldTimer: 0,
    handoffCd: 0,
    saveOobCd: 0,
    popSpotX: undefined,
    popSpotY: undefined,
    hotStreak: 0,
    flopCd: 0,
    points: 0,
    rebounds: 0,
    fgMade: 0,
    fgAttempted: 0,
    threeMade: 0,
    threeAttempted: 0,
    freeThrowsMade: 0,
    ownGoals: 0,
    assists: 0,
    steals: 0,
    blocks: 0,
    dunks: 0,
    dirtyPlays: 0,
  };
}

export function createMatch(opts: CreateMatchOptions = {}): MatchState {
  const rng = opts.rng ?? Math.random;
  const usedNames = new Set<string>();

  if (opts.rosterA) {
    for (const p of opts.rosterA) usedNames.add(p.name);
  }
  if (opts.rosterB) {
    for (const p of opts.rosterB) usedNames.add(p.name);
  }

  const team0Players =
    opts.rosterA && opts.rosterA.length >= 5
      ? opts.rosterA.slice(0, 5).map((p, idx) => resetPlayerForMatch(p, 0, idx, false))
      : Array.from({ length: 5 }, (_, i) => generateRandomPlayer(0, i, usedNames, rng, false));

  const providedBench0 = opts.rosterA ? opts.rosterA.slice(5, 8) : [];
  const team0Bench: PlayerEntity[] = Array.from({ length: 3 }, (_, bIdx) =>
    providedBench0[bIdx]
      ? resetPlayerForMatch(providedBench0[bIdx], 0, bIdx, true)
      : generateRandomPlayer(0, bIdx, usedNames, rng, true)
  );

  const team1Players =
    opts.rosterB && opts.rosterB.length >= 5
      ? opts.rosterB.slice(0, 5).map((p, idx) => resetPlayerForMatch(p, 1, idx, false))
      : Array.from({ length: 5 }, (_, i) => generateRandomPlayer(1, i, usedNames, rng, false));

  const providedBench1 = opts.rosterB ? opts.rosterB.slice(5, 8) : [];
  const team1Bench: PlayerEntity[] = Array.from({ length: 3 }, (_, bIdx) =>
    providedBench1[bIdx]
      ? resetPlayerForMatch(providedBench1[bIdx], 1, bIdx, true)
      : generateRandomPlayer(1, bIdx, usedNames, rng, true)
  );

  const players = [...team0Players, ...team1Players];
  const bench = [...team0Bench, ...team1Bench];
  const modifiers = opts.modifiers ? [...opts.modifiers] : [];
  const quarterDuration = opts.quarterDuration ?? 45;

  const balls: BallState[] = modifiers.includes('two_balls')
    ? [createTipoffBall(1, -48), createTipoffBall(2, 48)]
    : [createTipoffBall(1, 0)];

  const particles: PooledParticle[] = Array.from({ length: PARTICLE_POOL_SIZE }, () => ({
    active: false,
    x: 0,
    y: 0,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    color: '#ffffff',
    size: 3,
    age: 0,
    maxAge: 0.5,
  }));

  const callouts: TacticalCallout[] = Array.from({ length: CALLOUT_POOL_SIZE }, () => ({
    active: false,
    x: 0,
    y: 0,
    text: '',
    color: '#ffffff',
    age: 0,
    duration: 1.2,
  }));

  const state: MatchState = {
    mode: opts.mode ?? 'exhibition',
    teamNames: opts.teamNames ?? [CLUB_PRESETS[0].name, CLUB_PRESETS[1].name],
    teamColors: opts.teamColors ?? [CLUB_PRESETS[0].primary, CLUB_PRESETS[1].primary],
    players,
    bench,
    subCooldown: [8.0, 8.0],
    substitutionsCount: 0,
    timeoutsRemaining: [2, 2],
    scoringRun: { team: null, points: 0 },
    activePlayCall: null,
    playCallCooldown: [0, 0],
    shotChart: [],
    balls,
    referee: {
      x: MID_X,
      y: COURT_BOTTOM + 18,
      vx: 0,
      vy: 0,
      gazeAngle: -Math.PI / 2,
      whistleTimer: 0,
      lastCallText: null,
      callsMade: 0,
      missedFouls: 0,
    },
    score: [0, 0],
    quarter: 1,
    quarterDuration,
    quarterTimeRemaining: quarterDuration,
    quarterScores: [[0, 0]],
    shotClock: 24,
    teamFouls: [0, 0],
    violationsCount: [0, 0],
    possessionTeam: null,
    possessionArrow: 0,
    possessionTimer: 0,
    frontcourtEstablished: false,
    backcourtTimer: 0,
    shotTouchedRim: false,
    shotClockExpiredInFlight: false,
    heldBallTieTimer: 0,
    inbound: null,
    freeThrow: null,
    aiCoachCooldown: 3.0,
    tactics: opts.tactics ?? ['balanced', 'balanced'],
    modifiers,
    celebrationTimer: 1.4,
    quarterBreakTimer: 0,
    lastBannerText: 'Q1 · СПОРНЫЙ МЯЧ!',
    lastBannerColor: '#fbbf24',
    hitstopTimer: 0,
    cameraShake: 0,
    netSwayLeft: 0,
    netSwayRight: 0,
    messiGoalsCount: 0,
    ownGoalsCount: 0,
    finished: false,
    summary: null,
    selectedPlayerId: players[0]?.id ?? null,
    particles,
    particleCursor: 0,
    callouts,
    calloutCursor: 0,
    events: [],
    nextEventId: 1,
  };

  pushEvent(state, 'Стартовый свисток! Четверть 1 началась (на скамейках по 3 запасных).', 'neutral', false);
  return state;
}

export function formatClock(sec: number): string {
  const clamped = Math.max(0, Math.ceil(sec));
  const m = Math.floor(clamped / 60);
  const s = clamped % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function getQuarterLabel(q: number): string {
  return q <= 4 ? `Q${q}` : `OT${q - 4}`;
}

export function pushEvent(
  state: MatchState,
  text: string,
  team: 0 | 1 | 'ref' | 'neutral',
  highlight = false
): void {
  const ev = {
    id: state.nextEventId++,
    quarterLabel: getQuarterLabel(state.quarter),
    timeFormatted: formatClock(state.quarterTimeRemaining),
    text,
    team,
    highlight,
  };
  state.events.unshift(ev);
  if (state.events.length > 24) {
    state.events.length = 24;
  }
}

export function spawnCallout(
  state: MatchState,
  x: number,
  y: number,
  text: string,
  color = '#fbbf24',
  duration = 1.25
): void {
  const slot = state.callouts[state.calloutCursor];
  state.calloutCursor = (state.calloutCursor + 1) % state.callouts.length;
  slot.active = true;
  slot.x = x;
  slot.y = y;
  slot.text = text;
  slot.color = color;
  slot.age = 0;
  slot.duration = duration;
}

export function spawnParticles(
  state: MatchState,
  x: number,
  y: number,
  z: number,
  color: string,
  count: number,
  speedScale = 1,
  rng: () => number = Math.random
): void {
  for (let i = 0; i < count; i++) {
    const p = state.particles[state.particleCursor];
    state.particleCursor = (state.particleCursor + 1) % state.particles.length;
    const ang = rng() * Math.PI * 2;
    const spd = (35 + rng() * 115) * speedScale;
    p.active = true;
    p.x = x;
    p.y = y;
    p.z = z;
    p.vx = Math.cos(ang) * spd;
    p.vy = Math.sin(ang) * spd;
    p.vz = (60 + rng() * 140) * speedScale;
    p.color = color;
    p.size = 2.5 + rng() * 3;
    p.age = 0;
    p.maxAge = 0.35 + rng() * 0.35;
  }
}

export function syncModifierBalls(state: MatchState): void {
  const wantsTwo = state.modifiers.includes('two_balls');
  if (wantsTwo && state.balls.length < 2) {
    state.balls.push(createTipoffBall(2, 36));
    spawnCallout(state, MID_X, HOOP_Y, 'ВТОРОЙ МЯЧ НА ПАРКЕТЕ!', '#38bdf8');
  } else if (!wantsTwo && state.balls.length > 1) {
    state.balls.splice(1);
  }
}

export function updatePlayerTraitsAndStatsInSandbox(
  state: MatchState,
  playerId: string,
  newTraitIds: string[],
  newBaseStats?: PlayerStats
): void {
  const p = findAnyPlayerById(state, playerId);
  if (!p) return;
  if (newBaseStats) {
    p.baseStats = { ...newBaseStats };
  }
  p.traitIds = [...newTraitIds];
  p.effectiveStats = computeEffectiveStats(p.baseStats, p.traitIds, isLastHopeBuffActive(p));
  p.bodyScale = computeBodyScale(p.traitIds);
}

function syncTeamBenchSeats(state: MatchState, team: 0 | 1): void {
  const teamBench = state.bench.filter((p) => p.team === team);
  teamBench.forEach((bp, idx) => {
    bp.benchSlot = idx;
    const seat = getTeamBenchSeatPos(team, idx, teamBench.length);
    bp.x = seat.x;
    bp.y = seat.y;
    bp.targetX = seat.x;
    bp.targetY = seat.y;
    bp.z = 0;
    bp.vx = 0;
    bp.vy = 0;
    bp.vz = 0;
    bp.heading = Math.PI / 2;
    bp.aiAction = 'team_bench';
  });
}

/**
 * Проводит официальную замену между игроком на площадке и запасным со скамейки той же команды.
 * Если выходящий запасной начинал матч на скамейке и владеет уникальной чертой «Последняя надежда»,
 * активирует мощнейший мега-бафф и зрелищный выход!
 */
export function substitutePlayer(
  state: MatchState,
  outCourtPlayerId: string,
  inBenchPlayerId: string,
  reason = 'Ротация состава',
  rng: () => number = Math.random
): PlayerEntity | null {
  const courtIdx = state.players.findIndex((p) => p.id === outCourtPlayerId);
  const benchIdx = state.bench.findIndex((p) => p.id === inBenchPlayerId);
  if (courtIdx < 0 || benchIdx < 0) return null;

  const outPlayer = state.players[courtIdx];
  const inPlayer = state.bench[benchIdx];
  if (outPlayer.team !== inPlayer.team) return null;
  if (inPlayer.disqualified || inPlayer.fouls >= 5) return null;

  // Передаём тактический слот и позицию (PG..C), чтобы пятёрка на паркете сохраняла строй!
  const swappedSlot = outPlayer.slotIndex;
  const swappedRole = outPlayer.role;
  const oldBenchSlot = inPlayer.benchSlot;

  const wasInboundPasser = state.inbound?.passerId === outPlayer.id;
  const entryX = wasInboundPasser
    ? state.inbound!.x
    : Math.max(COURT_LEFT + 32, Math.min(COURT_RIGHT - 32, outPlayer.x));
  const entryY = wasInboundPasser
    ? state.inbound!.y
    : outPlayer.penaltyTimer > 0 || isPointOutOfBounds(outPlayer.x, outPlayer.y, -4)
    ? COURT_TOP + 28
    : Math.max(COURT_TOP + 24, Math.min(COURT_BOTTOM - 24, outPlayer.y));

  outPlayer.windupTimer = 0;
  outPlayer.windupBallId = null;
  outPlayer.windupTargetHoop = null;

  inPlayer.slotIndex = swappedSlot;
  inPlayer.role = swappedRole;
  inPlayer.x = entryX;
  inPlayer.y = entryY;
  inPlayer.targetX = entryX;
  inPlayer.targetY = entryY;
  inPlayer.z = 0;
  inPlayer.vx = 0;
  inPlayer.vy = 0;
  inPlayer.vz = 0;
  inPlayer.heading = inPlayer.team === 0 ? 0 : Math.PI;
  inPlayer.penaltyTimer = 0;
  inPlayer.trippedTimer = 0;
  inPlayer.paintTimer = 0;
  inPlayer.closelyGuardedTimer = 0;
  inPlayer.airborneWithBall = false;
  inPlayer.hasStoppedDribble = false;
  inPlayer.dribbleLiveTimer = 0;
  inPlayer.onCourt = true;
  inPlayer.subbedInOnce = true;
  inPlayer.aiAction = wasInboundPasser ? 'inbound_pass' : 'cut_open';
  inPlayer.aiThought = 'Свежий выход со скамейки!';

  const lastHopeTriggered = isLastHopeBuffActive(inPlayer);
  inPlayer.effectiveStats = computeEffectiveStats(
    inPlayer.baseStats,
    inPlayer.traitIds,
    lastHopeTriggered
  );

  // Передаём мяч / ввод / штрафной новому игроку, если заменяемый держал мяч
  for (const b of state.balls) {
    if (b.holderId === outPlayer.id) {
      b.holderId = inPlayer.id;
    }
  }
  if (state.inbound && state.inbound.passerId === outPlayer.id) {
    state.inbound.passerId = inPlayer.id;
  }
  if (state.freeThrow && state.freeThrow.shooterId === outPlayer.id) {
    state.freeThrow.shooterId = inPlayer.id;
  }
  if (state.selectedPlayerId === outPlayer.id) {
    state.selectedPlayerId = inPlayer.id;
  }

  // Сажаем заменённого игрока на командную скамейку запасных
  outPlayer.onCourt = false;
  outPlayer.benchSlot = oldBenchSlot;
  outPlayer.penaltyTimer = 0;
  outPlayer.trippedTimer = 0;
  outPlayer.paintTimer = 0;
  outPlayer.closelyGuardedTimer = 0;
  outPlayer.airborneWithBall = false;
  outPlayer.hasStoppedDribble = false;
  outPlayer.aiAction = 'team_bench';
  outPlayer.aiThought = outPlayer.disqualified
    ? 'Удалён до конца матча (5 фолов)'
    : 'Переводит дух на скамейке';
  outPlayer.effectiveStats = computeEffectiveStats(
    outPlayer.baseStats,
    outPlayer.traitIds,
    false
  );

  state.players[courtIdx] = inPlayer;
  state.bench[benchIdx] = outPlayer;
  syncTeamBenchSeats(state, inPlayer.team);
  state.substitutionsCount++;

  if (lastHopeTriggered && !inPlayer.lastHopeAnnounced) {
    inPlayer.lastHopeAnnounced = true;
    inPlayer.energy = 100;
    inPlayer.aiThought = 'ПОСЛЕДНЯЯ НАДЕЖДА! РВЁТ ПАРКЕТ!';
    spawnParticles(state, inPlayer.x, inPlayer.y, 18, '#fbbf24', 26, 1.2, rng);
    spawnCallout(
      state,
      inPlayer.x,
      inPlayer.y - 24,
      '★ ПОСЛЕДНЯЯ НАДЕЖДА! МЕГА-БАФФ!',
      '#fbbf24',
      1.9
    );
    state.lastBannerText = `★ ПОСЛЕДНЯЯ НАДЕЖДА: ${inPlayer.name} В ИГРЕ!`;
    state.lastBannerColor = '#fbbf24';
    state.celebrationTimer = Math.max(state.celebrationTimer, 1.5);
    pushEvent(
      state,
      `★ ДЖОКЕР СО СКАМЕЙКИ! Запасной ${inPlayer.name} [Последняя надежда] выходит вместо ${outPlayer.name} и получает ОГРОМНЫЕ БАФФЫ!`,
      inPlayer.team,
      true
    );
  } else {
    spawnCallout(state, inPlayer.x, inPlayer.y - 20, `⇄ ЗАМЕНА: ${inPlayer.name}`, '#38bdf8', 1.2);
    pushEvent(
      state,
      `Замена у «${state.teamNames[inPlayer.team]}» (${reason}): ${inPlayer.name} выходит вместо ${outPlayer.name}.`,
      inPlayer.team,
      false
    );
  }

  return inPlayer;
}

/**
 * Удобная замена в 1 клик из HUD или Песочницы:
 * - если выбран игрок на паркете — меняет его на лучшего доступного запасного своей команды;
 * - если выбран запасной на скамейке — выпускает его на паркет вместо самого уставшего/нафолившего партнёра!
 */
export function requestPlayerSubstitution(
  state: MatchState,
  targetPlayerId: string,
  rng: () => number = Math.random
): PlayerEntity | null {
  const onCourtPlayer = state.players.find((p) => p.id === targetPlayerId);
  if (onCourtPlayer) {
    const eligibleBench = state.bench
      .filter((b) => b.team === onCourtPlayer.team && !b.disqualified && b.fouls < 5)
      .sort((a, b) => {
        const scoreB = (pl: PlayerEntity) =>
          pl.energy * 0.65 +
          getPlayerOverall(pl) * 0.6 +
          (pl.traitIds.includes('last_hope') && pl.startedOnBench && !pl.subbedInOnce ? 32 : 0);
        return scoreB(b) - scoreB(a);
      });
    if (eligibleBench.length === 0) return null;
    const res = substitutePlayer(
      state,
      onCourtPlayer.id,
      eligibleBench[0].id,
      'Решение тренера',
      rng
    );
    if (res) {
      state.selectedPlayerId = res.id;
    }
    return res;
  }

  const benchPlayer = state.bench.find((b) => b.id === targetPlayerId);
  if (benchPlayer && !benchPlayer.disqualified && benchPlayer.fouls < 5) {
    const courtTeammates = state.players
      .filter((c) => c.team === benchPlayer.team)
      .sort((a, b) => {
        const scoreOut = (pl: PlayerEntity) =>
          (100 - pl.energy) +
          pl.fouls * 14 +
          (pl.role === benchPlayer.role ? 15 : 0) -
          (isLastHopeBuffActive(pl) ? 45 : 0) -
          (state.freeThrow?.shooterId === pl.id ? 200 : 0);
        return scoreOut(b) - scoreOut(a);
      });
    if (courtTeammates.length === 0) return null;
    const res = substitutePlayer(
      state,
      courtTeammates[0].id,
      benchPlayer.id,
      'Выход со скамейки',
      rng
    );
    if (res) {
      state.selectedPlayerId = res.id;
    }
    return res;
  }

  return null;
}

export const PLAY_CALL_LABELS: Record<CoachPlayCall, string> = {
  iso_star: 'Изоляция 1-в-1',
  pick_and_roll: 'Пик-н-ролл',
  alley_oop: 'Навес Алли-уп',
  feed_sniper: 'На снайпера',
};

/**
 * Тренерский минутный перерыв (Тайм-аут):
 * - восстанавливает дыхание пятёрке на площадке (+22% энергии) и скамейке (+14%);
 * - сбивает безответный рывок соперника и охлаждает кураж его стрелков;
 * - проводит ротацию уставших игроков со скамейки запасных!
 */
export function callTeamTimeout(
  state: MatchState,
  team: 0 | 1,
  rng: () => number = Math.random
): boolean {
  if (state.finished || state.timeoutsRemaining[team] <= 0) return false;

  state.timeoutsRemaining[team]--;
  const oppTeam: 0 | 1 = team === 0 ? 1 : 0;

  for (const p of state.players) {
    if (p.team === team) {
      p.energy = Math.min(100, p.energy + 22);
      if (p.hotStreak < 0) p.hotStreak = 0; // снимает мандраж промахов у своих!
    } else if (p.hotStreak >= 3) {
      p.hotStreak = 1; // тренерский тайм-аут остужает горячую руку соперника!
    }
  }
  for (const bp of state.bench) {
    if (bp.team === team && !bp.disqualified) {
      bp.energy = Math.min(100, bp.energy + 14);
    }
  }

  if (state.scoringRun.team === oppTeam) {
    state.scoringRun = { team: null, points: 0 };
  }

  performCoachSubstitutions(state, team, true, rng);

  sound.playWhistle();
  state.referee.whistleTimer = 0.95;
  state.lastBannerText = `ТАЙМ-АУТ · «${state.teamNames[team]}» (осталось: ${state.timeoutsRemaining[team]})`;
  state.lastBannerColor = state.teamColors[team];
  state.celebrationTimer = Math.max(state.celebrationTimer, 1.45);
  pushEvent(
    state,
    `ТАЙМ-АУТ! Тренер «${state.teamNames[team]}» берёт минутный перерыв: пятёрка перевела дух (+22% сил), темп соперника сбит!`,
    team,
    true
  );
  return true;
}

/**
 * Заказ живой комбинации в атаке с тренерского планшета (действует 9 секунд, КД 10 секунд).
 */
export function triggerCoachPlayCall(
  state: MatchState,
  team: 0 | 1,
  play: CoachPlayCall
): boolean {
  if (state.finished || state.playCallCooldown[team] > 0) return false;

  state.activePlayCall = {
    team,
    play,
    timer: 9.0,
  };
  state.playCallCooldown[team] = 10.0;

  const leader =
    state.players.find((p) => p.team === team && state.balls.some((b) => b.holderId === p.id)) ??
    state.players.find((p) => p.team === team && p.penaltyTimer <= 0) ??
    null;
  if (leader) {
    spawnCallout(
      state,
      leader.x,
      leader.y - 24,
      `ПЛАНШЕТ: ${PLAY_CALL_LABELS[play].toUpperCase()}!`,
      '#fde047',
      1.35
    );
  }
  pushEvent(
    state,
    `Тренер «${state.teamNames[team]}» кричит с бровки комбинацию: «${PLAY_CALL_LABELS[play]}»!`,
    team,
    false
  );
  return true;
}

/**
 * Автоматическая тренерская ротация на мёртвом мяче (между четвертями или перед вводом из-за линии):
 * выпускает свежих запасных при усталости, переборе фолов или прибережённого джокера «Последняя надежда».
 */
function performCoachSubstitutions(
  state: MatchState,
  team: 0 | 1,
  isPeriodBreak: boolean,
  rng: () => number
): void {
  if (!isPeriodBreak && state.subCooldown[team] > 0) return;

  const eligibleBench = state.bench.filter(
    (b) => b.team === team && !b.disqualified && b.fouls < 5 && b.energy >= 66
  );
  if (eligibleBench.length === 0) return;

  const oppTeam: 0 | 1 = team === 0 ? 1 : 0;
  const teamCourt = state.players.filter((p) => p.team === team && p.penaltyTimer <= 0);
  if (teamCourt.length === 0) return;

  // 1) Выпуск джокера «Последняя надежда» (начиная со 2-й четверти, либо если команда уступает в счёте / концовка Q1)
  const lastHopeSub = eligibleBench.find(
    (b) => b.traitIds.includes('last_hope') && b.startedOnBench && !b.subbedInOnce
  );
  if (
    lastHopeSub &&
    (state.quarter >= 2 ||
      state.score[team] < state.score[oppTeam] ||
      state.quarterTimeRemaining < 18)
  ) {
    const outTarget = [...teamCourt].sort(
      (a, b) =>
        a.energy +
        getPlayerOverall(a) * 0.5 +
        (isLastHopeBuffActive(a) ? 90 : 0) -
        (b.energy + getPlayerOverall(b) * 0.5 + (isLastHopeBuffActive(b) ? 90 : 0))
    )[0];
    if (outTarget && !isLastHopeBuffActive(outTarget)) {
      substitutePlayer(state, outTarget.id, lastHopeSub.id, 'Выход джокера', rng);
      state.subCooldown[team] = 11.0;
      return;
    }
  }

  // 2) Замена игрока на фол-трабле (3+ фола в первой половине или 4 фола в Q3)
  const foulTroublePlayer = teamCourt.find(
    (c) =>
      !isLastHopeBuffActive(c) &&
      ((state.quarter <= 2 && c.fouls >= 3) || (state.quarter === 3 && c.fouls >= 4))
  );
  if (foulTroublePlayer) {
    const safeSub = [...eligibleBench]
      .filter((b) => b.fouls <= foulTroublePlayer.fouls - 2)
      .sort((a, b) => b.energy + getPlayerOverall(b) * 0.4 - (a.energy + getPlayerOverall(a) * 0.4))[0];
    if (safeSub) {
      substitutePlayer(state, foulTroublePlayer.id, safeSub.id, 'Бережём от 5 фолов', rng);
      state.subCooldown[team] = 11.0;
      return;
    }
  }

  // 2.5) Умная тактическая замена под концовку 4-й четверти (снайпер под камбэк или замок в защиту!)
  if (state.quarter >= 4 && state.quarterTimeRemaining < 22) {
    const scoreDiff = state.score[team] - state.score[oppTeam];
    if (scoreDiff <= -3) {
      const benchSniper = [...eligibleBench]
        .filter((b) => b.effectiveStats.shooting >= 74 || b.traitIds.includes('parking_sniper'))
        .sort((a, b) => b.effectiveStats.shooting - a.effectiveStats.shooting)[0];
      const brickStarter = [...teamCourt]
        .filter((c) => !isLastHopeBuffActive(c) && c.effectiveStats.shooting < 56)
        .sort((a, b) => a.effectiveStats.shooting - b.effectiveStats.shooting)[0];
      if (benchSniper && brickStarter) {
        substitutePlayer(state, brickStarter.id, benchSniper.id, 'Снайпер на спасение', rng);
        state.subCooldown[team] = 11.0;
        return;
      }
    } else if (scoreDiff >= 2 && state.quarterTimeRemaining < 16) {
      const benchStopper = [...eligibleBench]
        .filter(
          (b) =>
            b.effectiveStats.steal >= 74 ||
            b.traitIds.includes('barrier_blocker') ||
            b.traitIds.includes('pickpocket')
        )
        .sort((a, b) => b.effectiveStats.steal - a.effectiveStats.steal)[0];
      const weakDef = [...teamCourt]
        .filter((c) => !isLastHopeBuffActive(c) && c.effectiveStats.steal < 52)
        .sort((a, b) => a.effectiveStats.steal - b.effectiveStats.steal)[0];
      if (benchStopper && weakDef) {
        substitutePlayer(state, weakDef.id, benchStopper.id, 'Замок в защиту на концовку', rng);
        state.subCooldown[team] = 11.0;
        return;
      }
    }
  }

  // 3) Плановая ротация по усталости
  const tiredThreshold = isPeriodBreak ? 62 : 44;
  const mostTired = [...teamCourt].sort((a, b) => a.energy - b.energy)[0];
  const freshestBench = [...eligibleBench].sort(
    (a, b) => b.energy + getPlayerOverall(b) * 0.35 - (a.energy + getPlayerOverall(a) * 0.35)
  )[0];
  if (
    mostTired &&
    freshestBench &&
    mostTired.energy < tiredThreshold &&
    freshestBench.energy >= mostTired.energy + 26
  ) {
    substitutePlayer(state, mostTired.id, freshestBench.id, 'Свежие ноги со скамейки', rng);
    state.subCooldown[team] = 11.0;
  }
}

/**
 * Передача мяча в руки конкретному игроку + проверка срабатывания черты «Месси» (7% на своей половине / 15% на вражеской!)
 */
export function giveBallToPlayer(
  state: MatchState,
  ball: BallState,
  player: PlayerEntity,
  rng: () => number,
  allowMessiTrigger = true,
  isOffensiveRebound = false
): void {
  // Освобождаем прежнего владельца этого мяча, если был в замахе
  for (const pl of state.players) {
    if (pl.windupBallId === ball.id) {
      pl.windupTimer = 0;
      pl.windupBallId = null;
      pl.windupTargetHoop = null;
    }
  }

  const wasTipoff = ball.isTipoffBall;
  // Если был пас от союзника — запомним для ассиста и проверим завершение Алли-упа!
  const wasAlliedPass = ball.shotType === 'pass' && ball.lastTouchTeam === player.team;
  const wasDirectOwnMissedShot = Boolean(ball.wasShotAttempt && ball.lastTouchTeam === player.team);
  const wasMissedOwnTeamShot = isOffensiveRebound || wasDirectOwnMissedShot;
  if (ball.wasShotAttempt) {
    recordMissedShotOutcome(state, ball);
    player.rebounds++;
    ball.wasShotAttempt = false;
  }
  const prevPasser = wasAlliedPass ? ball.lastPasserId : null;
  const catchZ = ball.z;

  ball.holderId = player.id;
  ball.lastTouchTeam = player.team;
  ball.shotType = 'none';
  ball.shotStyle = 'standard';
  ball.targetHoop = null;
  ball.isGuaranteedSwish = false;
  ball.isTipoffBall = false;
  ball.vx = 0;
  ball.vy = 0;
  ball.vz = 0;
  ball.lastPasserId = prevPasser;
  ball.passReceiverId = null;
  player.decisionTimer = Math.max(player.decisionTimer, 0.16);
  player.airborneWithBall = false;
  player.caughtInAir = player.z > 0 || player.vz > 0;
  player.hasStoppedDribble = false;
  player.dribbleLiveTimer = 0;
  player.closelyGuardedTimer = 0;

  // Стрелка поочерёдного владения достаётся команде, уступившей стартовый спорный мяч
  if (wasTipoff) {
    state.possessionArrow = player.team === 0 ? 1 : 0;
  }

  const clearlyInFrontcourt =
    player.team === 0 ? player.x > MID_X + 26 : player.x < MID_X - 26;
  const wasCenterTipoff =
    wasTipoff && Math.hypot(player.x - MID_X, player.y - HOOP_Y) <= CENTER_CIRCLE_RADIUS + 34;

  if (state.possessionTeam !== player.team) {
    state.possessionTeam = player.team;
    state.possessionTimer = 0;
    state.shotClock = 24;
    // По правилам FIBA/NBA на стартовом спорном мяче в центральном круге разрешено откинуть первый пас назад в тыловую зону!
    state.frontcourtEstablished =
      !wasCenterTipoff &&
      state.inbound === null &&
      state.freeThrow === null &&
      clearlyInFrontcourt;
    state.backcourtTimer = 0;
    state.shotTouchedRim = false;
    state.shotClockExpiredInFlight = false;
    state.heldBallTieTimer = 0;
    for (const pl of state.players) {
      pl.paintTimer = 0;
    }
  } else {
    // Правило 14 секунд FIBA/NBA: при подборе в нападении после касания дужки кольца счётчик сбрасывается до 14с (если было меньше)!
    if (state.shotTouchedRim) {
      if (state.shotClock < 14) {
        state.shotClock = 14;
        spawnCallout(state, player.x, player.y - 20, 'ПОДБОР В АТАКЕ · 14 СЕК!', '#38bdf8', 1.15);
      }
      state.shotTouchedRim = false;
      state.shotClockExpiredInFlight = false;
    }
    if (
      !state.frontcourtEstablished &&
      !wasCenterTipoff &&
      state.inbound === null &&
      state.freeThrow === null &&
      clearlyInFrontcourt
    ) {
      state.frontcourtEstablished = true;
      state.backcourtTimer = 0;
    }
  }

  // Проверка срабатывания уникальной черты «Месси»!
  if (allowMessiTrigger && player.traitIds.includes('messi')) {
    const ownHalf = isOnOwnHalf(player);
    const procChance = ownHalf ? 0.07 : 0.15;
    if (rng() < procChance) {
      const nearestHoop = getNearestHoop(player.x, player.y);
      const isOwnGoalTarget = nearestHoop === getDefendingHoop(player.team);
      player.aiAction = 'messi_windup';
      player.windupTimer = 0.14;
      player.windupTargetHoop = nearestHoop;
      player.windupBallId = ball.id;
      player.aiThought = isOwnGoalTarget
        ? 'МЕССИ БЬЁТ В СВОЁ КОЛЬЦО?!'
        : 'УДАР МЕССИ ЧЕРЕЗ ПОЛПОЛЯ!';
      spawnCallout(
        state,
        player.x,
        player.y - 24,
        isOwnGoalTarget ? 'МЕССИ?! НЕ ТУДА!' : 'МАГИЯ МЕССИ!',
        isOwnGoalTarget ? '#f87171' : '#fbbf24',
        1.5
      );
      pushEvent(
        state,
        isOwnGoalTarget
          ? `${player.name} [Месси] ловит кураж на своей половине и целится в СВОЁ кольцо!`
          : `${player.name} [Месси] запускает фирменную радугу через полплощадки!`,
        player.team,
        true
      );
      return;
    }
  }

  // Завершение навеса «Алли-уп» в прыжке прямо в кольцо!
  if (
    allowMessiTrigger &&
    wasAlliedPass &&
    (catchZ >= 52 || (catchZ >= 40 && player.traitIds.includes('sixth_sense_alley')))
  ) {
    const atkHoopIdx = getAttackingHoop(player.team);
    const atkHoop = getHoopPosition(atkHoopIdx);
    const distAtk = Math.hypot(player.x - atkHoop.x, player.y - atkHoop.y);
    const maxAlleyDist = player.traitIds.includes('sixth_sense_alley') ? 162 : 120;
    const canFinishAlleyOop =
      distAtk < maxAlleyDist &&
      (player.traitIds.includes('gorilla_dunk') ||
        player.traitIds.includes('grasshopper') ||
        player.traitIds.includes('sixth_sense_alley') ||
        player.effectiveStats.jump >= 72);

    if (canFinishAlleyOop) {
      player.vz = Math.max(player.vz, 235);
      player.aiThought = 'ВКОЛАЧИВАЕТ АЛЛИ-УП!';
      spawnCallout(state, player.x, player.y - 22, 'АЛЛИ-УП!', '#fbbf24', 1.35);
      launchBallShot(state, ball, player, atkHoopIdx, 'dunk', rng);
      return;
    }
  }

  // Добивание в прыжке на чужом щите (Putback Dunk / Tip-In) сразу после промаха!
  if (allowMessiTrigger && !wasAlliedPass && wasMissedOwnTeamShot) {
    const atkHoopIdx = getAttackingHoop(player.team);
    const atkHoop = getHoopPosition(atkHoopIdx);
    const distAtk = Math.hypot(player.x - atkHoop.x, player.y - atkHoop.y);
    const highEnough = catchZ >= 28 || player.z >= 14;
    const canPutback =
      distAtk < 102 &&
      highEnough &&
      (player.traitIds.includes('gorilla_dunk') ||
        player.traitIds.includes('grasshopper') ||
        player.traitIds.includes('brick_magnet_board') ||
        player.traitIds.includes('sixth_sense_alley') ||
        player.effectiveStats.jump >= 66 ||
        player.effectiveStats.iq >= 68);

    if (canPutback) {
      const isSlamPutback =
        player.traitIds.includes('gorilla_dunk') ||
        player.traitIds.includes('grasshopper') ||
        player.effectiveStats.jump >= 74;
      player.vz = Math.max(player.vz, isSlamPutback ? 225 : 155);
      player.aiThought = isSlamPutback
        ? 'ДОБИВАНИЕ СВЕРХУ С ОТСКОКА!'
        : 'ДОБИВАНИЕ В КАСАНИЕ НА ЩИТЕ!';
      spawnCallout(
        state,
        player.x,
        player.y - 22,
        isSlamPutback ? 'ДОБИВАНИЕ СВЕРХУ!' : 'В КАСАНИЕ!',
        '#fbbf24',
        1.35
      );
      pushEvent(
        state,
        isSlamPutback
          ? `${player.name} взмывает на чужом щите и вколачивает добивание сверху!`
          : `${player.name} снимает подбор в воздухе и добавляет мяч в кольцо одним касанием!`,
        player.team,
        true
      );
      launchBallShot(state, ball, player, atkHoopIdx, isSlamPutback ? 'dunk' : 'normal', rng);
    }
  }
}

function recordShotChartPoint(
  state: MatchState,
  ball: BallState,
  shooter: PlayerEntity | null,
  targetHoop: 0 | 1,
  made: boolean
): void {
  if (ball.shotChartRecorded) return;
  ball.shotChartRecorded = true;
  if (!shooter || targetHoop !== getAttackingHoop(shooter.team)) return;
  if (state.shotChart.length >= 160) return;

  const isThree =
    ball.shotType !== 'dunk' &&
    isThreePointShot(ball.shotOriginX, ball.shotOriginY, targetHoop);
  state.shotChart.push({
    x: ball.shotOriginX,
    y: ball.shotOriginY,
    team: shooter.team,
    made,
    isThree,
    shooterName: shooter.name,
    quarter: state.quarter,
  });
}

function recordMissedShotOutcome(state: MatchState, ball: BallState): void {
  if (!ball.wasShotAttempt || ball.shotChartRecorded) return;
  const shooter = findAnyPlayerById(state, ball.lastShooterId);
  const atkHoop = shooter ? getAttackingHoop(shooter.team) : (ball.targetHoop ?? 1);
  recordShotChartPoint(state, ball, shooter, atkHoop, false);
  if (shooter && atkHoop === getAttackingHoop(shooter.team)) {
    shooter.hotStreak = shooter.hotStreak < 0 ? shooter.hotStreak - 1 : -1;
  }
}

export function launchBallShot(
  state: MatchState,
  ball: BallState,
  shooter: PlayerEntity,
  targetHoop: 0 | 1,
  kind: 'normal' | 'messi' | 'dunk' | 'panic',
  rng: () => number,
  shotStyle: ShotStyle = 'standard'
): void {
  const hoop = getHoopPosition(targetHoop);
  const gravity = getEffectiveGravity(state.modifiers);
  const effectiveStyle: ShotStyle = kind === 'normal' ? shotStyle : 'standard';

  shooter.pickupLockout = 0.45;
  shooter.actionCooldown = 0.45;
  shooter.windupTimer = 0;
  shooter.windupBallId = null;
  shooter.windupTargetHoop = null;
  shooter.windupShotStyle = 'standard';

  ball.holderId = null;
  ball.lastTouchTeam = shooter.team;
  ball.lastShooterId = shooter.id;
  ball.shotType = kind;
  ball.shotStyle = effectiveStyle;
  ball.targetHoop = targetHoop;
  ball.shotOriginX = shooter.x;
  ball.shotOriginY = shooter.y;
  ball.rimBounceCount = 0;
  ball.wasShotAttempt = true;
  ball.shotChartRecorded = false;

  // Учёт броска с игры (FG / 3PT) в официальном протоколе матча (только по чужому кольцу)
  if (targetHoop === getAttackingHoop(shooter.team)) {
    shooter.fgAttempted++;
    if (kind !== 'dunk' && isThreePointShot(shooter.x, shooter.y, targetHoop)) {
      shooter.threeAttempted++;
    }
  }

  if (kind === 'messi') {
    // 100% точный высокий навес Месси в ближайшее кольцо!
    const arc = computeArcVelocity(
      shooter.x,
      shooter.y,
      shooter.z + 32,
      hoop.x,
      hoop.y,
      RIM_Z,
      232,
      gravity
    );
    ball.x = shooter.x;
    ball.y = shooter.y;
    ball.z = shooter.z + 32;
    ball.vx = arc.vx;
    ball.vy = arc.vy;
    ball.vz = arc.vz;
    ball.isGuaranteedSwish = true;
    return;
  }

  if (kind === 'dunk') {
    shooter.dunks++;
    // Данк-Горилла расталкивает защитников ударной волной!
    if (shooter.traitIds.includes('gorilla_dunk')) {
      for (const opp of state.players) {
        if (opp.team !== shooter.team && opp.penaltyTimer <= 0) {
          const dx = opp.x - hoop.x;
          const dy = opp.y - hoop.y;
          const d = Math.hypot(dx, dy);
          if (d < 115 && d > 1e-2) {
            opp.vx += (dx / d) * 290;
            opp.vy += (dy / d) * 290;
          }
        }
      }
      spawnCallout(state, shooter.x, shooter.y - 24, 'ГОРИЛЛА-ДАНК!', '#f97316');
      state.cameraShake = Math.max(state.cameraShake, 0.55);
      state.hitstopTimer = 0.05;
    }
    const arc = computeArcVelocity(
      shooter.x,
      shooter.y,
      Math.max(shooter.z + 36, 85),
      hoop.x,
      hoop.y,
      RIM_Z,
      128,
      gravity
    );
    ball.x = shooter.x;
    ball.y = shooter.y;
    ball.z = Math.max(shooter.z + 36, 85);
    ball.vx = arc.vx;
    ball.vy = arc.vy;
    ball.vz = arc.vz;
    ball.isGuaranteedSwish = true;
    return;
  }

  // Обычный, флоутер, крюк или панический бросок: расчёт вероятности попадания (защитники, улетевшие на ложном замахе, не мешают!)
  const dist = Math.hypot(shooter.x - hoop.x, shooter.y - hoop.y);
  let closeDefenders = 0;
  for (const opp of state.players) {
    if (
      opp.team !== shooter.team &&
      opp.penaltyTimer <= 0 &&
      opp.trippedTimer <= 0 &&
      (opp.bittenByFakeTimer ?? 0) <= 0
    ) {
      if (Math.hypot(opp.x - shooter.x, opp.y - shooter.y) < 68) {
        closeDefenders++;
      }
    }
  }

  // Бонус «Адреналиновый псих», когда команда проигрывает, и мега-точность «Последней надежды» со скамейки!
  const losing = state.score[shooter.team] < state.score[shooter.team === 0 ? 1 : 0];
  const clutchBonus = shooter.traitIds.includes('clutch_psycho') && losing ? 0.16 : 0;
  const lastHopeActive = isLastHopeBuffActive(shooter);
  const lastHopeShotBonus = lastHopeActive ? 0.18 : 0;
  const fatiguePenalty = shooter.energy < 28 ? 0.16 : 0;
  let contestFactor = shooter.traitIds.includes('egoisto')
    ? 0.045
    : lastHopeActive
    ? 0.06
    : 0.13;
  if (effectiveStyle === 'floater') {
    contestFactor *= 0.52;
  } else if (effectiveStyle === 'hook') {
    contestFactor *= 0.48;
  }
  const styleBonus = effectiveStyle === 'floater' ? 0.08 : effectiveStyle === 'hook' ? 0.1 : 0;
  const distPenalty = Math.max(0, (dist - 120) / 720);

  const hotBonus = shooter.hotStreak >= 3 ? 0.11 : shooter.hotStreak <= -3 ? -0.04 : 0;
  const isBankShotAngle =
    effectiveStyle === 'standard' &&
    shooter.traitIds.includes('brick_magnet_board') &&
    dist >= 80 &&
    dist <= 210 &&
    Math.abs(shooter.y - hoop.y) >= 38;
  const bankBonus = isBankShotAngle ? 0.12 : 0;
  if (effectiveStyle === 'floater') {
    spawnCallout(state, shooter.x, shooter.y - 20, 'ФЛОУТЕР!', '#38bdf8', 1.0);
  } else if (effectiveStyle === 'hook') {
    spawnCallout(state, shooter.x, shooter.y - 20, 'КРЮК ИЗ ПОСТА!', '#fbbf24', 1.05);
  } else if (isBankShotAngle) {
    spawnCallout(state, shooter.x, shooter.y - 20, 'ОТ ЩИТА!', '#fde68a', 0.95);
  }

  let hitProb =
    0.26 +
    (shooter.effectiveStats.shooting / 100) * 0.54 +
    clutchBonus +
    lastHopeShotBonus +
    hotBonus +
    bankBonus +
    styleBonus -
    fatiguePenalty -
    closeDefenders * contestFactor -
    distPenalty;

  if (shooter.traitIds.includes('parking_sniper')) {
    hitProb += dist >= 245 ? 0.16 : -0.22;
  }
  if (kind === 'panic') {
    hitProb *= 0.35;
  }
  hitProb = Math.max(0.06, Math.min(lastHopeActive ? 0.96 : 0.92, hitProb));

  const willMake = rng() < hitProb;
  const scatter = willMake ? 0 : 24 + rng() * 24;
  const scatterAng = rng() * Math.PI * 2;
  const targetX = hoop.x + Math.cos(scatterAng) * scatter;
  const targetY = hoop.y + Math.sin(scatterAng) * scatter;
  const peakZ =
    effectiveStyle === 'floater'
      ? Math.max(206, Math.min(232, 176 + dist * 0.22))
      : effectiveStyle === 'hook'
      ? Math.max(162, Math.min(205, 142 + dist * 0.16))
      : Math.min(205, 132 + dist * 0.14);
  const releaseZ =
    effectiveStyle === 'floater'
      ? shooter.z + 38
      : effectiveStyle === 'hook'
      ? shooter.z + 36
      : shooter.z + 30;

  const arc = computeArcVelocity(
    shooter.x,
    shooter.y,
    releaseZ,
    targetX,
    targetY,
    RIM_Z,
    peakZ,
    gravity
  );
  ball.x = shooter.x;
  ball.y = shooter.y;
  ball.z = releaseZ;
  ball.vx = arc.vx;
  ball.vy = arc.vy;
  ball.vz = arc.vz;
  ball.isGuaranteedSwish = willMake;
}

function launchPass(
  state: MatchState,
  ball: BallState,
  passer: PlayerEntity,
  receiver: PlayerEntity,
  isAlleyOop = false,
  isSafeHighLob = false
): void {
  const gravity = getEffectiveGravity(state.modifiers);
  passer.pickupLockout = 0.35;
  passer.actionCooldown = 0.3;
  ball.holderId = null;
  ball.lastTouchTeam = passer.team;
  ball.lastPasserId = passer.id;
  ball.passReceiverId = receiver.id;
  ball.shotType = 'pass';
  ball.targetHoop = null;
  ball.isGuaranteedSwish = false;

  const releaseX = Math.max(COURT_LEFT + 6, Math.min(COURT_RIGHT - 6, passer.x));
  const releaseY = Math.max(COURT_TOP + 6, Math.min(COURT_BOTTOM - 6, passer.y));
  const dist = Math.hypot(receiver.x - releaseX, receiver.y - releaseY);
  let targetX = receiver.x;
  let targetY = receiver.y;
  let targetZ = isSafeHighLob ? 34 : 24;
  let peakZ = isSafeHighLob
    ? Math.max(108, 84 + dist * 0.12)
    : passer.traitIds.includes('pass_telepath')
    ? Math.max(44, 34 + dist * 0.045)
    : Math.max(54, 42 + dist * 0.095);

  if (isAlleyOop) {
    const atkHoop = getHoopPosition(getAttackingHoop(passer.team));
    const dirBack = passer.team === 0 ? -1 : 1;
    targetX = atkHoop.x + dirBack * 34;
    targetY = HOOP_Y + (receiver.y - HOOP_Y) * 0.35;
    targetZ = 76;
    peakZ = 136;
  } else {
    // Упреждающий пас на ход бегущему партнёру с учётом баскетбольного ИИ пасующего!
    const estArc = computeArcVelocity(
      releaseX,
      releaseY,
      passer.z + 26,
      receiver.x,
      receiver.y,
      24,
      peakZ,
      gravity
    );
    const leadAccuracy = passer.traitIds.includes('pass_telepath')
      ? 0.88
      : 0.35 + (passer.effectiveStats.iq / 100) * 0.45;
    const leadT = Math.min(0.42, estArc.flightTime * leadAccuracy);
    targetX = Math.max(
      COURT_LEFT + 24,
      Math.min(COURT_RIGHT - 24, receiver.x + receiver.vx * leadT)
    );
    targetY = Math.max(
      COURT_TOP + 24,
      Math.min(COURT_BOTTOM - 24, receiver.y + receiver.vy * leadT)
    );
  }

  const arc = computeArcVelocity(
    releaseX,
    releaseY,
    passer.z + 26,
    targetX,
    targetY,
    targetZ,
    peakZ,
    gravity
  );
  ball.x = releaseX;
  ball.y = releaseY;
  ball.z = passer.z + 26;
  ball.vx = arc.vx;
  ball.vy = arc.vy;
  ball.vz = arc.vz;
}

/**
 * Начинает официальный ввод мяча из-за боковой или лицевой линии (с правилом 5 секунд на ввод!).
 */
export function startInboundPlay(
  state: MatchState,
  receivingTeam: 0 | 1,
  spotX: number,
  spotY: number,
  reason: string,
  shotClockValue: number,
  rng: () => number,
  isBaselineAfterScore = false
): void {
  state.freeThrow = null;
  state.shotTouchedRim = false;
  state.shotClockExpiredInFlight = false;
  state.heldBallTieTimer = 0;
  state.frontcourtEstablished = false;
  state.backcourtTimer = 0;

  for (const pl of state.players) {
    pl.windupTimer = 0;
    pl.windupBallId = null;
    pl.windupTargetHoop = null;
    pl.paintTimer = 0;
    pl.closelyGuardedTimer = 0;
    pl.airborneWithBall = false;
    pl.hasStoppedDribble = false;
  }

  // На мёртвом мяче перед вводом оба тренера могут освежить состав со скамейки запасных!
  if (state.balls.length === 1) {
    performCoachSubstitutions(state, 0, false, rng);
    performCoachSubstitutions(state, 1, false, rng);
  }

  const candidates = state.players
    .filter((p) => p.team === receivingTeam && p.penaltyTimer <= 0 && p.trippedTimer <= 0)
    .sort((a, b) => {
      const scoreP = (p: PlayerEntity) =>
        (p.role === 'SF' || p.role === 'SG' ? 28 : p.role === 'PG' ? 18 : 0) +
        p.effectiveStats.iq * 0.4 +
        p.effectiveStats.teamwork * 0.4 -
        (p.traitIds.includes('messi') ? 90 : 0) -
        (p.traitIds.includes('panicker') ? 35 : 0);
      return scoreP(b) - scoreP(a);
    });
  const inbounder = candidates[0];

  const ball = state.balls[0];
  if (!inbounder || !ball) return;

  // В безумном режиме «Два мяча» не останавливаем секундомер для второго мяча — мгновенно отдаём у бровки
  if (state.balls.length > 1) {
    inbounder.x = Math.max(COURT_LEFT + 34, Math.min(COURT_RIGHT - 34, spotX));
    inbounder.y = Math.max(COURT_TOP + 34, Math.min(COURT_BOTTOM - 34, spotY));
    inbounder.z = 0;
    inbounder.vx = 0;
    inbounder.vy = 0;
    state.inbound = null;
    giveBallToPlayer(state, ball, inbounder, rng, false);
    state.shotClock = shotClockValue;
    return;
  }

  let inbX = spotX;
  let inbY = spotY;

  if (isBaselineAfterScore) {
    inbX = receivingTeam === 0 ? COURT_LEFT - 16 : COURT_RIGHT + 16;
    inbY = HOOP_Y - 58;
  } else if (spotX <= COURT_LEFT + 18) {
    inbX = COURT_LEFT - 16;
    inbY = Math.max(COURT_TOP + 46, Math.min(COURT_BOTTOM - 46, spotY));
  } else if (spotX >= COURT_RIGHT - 18) {
    inbX = COURT_RIGHT + 16;
    inbY = Math.max(COURT_TOP + 46, Math.min(COURT_BOTTOM - 46, spotY));
  } else {
    inbX = Math.max(COURT_LEFT + 70, Math.min(COURT_RIGHT - 70, spotX));
    inbY = spotY <= HOOP_Y ? COURT_TOP - 15 : COURT_BOTTOM + 14;
  }

  inbounder.x = inbX;
  inbounder.y = inbY;
  inbounder.z = 0;
  inbounder.vx = 0;
  inbounder.vy = 0;
  inbounder.heading = Math.atan2(HOOP_Y - inbY, MID_X - inbX);

  state.inbound = {
    team: receivingTeam,
    ballId: ball.id,
    passerId: inbounder.id,
    x: inbX,
    y: inbY,
    timer: 5.0,
    reason,
    isAfterScore: isBaselineAfterScore,
  };

  giveBallToPlayer(state, ball, inbounder, rng, false);
  state.shotClock = shotClockValue;
  inbounder.aiAction = 'inbound_pass';
  inbounder.aiThought = `Ввод мяча (${reason})`;
  inbounder.actionCooldown = 0.25;

  // Гарантируем открытого разыгрывающего под приём ввода мяча
  const mates = state.players.filter(
    (p) =>
      p.team === receivingTeam &&
      p.id !== inbounder.id &&
      p.penaltyTimer <= 0 &&
      p.trippedTimer <= 0
  );
  const minMateDist = mates.reduce(
    (best, m) => Math.min(best, Math.hypot(m.x - inbX, m.y - inbY)),
    9999
  );
  if ((isBaselineAfterScore || minMateDist > 235) && mates.length > 0) {
    const guard = mates.find((m) => m.role === 'PG' || m.role === 'SG') ?? mates[0];
    const dirX = inbX < COURT_LEFT ? 1 : inbX > COURT_RIGHT ? -1 : receivingTeam === 0 ? -0.35 : 0.35;
    const dirY = inbY < COURT_TOP ? 1 : inbY > COURT_BOTTOM ? -1 : 0;
    guard.x = Math.max(COURT_LEFT + 62, Math.min(COURT_RIGHT - 62, inbX + dirX * 135));
    guard.y = Math.max(
      COURT_TOP + 54,
      Math.min(
        COURT_BOTTOM - 54,
        isBaselineAfterScore ? HOOP_Y - 108 : dirY !== 0 ? inbY + dirY * 120 : HOOP_Y - 62
      )
    );
    guard.vx = 0;
    guard.vy = 0;
  }

  // После забитого мяча забившая команда выходит из чужой трёхсекундной зоны в сторону центра, а на обычном ауте соблюдает буфер 78px!
  const oppTeam: 0 | 1 = receivingTeam === 0 ? 1 : 0;
  const oppPressing = state.tactics[oppTeam] === 'blitz';
  for (const opp of state.players) {
    if (opp.team === receivingTeam || opp.penaltyTimer > 0) continue;

    if (isBaselineAfterScore) {
      const clearDepth = oppPressing && (opp.role === 'PG' || opp.role === 'SG') ? 165 : 235;
      if (receivingTeam === 0 && opp.x < COURT_LEFT + clearDepth) {
        opp.x = COURT_LEFT + clearDepth + (opp.slotIndex % 3) * 28;
        opp.vx = 0;
      } else if (receivingTeam === 1 && opp.x > COURT_RIGHT - clearDepth) {
        opp.x = COURT_RIGHT - clearDepth - (opp.slotIndex % 3) * 28;
        opp.vx = 0;
      }
    }

    const dx = opp.x - inbX;
    const dy = opp.y - inbY;
    const d = Math.hypot(dx, dy);
    const minBuf = isBaselineAfterScore ? 96 : 76;
    if (d < minBuf) {
      const pushX = d > 1e-2 ? (dx / d) * (minBuf + 4) : opp.team === 0 ? -minBuf : minBuf;
      const pushY = d > 1e-2 ? (dy / d) * (minBuf + 4) : 0;
      opp.x = Math.max(COURT_LEFT + 22, Math.min(COURT_RIGHT - 22, inbX + pushX));
      opp.y = Math.max(COURT_TOP + 22, Math.min(COURT_BOTTOM - 22, inbY + pushY));
    }
  }
}

/**
 * Свисток судьи при нарушении баскетбольных правил (аут, пробежка, двойное ведение, зона, 8с, 3с, 5с, игра ногой).
 */
export function callRuleViolation(
  state: MatchState,
  violatingTeam: 0 | 1,
  shortCall: string,
  eventText: string,
  spotX: number,
  spotY: number,
  rng: () => number,
  shotClockOverride?: number
): void {
  const receivingTeam: 0 | 1 = violatingTeam === 0 ? 1 : 0;
  sound.playWhistle();
  state.referee.whistleTimer = 1.05;
  state.referee.callsMade++;
  state.referee.lastCallText = shortCall;
  state.violationsCount[violatingTeam]++;

  const nextShotClock = shotClockOverride ?? 24;
  spawnCallout(
    state,
    Math.max(COURT_LEFT + 36, Math.min(COURT_RIGHT - 36, spotX)),
    Math.max(COURT_TOP + 32, Math.min(COURT_BOTTOM - 24, spotY - 18)),
    shortCall,
    '#f87171',
    1.4
  );
  pushEvent(state, eventText, 'ref', false);

  startInboundPlay(state, receivingTeam, spotX, spotY, shortCall, nextShotClock, rng, false);
}

function arrangeFreeThrowLineup(
  state: MatchState,
  shooter: PlayerEntity,
  targetHoop: 0 | 1
): void {
  const ftPos = getFreeThrowLinePos(targetHoop);
  shooter.x = ftPos.x;
  shooter.y = ftPos.y;
  shooter.z = 0;
  shooter.vx = 0;
  shooter.vy = 0;
  shooter.heading = targetHoop === 0 ? Math.PI : 0;
  shooter.targetX = ftPos.x;
  shooter.targetY = ftPos.y;
  shooter.aiAction = 'free_throw';

  const hoopX = targetHoop === 0 ? HOOP_LEFT_X : HOOP_RIGHT_X;
  const towardCenter = targetHoop === 0 ? 1 : -1;

  const defenders = state.players.filter(
    (p) => p.team !== shooter.team && p.penaltyTimer <= 0 && p.id !== shooter.id
  );
  const teammates = state.players.filter(
    (p) => p.team === shooter.team && p.penaltyTimer <= 0 && p.id !== shooter.id
  );

  // Классическая расстановка на «усах» вдоль трёхсекундной трапеции: защитники ближе к щиту, затем атакующие
  const defSpots = [
    { x: hoopX + towardCenter * 52, y: HOOP_Y - 94 },
    { x: hoopX + towardCenter * 52, y: HOOP_Y + 94 },
    { x: hoopX + towardCenter * 124, y: HOOP_Y - 94 },
    { x: MID_X - towardCenter * 75, y: HOOP_Y - 70 },
    { x: MID_X - towardCenter * 75, y: HOOP_Y + 70 },
  ];
  const atkSpots = [
    { x: hoopX + towardCenter * 88, y: HOOP_Y - 94 },
    { x: hoopX + towardCenter * 88, y: HOOP_Y + 94 },
    { x: MID_X - towardCenter * 35, y: HOOP_Y - 55 },
    { x: MID_X - towardCenter * 35, y: HOOP_Y + 55 },
  ];

  defenders.forEach((d, idx) => {
    const sp = defSpots[idx % defSpots.length];
    d.x = sp.x;
    d.y = sp.y;
    d.z = 0;
    d.vx = 0;
    d.vy = 0;
    d.trippedTimer = 0;
    d.targetX = sp.x;
    d.targetY = sp.y;
    d.heading = Math.atan2(HOOP_Y - d.y, hoopX - d.x);
    d.aiAction = 'rebound_position';
    d.aiThought = 'Ждёт отскока на штрафном';
  });

  teammates.forEach((m, idx) => {
    const sp = atkSpots[idx % atkSpots.length];
    m.x = sp.x;
    m.y = sp.y;
    m.z = 0;
    m.vx = 0;
    m.vy = 0;
    m.trippedTimer = 0;
    m.targetX = sp.x;
    m.targetY = sp.y;
    m.heading = Math.atan2(HOOP_Y - m.y, hoopX - m.x);
    m.aiAction = idx < 2 ? 'rebound_position' : 'safety_back';
    m.aiThought = idx < 2 ? 'На усах под штрафной' : 'Страхует центр на штрафном';
  });
}

/**
 * Назначает серию штрафных бросков (1, 2 или 3 броска с реальной баллистикой и живым подбором на последнем промахе!).
 */
export function startFreeThrowSequence(
  state: MatchState,
  shooter: PlayerEntity,
  totalShots: number,
  retainPossessionTeam: 0 | 1 | null,
  rng: () => number
): void {
  const shootingTeam = shooter.team;
  const targetHoop = getAttackingHoop(shootingTeam);
  const ball = state.balls[0];
  if (!ball || totalShots <= 0) return;

  state.inbound = null;
  state.shotTouchedRim = false;
  state.shotClockExpiredInFlight = false;
  state.heldBallTieTimer = 0;

  for (const pl of state.players) {
    pl.windupTimer = 0;
    pl.windupBallId = null;
    pl.windupTargetHoop = null;
    pl.paintTimer = 0;
    pl.closelyGuardedTimer = 0;
    pl.airborneWithBall = false;
    pl.hasStoppedDribble = false;
  }
  shooter.trippedTimer = 0;

  // В режиме «Два мяча» пробиваем мгновенно, чтобы второй мяч продолжал движение без паузы
  if (state.balls.length > 1) {
    const acc = Math.max(0.35, Math.min(0.94, 0.48 + (shooter.effectiveStats.shooting / 100) * 0.44));
    let made = 0;
    for (let i = 0; i < totalShots; i++) {
      if (rng() < acc) made++;
    }
    if (made > 0) {
      state.score[shootingTeam] += made;
      const qIdx = state.quarterScores.length - 1;
      if (qIdx >= 0) state.quarterScores[qIdx][shootingTeam] += made;
      shooter.points += made;
      shooter.freeThrowsMade += made;
    }
    const nextTeam: 0 | 1 = retainPossessionTeam ?? (shootingTeam === 0 ? 1 : 0);
    const rec = state.players.find((p) => p.team === nextTeam && p.penaltyTimer <= 0) ?? shooter;
    giveBallToPlayer(state, ball, rec, rng, false);
    return;
  }

  arrangeFreeThrowLineup(state, shooter, targetHoop);
  giveBallToPlayer(state, ball, shooter, rng, false);
  shooter.aiAction = 'free_throw';
  shooter.aiThought = `Штрафной бросок (1/${totalShots})`;
  state.possessionTeam = shootingTeam;
  state.frontcourtEstablished = true;
  state.backcourtTimer = 0;
  state.shotClock = 24;

  state.freeThrow = {
    shooterId: shooter.id,
    shootingTeam,
    targetHoop,
    ballId: ball.id,
    shotsRemaining: totalShots,
    totalShots,
    phase: 'lineup',
    phaseTimer: 0.46,
    retainPossessionTeam,
  };
}

export type PersonalFoulKind =
  | 'reach_in'
  | 'shove'
  | 'shooting_2'
  | 'shooting_3'
  | 'and_one'
  | 'flagrant_trip'
  | 'offensive';

/**
 * Единая судейская процедура персонального фола по правилам FIBA/NBA:
 * - Фол в нападении: 0 штрафных (даже при 5+ командных фолах!), переход владения вводом сбоку;
 * - Фол на броске: 2 штрафных (2-очковый), 3 штрафных (из-за дуги), либо 1 штрафной при попадании с фолом (And-One);
 * - Бонус командных фолов (5-й и далее фол команды в четверти): 2 штрафных броска;
 * - Неспортивный фол (подножка): 2 штрафных + сохранение владения пострадавшей командой + 8с скамьи штрафников!
 */
export function adjudicatePersonalFoul(
  state: MatchState,
  offender: PlayerEntity,
  victim: PlayerEntity | null,
  foulKind: PersonalFoulKind,
  rng: () => number
): void {
  sound.playWhistle();
  state.referee.whistleTimer = 1.15;
  state.referee.callsMade++;
  state.referee.gazeAngle = Math.atan2(offender.y - state.referee.y, offender.x - state.referee.x);

  const victimTeam: 0 | 1 = offender.team === 0 ? 1 : 0;
  const wasTeamControlFoul =
    foulKind === 'offensive' ||
    (state.possessionTeam === offender.team &&
      foulKind !== 'flagrant_trip' &&
      foulKind !== 'shooting_2' &&
      foulKind !== 'shooting_3' &&
      foulKind !== 'and_one');

  offender.fouls++;
  state.teamFouls[offender.team]++;

  // Неспортивный фол: скамейка штрафников на 8с (максимум 1 штрафник на команду)
  let sentToBench = false;
  if (foulKind === 'flagrant_trip') {
    const currentBenched = state.players.filter(
      (p) => p.team === offender.team && p.penaltyTimer > 0
    ).length;
    if (currentBenched < 1) {
      offender.penaltyTimer = 8.0;
      const benchPos = PENALTY_BENCH_POS[offender.team];
      offender.x = benchPos.x;
      offender.y = benchPos.y;
      offender.z = 0;
      offender.vx = 0;
      offender.vy = 0;
      sentToBench = true;
    }
  }

  // Проверка дисквалификации за 5 персональных фолов: выпускаем настоящего запасного со скамейки!
  let disqualifiedName: string | null = null;
  let subbedInName: string | null = null;
  if (offender.fouls >= 5 && !offender.isReserveSub) {
    const idx = state.players.findIndex((p) => p.id === offender.id);
    if (idx >= 0) {
      disqualifiedName = offender.name;
      offender.disqualified = true;
      const eligibleBench = state.bench
        .filter((b) => b.team === offender.team && !b.disqualified && b.fouls < 5)
        .sort((a, b) => {
          const val = (pl: PlayerEntity) =>
            getPlayerOverall(pl) +
            (pl.traitIds.includes('last_hope') && pl.startedOnBench ? 30 : 0);
          return val(b) - val(a);
        });
      if (eligibleBench.length > 0) {
        const subIn = substitutePlayer(
          state,
          offender.id,
          eligibleBench[0].id,
          '5 персональных фолов',
          rng
        );
        subbedInName = subIn?.name ?? null;
      } else {
        const reserve = createClumsyReservePlayer(offender);
        state.players[idx] = reserve;
        subbedInName = reserve.name;
        if (state.selectedPlayerId === offender.id) {
          state.selectedPlayerId = reserve.id;
        }
      }
    }
  }

  const ftShooter =
    (victim && victim.team === victimTeam && victim.penaltyTimer <= 0 ? victim : null) ??
    state.players.find((p) => p.team === victimTeam && p.penaltyTimer <= 0) ??
    null;

  if (wasTeamControlFoul) {
    const callMsg = `ФОЛ В АТАКЕ! ${offender.name} — переход владения`;
    state.referee.lastCallText = callMsg;
    spawnCallout(state, offender.x, offender.y - 22, 'ФОЛ В НАПАДЕНИИ!', '#ef4444', 1.4);
    pushEvent(state, callMsg, 'ref', false);
    startInboundPlay(
      state,
      victimTeam,
      offender.x,
      offender.y,
      'Фол в нападении',
      24,
      rng,
      false
    );
  } else {
    const inBonus = state.teamFouls[offender.team] >= 5;
    let freeThrows = 0;
    let retainTeam: 0 | 1 | null = null;

    if (foulKind === 'shooting_3') freeThrows = 3;
    else if (foulKind === 'shooting_2') freeThrows = 2;
    else if (foulKind === 'and_one') freeThrows = 1;
    else if (foulKind === 'flagrant_trip') {
      freeThrows = 2;
      retainTeam = victimTeam;
    } else if (inBonus) {
      freeThrows = 2;
    }

    if (freeThrows > 0 && ftShooter) {
      const foulLabel =
        foulKind === 'and_one'
          ? `ГОЛ С ФОЛОМ! ${ftShooter.name} бьёт +1 штрафной!`
          : foulKind === 'shooting_3'
          ? `ФОЛ НА ДУГЕ! ${ftShooter.name} пробивает 3 штрафных!`
          : foulKind === 'shooting_2'
          ? `ФОЛ ПРИ БРОСКЕ! ${ftShooter.name} пробивает 2 штрафных!`
          : foulKind === 'flagrant_trip'
          ? `НЕСПОРТИВНЫЙ ФОЛ! ${offender.name}${sentToBench ? ' (8с скамьи)' : ''} — 2 штрафных + владение!`
          : `5+ КОМАНДНЫХ ФОЛОВ! ${ftShooter.name} пробивает 2 штрафных!`;

      state.referee.lastCallText = foulLabel;
      spawnCallout(
        state,
        ftShooter.x,
        ftShooter.y - 22,
        freeThrows === 1 ? 'МЯЧ + 1 ШТРАФНОЙ!' : `${freeThrows} ШТРАФНЫХ!`,
        '#fbbf24',
        1.45
      );
      pushEvent(state, foulLabel, 'ref', true);
      startFreeThrowSequence(state, ftShooter, freeThrows, retainTeam, rng);
    } else {
      // Обычный небросковый фол защиты до бонуса: ввод из-за линии (сброс до 14с в передовой зоне или 24с в тыловой!)
      const spotX = victim ? victim.x : offender.x;
      const spotY = victim ? victim.y : offender.y;
      const resetClock = isInFrontcourt(victimTeam, spotX)
        ? Math.max(state.shotClock, 14)
        : 24;
      const msg = `ФОЛ В ЗАЩИТЕ! ${offender.name} (командные: ${state.teamFouls[offender.team]})`;
      state.referee.lastCallText = msg;
      spawnCallout(state, offender.x, offender.y - 22, 'СВИСТОК · ФОЛ!', '#ef4444', 1.3);
      pushEvent(state, msg, 'ref', false);
      startInboundPlay(state, victimTeam, spotX, spotY, 'Фол в защите', resetClock, rng, false);
    }
  }

  if (disqualifiedName) {
    pushEvent(
      state,
      `5 ПЕРСОНАЛЬНЫХ ФОЛОВ! ${disqualifiedName} удалён до конца игры${subbedInName ? ` — со скамейки выходит ${subbedInName}` : ''}!`,
      'ref',
      true
    );
  }
}

function awardPoints(
  state: MatchState,
  ball: BallState,
  scoredHoop: 0 | 1,
  rng: () => number,
  andOneFouler?: PlayerEntity | null
): void {
  // Кольцо 0 защищает команда 0 -> гол в кольцо 0 приносит очки команде 1!
  // Кольцо 1 защищает команда 1 -> гол в кольцо 1 приносит очки команде 0!
  const scoringTeam: 0 | 1 = scoredHoop === 0 ? 1 : 0;
  const threePt = isThreePointShot(ball.shotOriginX, ball.shotOriginY, scoredHoop);
  const pts = ball.shotType === 'dunk' ? 2 : threePt ? 3 : 2;

  state.score[scoringTeam] += pts;
  const qIdx = state.quarterScores.length - 1;
  if (qIdx >= 0) {
    state.quarterScores[qIdx][scoringTeam] += pts;
  }

  const shooter = findAnyPlayerById(state, ball.lastShooterId);
  const isOwnGoal = shooter !== null && shooter.team !== scoringTeam;

  if (scoredHoop === 0) state.netSwayLeft = 1.0;
  else state.netSwayRight = 1.0;

  if (ball.shotType === 'messi') {
    state.messiGoalsCount++;
  }

  ball.wasShotAttempt = false;
  if (state.activePlayCall?.team === scoringTeam) {
    state.activePlayCall = null;
  }

  // Учёт безответного рывка очков (например 8:0!)
  if (state.scoringRun.team === scoringTeam) {
    state.scoringRun.points += pts;
  } else {
    state.scoringRun = { team: scoringTeam, points: pts };
  }

  if (isOwnGoal && shooter) {
    shooter.ownGoals++;
    shooter.hotStreak = -1;
    state.ownGoalsCount++;
    sound.playOwnGoalTrombone();
    state.lastBannerText = `АВТОГОЛ! ${shooter.name} +${pts} СОПЕРНИКУ!`;
    state.lastBannerColor = '#f87171';
    spawnCallout(state, shooter.x, shooter.y - 22, `АВТОГОЛ +${pts}!`, '#f87171', 1.7);
    pushEvent(
      state,
      `КУРЬЁЗ! ${shooter.name} уложил мяч в СВОЁ кольцо (+${pts} для «${state.teamNames[scoringTeam]}»)!`,
      scoringTeam,
      true
    );
  } else {
    if (shooter) {
      shooter.points += pts;
      shooter.fgMade++;
      if (pts === 3) {
        shooter.threeMade++;
      }
      shooter.hotStreak = shooter.hotStreak > 0 ? shooter.hotStreak + 1 : 1;
      recordShotChartPoint(state, ball, shooter, scoredHoop, true);

      if (shooter.hotStreak === 3) {
        spawnCallout(state, shooter.x, shooter.y - 26, '🔥 В ОГНЕ! 3 ПОДРЯД!', '#f97316', 1.65);
        pushEvent(
          state,
          `🔥 ГОРЯЧАЯ РУКА! ${shooter.name} забивает 3-й мяч подряд и входит в режим «В ОГНЕ»!`,
          scoringTeam,
          true
        );
      }

      if (ball.lastPasserId && ball.lastPasserId !== shooter.id) {
        const passer = findAnyPlayerById(state, ball.lastPasserId);
        if (passer && passer.team === scoringTeam) {
          passer.assists++;
        }
      }
    }
    sound.playSwish(ball.shotType === 'messi');
    const runTag =
      state.scoringRun.points >= 8 ? ` · РЫВОК ${state.scoringRun.points}:0!` : '';
    const label =
      ball.shotType === 'messi'
        ? `ШЕДЕВР МЕССИ! +${pts}`
        : ball.shotType === 'dunk'
        ? `СЛЭМ-ДАНК! +${pts}`
        : threePt
        ? `ТРЁХОЧКОВЫЙ! +${pts}`
        : ball.shotStyle === 'floater'
        ? `ФЛОУТЕР! +${pts}`
        : ball.shotStyle === 'hook'
        ? `КРЮК ИЗ ПОСТА! +${pts}`
        : `ТОЧНО В ЦЕЛЬ! +${pts}`;
    state.lastBannerText = `${state.teamNames[scoringTeam]} · ${label}${runTag}`;
    state.lastBannerColor = state.teamColors[scoringTeam];
    pushEvent(
      state,
      `${shooter ? shooter.name : state.teamNames[scoringTeam]} забивает (${label}${runTag})! Счёт ${state.score[0]}:${state.score[1]}`,
      scoringTeam,
      pts === 3 || ball.shotType === 'messi' || ball.shotType === 'dunk' || state.scoringRun.points >= 8
    );
  }

  const hoop = getHoopPosition(scoredHoop);
  spawnParticles(state, hoop.x, hoop.y, RIM_Z, state.teamColors[scoringTeam], 22, 1.2, rng);

  // Модификатор «Взрывной мяч»
  if (state.modifiers.includes('explosive_ball')) {
    state.cameraShake = 0.7;
    state.hitstopTimer = 0.06;
    for (const pl of state.players) {
      if (pl.penaltyTimer > 0) continue;
      const dx = pl.x - hoop.x;
      const dy = pl.y - hoop.y;
      const d = Math.hypot(dx, dy);
      if (d < 220 && d > 1) {
        const f = ((220 - d) / 220) * 380;
        pl.vx += (dx / d) * f;
        pl.vy += (dy / d) * f;
        pl.vz = Math.max(pl.vz, 160);
      }
    }
  } else {
    state.cameraShake = Math.max(state.cameraShake, 0.28);
  }

  state.celebrationTimer = 0.65;
  state.shotClock = 24;
  state.shotTouchedRim = false;
  state.shotClockExpiredInFlight = false;

  // Если при результативном броске защитник сфолил — пробивается бонусный 1 штрафной (And-One)!
  if (!isOwnGoal && shooter && andOneFouler && andOneFouler.team !== scoringTeam) {
    adjudicatePersonalFoul(state, andOneFouler, shooter, 'and_one', rng);
    return;
  }

  // По правилам баскетбола пропустившая команда вводит мяч из-за СВОЕЙ лицевой линии за 5 секунд!
  const concedingTeam: 0 | 1 = scoringTeam === 0 ? 1 : 0;
  startInboundPlay(
    state,
    concedingTeam,
    concedingTeam === 0 ? COURT_LEFT : COURT_RIGHT,
    HOOP_Y,
    'Ввод из-за лицевой',
    24,
    rng,
    true
  );
}

function handleDirtyTrick(
  state: MatchState,
  offender: PlayerEntity,
  kind: 'trip' | 'kick_ball' | 'shove',
  rng: () => number,
  victim?: PlayerEntity,
  ball?: BallState
): void {
  offender.dirtyTrickCd = 6.0;
  offender.energy = Math.max(0, offender.energy - 13);
  offender.dirtyPlays++;
  offender.aiAction = 'dirty_play';

  // Сначала применяем физический эффект самой пакости
  if (kind === 'trip' && victim) {
    victim.trippedTimer = 2.2;
    victim.vx *= 0.3;
    victim.vy *= 0.3;
    const droppedBall = state.balls.find((b) => b.holderId === victim.id);
    if (droppedBall) {
      droppedBall.holderId = null;
      droppedBall.vx = (rng() - 0.5) * 180;
      droppedBall.vy = (rng() - 0.5) * 180;
      droppedBall.vz = 110;
      droppedBall.shotType = 'none';
    }
    spawnParticles(state, victim.x, victim.y, 10, '#f87171', 10, 0.9, rng);
  } else if (kind === 'kick_ball' && ball) {
    sound.playKick();
    const atkDir = offender.team === 0 ? 1 : -1;
    ball.holderId = null;
    ball.shotType = 'kicked';
    ball.vx = atkDir * (340 + rng() * 140);
    ball.vy = (rng() - 0.5) * 260;
    ball.vz = 190 + rng() * 90;
    ball.lastTouchTeam = offender.team;
    spawnParticles(state, ball.x, ball.y, 12, '#fbbf24', 8, 1.0, rng);
  } else if (kind === 'shove' && victim) {
    const dx = victim.x - offender.x;
    const dy = victim.y - offender.y;
    const d = Math.max(1, Math.hypot(dx, dy));
    victim.vx += (dx / d) * 260;
    victim.vy += (dy / d) * 260;
  }

  // Проверка внимания Судьи (80% спереди / 50% за спиной, -45% Незаметный, +10% за подозрительность)
  const { chance } = computeRefereeDetectionChance(state.referee, offender);
  offender.refSuspicion = Math.min(5, offender.refSuspicion + 1);

  if (rng() >= chance) {
    state.referee.missedFouls++;
    const trickLabel =
      kind === 'trip' ? 'Тихая подножка!' : kind === 'kick_ball' ? 'Пнул ногой!' : 'Толчок исподтишка!';
    spawnCallout(state, offender.x, offender.y - 20, `${trickLabel} (Судья проспал)`, '#a3e635', 1.3);
    pushEvent(
      state,
      `${offender.name} хитрит (${trickLabel.toLowerCase()}), но судья смотрел не туда!`,
      offender.team,
      false
    );
    return;
  }

  // По правилам баскетбола умышленная игра ногой — это НАРУШЕНИЕ (Violation), а не персональный фол!
  if (kind === 'kick_ball') {
    const otherTeam: 0 | 1 = offender.team === 0 ? 1 : 0;
    const resetClock =
      state.possessionTeam === otherTeam && isInFrontcourt(otherTeam, offender.x)
        ? Math.max(state.shotClock, 14)
        : 24;
    callRuleViolation(
      state,
      offender.team,
      'ИГРА НОГОЙ!',
      `СВИСТОК! Умышленная игра ногой у ${offender.name} — ввод из-за боковой (${resetClock}с).`,
      offender.x,
      offender.y,
      rng,
      resetClock
    );
    return;
  }

  // Подножка = неспортивный фол (2 штрафных + владение + 8с скамьи), толчок = персональный фол (в защите или в атаке)
  adjudicatePersonalFoul(
    state,
    offender,
    victim ?? null,
    kind === 'trip' ? 'flagrant_trip' : 'shove',
    rng
  );
}

function startNextPeriod(state: MatchState, rng: () => number = Math.random): void {
  state.quarter++;
  const isOT = state.quarter > 4;
  state.quarterTimeRemaining = isOT ? 30 : state.quarterDuration;
  state.quarterScores.push([0, 0]);
  state.shotClock = 24;
  state.teamFouls = [0, 0];
  state.inbound = null;
  state.freeThrow = null;
  state.frontcourtEstablished = false;
  state.backcourtTimer = 0;
  state.shotTouchedRim = false;
  state.shotClockExpiredInFlight = false;
  state.heldBallTieTimer = 0;
  state.activePlayCall = null;

  // По правилам FIBA во второй половине (Q3) командам выделяется по 3 тайм-аута, а на каждый овертайм — по 1!
  if (state.quarter === 3) {
    state.timeoutsRemaining = [3, 3];
  } else if (isOT) {
    state.timeoutsRemaining = [
      Math.max(1, state.timeoutsRemaining[0]),
      Math.max(1, state.timeoutsRemaining[1]),
    ];
  }

  for (const bp of state.bench) {
    if (!bp.disqualified) {
      bp.energy = Math.min(100, bp.energy + 45);
      bp.refSuspicion = 0;
    }
  }

  // Тренерские замены в перерыве между четвертями и перед овертаймом!
  performCoachSubstitutions(state, 0, true, rng);
  performCoachSubstitutions(state, 1, true, rng);

  for (const p of state.players) {
    const home = getHomeFormationCoords(p.team, p.slotIndex);
    p.x = home.x;
    p.y = home.y;
    p.z = 0;
    p.vx = 0;
    p.vy = 0;
    p.vz = 0;
    p.penaltyTimer = 0;
    p.trippedTimer = 0;
    p.refSuspicion = 0;
    p.paintTimer = 0;
    p.closelyGuardedTimer = 0;
    p.airborneWithBall = false;
    p.hasStoppedDribble = false;
    p.energy = Math.min(100, p.energy + 35);
    p.windupTimer = 0;
    p.windupBallId = null;
    p.cutTimer = 0;
    p.screenTargetId = null;
    p.aiAction = 'tipoff';
  }
  syncTeamBenchSeats(state, 0);
  syncTeamBenchSeats(state, 1);

  state.possessionTeam = null;
  state.possessionTimer = 0;

  const qLabel = getQuarterLabel(state.quarter);

  // По правилам FIBA Q2, Q3 и Q4 начинаются вводом с центральной линии по стрелке поочерёдного владения, а Q1 и Овертайм — спорным мячом!
  if (!isOT && state.quarter >= 2 && state.quarter <= 4 && !state.modifiers.includes('two_balls')) {
    const arrowTeam = state.possessionArrow;
    state.possessionArrow = arrowTeam === 0 ? 1 : 0;
    state.balls = [createTipoffBall(1, 0)];
    startInboundPlay(
      state,
      arrowTeam,
      MID_X,
      COURT_TOP - 15,
      `Начало ${qLabel} по стрелке владения`,
      24,
      rng,
      false
    );
    state.lastBannerText = `${qLabel} · ВЛАДЕНИЕ «${state.teamNames[arrowTeam]}»!`;
    state.lastBannerColor = state.teamColors[arrowTeam];
    state.celebrationTimer = 1.4;
    pushEvent(
      state,
      `Началась четверть ${qLabel}! Ввод с центра по стрелке владения за «${state.teamNames[arrowTeam]}».`,
      arrowTeam,
      true
    );
    return;
  }

  state.balls = state.modifiers.includes('two_balls')
    ? [createTipoffBall(1, -48), createTipoffBall(2, 48)]
    : [createTipoffBall(1, 0)];

  state.lastBannerText = isOT ? `${qLabel} · ОВЕРТАЙМ!` : `${qLabel} · НАЧАЛО ЧЕТВЕРТИ!`;
  state.lastBannerColor = '#fbbf24';
  state.celebrationTimer = 1.4;
  pushEvent(state, `Начался период ${qLabel}! Спорный мяч в центре.`, 'neutral', true);
}

function finalizeMatchSummary(state: MatchState): MatchSummarySnapshot {
  const winner: 0 | 1 = state.score[0] >= state.score[1] ? 0 : 1;
  const allRoster = [...state.players, ...state.bench];
  let mvp = allRoster[0];
  let bestScore = -999;
  for (const p of allRoster) {
    const impact =
      p.points * 2 +
      p.rebounds * 1.3 +
      p.assists * 1.5 +
      p.steals * 2 +
      p.blocks * 2.2 -
      p.ownGoals * 3;
    if (impact > bestScore) {
      bestScore = impact;
      mvp = p;
    }
  }

  const totalSteals = allRoster.reduce((s, p) => s + p.steals, 0);
  const totalBlocks = allRoster.reduce((s, p) => s + p.blocks, 0);
  const fouls0 = allRoster.filter((p) => p.team === 0).reduce((s, p) => s + p.fouls, 0);
  const fouls1 = allRoster.filter((p) => p.team === 1).reduce((s, p) => s + p.fouls, 0);

  return {
    teamNames: [...state.teamNames] as [string, string],
    teamColors: [...state.teamColors] as [string, string],
    score: [...state.score] as [number, number],
    quarterScores: state.quarterScores.map((q) => [...q] as [number, number]),
    winner,
    mvpName: mvp.name,
    mvpTeam: mvp.team,
    mvpPoints: mvp.points,
    mvpRebounds: mvp.rebounds,
    mvpAssists: mvp.assists,
    mvpTraits: [...mvp.traitIds],
    totalSteals,
    totalBlocks,
    totalFouls: [fouls0, fouls1],
    refCalls: state.referee.callsMade,
    refMissed: state.referee.missedFouls,
    messiGoals: state.messiGoalsCount,
    ownGoals: state.ownGoalsCount,
  };
}

function stepFreeThrowSequence(
  state: MatchState,
  dt: number,
  gravity: number,
  rng: () => number
): void {
  const ft = state.freeThrow;
  if (!ft) return;

  const shooter =
    state.players.find((p) => p.id === ft.shooterId && p.penaltyTimer <= 0) ??
    state.players.find((p) => p.team === ft.shootingTeam && p.penaltyTimer <= 0) ??
    null;
  const ball = state.balls.find((b) => b.id === ft.ballId) ?? state.balls[0] ?? null;

  if (!shooter || !ball) {
    state.freeThrow = null;
    return;
  }

  // Судья стоит напротив линии штрафного броска и внимательно следит за заступами
  state.referee.x += (shooter.x - state.referee.x) * Math.min(1, 4.5 * dt);
  state.referee.y = COURT_BOTTOM + 16;
  state.referee.gazeAngle = Math.atan2(shooter.y - state.referee.y, shooter.x - state.referee.x);

  const shotIdx = ft.totalShots - ft.shotsRemaining + 1;

  if (ft.phase === 'lineup') {
    ft.phaseTimer -= dt;
    const ftPos = getFreeThrowLinePos(ft.targetHoop);
    shooter.x = ftPos.x;
    shooter.y = ftPos.y;
    shooter.z = 0;
    shooter.aiAction = 'free_throw';
    shooter.aiThought = `Готовится к штрафному (${shotIdx}/${ft.totalShots})`;
    ball.holderId = shooter.id;
    ball.x = shooter.x + Math.cos(shooter.heading) * 13;
    ball.y = shooter.y + Math.sin(shooter.heading) * 13;
    ball.z = 14 + Math.abs(Math.sin(ft.phaseTimer * 9)) * 10;

    if (ft.phaseTimer <= 0) {
      ft.phase = 'shooting';
      ft.phaseTimer = 0.18;
    }
    return;
  }

  if (ft.phase === 'shooting') {
    ft.phaseTimer -= dt;
    shooter.aiAction = 'free_throw';
    shooter.aiThought = `Бросает штрафной (${shotIdx}/${ft.totalShots})!`;
    ball.holderId = shooter.id;
    ball.x = shooter.x + Math.cos(shooter.heading) * 11;
    ball.y = shooter.y + Math.sin(shooter.heading) * 11;
    ball.z = 34;

    if (ft.phaseTimer <= 0) {
      const hoop = getHoopPosition(ft.targetHoop);
      const lhBonus = isLastHopeBuffActive(shooter) ? 0.12 : 0;
      const acc = Math.max(
        0.35,
        Math.min(0.96, 0.48 + (shooter.effectiveStats.shooting / 100) * 0.44 + lhBonus)
      );
      const willMake = rng() < acc;
      const scatter = willMake ? 0 : 25 + rng() * 14;
      const ang = rng() * Math.PI * 2;
      const tx = hoop.x + Math.cos(ang) * scatter;
      const ty = hoop.y + Math.sin(ang) * scatter;
      const arc = computeArcVelocity(shooter.x, shooter.y, 34, tx, ty, RIM_Z, 152, gravity);

      shooter.pickupLockout = 0.55;
      ball.holderId = null;
      ball.lastTouchTeam = ft.shootingTeam;
      ball.lastShooterId = shooter.id;
      ball.shotType = 'normal';
      ball.targetHoop = ft.targetHoop;
      ball.shotOriginX = shooter.x;
      ball.shotOriginY = shooter.y;
      ball.x = shooter.x;
      ball.y = shooter.y;
      ball.z = 34;
      ball.vx = arc.vx;
      ball.vy = arc.vy;
      ball.vz = arc.vz;
      ball.isGuaranteedSwish = willMake;

      ft.phase = 'in_flight';
      ft.phaseTimer = arc.flightTime + 0.55;
    }
    return;
  }

  // Фаза полёта штрафного броска к кольцу
  const prevZ = ball.z;
  ball.vz -= gravity * dt;
  ball.x += ball.vx * dt;
  ball.y += ball.vy * dt;
  ball.z += ball.vz * dt;
  ball.spinAngle += Math.hypot(ball.vx, ball.vy) * dt * 0.04;
  ft.phaseTimer -= dt;

  const tp = ball.trail[ball.trailHead];
  ball.trailHead = (ball.trailHead + 1) % ball.trail.length;
  tp.x = ball.x;
  tp.y = ball.y;
  tp.z = ball.z;
  tp.alpha = 1;
  tp.isMessi = false;

  const crossedRimPlane = prevZ >= RIM_Z && ball.z <= RIM_Z && ball.vz < 0;
  if (!crossedRimPlane && ft.phaseTimer > 0) {
    return;
  }

  const made = ball.isGuaranteedSwish;
  ft.shotsRemaining--;
  const hx = ft.targetHoop === 0 ? HOOP_LEFT_X : HOOP_RIGHT_X;

  if (made) {
    state.score[ft.shootingTeam] += 1;
    const qIdx = state.quarterScores.length - 1;
    if (qIdx >= 0) {
      state.quarterScores[qIdx][ft.shootingTeam] += 1;
    }
    shooter.points += 1;
    shooter.freeThrowsMade += 1;
    if (ft.targetHoop === 0) state.netSwayLeft = 0.85;
    else state.netSwayRight = 0.85;
    sound.playSwish(false);
    spawnParticles(state, hx, HOOP_Y, RIM_Z, state.teamColors[ft.shootingTeam], 12, 0.9, rng);
    spawnCallout(state, hx, HOOP_Y - 22, 'ШТРАФНОЙ +1!', '#fbbf24', 1.05);
    pushEvent(
      state,
      `${shooter.name} реализует штрафной (${shotIdx}/${ft.totalShots})! Счёт ${state.score[0]}:${state.score[1]}`,
      ft.shootingTeam,
      false
    );

    if (ft.shotsRemaining > 0) {
      arrangeFreeThrowLineup(state, shooter, ft.targetHoop);
      giveBallToPlayer(state, ball, shooter, rng, false);
      ft.phase = 'lineup';
      ft.phaseTimer = 0.4;
    } else {
      const retain = ft.retainPossessionTeam;
      state.freeThrow = null;
      if (retain !== null) {
        startInboundPlay(
          state,
          retain,
          MID_X,
          COURT_TOP - 15,
          'Владение после неспортивного фола',
          14,
          rng,
          false
        );
      } else {
        const oppTeam: 0 | 1 = ft.shootingTeam === 0 ? 1 : 0;
        startInboundPlay(
          state,
          oppTeam,
          oppTeam === 0 ? COURT_LEFT : COURT_RIGHT,
          HOOP_Y,
          'Ввод из-за лицевой после штрафного',
          24,
          rng,
          true
        );
      }
    }
  } else {
    sound.playBounce(0.85);
    spawnParticles(state, hx, HOOP_Y, RIM_Z, '#e5e7eb', 6, 0.7, rng);

    if (ft.shotsRemaining > 0) {
      spawnCallout(state, hx, HOOP_Y - 22, 'ПРОМАХ СО ШТРАФНОГО!', '#fca5a5', 0.95);
      arrangeFreeThrowLineup(state, shooter, ft.targetHoop);
      giveBallToPlayer(state, ball, shooter, rng, false);
      ft.phase = 'lineup';
      ft.phaseTimer = 0.42;
    } else {
      const retain = ft.retainPossessionTeam;
      state.freeThrow = null;
      if (retain !== null) {
        startInboundPlay(
          state,
          retain,
          MID_X,
          COURT_TOP - 15,
          'Владение после неспортивного фола',
          14,
          rng,
          false
        );
      } else {
        // Последний штрафной мимо — живой отскок от дужки и борьба за подбор!
        const outAng = (ft.targetHoop === 0 ? 0 : Math.PI) + (rng() - 0.5) * 1.1;
        ball.x = hx + (ft.targetHoop === 0 ? 12 : -12);
        ball.y = HOOP_Y + (rng() - 0.5) * 20;
        ball.z = RIM_Z + 4;
        ball.vx = Math.cos(outAng) * (105 + rng() * 60);
        ball.vy = Math.sin(outAng) * (85 + rng() * 60);
        ball.vz = 112;
        ball.shotType = 'none';
        ball.targetHoop = null;
        ball.isGuaranteedSwish = false;
        state.shotTouchedRim = true;
        state.shotClock = 24;
        spawnCallout(state, hx, HOOP_Y - 22, 'МИМО · ЖИВОЙ ПОДБОР!', '#38bdf8', 1.1);
      }
    }
  }
}

export function stepMatch(
  state: MatchState,
  rawDt: number,
  rng: () => number = Math.random
): void {
  if (state.finished) return;

  const dt = Math.min(rawDt, 0.05);

  // Обновление визуальных затуханий (работают даже в hitstop)
  state.cameraShake = Math.max(0, state.cameraShake - dt * 1.8);
  state.netSwayLeft = Math.max(0, state.netSwayLeft - dt * 2.4);
  state.netSwayRight = Math.max(0, state.netSwayRight - dt * 2.4);
  state.referee.whistleTimer = Math.max(0, state.referee.whistleTimer - dt);

  for (const c of state.callouts) {
    if (!c.active) continue;
    c.age += dt;
    if (c.age >= c.duration) c.active = false;
  }

  const gravity = getEffectiveGravity(state.modifiers);
  for (const p of state.particles) {
    if (!p.active) continue;
    p.age += dt;
    if (p.age >= p.maxAge) {
      p.active = false;
      continue;
    }
    p.vz -= gravity * 0.65 * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.z = Math.max(0, p.z + p.vz * dt);
  }

  if (state.hitstopTimer > 0) {
    state.hitstopTimer = Math.max(0, state.hitstopTimer - dt);
    return;
  }

  if (state.quarterBreakTimer > 0) {
    state.quarterBreakTimer = Math.max(0, state.quarterBreakTimer - dt);
    if (state.quarterBreakTimer <= 0) {
      startNextPeriod(state, rng);
    }
    return;
  }

  if (state.celebrationTimer > 0) {
    state.celebrationTimer = Math.max(0, state.celebrationTimer - dt);
    if (state.celebrationTimer <= 0) {
      state.lastBannerText = null;
    }
  }

  // ==================== СЕРИЯ ШТРАФНЫХ БРОСКОВ (СЕКУНДОМЕР ОСТАНОВЛЕН) ====================
  if (state.freeThrow) {
    stepFreeThrowSequence(state, dt, gravity, rng);
    return;
  }

  // ==================== КОНТРОЛЬ ПРАВИЛА 5 СЕКУНД НА ВВОД ИЗ АУТА ====================
  if (state.inbound) {
    const inb = state.inbound;
    const inbBall = state.balls.find((b) => b.id === inb.ballId) ?? state.balls[0];
    if (!inbBall || inbBall.holderId !== inb.passerId) {
      // Мяч уже выпущен из рук вводящего в площадку — игра живая, часы пошли!
      state.inbound = null;
    } else {
      inb.timer -= dt;
      if (inb.timer <= 0) {
        const passer = state.players.find((p) => p.id === inb.passerId);
        state.inbound = null;
        callRuleViolation(
          state,
          inb.team,
          '5 СЕКУНД НА ВВОД!',
          `СВИСТОК! ${passer ? passer.name : state.teamNames[inb.team]} не ввёл мяч из-за линии за 5 секунд — потеря!`,
          inb.x,
          inb.y,
          rng
        );
        return;
      }
    }
  }

  // ==================== ТАЙМЕР ЧЕТВЕРТИ (ОСТАНАВЛИВАЕТСЯ НА МЁРТВОМ МЯЧЕ ПРИ ВВОДЕ) ====================
  if (!state.inbound) {
    state.quarterTimeRemaining -= dt;
    if (state.quarterTimeRemaining <= 0) {
      // Правило броска под сирену (Buzzer Beater): если мяч уже выпущен в кольцо до сирены, даём ему долететь!
      const shotInAirAtBuzzer = state.balls.some(
        (b) =>
          b.holderId === null &&
          b.targetHoop !== null &&
          (b.shotType === 'normal' ||
            b.shotType === 'panic' ||
            b.shotType === 'dunk' ||
            b.shotType === 'messi') &&
          b.z > 20
      );

      if (shotInAirAtBuzzer) {
        state.quarterTimeRemaining = 0.001;
      } else {
        state.quarterTimeRemaining = 0;
        sound.playBuzzer();

        if (state.quarter >= 4 && state.score[0] !== state.score[1]) {
          state.finished = true;
          state.summary = finalizeMatchSummary(state);
          pushEvent(
            state,
            `Финальная сирена! Победа «${state.teamNames[state.summary.winner]}» (${state.score[0]}:${state.score[1]})!`,
            state.summary.winner,
            true
          );
          return;
        } else {
          state.quarterBreakTimer = 1.1;
          state.lastBannerText =
            state.quarter >= 4
              ? `НИЧЬЯ ${state.score[0]}:${state.score[1]} · БУДЕТ ОВЕРТАЙМ!`
              : `КОНЕЦ Q${state.quarter} (${state.score[0]}:${state.score[1]})`;
          state.lastBannerColor = '#fbbf24';
          return;
        }
      }
    }
  }

  // Кулдауны замен, комбинаций с планшета и восстановление сил запасных на скамейках
  state.subCooldown[0] = Math.max(0, state.subCooldown[0] - dt);
  state.subCooldown[1] = Math.max(0, state.subCooldown[1] - dt);
  state.playCallCooldown[0] = Math.max(0, state.playCallCooldown[0] - dt);
  state.playCallCooldown[1] = Math.max(0, state.playCallCooldown[1] - dt);
  if (state.activePlayCall) {
    state.activePlayCall.timer -= dt;
    if (
      state.activePlayCall.timer <= 0 ||
      (state.possessionTeam !== null && state.possessionTeam !== state.activePlayCall.team)
    ) {
      state.activePlayCall = null;
    }
  }

  for (const bp of state.bench) {
    const seat = getTeamBenchSeatPos(bp.team, bp.benchSlot, 3);
    bp.x = seat.x;
    bp.y = seat.y;
    bp.z = 0;
    bp.vx = 0;
    bp.vy = 0;
    bp.heading = Math.PI / 2;
    bp.aiAction = 'team_bench';
    if (!bp.disqualified) {
      bp.energy = Math.min(100, bp.energy + 9.5 * dt);
    }
  }

  // ИИ-тренер второй команды: тактика, своевременные тайм-ауты против рывка 8:0 и живые комбинации!
  state.aiCoachCooldown -= dt;
  if (state.aiCoachCooldown <= 0) {
    state.aiCoachCooldown = 6.5;

    // 1) Умный тайм-аут ИИ при безответном рывке соперника >= 8 очков или выдохшейся основе
    const aiCourt = state.players.filter((p) => p.team === 1 && p.penaltyTimer <= 0);
    const avgAiEnergy =
      aiCourt.reduce((s, p) => s + p.energy, 0) / Math.max(1, aiCourt.length);
    if (
      state.timeoutsRemaining[1] > 0 &&
      ((state.scoringRun.team === 0 && state.scoringRun.points >= 8) || avgAiEnergy < 38)
    ) {
      callTeamTimeout(state, 1, rng);
    }

    // 2) Перестройка тактической схемы
    const nextAiTactic = chooseAdaptiveAiTactic(state, 1);
    if (nextAiTactic !== state.tactics[1]) {
      state.tactics[1] = nextAiTactic;
      const tacticRu: Record<TeamTactic, string> = {
        balanced: 'Баланс',
        blitz: 'Прессинг',
        perimeter: 'Трёшки',
        dirty: 'Грязная игра',
      };
      pushEvent(
        state,
        `Тренер «${state.teamNames[1]}» перестроил схему: ${tacticRu[nextAiTactic]}!`,
        1,
        false
      );
    }

    // 3) Заказ атакующей комбинации, если мяч у команды 1
    if (state.possessionTeam === 1 && state.playCallCooldown[1] <= 0 && !state.activePlayCall) {
      const hasFlyer = aiCourt.some(
        (p) =>
          p.traitIds.includes('sixth_sense_alley') ||
          p.traitIds.includes('gorilla_dunk') ||
          p.traitIds.includes('grasshopper')
      );
      const hasSniper = aiCourt.some(
        (p) => p.traitIds.includes('parking_sniper') || p.effectiveStats.shooting >= 78
      );
      const aiPlay: CoachPlayCall = hasFlyer
        ? 'alley_oop'
        : hasSniper
        ? 'feed_sniper'
        : rng() < 0.55
        ? 'pick_and_roll'
        : 'iso_star';
      triggerCoachPlayCall(state, 1, aiPlay);
    }
  }

  // ==================== СЧЁТЧИК 24 СЕКУНД, ПРАВИЛО 8 СЕКУНД И ЗОНЫ (OVER-AND-BACK) ====================
  if (!state.inbound && state.possessionTeam !== null) {
    state.possessionTimer += dt;

    // Счётчик 24 секунд тикает во время ведения, передач и полёта мяча до касания дужки кольца!
    if (!state.shotClockExpiredInFlight) {
      state.shotClock = Math.max(0, state.shotClock - dt);
      if (state.shotClock <= 0) {
        const liveShotInFlight = state.balls.some(
          (b) =>
            b.holderId === null &&
            b.targetHoop !== null &&
            (b.shotType === 'normal' ||
              b.shotType === 'panic' ||
              b.shotType === 'dunk' ||
              b.shotType === 'messi')
        );
        if (liveShotInFlight) {
          // Мяч уже в воздухе в момент сирены 24с — ждём, попадёт ли или коснётся ли кольца!
          state.shotClockExpiredInFlight = true;
        } else {
          const fb = state.balls[0];
          callRuleViolation(
            state,
            state.possessionTeam,
            '24 СЕКУНДЫ!',
            `СВИСТОК! Истекли 24 секунды на атаку у «${state.teamNames[state.possessionTeam]}» — переход владения!`,
            fb ? fb.x : MID_X,
            fb ? fb.y : HOOP_Y,
            rng
          );
          return;
        }
      }
    }

    // Правило 8 секунд вывода со своей половины и Нарушение зоны (возврат в тыловую зону) в режиме 1 мяча
    if (state.balls.length === 1) {
      const b = state.balls[0];
      const holder =
        b.holderId !== null
          ? state.players.find((p) => p.id === b.holderId && p.penaltyTimer <= 0) ?? null
          : null;

      if (holder && holder.team === state.possessionTeam) {
        const strictlyFront =
          (holder.team === 0 ? holder.x > MID_X + 26 : holder.x < MID_X - 26) &&
          state.possessionTimer >= 0.35;
        const strictlyBack = holder.team === 0 ? holder.x < MID_X - 8 : holder.x > MID_X + 8;

        if (!state.frontcourtEstablished) {
          if (strictlyFront) {
            state.frontcourtEstablished = true;
            state.backcourtTimer = 0;
          } else if (!isInFrontcourt(holder.team, holder.x)) {
            state.backcourtTimer += dt;
            if (state.backcourtTimer >= 8.0) {
              callRuleViolation(
                state,
                holder.team,
                'НАРУШЕНИЕ 8 СЕКУНД!',
                `СВИСТОК! «${state.teamNames[holder.team]}» не вывела мяч в передовую зону за 8 секунд!`,
                MID_X,
                holder.y,
                rng
              );
              return;
            }
          }
        } else if (strictlyBack) {
          callRuleViolation(
            state,
            holder.team,
            'НАРУШЕНИЕ ЗОНЫ!',
            `СВИСТОК! Нарушение зоны у ${holder.name} (вернулся с мячом на свою половину)!`,
            MID_X,
            holder.y,
            rng
          );
          return;
        }
      } else if (!holder && b.targetHoop === null && !state.frontcourtEstablished) {
        if (b.shotType === 'pass' && b.lastTouchTeam === state.possessionTeam) {
          // По правилам FIBA (ст. 30.1.1) и NBA мяч в воздухе над центральной линией ещё НЕ устанавливает
          // статус передовой зоны до касания игрока в передовой зоне, чтобы пас со спорного в районе круга не давал ложную «зону»
          if (!isInFrontcourt(state.possessionTeam, b.x)) {
            state.backcourtTimer += dt;
            if (state.backcourtTimer >= 8.0) {
              callRuleViolation(
                state,
                state.possessionTeam,
                'НАРУШЕНИЕ 8 СЕКУНД!',
                `СВИСТОК! Истекли 8 секунд на вывод мяча через центральную линию!`,
                MID_X,
                b.y,
                rng
              );
              return;
            }
          }
        }
      }
    }
  }

  // ==================== ДВИЖЕНИЕ И ВЗГЛЯД СУДЬИ ====================
  const focalBall = state.balls[0];
  if (focalBall) {
    // Если ввод мяча идёт с нижней бровки рядом с судьёй — судья отходит в сторону, чтобы не мешать вводящему
    let desiredRefX = Math.max(COURT_LEFT + 90, Math.min(COURT_RIGHT - 90, focalBall.x));
    if (state.inbound && state.inbound.y > HOOP_Y && Math.abs(desiredRefX - state.inbound.x) < 52) {
      desiredRefX = state.inbound.x < MID_X ? state.inbound.x + 65 : state.inbound.x - 65;
    }
    const desiredRefY = COURT_BOTTOM + 16;
    state.referee.vx = (desiredRefX - state.referee.x) * 3.2;
    state.referee.x += state.referee.vx * dt;
    state.referee.y = desiredRefY;

    if (state.referee.whistleTimer <= 0) {
      const targetAngle = Math.atan2(
        focalBall.y - state.referee.y,
        focalBall.x - state.referee.x
      );
      const scanOffset = Math.sin(state.quarterTimeRemaining * 1.9) * 0.65;
      state.referee.gazeAngle = targetAngle + scanOffset;
    }
  }

  // ==================== СПОРНЫЙ МЯЧ И ЧЕРТА «РУКИ-ЗАГРЕБУКИ» ====================
  for (const ball of state.balls) {
    if (ball.isTipoffBall && ball.holderId === null && ball.vz <= 65) {
      const grabbers = state.players.filter(
        (p) => p.penaltyTimer <= 0 && p.trippedTimer <= 0 && p.traitIds.includes('grabby_hands')
      );
      if (grabbers.length > 0) {
        grabbers.sort(
          (a, b) =>
            b.effectiveStats.jump +
            b.effectiveStats.steal -
            (a.effectiveStats.jump + a.effectiveStats.steal)
        );
        const winner = grabbers[0];
        winner.x = ball.x + (winner.team === 0 ? -16 : 16);
        winner.y = ball.y;
        winner.vz = 210;
        giveBallToPlayer(state, ball, winner, rng, true);
        spawnCallout(state, winner.x, winner.y - 22, 'РУКИ-ЗАГРЕБУКИ!', '#38bdf8', 1.25);
        pushEvent(
          state,
          `${winner.name} [Руки-загребуки] мгновенно забирает стартовый мяч!`,
          winner.team,
          false
        );
      } else {
        ball.isTipoffBall = false;
      }
    }
  }

  // ==================== ИИ И ФИЗИКА ИГРОКОВ + ПЕРСОНАЛЬНЫЕ ПРАВИЛА БАСКЕТБОЛА ====================
  const isIce = state.modifiers.includes('ice_court');
  const hasTrampolines = state.modifiers.includes('paint_trampolines');
  const anyTipoffActive = state.balls.some((b) => b.isTipoffBall && b.holderId === null);
  const anyShotInAir = state.balls.some((b) => b.holderId === null && b.targetHoop !== null);

  for (const p of state.players) {
    p.actionCooldown = Math.max(0, p.actionCooldown - dt);
    p.stealCooldown = Math.max(0, p.stealCooldown - dt);
    p.pickupLockout = Math.max(0, p.pickupLockout - dt);
    p.superJumpCd = Math.max(0, p.superJumpCd - dt);
    p.dirtyTrickCd = Math.max(0, p.dirtyTrickCd - dt);
    p.dodgeCommitTimer = Math.max(0, p.dodgeCommitTimer - dt);
    p.driveLockTimer = Math.max(0, (p.driveLockTimer ?? 0) - dt);
    p.flopCd = Math.max(0, (p.flopCd ?? 0) - dt);
    p.saveOobCd = Math.max(0, (p.saveOobCd ?? 0) - dt);

    if (p.penaltyTimer > 0) {
      if (!state.inbound) {
        p.penaltyTimer = Math.max(0, p.penaltyTimer - dt);
      }
      const bench = PENALTY_BENCH_POS[p.team];
      p.x = bench.x;
      p.y = bench.y;
      p.z = 0;
      if (p.penaltyTimer <= 0) {
        p.y = COURT_TOP + 26;
        p.aiThought = 'Вернулся со скамьи!';
        spawnCallout(state, p.x, p.y, 'В ИГРЕ!', '#38bdf8', 0.9);
      }
      continue;
    }

    if (p.trippedTimer > 0) {
      p.trippedTimer = Math.max(0, p.trippedTimer - dt);
      const damp = isIce ? 0.985 : 0.9;
      p.vx *= Math.pow(damp, dt * 60);
      p.vy *= Math.pow(damp, dt * 60);
      p.x = Math.max(COURT_LEFT + 16, Math.min(COURT_RIGHT - 16, p.x + p.vx * dt));
      p.y = Math.max(COURT_TOP + 16, Math.min(COURT_BOTTOM - 16, p.y + p.vy * dt));
      continue;
    }

    // Завершение замаха броска + проверка фола на бросающем (Shooting Foul / 2, 3 или And-One!)
    if (p.windupTimer > 0) {
      p.windupTimer -= dt;
      if (p.windupTimer <= 0) {
        const wBall = state.balls.find((b) => b.id === p.windupBallId && b.holderId === p.id);
        if (wBall && p.windupTargetHoop !== null) {
          const targetHoop = p.windupTargetHoop;
          const shotKind = p.aiAction === 'messi_windup' ? 'messi' : 'normal';
          const releaseStyle = p.windupShotStyle ?? 'standard';
          launchBallShot(state, wBall, p, targetHoop, shotKind, rng, releaseStyle);

          // Проверка фола защитника на бросающем игроке при плотном контакте!
          if (shotKind === 'normal' && state.balls.length === 1 && !state.inbound) {
            const tightContestOpp = state.players.find(
              (opp) =>
                opp.team !== p.team &&
                opp.penaltyTimer <= 0 &&
                opp.trippedTimer <= 0 &&
                Math.hypot(opp.x - p.x, opp.y - p.y) < 33
            );
            if (tightContestOpp) {
              const foulChance =
                0.08 +
                (tightContestOpp.effectiveStats.deviance / 100) * 0.11 -
                (tightContestOpp.effectiveStats.iq / 100) * 0.04;
              if (
                rng() < foulChance &&
                rng() < computeRefereeDetectionChance(state.referee, tightContestOpp).chance
              ) {
                if (wBall.isGuaranteedSwish) {
                  // Мяч забит с фолом (And-One: очки сразу засчитываются + 1 штрафной!)
                  awardPoints(state, wBall, targetHoop, rng, tightContestOpp);
                  return;
                } else {
                  const isThree = isThreePointShot(wBall.shotOriginX, wBall.shotOriginY, targetHoop);
                  adjudicatePersonalFoul(
                    state,
                    tightContestOpp,
                    p,
                    isThree ? 'shooting_3' : 'shooting_2',
                    rng
                  );
                  return;
                }
              }
            }
          }
        } else {
          p.windupBallId = null;
          p.windupTargetHoop = null;
        }
      }
    }

    let foulInterruptedStep = false;

    // Шаг принятия решений ИИ
    stepPlayerAI(state, p, dt, rng, {
      onShoot: (shooter, ball, targetHoop, isDunk, isPanic, shotStyle = 'standard') => {
        if (isDunk) {
          shooter.vz = Math.max(shooter.vz, 275);
          launchBallShot(state, ball, shooter, targetHoop, 'dunk', rng, 'standard');
        } else if (isPanic) {
          launchBallShot(state, ball, shooter, targetHoop, 'panic', rng, 'standard');
        } else {
          shooter.aiAction =
            shotStyle === 'floater'
              ? 'floater'
              : shotStyle === 'hook'
              ? 'post_up'
              : 'shoot_windup';
          shooter.windupShotStyle = shotStyle;
          shooter.windupTimer =
            shotStyle === 'floater'
              ? 0.12
              : shooter.traitIds.includes('showboat')
              ? 0.42
              : 0.16;
          shooter.windupTargetHoop = targetHoop;
          shooter.windupBallId = ball.id;
          if (shooter.z === 0) {
            shooter.vz = shotStyle === 'floater' ? 135 : 155;
          }
        }
      },
      onPass: (passer, ball, receiver, isAlleyOop, isSafeHighLob) => {
        const wasInboundPass = state.inbound?.passerId === passer.id;
        passer.aiAction = 'pass';
        launchPass(state, ball, passer, receiver, Boolean(isAlleyOop), Boolean(isSafeHighLob));
        if (wasInboundPass) {
          state.inbound = null;
        } else {
          // Умная связка «Дай и беги» (Give-and-Go): отдав пас на периметре, командный игрок тут же врывается под кольцо за ответкой!
          const atkHoop = getHoopPosition(getAttackingHoop(passer.team));
          const dHoop = Math.hypot(passer.x - atkHoop.x, passer.y - atkHoop.y);
          if (
            dHoop > 170 &&
            dHoop < 460 &&
            passer.effectiveStats.iq >= 50 &&
            passer.effectiveStats.teamwork >= 46 &&
            !passer.traitIds.includes('parking_sniper') &&
            !passer.traitIds.includes('sleepy_fly') &&
            rng() < 0.58
          ) {
            passer.aiAction = 'give_and_go';
            passer.cutTimer = 1.45;
            passer.aiThought = 'Отдал и врывается под ответку (Дай и беги)!';
          }
        }
      },
      onPumpFake: (shooter, defender) => {
        shooter.pumpFakeCd = 5.2;
        shooter.pumpFakeTimer = 0.22;
        shooter.decisionTimer = 0.24;
        if ((shooter.dribbleLiveTimer ?? 0) > 0.38) {
          shooter.hasStoppedDribble = true;
        }
        shooter.aiAction = 'pump_fake';
        shooter.aiThought = 'Ложный замах броска!';
        spawnCallout(state, shooter.x, shooter.y - 20, 'ЛОЖНЫЙ ЗАМАХ!', '#fde047', 0.95);

        const biteProb = Math.max(
          0.1,
          Math.min(
            0.94,
            0.66 -
              (defender.effectiveStats.iq / 100) * 0.44 +
              (shooter.effectiveStats.iq / 100) * 0.22 +
              (defender.traitIds.includes('panicker') ? 0.18 : 0) +
              (defender.aiAction === 'closeout' ? 0.16 : 0) +
              (shooter.traitIds.includes('showboat') ? 0.12 : 0)
          )
        );
        if (rng() < biteProb && defender.z === 0) {
          defender.vz = computeJumpVelocity(defender, false) * 0.92;
          defender.bittenByFakeTimer = 0.95;
          const lungeAng =
            Math.atan2(shooter.y - defender.y, shooter.x - defender.x) +
            (defender.slotIndex % 2 === 0 ? 0.36 : -0.36);
          defender.vx += Math.cos(lungeAng) * 145;
          defender.vy += Math.sin(lungeAng) * 145;
          defender.aiThought = 'Ай! Купился на ложный замах!';
          spawnCallout(state, defender.x, defender.y - 18, 'КУПИЛСЯ НА ФЕЙК!', '#38bdf8', 0.95);
          sound.playBounce(0.6);
        } else {
          defender.aiThought = 'Не купился на фейк — стоит на ногах!';
        }
      },
      onGhostPumpFake: (shooter) => {
        shooter.pumpFakeCd = 5.5;
        shooter.pumpFakeTimer = 0.28;
        shooter.decisionTimer = 0.28;
        shooter.aiAction = 'pump_fake';
        shooter.aiThought = 'Качает воздух ложным замахом — вокруг ни души!';
        spawnCallout(state, shooter.x, shooter.y - 20, 'КАЧАЕТ ВОЗДУХ!', '#fde047', 1.1);
        pushEvent(
          state,
          `${shooter.name} грозно качнул воздух ложным замахом — хотя в трёх метрах вокруг никого не было!`,
          shooter.team,
          false
        );
      },
      onHandoffMove: (giver, receiver, ball) => {
        giver.handoffCd = 5.0;
        giver.handoffHoldTimer = 1.15;
        giver.pickupLockout = 0.65;
        giver.actionCooldown = 0.45;
        giver.vx *= 0.2;
        giver.vy *= 0.2;
        giver.aiAction = 'screen_set';
        giver.aiThought = `Отдал из рук в руки ${receiver.name.split(' ')[0]} и встал в заслон!`;

        giveBallToPlayer(state, ball, receiver, rng, false);
        ball.lastPasserId = giver.id;
        receiver.actionCooldown = 0.12;
        receiver.decisionTimer = 0.1;
        receiver.aiAction = 'drive';
        receiver.aiThought = 'Забрал мяч из рук в руки под заслон!';
        spawnCallout(state, giver.x, giver.y - 20, 'ИЗ РУК В РУКИ!', '#38bdf8', 1.0);
        sound.playBounce(0.6);
      },
      onIntentionalFoul: (defender, carrier) => {
        defender.aiAction = 'intentional_foul';
        defender.aiThought = 'Умышленный тактический фол — останавливает время!';
        spawnCallout(state, defender.x, defender.y - 22, 'ТАКТИЧЕСКИЙ ФОЛ!', '#f97316', 1.35);
        pushEvent(
          state,
          `ТАКТИКА КОНЦОВКИ! ${defender.name} умышленно фолит на ${carrier.name}, чтобы остановить секундомер!`,
          defender.team,
          true
        );
        adjudicatePersonalFoul(state, defender, carrier, 'reach_in', rng);
        foulInterruptedStep = true;
      },
      onStepBackMove: (carrier, defender, ball, targetHoop) => {
        carrier.stepBackCd = 6.2;
        carrier.aiAction = 'step_back';
        const atkHoop = getHoopPosition(targetHoop);
        const dxAway = carrier.x - atkHoop.x;
        const dyAway = carrier.y - atkHoop.y;
        const curDist = Math.max(1, Math.hypot(dxAway, dyAway));
        const targetDist = Math.max(curDist + 42, THREE_PT_RADIUS + 20);
        const ux = dxAway / curDist;
        const uy = dyAway / curDist;
        const side = getCommittedBypassSide(carrier);
        let stepX = atkHoop.x + ux * targetDist - uy * side * 14;
        let stepY = atkHoop.y + uy * targetDist + ux * side * 14;
        stepX = Math.max(COURT_LEFT + 22, Math.min(COURT_RIGHT - 22, stepX));
        if (state.frontcourtEstablished) {
          stepX = carrier.team === 0 ? Math.max(MID_X + 24, stepX) : Math.min(MID_X - 24, stepX);
        }
        stepY = Math.max(COURT_TOP + 22, Math.min(COURT_BOTTOM - 22, stepY));

        carrier.x = stepX;
        carrier.y = stepY;
        carrier.targetX = stepX;
        carrier.targetY = stepY;
        carrier.vx = ux * 95;
        carrier.vy = uy * 95;
        carrier.vz = Math.max(carrier.vz, 165);
        carrier.windupTimer = 0.14;
        carrier.windupTargetHoop = targetHoop;
        carrier.windupBallId = ball.id;
        carrier.hasStoppedDribble = true;
        carrier.aiThought = 'Степ-бэк за дугу и трёхочковый!';
        defender.stealCooldown = Math.max(defender.stealCooldown, 0.55);
        spawnCallout(state, carrier.x, carrier.y - 20, 'СТЕП-БЭК!', '#fbbf24', 1.05);
        sound.playBounce(0.7);
      },
      onEurostepMove: (carrier, defender, _ball, targetHoop) => {
        carrier.eurostepCd = 5.2;
        carrier.aiAction = 'eurostep';
        carrier.actionCooldown = 0.12;
        const atkHoop = getHoopPosition(targetHoop);
        const toHoopAng = Math.atan2(atkHoop.y - carrier.y, atkHoop.x - carrier.x);
        const defRelAng = wrapAngle(
          Math.atan2(defender.y - carrier.y, defender.x - carrier.x) - toHoopAng
        );
        const jukeSide =
          Math.abs(defRelAng) > 0.08
            ? defRelAng > 0
              ? -1
              : 1
            : getCommittedBypassSide(carrier);
        const slashAng = toHoopAng + jukeSide * 0.68;
        const nextX = Math.max(
          COURT_LEFT + 20,
          Math.min(COURT_RIGHT - 20, carrier.x + Math.cos(slashAng) * 34)
        );
        const nextY = Math.max(
          COURT_TOP + 20,
          Math.min(COURT_BOTTOM - 20, carrier.y + Math.sin(slashAng) * 34)
        );
        carrier.x = nextX;
        carrier.y = nextY;
        carrier.vx = Math.cos(toHoopAng + jukeSide * 0.25) * 165;
        carrier.vy = Math.sin(toHoopAng + jukeSide * 0.25) * 165;
        carrier.targetX = atkHoop.x;
        carrier.targetY = Math.max(
          COURT_TOP + 28,
          Math.min(COURT_BOTTOM - 28, atkHoop.y + jukeSide * 26)
        );
        carrier.decisionTimer = 0.08;
        defender.stealCooldown = Math.max(defender.stealCooldown, 0.6);
        carrier.aiThought = 'Обходит корпус защитника евростепом!';
        spawnCallout(state, carrier.x, carrier.y - 20, 'ЕВРОСТЕП!', '#38bdf8', 0.95);
        sound.playBounce(0.65);
      },
      onCrossoverMove: (carrier, defender, ball) => {
        carrier.crossoverCd = 4.6;

        // Комедия низкого дриблинга: неуклюжий балбес пытается выдать финт и стукает мячом себе в кроссовок!
        if (
          (carrier.effectiveStats.dribble <= 28 || carrier.traitIds.includes('butterfingers')) &&
          carrier.effectiveStats.iq <= 40
        ) {
          carrier.pickupLockout = 0.75;
          carrier.actionCooldown = 0.5;
          ball.holderId = null;
          ball.shotType = 'none';
          ball.targetHoop = null;
          ball.z = 14;
          const bAng = carrier.heading + (rng() - 0.5) * 1.4;
          ball.vx = Math.cos(bAng) * 145;
          ball.vy = Math.sin(bAng) * 145;
          ball.vz = 65;
          carrier.aiThought = 'Ой! Стукнул мячом себе в кроссовок!';
          spawnCallout(state, carrier.x, carrier.y - 20, 'В СВОЮ НОГУ!', '#fca5a5', 1.1);
          pushEvent(
            state,
            `${carrier.name} попытался финтить, но стукнул мячом себе в кроссовок!`,
            carrier.team,
            false
          );
          return;
        }

        carrier.aiAction = 'ankle_breaker';
        const side = getCommittedBypassSide(carrier);
        const burstAng = carrier.heading + side * 0.72;
        carrier.vx += Math.cos(burstAng) * 135;
        carrier.vy += Math.sin(burstAng) * 135;

        const ankleScore =
          carrier.effectiveStats.dribble * 0.62 +
          carrier.effectiveStats.speed * 0.28 -
          defender.effectiveStats.iq * 0.42 -
          defender.effectiveStats.steal * 0.25 +
          (isIce ? 22 : 0) +
          (carrier.traitIds.includes('showboat') ? 10 : 0);
        const ankleProb = Math.max(0.06, Math.min(0.55, 0.14 + ankleScore / 140));

        if (rng() < ankleProb) {
          defender.trippedTimer = 0.85;
          defender.vx -= Math.cos(burstAng) * 115;
          defender.vy -= Math.sin(burstAng) * 115;
          defender.aiAction = 'tripped_down';
          defender.aiThought = 'Упал от кроссовера — лодыжки хрустнули!';
          carrier.aiThought = 'Сломал лодыжки защитнику кроссовером!';
          spawnCallout(state, carrier.x, carrier.y - 22, 'ЛОДЫЖКИ ХРУСТНУЛИ!', '#f97316', 1.25);
          spawnParticles(state, defender.x, defender.y, 6, '#fbbf24', 12, 0.9, rng);
          pushEvent(
            state,
            `${carrier.name} резким кроссовером уложил ${defender.name} на паркет без фола!`,
            carrier.team,
            true
          );
        } else {
          carrier.aiThought = 'Резкий кроссовер на скорости!';
          spawnCallout(state, carrier.x, carrier.y - 18, 'КРОССОВЕР!', '#fde047', 0.8);
        }
      },
      onJump: (jumper, useSuperJump, reason) => {
        if (jumper.z > 0) return;
        const actualSuper =
          useSuperJump &&
          jumper.traitIds.includes('grasshopper') &&
          jumper.superJumpCd <= 0 &&
          jumper.energy >= 15;
        jumper.vz = computeJumpVelocity(jumper, actualSuper);
        if (actualSuper) {
          jumper.superJumpCd = 3.5;
          jumper.energy = Math.max(0, jumper.energy - 16);
          sound.playSpringJump();
          spawnCallout(state, jumper.x, jumper.y - 18, 'КУЗНЕЧИК!', '#4ade80', 1.1);
          spawnParticles(state, jumper.x, jumper.y, 2, '#4ade80', 10, 0.85, rng);
        } else {
          jumper.energy = Math.max(0, jumper.energy - 6);
        }
        jumper.aiThought = reason;
      },
      onAttemptSteal: (defender, carrier, ball) => {
        defender.stealCooldown = 1.05;
        const stealScore =
          defender.effectiveStats.steal * 0.58 - carrier.effectiveStats.dribble * 0.48;
        let prob = Math.max(0.05, Math.min(0.48, 0.12 + stealScore / 160));
        // После подбора в защите или приёма ввода на своей половине игрок укрывает мяч корпусом и локтями!
        if (
          isOnOwnHalf(carrier) &&
          state.possessionTimer < 1.4 &&
          !carrier.traitIds.includes('butterfingers')
        ) {
          prob *= 0.38;
        }
        if (rng() < prob) {
          defender.steals++;
          carrier.pickupLockout = 0.65;
          giveBallToPlayer(state, ball, defender, rng, true);
          spawnCallout(state, defender.x, defender.y - 18, 'ПЕРЕХВАТ!', '#38bdf8', 1.0);
          pushEvent(state, `${defender.name} чисто выкрал мяч у ${carrier.name}!`, defender.team, false);
        } else if (!state.inbound && state.balls.length === 1) {
          // Фол при неудачной попытке выбить мяч по рукам (Reach-in Foul)!
          const reachFoulProb =
            0.07 +
            (defender.effectiveStats.deviance / 100) * 0.11 -
            (defender.effectiveStats.iq / 100) * 0.04;
          if (
            rng() < reachFoulProb &&
            rng() < computeRefereeDetectionChance(state.referee, defender).chance
          ) {
            adjudicatePersonalFoul(state, defender, carrier, 'reach_in', rng);
            foulInterruptedStep = true;
          }
        }
      },
      onAttemptDirtyTrick: (offender, kind, victim, ball) => {
        handleDirtyTrick(state, offender, kind, rng, victim, ball);
        if (state.freeThrow || state.inbound) {
          foulInterruptedStep = true;
        }
      },
    });

    if (foulInterruptedStep) {
      return;
    }

    // Если этот игрок сейчас вводит мяч из-за боковой/лицевой линии — он стоит за линией до паса!
    if (state.inbound && state.inbound.passerId === p.id) {
      p.x = state.inbound.x;
      p.y = state.inbound.y;
      p.vx = 0;
      p.vy = 0;
      p.z = 0;
      continue;
    }

    // Запоминаем позицию до шага перемещения и столкновений (для точной скорости смещения на паркете!)
    p.prevStepX = p.x;
    p.prevStepY = p.y;

    // Перемещение по площадке к (targetX, targetY) с фиксированной стороной обхода препятствий (без кручения!)
    const dx = p.targetX - p.x;
    const dy = p.targetY - p.y;
    const dist = Math.hypot(dx, dy);

    const losing = state.score[p.team] < state.score[p.team === 0 ? 1 : 0];
    const clutchSpeed = p.traitIds.includes('clutch_psycho') && losing ? 1.22 : 1.0;
    const schizoBoost =
      p.traitIds.includes('schizo_tactician') && !p.schizoBreadMode ? 1.3 : 1.0;
    const fatigueFactor = p.energy < 22 ? 0.68 : p.energy < 45 ? 0.86 : 1.0;
    const carryingBall = state.balls.some((b) => b.holderId === p.id);
    const carryFactor = carryingBall ? 0.88 + (p.effectiveStats.dribble / 100) * 0.1 : 1.0;
    const chasedownBoost = p.aiAction === 'chasedown_block' ? 1.14 : 1.0;

    const maxSpeed =
      (108 + (p.effectiveStats.speed / 100) * 126) *
      clutchSpeed *
      schizoBoost *
      fatigueFactor *
      carryFactor *
      chasedownBoost;

    let desiredVx = 0;
    let desiredVy = 0;
    let desiredAng = p.heading;
    if (dist > 8) {
      desiredAng = Math.atan2(dy, dx);

      // Если прямо по курсу бежит или стоит другой игрок (симметричная помеха) — выбираем и держим одну сторону обхода!
      if (dist > 20 && !anyTipoffActive && p.aiAction !== 'screen_set' && p.aiAction !== 'offball_screen') {
        let blockerDist = 999;
        for (const other of state.players) {
          if (other.id === p.id || other.penaltyTimer > 0 || other.trippedTimer > 0) continue;
          const ox = other.x + other.vx * 0.1;
          const oy = other.y + other.vy * 0.1;
          const dOther = Math.hypot(ox - p.x, oy - p.y);
          if (dOther > 1 && dOther < 54 && dOther < dist - 8 && dOther < blockerDist) {
            const angToOther = Math.atan2(oy - p.y, ox - p.x);
            if (Math.abs(wrapAngle(angToOther - desiredAng)) < 0.72) {
              blockerDist = dOther;
            }
          }
        }
        if (blockerDist < 54) {
          const side = getCommittedBypassSide(p); // -1 = обход выше по Y, +1 = обход ниже по Y
          const sinPlus = Math.sin(desiredAng + 0.65);
          const sinMinus = Math.sin(desiredAng - 0.65);
          const rotSign =
            Math.abs(sinPlus - sinMinus) > 0.15
              ? (sinPlus - sinMinus) * side >= 0
                ? 1
                : -1
              : side;
          const deflect = rotSign * (0.44 + (1 - blockerDist / 54) * 0.46);
          desiredAng = wrapAngle(desiredAng + deflect);
        }
      }

      const arrivalScale = dist < 24 ? dist / 24 : 1;
      desiredVx = Math.cos(desiredAng) * maxSpeed * arrivalScale;
      desiredVy = Math.sin(desiredAng) * maxSpeed * arrivalScale;
    }

    const accel = isIce ? 2.3 : 11.5;
    p.vx += (desiredVx - p.vx) * Math.min(1, accel * dt);
    p.vy += (desiredVy - p.vy) * Math.min(1, accel * dt);

    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Плавный поворот корпуса с ограничением угловой скорости (исключает бешеное вращение на месте!)
    const speedAfter = Math.hypot(p.vx, p.vy);
    let targetHeading = p.heading;
    if (speedAfter > 22) {
      targetHeading = Math.atan2(p.vy, p.vx);
    } else if (dist > 12) {
      targetHeading = desiredAng;
    } else if (
      p.aiAction === 'guard_carrier' ||
      p.aiAction === 'take_charge' ||
      p.aiAction === 'closeout'
    ) {
      const heldBall = state.balls.find((b) => b.holderId !== null);
      const carrierPl = heldBall ? state.players.find((pl) => pl.id === heldBall.holderId) : null;
      if (carrierPl && Math.hypot(carrierPl.x - p.x, carrierPl.y - p.y) > 8) {
        targetHeading = Math.atan2(carrierPl.y - p.y, carrierPl.x - p.x);
      }
    }
    p.heading = rotateAngleToward(p.heading, targetHeading, 10.5 * dt);

    // Правило центрального круга на вбрасывании: не-центровые не могут входить внутрь круга до касания мяча!
    if (anyTipoffActive && p.role !== 'C' && !p.traitIds.includes('grabby_hands')) {
      const cdx = p.x - MID_X;
      const cdy = p.y - HOOP_Y;
      const cd = Math.hypot(cdx, cdy);
      const minCircleR = CENTER_CIRCLE_RADIUS + 5;
      if (cd < minCircleR && cd > 1e-2) {
        p.x = MID_X + (cdx / cd) * minCircleR;
        p.y = HOOP_Y + (cdy / cd) * minCircleR;
      }
    }

    // Проверка заступа игрока с мячом в АУТ (на гололёде или под прессингом у боковой линии)!
    if (
      carryingBall &&
      !state.inbound &&
      state.balls.length === 1 &&
      isPointOutOfBounds(p.x, p.y, 4)
    ) {
      callRuleViolation(
        state,
        p.team,
        'ЗАСТУП В АУТ!',
        `СВИСТОК! ${p.name} заступил с мячом на ограничительную линию площадки (Аут)!`,
        p.x,
        p.y,
        rng
      );
      return;
    }

    // Границы паркета для полевых игроков
    const boundMargin = carryingBall && isIce ? 2 : 14;
    if (p.x < COURT_LEFT + boundMargin) {
      p.x = COURT_LEFT + boundMargin;
      if (isIce) p.vx = Math.abs(p.vx) * 0.65;
    } else if (p.x > COURT_RIGHT - boundMargin) {
      p.x = COURT_RIGHT - boundMargin;
      if (isIce) p.vx = -Math.abs(p.vx) * 0.65;
    }
    if (p.y < COURT_TOP + boundMargin) {
      p.y = COURT_TOP + boundMargin;
      if (isIce) p.vy = Math.abs(p.vy) * 0.65;
    } else if (p.y > COURT_BOTTOM - boundMargin) {
      p.y = COURT_BOTTOM - boundMargin;
      if (isIce) p.vy = -Math.abs(p.vy) * 0.65;
    }

    // Батуты в краске!
    if (hasTrampolines && p.z === 0) {
      for (const pad of TRAMPOLINE_PADS) {
        if (Math.hypot(p.x - pad.x, p.y - pad.y) < pad.radius) {
          p.vz = 420;
          sound.playSpringJump();
          break;
        }
      }
    }

    // Вертикальная физика прыжка + правило ПРОБЕЖКИ (прыгнул и приземлился с мячом в руках!)
    if (p.z > 0 || p.vz > 0) {
      p.vz -= gravity * dt;
      p.z += p.vz * dt;

      const stillCarrying = state.balls.some((b) => b.holderId === p.id);
      if (stillCarrying && !p.caughtInAir && p.z > 16 && p.windupTimer <= 0) {
        p.airborneWithBall = true;
      }

      if (p.z <= 0) {
        p.z = 0;
        p.vz = 0;
        p.caughtInAir = false;

        // Если игрок бросал в прыжке (Понторез) — выпускает мяч перед самым касанием паркета
        if (stillCarrying && p.windupTimer > 0) {
          const wBall = state.balls.find((b) => b.id === p.windupBallId && b.holderId === p.id);
          if (wBall && p.windupTargetHoop !== null) {
            p.windupTimer = 0;
            launchBallShot(
              state,
              wBall,
              p,
              p.windupTargetHoop,
              p.aiAction === 'messi_windup' ? 'messi' : 'normal',
              rng,
              p.windupShotStyle ?? 'standard'
            );
          }
        } else if (
          stillCarrying &&
          p.airborneWithBall &&
          !state.inbound &&
          state.balls.length === 1 &&
          !hasTrampolines
        ) {
          p.airborneWithBall = false;
          callRuleViolation(
            state,
            p.team,
            'ПРОБЕЖКА!',
            `СВИСТОК! Пробежка у ${p.name} — выпрыгнул с мячом и приземлился, не выпустив его из рук!`,
            p.x,
            p.y,
            rng
          );
          return;
        }
      }
    }

    const nowCarrying = state.balls.some((b) => b.holderId === p.id);
    if (!nowCarrying) {
      p.airborneWithBall = false;
      p.closelyGuardedTimer = 0;
    }

    // ==================== ПРАВИЛО 3 СЕКУНД В ЧУЖОЙ КРАСКЕ ====================
    if (
      !state.inbound &&
      state.balls.length === 1 &&
      state.possessionTeam === p.team &&
      state.frontcourtEstablished &&
      !anyShotInAir &&
      !state.shotTouchedRim &&
      p.windupTimer <= 0 &&
      isInAttackingPaint(p)
    ) {
      p.paintTimer += dt;
      if (p.paintTimer >= 3.0) {
        p.paintTimer = 0;
        callRuleViolation(
          state,
          p.team,
          '3 СЕКУНДЫ В КРАСКЕ!',
          `СВИСТОК! Нарушение правила 3 секунд в трёхсекундной зоне у ${p.name}!`,
          p.x,
          p.y,
          rng
        );
        return;
      }
    } else {
      p.paintTimer = 0;
    }

    // Расход и восстановление выносливости («Последняя надежда» даёт второе дыхание -45% расхода сил!)
    const moveMag = Math.hypot(p.vx, p.vy);
    p.stridePhase += moveMag * dt * 0.085;
    if (moveMag > 45) {
      const lastHopeDrainFactor = isLastHopeBuffActive(p) ? 0.55 : 1.0;
      const burnRate =
        (p.traitIds.includes('turbo_hamster') ? 6.8 : 3.6) *
        (1.32 - (p.effectiveStats.stamina / 100) * 0.65) *
        lastHopeDrainFactor;
      p.energy = Math.max(5, p.energy - burnRate * dt);
    } else {
      const recRate = 5.5 + (p.effectiveStats.stamina / 100) * 6.5;
      p.energy = Math.min(100, p.energy + recRate * dt);
    }

    // Черта «Руки-масло»: шанс выронить мяч под плотным прессингом
    if (nowCarrying && p.traitIds.includes('butterfingers')) {
      const pressured = state.players.some(
        (opp) =>
          opp.team !== p.team &&
          opp.penaltyTimer <= 0 &&
          opp.trippedTimer <= 0 &&
          Math.hypot(opp.x - p.x, opp.y - p.y) < 42
      );
      if (pressured && rng() < 0.55 * dt) {
        const held = state.balls.find((b) => b.holderId === p.id);
        if (held) {
          held.holderId = null;
          held.vx = (rng() - 0.5) * 160;
          held.vy = (rng() - 0.5) * 160;
          held.vz = 80;
          p.pickupLockout = 0.65;
          spawnCallout(state, p.x, p.y - 18, 'РУКИ-МАСЛО! УРОНИЛ!', '#fb923c', 1.15);
        }
      }
    }
  }

  // Физика контактов на паркете, скольжение корпусом в борьбе и жёсткие заслоны (Пик-н-ролл!)
  for (let i = 0; i < state.players.length; i++) {
    const a = state.players[i];
    if (a.penaltyTimer > 0 || state.inbound?.passerId === a.id) continue;
    for (let j = i + 1; j < state.players.length; j++) {
      const b = state.players[j];
      if (b.penaltyTimer > 0 || state.inbound?.passerId === b.id) continue;
      if (Math.abs(a.z - b.z) > 38) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.hypot(dx, dy);
      const aScreening =
        (a.aiAction === 'screen_set' || a.aiAction === 'offball_screen') && a.team !== b.team;
      const bScreening =
        (b.aiAction === 'screen_set' || b.aiAction === 'offball_screen') && b.team !== a.team;
      const minDist = aScreening || bScreening ? 29 : 24;
      if (d > 0.01 && d < minDist) {
        const overlap = minDist - d;
        const nx = dx / d;
        const ny = dy / d;
        if (aScreening && !bScreening) {
          a.x -= nx * overlap * 0.14;
          a.y -= ny * overlap * 0.14;
          b.x += nx * overlap * 0.86;
          b.y += ny * overlap * 0.86;
          b.vx *= 0.62;
          b.vy *= 0.62;
        } else if (bScreening && !aScreening) {
          a.x -= nx * overlap * 0.86;
          a.y -= ny * overlap * 0.86;
          b.x += nx * overlap * 0.14;
          b.y += ny * overlap * 0.14;
          a.vx *= 0.62;
          a.vy *= 0.62;
        } else {
          const push = overlap * 0.5;
          a.x -= nx * push;
          a.y -= ny * push;
          b.x += nx * push;
          b.y += ny * push;

          // При столкновении игроков разных команд гасим лобовое вдавливание и даём боковое соскальзывание плечом, чтобы никогда не застревали лоб в лоб!
          if (a.team !== b.team) {
            const preSpeedA = Math.hypot(a.vx, a.vy);
            const preSpeedB = Math.hypot(b.vx, b.vy);
            const relVn = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
            if (relVn < 0) {
              const cancel = relVn * 0.5;
              a.vx += nx * cancel;
              a.vy += ny * cancel;
              b.vx -= nx * cancel;
              b.vy -= ny * cancel;
            }
            const tx = -ny;
            const ty = nx;
            const relVt = (a.vx - b.vx) * tx + (a.vy - b.vy) * ty;
            const slideDir =
              Math.abs(relVt) > 28 ? Math.sign(relVt) : getCommittedBypassSide(a);
            const slideStep = slideDir * 48 * dt;
            a.x = Math.max(COURT_LEFT + 14, Math.min(COURT_RIGHT - 14, a.x + tx * slideStep));
            a.y = Math.max(COURT_TOP + 14, Math.min(COURT_BOTTOM - 14, a.y + ty * slideStep));
            b.x = Math.max(COURT_LEFT + 14, Math.min(COURT_RIGHT - 14, b.x - tx * slideStep));
            b.y = Math.max(COURT_TOP + 14, Math.min(COURT_BOTTOM - 14, b.y - ty * slideStep));

            // Подстановка под фол в атаке (Take Charge) ИЛИ актёрская симуляция «Актёра Оскара» (flopper)!
            if (!state.inbound && !state.freeThrow && state.possessionTeam !== null) {
              const atkIsA = a.team === state.possessionTeam;
              const atkPlayer = atkIsA ? a : b;
              const defPlayer = atkIsA ? b : a;
              const atkPreSpeed = atkIsA ? preSpeedA : preSpeedB;
              const defPreSpeed = atkIsA ? preSpeedB : preSpeedA;
              const atkSpeed = Math.hypot(atkPlayer.vx, atkPlayer.vy);
              const atkHasBall = state.balls.some((bl) => bl.holderId === atkPlayer.id);

              if (
                atkHasBall &&
                atkPlayer.aiAction !== 'eurostep' &&
                (defPlayer.aiAction === 'take_charge' || (defPlayer.chargePlantTimer ?? 0) > 0) &&
                defPlayer.z <= 2 &&
                defPreSpeed < 44 &&
                atkPreSpeed >= 96 &&
                rng() < computeRefereeDetectionChance(state.referee, atkPlayer).chance
              ) {
                defPlayer.chargePlantTimer = 0;
                defPlayer.trippedTimer = 0.95;
                defPlayer.vx = (defPlayer.x - atkPlayer.x) * 3.0;
                defPlayer.vy = (defPlayer.y - atkPlayer.y) * 3.0;
                spawnCallout(state, defPlayer.x, defPlayer.y - 22, 'ФОЛ В АТАКЕ (ТАРАН)!', '#38bdf8', 1.4);
                pushEvent(
                  state,
                  `ГРАМОТНАЯ ЗАЩИТА! ${defPlayer.name} вовремя подставился на дуге — фол в нападении (таран) у ${atkPlayer.name}!`,
                  defPlayer.team,
                  true
                );
                adjudicatePersonalFoul(state, atkPlayer, defPlayer, 'offensive', rng);
                return;
              }

              if (
                defPlayer.traitIds.includes('flopper') &&
                (defPlayer.flopCd ?? 0) <= 0 &&
                defPlayer.z <= 2 &&
                atkSpeed >= 92 &&
                rng() < 0.34
              ) {
                defPlayer.flopCd = 9.0;
                defPlayer.trippedTimer = 1.35;
                defPlayer.vx = (defPlayer.x - atkPlayer.x) * 3.5;
                defPlayer.vy = (defPlayer.y - atkPlayer.y) * 3.5;

                // Верит ли судья спектаклю? Высокий IQ помогает продать контакт, но рост refSuspicion выдаёт симулянта!
                const sellProb = Math.max(
                  0.12,
                  Math.min(0.76, 0.44 + (defPlayer.effectiveStats.iq / 100) * 0.28 - defPlayer.refSuspicion * 0.14)
                );
                const rollFlop = rng();
                defPlayer.refSuspicion = Math.min(5, defPlayer.refSuspicion + 1);

                if (rollFlop < sellProb) {
                  spawnCallout(state, defPlayer.x, defPlayer.y - 22, 'ОСКАР! ПРОДАЛ ФОЛ В АТАКЕ!', '#fde047', 1.45);
                  pushEvent(
                    state,
                    `ТЕАТР! ${defPlayer.name} [Актёр Оскара] картинно отлетает от контакта — судья даёт фол в нападении ${atkPlayer.name}!`,
                    defPlayer.team,
                    true
                  );
                  adjudicatePersonalFoul(state, atkPlayer, defPlayer, 'offensive', rng);
                  return;
                } else if (rollFlop < sellProb + 0.36) {
                  // Судья раскусил симуляцию — технический фол за флоп (1 штрафной сопернику + сохранение владения)!
                  sound.playWhistle();
                  state.referee.whistleTimer = 1.1;
                  state.referee.callsMade++;
                  defPlayer.fouls++;
                  const callMsg = `ТЕХНИЧЕСКИЙ ЗА СИМУЛЯЦИЮ! ${defPlayer.name} [Актёр Оскара] переиграл — 1 штрафной!`;
                  state.referee.lastCallText = callMsg;
                  spawnCallout(state, defPlayer.x, defPlayer.y - 22, 'СИМУЛЯЦИЯ! ТЕХНАРЬ!', '#ef4444', 1.45);
                  pushEvent(state, callMsg, 'ref', true);
                  startFreeThrowSequence(state, atkPlayer, 1, atkPlayer.team, rng);
                  return;
                } else {
                  spawnCallout(state, defPlayer.x, defPlayer.y - 20, 'ВСТАВАЙ, АРТИСТ!', '#a8a29e', 1.15);
                }
              }
            }
          }
        }
      }
    }
  }

  // ==================== ПОСЛЕ СТОЛКНОВЕНИЙ: ПРАВИЛО 5 СЕКУНД ПОД ОПЕКОЙ, СПОРНЫЙ ЗАХВАТ И ДВОЙНОЕ ВЕДЕНИЕ ====================
  if (!state.inbound && state.balls.length === 1) {
    const carrierBall = state.balls[0];
    const carrier =
      carrierBall && carrierBall.holderId !== null
        ? state.players.find(
            (pl) => pl.id === carrierBall.holderId && pl.penaltyTimer <= 0 && pl.trippedTimer <= 0
          ) ?? null
        : null;

    if (carrier) {
      let minDefDist = 9999;
      for (const opp of state.players) {
        if (opp.team !== carrier.team && opp.penaltyTimer <= 0 && opp.trippedTimer <= 0) {
          const d = Math.hypot(opp.x - carrier.x, opp.y - carrier.y);
          if (d < minDefDist) minDefDist = d;
        }
      }

      // Реальная скорость смещения игрока на паркете за кадр (учитывает упор в защитника!)
      const actualSpeed =
        Math.hypot(
          carrier.x - (carrier.prevStepX ?? carrier.x),
          carrier.y - (carrier.prevStepY ?? carrier.y)
        ) / Math.max(dt, 1e-4);

      // 1) 5 секунд под плотной опекой без продвижения/паса/броска
      if (minDefDist < 46 && actualSpeed < 45 && carrier.windupTimer <= 0) {
        carrier.closelyGuardedTimer += dt;
        if (carrier.closelyGuardedTimer >= 5.0) {
          carrier.closelyGuardedTimer = 0;
          callRuleViolation(
            state,
            carrier.team,
            '5 СЕК ПОД ОПЕКОЙ!',
            `СВИСТОК! ${carrier.name} удерживал мяч 5 секунд под плотной опекой без продвижения!`,
            carrier.x,
            carrier.y,
            rng
          );
          return;
        }
      } else {
        carrier.closelyGuardedTimer = Math.max(0, carrier.closelyGuardedTimer - dt * 1.4);
      }

      // 2) Взаимный захват мяча (Спорный мяч -> владение по стрелке FIBA!)
      if (minDefDist < 31 && actualSpeed < 34 && carrier.windupTimer <= 0) {
        state.heldBallTieTimer += dt;
        if (state.heldBallTieTimer >= 1.8) {
          state.heldBallTieTimer = 0;
          const arrowTeam = state.possessionArrow;
          state.possessionArrow = arrowTeam === 0 ? 1 : 0;
          sound.playWhistle();
          state.referee.whistleTimer = 1.05;
          state.referee.callsMade++;
          state.referee.lastCallText = `СПОРНЫЙ МЯЧ! Владение «${state.teamNames[arrowTeam]}»`;
          spawnCallout(state, carrier.x, carrier.y - 20, 'СПОРНЫЙ · ПО СТРЕЛКЕ!', '#fbbf24', 1.35);
          pushEvent(
            state,
            `СВИСТОК! Спорный мяч в захвате — по стрелке поочерёдного владения вводит «${state.teamNames[arrowTeam]}»!`,
            'ref',
            false
          );
          const nextClk =
            arrowTeam === carrier.team && isInFrontcourt(arrowTeam, carrier.x)
              ? Math.max(state.shotClock, 14)
              : 24;
          startInboundPlay(
            state,
            arrowTeam,
            carrier.x,
            carrier.y,
            'Спорный мяч (стрелка)',
            nextClk,
            rng,
            false
          );
          return;
        }
      } else {
        state.heldBallTieTimer = Math.max(0, state.heldBallTieTimer - dt * 1.5);
      }

      // 3) Контроль ведения и запрет повторного старта после остановки (Двойное ведение!)
      if (!carrier.hasStoppedDribble) {
        if (actualSpeed > 48) {
          carrier.dribbleLiveTimer = (carrier.dribbleLiveTimer ?? 0) + dt;
        } else if (
          (carrier.dribbleLiveTimer ?? 0) > 0.65 &&
          minDefDist < 42 &&
          actualSpeed < 22 &&
          carrier.windupTimer <= 0
        ) {
          carrier.hasStoppedDribble = true;
        }
      } else if (
        carrier.windupTimer <= 0 &&
        carrier.z === 0 &&
        actualSpeed > 95
      ) {
        carrier.hasStoppedDribble = false;
        callRuleViolation(
          state,
          carrier.team,
          'ДВОЙНОЕ ВЕДЕНИЕ!',
          `СВИСТОК! Двойное ведение у ${carrier.name} — снова побежал с ведением после остановки!`,
          carrier.x,
          carrier.y,
          rng
        );
        return;
      }

      // 4) Ошибка ведения на скорости у совсем неуклюжих дриблёров под прессингом
      if (
        carrier.effectiveStats.dribble < 32 &&
        minDefDist < 38 &&
        actualSpeed > 85 &&
        rng() < 0.075 * dt
      ) {
        const isDoubleDrib = rng() < 0.5;
        callRuleViolation(
          state,
          carrier.team,
          isDoubleDrib ? 'ДВОЙНОЕ ВЕДЕНИЕ!' : 'ПРОБЕЖКА!',
          `СВИСТОК! ${isDoubleDrib ? 'Двойное ведение' : 'Пробежка на шагах'} у ${carrier.name} под прессингом!`,
          carrier.x,
          carrier.y,
          rng
        );
        return;
      }
    }
  }

  // ==================== ФИЗИКА МЯЧЕЙ, АУТЫ, ПОМЕХА ПОПАДАНИЮ, ПЕРЕХВАТЫ И ГОЛЫ ====================
  for (const ball of state.balls) {
    if (ball.holderId !== null) {
      const holder = state.players.find((p) => p.id === ball.holderId && p.penaltyTimer <= 0);
      if (!holder) {
        ball.holderId = null;
      } else {
        const isInbPasser = state.inbound?.passerId === holder.id;
        const dribbleOsc = isInbPasser
          ? 0
          : Math.abs(Math.sin(state.quarterTimeRemaining * 10 + ball.id)) * 18;
        ball.x = holder.x + Math.cos(holder.heading) * 14;
        ball.y = holder.y + Math.sin(holder.heading) * 14;
        ball.z = isInbPasser ? 24 : holder.z > 2 ? holder.z + 24 : 6 + dribbleOsc;
        continue;
      }
    }

    // Черта «Магнит в шортах»: притягивает свободные мячи (кроме высоких навесов Месси)
    if (ball.shotType !== 'messi' && ball.z < 95) {
      for (const p of state.players) {
        if (p.penaltyTimer > 0 || p.trippedTimer > 0 || !p.traitIds.includes('magnet')) continue;
        const dx = p.x - ball.x;
        const dy = p.y - ball.y;
        const d = Math.hypot(dx, dy);
        if (d > 8 && d < 155) {
          ball.vx += (dx / d) * 320 * dt;
          ball.vy += (dy / d) * 320 * dt;
        }
      }
    }

    const prevZ = ball.z;
    ball.vz -= gravity * dt;
    ball.x += ball.vx * dt;
    ball.y += ball.vy * dt;
    ball.z += ball.vz * dt;
    ball.spinAngle += Math.hypot(ball.vx, ball.vy) * dt * 0.04;

    // Запись шлейфа мяча (кольцевой буфер без аллокаций)
    const tp = ball.trail[ball.trailHead];
    ball.trailHead = (ball.trailHead + 1) % ball.trail.length;
    tp.x = ball.x;
    tp.y = ball.y;
    tp.z = ball.z;
    tp.alpha = 1;
    tp.isMessi = ball.shotType === 'messi';

    // Физический отскок промахов от баскетбольного ЩИТА (чтобы броски не пролетали сквозь стекло в аут!)
    if (!ball.isGuaranteedSwish && ball.z >= 74 && ball.z <= 158 && Math.abs(ball.y - HOOP_Y) <= 38) {
      const leftBoardX = HOOP_LEFT_X - 12;
      const rightBoardX = HOOP_RIGHT_X + 12;
      if (ball.x <= leftBoardX && ball.vx < 0) {
        recordMissedShotOutcome(state, ball);
        ball.x = leftBoardX + 2;
        ball.vx = Math.abs(ball.vx) * 0.54 + 35;
        ball.vy *= 0.68;
        ball.shotType = 'none';
        ball.targetHoop = null;
        sound.playBounce(0.75);
      } else if (ball.x >= rightBoardX && ball.vx > 0) {
        recordMissedShotOutcome(state, ball);
        ball.x = rightBoardX - 2;
        ball.vx = -(Math.abs(ball.vx) * 0.54 + 35);
        ball.vy *= 0.68;
        ball.shotType = 'none';
        ball.targetHoop = null;
        sound.playBounce(0.75);
      }
    }

    // Настоящее правило АУТА + героический прыжок за уходящим мячом («СПАС МЯЧ ИЗ АУТА!»)!
    if (!state.inbound && state.balls.length === 1) {
      if (isPointOutOfBounds(ball.x, ball.y, 0)) {
        if (!ball.isGuaranteedSwish && ball.z < 58) {
          const saverCandidates = state.players
            .filter(
              (pl) =>
                pl.penaltyTimer <= 0 &&
                pl.trippedTimer <= 0 &&
                (pl.saveOobCd ?? 0) <= 0 &&
                pl.z < 28 &&
                Math.hypot(pl.x - ball.x, pl.y - ball.y) < 42
            )
            .sort(
              (a, b) =>
                Math.hypot(a.x - ball.x, a.y - ball.y) - Math.hypot(b.x - ball.x, b.y - ball.y)
            );
          const saver = saverCandidates[0] ?? null;

          if (saver) {
            recordMissedShotOutcome(state, ball);
            ball.wasShotAttempt = false;
            const safeX = Math.max(COURT_LEFT + 18, Math.min(COURT_RIGHT - 18, ball.x));
            const safeY = Math.max(COURT_TOP + 18, Math.min(COURT_BOTTOM - 18, ball.y));

            if (saver.effectiveStats.iq >= 34) {
              const allyPool = state.players.filter(
                (m) =>
                  m.team === saver.team &&
                  m.id !== saver.id &&
                  m.penaltyTimer <= 0 &&
                  m.trippedTimer <= 0 &&
                  (!state.frontcourtEstablished ||
                    state.possessionTeam !== saver.team ||
                    isInFrontcourt(saver.team, m.x))
              );
              const fallbackAllies = state.players.filter(
                (m) => m.team === saver.team && m.id !== saver.id && m.penaltyTimer <= 0
              );
              const bestAlly = (allyPool.length > 0 ? allyPool : fallbackAllies).sort((a, b) => {
                const openA = state.players
                  .filter((o) => o.team !== saver.team && o.penaltyTimer <= 0)
                  .reduce((minD, o) => Math.min(minD, Math.hypot(o.x - a.x, o.y - a.y)), 999);
                const openB = state.players
                  .filter((o) => o.team !== saver.team && o.penaltyTimer <= 0)
                  .reduce((minD, o) => Math.min(minD, Math.hypot(o.x - b.x, o.y - b.y)), 999);
                return openB - openA;
              })[0];

              if (bestAlly) {
                const saveArc = computeArcVelocity(
                  safeX,
                  safeY,
                  22,
                  bestAlly.x,
                  bestAlly.y,
                  26,
                  96,
                  gravity
                );
                ball.x = safeX;
                ball.y = safeY;
                ball.z = 22;
                ball.vx = saveArc.vx;
                ball.vy = saveArc.vy;
                ball.vz = saveArc.vz;
                ball.holderId = null;
                ball.lastTouchTeam = saver.team;
                ball.lastPasserId = saver.id;
                ball.passReceiverId = bestAlly.id;
                ball.shotType = 'pass';
                ball.shotStyle = 'standard';
                ball.targetHoop = null;
                ball.isGuaranteedSwish = false;

                saver.saveOobCd = 6.0;
                saver.pickupLockout = 0.85;
                saver.trippedTimer = 0.65;
                saver.aiAction = 'tripped_down';
                saver.aiThought = `В прыжке спас мяч из аута на ${bestAlly.name.split(' ')[0]}!`;
                spawnParticles(state, safeX, safeY, 12, '#4ade80', 12, 0.9, rng);
                spawnCallout(state, saver.x, saver.y - 22, 'СПАС МЯЧ ИЗ АУТА!', '#4ade80', 1.4);
                pushEvent(
                  state,
                  `ГЕРОИЗМ! ${saver.name} в отчаянном прыжке достал уходящий в аут мяч и отбросил партнёру ${bestAlly.name}!`,
                  saver.team,
                  true
                );
                sound.playBounce(0.75);
                continue;
              }
            } else {
              // Комедия глупого балбеса (IQ < 34): героически спасает мяч из аута... прямо в руки сопернику!
              const oppGift = state.players
                .filter((o) => o.team !== saver.team && o.penaltyTimer <= 0)
                .sort(
                  (a, b) =>
                    Math.hypot(a.x - safeX, a.y - safeY) - Math.hypot(b.x - safeX, b.y - safeY)
                )[0];
              if (oppGift) {
                const oopsArc = computeArcVelocity(
                  safeX,
                  safeY,
                  22,
                  oppGift.x,
                  oppGift.y,
                  24,
                  92,
                  gravity
                );
                ball.x = safeX;
                ball.y = safeY;
                ball.z = 22;
                ball.vx = oopsArc.vx;
                ball.vy = oopsArc.vy;
                ball.vz = oopsArc.vz;
                ball.holderId = null;
                ball.lastTouchTeam = saver.team;
                ball.lastPasserId = saver.id;
                ball.passReceiverId = oppGift.id;
                ball.shotType = 'pass';
                ball.shotStyle = 'standard';
                ball.targetHoop = null;
                ball.isGuaranteedSwish = false;

                saver.saveOobCd = 6.0;
                saver.pickupLockout = 0.85;
                saver.trippedTimer = 0.65;
                saver.aiAction = 'tripped_down';
                saver.aiThought = 'Героически спас мяч из аута... ой, это же чужой!';
                spawnParticles(state, safeX, safeY, 12, '#fca5a5', 10, 0.9, rng);
                spawnCallout(state, saver.x, saver.y - 22, 'СПАС ИЗ АУТА... ЧУЖОМУ!', '#fca5a5', 1.45);
                pushEvent(
                  state,
                  `КУРЬЁЗ! ${saver.name} в падении спас мяч из аута и подарил точно в руки сопернику ${oppGift.name}!`,
                  saver.team,
                  true
                );
                sound.playBounce(0.75);
                continue;
              }
            }
          }
        }

        recordMissedShotOutcome(state, ball);
        ball.wasShotAttempt = false;
        const lastTeam: 0 | 1 = ball.lastTouchTeam ?? (ball.x < MID_X ? 0 : 1);
        callRuleViolation(
          state,
          lastTeam,
          'АУТ!',
          `СВИСТОК! Мяч ушёл в аут от «${state.teamNames[lastTeam]}» — ввод из-за линии!`,
          ball.x,
          ball.y,
          rng
        );
        return;
      }
    } else {
      // В хаотичном режиме «Два мяча» оставляем упругие борта, чтобы второй мяч оставался в игре
      if (ball.x < COURT_LEFT + 10) {
        ball.x = COURT_LEFT + 10;
        ball.vx = Math.abs(ball.vx) * 0.72;
        ball.isGuaranteedSwish = false;
      } else if (ball.x > COURT_RIGHT - 10) {
        ball.x = COURT_RIGHT - 10;
        ball.vx = -Math.abs(ball.vx) * 0.72;
        ball.isGuaranteedSwish = false;
      }
      if (ball.y < COURT_TOP + 10) {
        ball.y = COURT_TOP + 10;
        ball.vy = Math.abs(ball.vy) * 0.72;
        ball.isGuaranteedSwish = false;
      } else if (ball.y > COURT_BOTTOM - 10) {
        ball.y = COURT_BOTTOM - 10;
        ball.vy = -Math.abs(ball.vy) * 0.72;
        ball.isGuaranteedSwish = false;
      }
    }

    // Отскок от пола
    if (ball.z <= 0) {
      ball.z = 0;
      recordMissedShotOutcome(state, ball);

      // Если 24 секунды истекли во время полёта броска, а мяч приземлился без касания дужки кольца (Airball) -> нарушение 24с!
      if (
        state.shotClockExpiredInFlight &&
        !state.shotTouchedRim &&
        state.possessionTeam !== null &&
        state.balls.length === 1
      ) {
        ball.wasShotAttempt = false;
        callRuleViolation(
          state,
          state.possessionTeam,
          '24 СЕКУНДЫ!',
          `СВИСТОК! Сирена 24 секунд — бросок «${state.teamNames[state.possessionTeam]}» не коснулся дужки кольца!`,
          ball.x,
          ball.y,
          rng
        );
        return;
      }

      if (Math.abs(ball.vz) > 45) {
        ball.vz = Math.abs(ball.vz) * 0.62;
        sound.playBounce(Math.min(1, Math.abs(ball.vz) / 220));
      } else {
        ball.vz = 0;
      }
      const floorFriction = isIce ? 0.992 : 0.965;
      ball.vx *= Math.pow(floorFriction, dt * 60);
      ball.vy *= Math.pow(floorFriction, dt * 60);
      ball.shotType = 'none';
      ball.targetHoop = null;
      ball.isGuaranteedSwish = false;
      ball.isTipoffBall = false;
    }

    // Проверка попадания в ОБА кольца (когда мяч падает сверху вниз через плоскость RIM_Z!)
    if (prevZ >= RIM_Z && ball.z <= RIM_Z && ball.vz < 0) {
      const hoops: Array<0 | 1> = [0, 1];
      let scored = false;
      for (const hIdx of hoops) {
        const hx = hIdx === 0 ? HOOP_LEFT_X : HOOP_RIGHT_X;
        const dRim = Math.hypot(ball.x - hx, ball.y - HOOP_Y);
        if (
          (ball.isGuaranteedSwish && ball.targetHoop === hIdx && dRim <= RIM_RADIUS + 14) ||
          dRim <= RIM_RADIUS - 2
        ) {
          awardPoints(state, ball, hIdx, rng);
          scored = true;
          break;
        } else if (dRim <= RIM_RADIUS + 14 && ball.rimBounceCount < 2) {
          // Звон дужки кольца: фиксируем промах в карте бросков и касание кольца для правила 14 секунд!
          recordMissedShotOutcome(state, ball);
          ball.rimBounceCount++;
          state.shotTouchedRim = true;
          state.frontcourtEstablished = false;
          state.backcourtTimer = 0;
          if (state.shotClockExpiredInFlight) {
            state.shotClockExpiredInFlight = false;
            state.shotClock = 14;
          }
          ball.z = RIM_Z + 2;
          ball.vz = Math.abs(ball.vz) * 0.52 + 85;
          const outAngle = rng() * Math.PI * 2;
          ball.vx = Math.cos(outAngle) * (95 + rng() * 90);
          ball.vy = Math.sin(outAngle) * (95 + rng() * 90);
          ball.shotType = 'none';
          ball.targetHoop = null;
          ball.isGuaranteedSwish = false;
          sound.playBounce(0.9);
          spawnParticles(state, hx, HOOP_Y, RIM_Z, '#e5e7eb', 7, 0.7, rng);
        }
      }
      if (scored) continue;
    }

    // ==================== ПОДБОР, БЛОК-ШОТЫ И ПРАВИЛО ПОМЕХИ ПОПАДАНИЮ (GOALTENDING) ====================
    let bestCatcher: PlayerEntity | null = null;
    let bestCatchScore = -999;

    for (const p of state.players) {
      if (p.penaltyTimer > 0 || p.trippedTimer > 0 || p.pickupLockout > 0) continue;
      if (state.inbound?.passerId === p.id) continue;
      if (state.balls.some((otherB) => otherB !== ball && otherB.holderId === p.id)) continue;

      const horizDist = Math.hypot(p.x - ball.x, p.y - ball.y);
      const maxR = computePlayerHorizontalInterceptRadius(p);
      if (horizDist > maxR) continue;

      const vertReach = computePlayerVerticalReach(p);
      if (ball.z > vertReach || ball.z < p.z - 22) continue;

      // Правила перехвата летящих бросков:
      if (
        ball.shotType === 'messi' ||
        ball.shotType === 'normal' ||
        ball.shotType === 'panic' ||
        ball.shotType === 'dunk'
      ) {
        const isEnemyShot = ball.lastTouchTeam !== p.team;
        const isAlliedOwnGoalShot =
          ball.lastTouchTeam === p.team && ball.targetHoop === getDefendingHoop(p.team);

        // Свой нормальный бросок во вражеское кольцо партнёры не трогают в воздухе (Offensive Interference)
        if (!isEnemyShot && !isAlliedOwnGoalShot) continue;

        // Высоколетящий навес «Месси» или крутая слеза «Флоутера» в верхней фазе требуют супер-прыжка «Кузнечика» или батута!
        if (
          (ball.shotType === 'messi' && ball.z > 125 ||
            ball.shotStyle === 'floater' && ball.z > 118) &&
          !p.traitIds.includes('grasshopper') &&
          p.z < 88
        ) {
          continue;
        }
      }

      const catchPriority =
        (p.traitIds.includes('grasshopper') && ball.z > 95 ? 80 : 0) +
        (ball.passReceiverId === p.id ? 45 : 0) +
        p.effectiveStats.steal * 0.4 +
        p.z * 0.35 -
        horizDist;

      if (catchPriority > bestCatchScore) {
        bestCatchScore = catchPriority;
        bestCatcher = p;
      }
    }

    if (bestCatcher) {
      // Проверка правила ПОМЕХИ ПОПАДАНИЮ (Goaltending): обычный бросок перехвачен на снижении над цилиндром кольца!
      if (
        (ball.shotType === 'normal' || ball.shotType === 'panic') &&
        ball.targetHoop !== null &&
        ball.lastTouchTeam !== null &&
        bestCatcher.team !== ball.lastTouchTeam &&
        ball.vz < -18 &&
        ball.z >= RIM_Z - 3
      ) {
        const targetHoopPos = getHoopPosition(ball.targetHoop);
        const distBallToHoop = Math.hypot(ball.x - targetHoopPos.x, ball.y - targetHoopPos.y);
        if (distBallToHoop <= 88) {
          sound.playWhistle();
          state.referee.whistleTimer = 1.1;
          state.referee.callsMade++;
          state.violationsCount[bestCatcher.team]++;
          state.referee.lastCallText = `ПОМЕХА ПОПАДАНИЮ! (${bestCatcher.name})`;
          spawnCallout(
            state,
            bestCatcher.x,
            bestCatcher.y - 24,
            'ПОМЕХА ПОПАДАНИЮ! ГОЛ ЗАСЧИТАН!',
            '#ef4444',
            1.6
          );
          pushEvent(
            state,
            `СВИСТОК! Помеха попаданию (Goaltending) у ${bestCatcher.name} — мяч шёл вниз в кольцо и засчитан!`,
            'ref',
            true
          );
          awardPoints(state, ball, ball.targetHoop, rng);
          continue;
        }
      }

      // Если мяч не коснулся кольца после сирены 24с и пойман игроком атакующей команды — нарушение 24с!
      if (
        state.shotClockExpiredInFlight &&
        !state.shotTouchedRim &&
        state.possessionTeam === bestCatcher.team &&
        state.balls.length === 1
      ) {
        callRuleViolation(
          state,
          bestCatcher.team,
          '24 СЕКУНДЫ!',
          `СВИСТОК! Истекли 24 секунды — мяч не задел дужку кольца до подбора!`,
          bestCatcher.x,
          bestCatcher.y,
          rng
        );
        return;
      }

      const wasHighShot =
        (ball.shotType === 'messi' || ball.shotType === 'normal') && ball.z > 82;
      const wasMessiShot = ball.shotType === 'messi';
      const savedOwnGoal =
        wasMessiShot &&
        ball.lastTouchTeam === bestCatcher.team &&
        ball.targetHoop === getDefendingHoop(bestCatcher.team);

      if (wasHighShot) {
        bestCatcher.blocks++;
        state.hitstopTimer = 0.045;
        state.cameraShake = Math.max(state.cameraShake, 0.35);
        spawnParticles(state, ball.x, ball.y, ball.z, '#4ade80', 16, 1.1, rng);
        const msg = savedOwnGoal
          ? 'СПАС ОТ АВТОГОЛА МЕССИ!'
          : wasMessiShot
          ? 'ПЕРЕХВАТ УДАРА МЕССИ!'
          : 'БЛОК-ПЕРЕХВАТ В НЕБЕ!';
        spawnCallout(state, bestCatcher.x, bestCatcher.y - 24, msg, '#4ade80', 1.55);
        pushEvent(
          state,
          savedOwnGoal
            ? `${bestCatcher.name} в космическом прыжке спас своё кольцо от автогола «Месси»!`
            : wasMessiShot
            ? `${bestCatcher.name} выпрыгнул под потолок и перехватил радугу «Месси»!`
            : `${bestCatcher.name} снял верховой бросок на взлёте!`,
          bestCatcher.team,
          true
        );
      }

      const wasOffensiveShotRebound = Boolean(
        ball.wasShotAttempt && ball.lastTouchTeam === bestCatcher.team
      );
      if (ball.wasShotAttempt) {
        recordMissedShotOutcome(state, ball);
        bestCatcher.rebounds++;
        ball.wasShotAttempt = false;
      }

      giveBallToPlayer(state, ball, bestCatcher, rng, true, wasOffensiveShotRebound);
    }
  }
}
