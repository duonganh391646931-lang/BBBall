import { MatchModifierId, PlayerEntity, RefereeEntity } from './types';

export const COURT_W = 1180;
export const COURT_H = 620;
export const COURT_LEFT = 50;
export const COURT_RIGHT = COURT_W - 50; // 1130
export const COURT_TOP = 64;
export const COURT_BOTTOM = COURT_H - 44; // 576

export const MID_X = COURT_W / 2; // 590
export const HOOP_Y = (COURT_TOP + COURT_BOTTOM) / 2; // 320
export const HOOP_LEFT_X = 96; // Защищает команда 0, атакует команда 1
export const HOOP_RIGHT_X = COURT_W - 96; // 1084 — защищает команда 1, атакует команда 0
export const RIM_Z = 95;
export const RIM_RADIUS = 19;
export const THREE_PT_RADIUS = 248;

export const PENALTY_BENCH_POS: Record<0 | 1, { x: number; y: number }> = {
  0: { x: 406, y: 28 },
  1: { x: 774, y: 28 },
};

export const TEAM_BENCH_BOX: Record<0 | 1, { x: number; y: number; width: number; height: number }> = {
  0: { x: 52, y: 12, width: 162, height: 32 },
  1: { x: 966, y: 12, width: 162, height: 32 },
};

export function getTeamBenchSeatPos(
  team: 0 | 1,
  seatIndex: number,
  totalSeats = 3
): { x: number; y: number } {
  const box = TEAM_BENCH_BOX[team];
  const count = Math.max(1, totalSeats);
  const step = box.width / (count + 1);
  return {
    x: box.x + step * (seatIndex + 1),
    y: box.y + box.height * 0.5,
  };
}

export const TRAMPOLINE_PADS = [
  { x: 215, y: HOOP_Y, radius: 34 },
  { x: COURT_W - 215, y: HOOP_Y, radius: 34 },
];

export const BASE_GRAVITY = 540;

export function getEffectiveGravity(modifiers: MatchModifierId[]): number {
  return modifiers.includes('low_gravity') ? 275 : BASE_GRAVITY;
}

export function getHoopPosition(hoopIndex: 0 | 1): { x: number; y: number; z: number } {
  return {
    x: hoopIndex === 0 ? HOOP_LEFT_X : HOOP_RIGHT_X,
    y: HOOP_Y,
    z: RIM_Z,
  };
}

export function getAttackingHoop(team: 0 | 1): 0 | 1 {
  return team === 0 ? 1 : 0;
}

export function getDefendingHoop(team: 0 | 1): 0 | 1 {
  return team === 0 ? 0 : 1;
}

export function getNearestHoop(x: number, y: number): 0 | 1 {
  const dLeft = Math.hypot(x - HOOP_LEFT_X, y - HOOP_Y);
  const dRight = Math.hypot(x - HOOP_RIGHT_X, y - HOOP_Y);
  return dLeft <= dRight ? 0 : 1;
}

export const KEY_WIDTH = 190;
export const KEY_HEIGHT = 164;
export const CENTER_CIRCLE_RADIUS = 66;

export function isOnOwnHalf(player: PlayerEntity): boolean {
  return player.team === 0 ? player.x < MID_X : player.x > MID_X;
}

export function isInFrontcourt(team: 0 | 1, x: number): boolean {
  return team === 0 ? x > MID_X : x < MID_X;
}

export function isInAttackingPaint(player: PlayerEntity): boolean {
  const halfH = KEY_HEIGHT / 2;
  if (Math.abs(player.y - HOOP_Y) > halfH) return false;
  if (player.team === 0) {
    // Команда 0 атакует ПРАВОЕ кольцо (краска: [COURT_RIGHT - KEY_WIDTH .. COURT_RIGHT])
    return player.x >= COURT_RIGHT - KEY_WIDTH && player.x <= COURT_RIGHT;
  }
  // Команда 1 атакует ЛЕВОЕ кольцо (краска: [COURT_LEFT .. COURT_LEFT + KEY_WIDTH])
  return player.x >= COURT_LEFT && player.x <= COURT_LEFT + KEY_WIDTH;
}

export function isPointOutOfBounds(x: number, y: number, margin = 0): boolean {
  return (
    x <= COURT_LEFT + margin ||
    x >= COURT_RIGHT - margin ||
    y <= COURT_TOP + margin ||
    y >= COURT_BOTTOM - margin
  );
}

export function getFreeThrowLinePos(targetHoop: 0 | 1): { x: number; y: number } {
  return {
    x: targetHoop === 0 ? COURT_LEFT + KEY_WIDTH - 2 : COURT_RIGHT - KEY_WIDTH + 2,
    y: HOOP_Y,
  };
}

export function computeJumpVelocity(player: PlayerEntity, useSuperJump: boolean): number {
  if (useSuperJump && player.traitIds.includes('grasshopper') && player.superJumpCd <= 0) {
    // Супер-прыжок Кузнечика: взмывает выше 225px над паркетом!
    return 505;
  }
  const jumpNorm = player.effectiveStats.jump / 100;
  return 195 + jumpNorm * 135;
}

export function computePlayerVerticalReach(player: PlayerEntity): number {
  const baseBodyHeight = 46 * player.bodyScale;
  const grasshopperBonus = player.traitIds.includes('grasshopper') ? 18 : 0;
  const barrierBonus = player.traitIds.includes('barrier_blocker') ? 10 : 0;
  const alleyBonus = player.traitIds.includes('sixth_sense_alley') ? 12 : 0;
  return player.z + baseBodyHeight + grasshopperBonus + barrierBonus + alleyBonus;
}

export function computePlayerHorizontalInterceptRadius(player: PlayerEntity): number {
  let r = 23 * player.bodyScale + (player.effectiveStats.steal / 100) * 10;
  if (player.traitIds.includes('grasshopper')) r += 14;
  if (player.traitIds.includes('barrier_blocker')) r += 8;
  if (player.traitIds.includes('magnet')) r += 10;
  if (player.traitIds.includes('brick_magnet_board')) r += 8;
  return r;
}

export interface ArcSolution {
  vx: number;
  vy: number;
  vz: number;
  flightTime: number;
}

export function computeArcVelocity(
  fromX: number,
  fromY: number,
  fromZ: number,
  toX: number,
  toY: number,
  toZ: number,
  desiredPeakZ: number,
  gravity: number = BASE_GRAVITY
): ArcSolution {
  const peakZ = Math.max(desiredPeakZ, Math.max(fromZ, toZ) + 18);
  const dzUp = Math.max(4, peakZ - fromZ);
  const dzDown = Math.max(4, peakZ - toZ);

  const vz = Math.sqrt(2 * gravity * dzUp);
  const tUp = vz / gravity;
  const tDown = Math.sqrt((2 * dzDown) / gravity);
  const flightTime = Math.max(0.18, tUp + tDown);

  const vx = (toX - fromX) / flightTime;
  const vy = (toY - fromY) / flightTime;

  return { vx, vy, vz, flightTime };
}

export function projectPointOntoSegment(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number
): { projX: number; projY: number; t: number; perpDist: number } {
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq < 1e-4) {
    return {
      projX: ax,
      projY: ay,
      t: 0,
      perpDist: Math.hypot(px - ax, py - ay),
    };
  }
  const tRaw = ((px - ax) * dx + (py - ay) * dy) / lenSq;
  const t = Math.max(0.08, Math.min(0.92, tRaw));
  const projX = ax + t * dx;
  const projY = ay + t * dy;
  return {
    projX,
    projY,
    t,
    perpDist: Math.hypot(px - projX, py - projY),
  };
}

export function isThreePointShot(originX: number, originY: number, targetHoop: 0 | 1): boolean {
  const hoop = getHoopPosition(targetHoop);
  return Math.hypot(originX - hoop.x, originY - hoop.y) >= THREE_PT_RADIUS;
}

/**
 * Вычисляет вероятность того, что Судья заметит нарушение правил игроком offender.
 * Базовый шанс: 80% перед взглядом судьи (dot >= 0) и 50% за спиной (dot < 0).
 * Черта «Незаметный» даёт -45%, а каждая единица подозрительности (refSuspicion) даёт +10%.
 */
export function computeRefereeDetectionChance(
  referee: RefereeEntity,
  offender: PlayerEntity
): { chance: number; inFrontOfRef: boolean } {
  const dx = offender.x - referee.x;
  const dy = offender.y - referee.y;
  const len = Math.hypot(dx, dy);
  let inFrontOfRef = true;
  if (len > 1e-3) {
    const gazeX = Math.cos(referee.gazeAngle);
    const gazeY = Math.sin(referee.gazeAngle);
    const dot = (dx / len) * gazeX + (dy / len) * gazeY;
    inFrontOfRef = dot >= 0;
  }

  let chance = inFrontOfRef ? 0.8 : 0.5;
  if (offender.traitIds.includes('stealthy')) {
    chance -= 0.45;
  }
  chance += offender.refSuspicion * 0.1;

  return {
    chance: Math.max(0.05, Math.min(0.98, chance)),
    inFrontOfRef,
  };
}
