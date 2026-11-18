import { getAttackingHoop, getDefendingHoop, getHoopPosition } from '../physics';
import { MatchState } from '../types';

export interface RewardConfig {
  /** Множитель дисконтирования для PBRS-потенциала (gamma * Phi(s') - Phi(s)) */
  gamma: number;
  /** Командная награда за 1 набранное/пропущенное очко (нулевая сумма) */
  teamPointWeight: number;
  /** Индивидуальный бонус автору забитого броска (за 1 очко) */
  individualPointBonus: number;
  /** Бонус за результативную передачу (ассист) */
  assistBonus: number;
  /** Бонус за подбор */
  reboundBonus: number;
  /** Бонус за чистый перехват */
  stealBonus: number;
  /** Бонус за блок-шот */
  blockBonus: number;
  /** Штраф игроку за персональный фол */
  personalFoulPenalty: number;
  /** Штраф всей команде за нарушение правил (аут, пробежка, 24с, 8с, 3с, 5с) */
  teamViolationPenalty: number;
  /** Коэффициент плотного PBRS-потенциала продвижения мяча и сдерживания в защите */
  shapingScale: number;
}

export const DEFAULT_REWARD_CONFIG: RewardConfig = {
  gamma: 0.995,
  teamPointWeight: 1.0,
  individualPointBonus: 0.25,
  assistBonus: 0.35,
  reboundBonus: 0.25,
  stealBonus: 0.38,
  blockBonus: 0.38,
  personalFoulPenalty: 0.32,
  teamViolationPenalty: 0.18,
  shapingScale: 0.12,
};

interface PlayerStatSnapshot {
  points: number;
  rebounds: number;
  assists: number;
  steals: number;
  blocks: number;
  fouls: number;
  ownGoals: number;
}

/**
 * Вычисляет скалярный потенциал состояния Phi(s) для команды team в диапазоне [-1, 1]:
 * поощряет приближение контролируемого мяча к кольцу соперника и плотную опеку вражеского дриблёра.
 */
export function computeTeamPotential(state: MatchState, team: 0 | 1): number {
  const atkHoop = getHoopPosition(getAttackingHoop(team));
  const defHoop = getHoopPosition(getDefendingHoop(team));
  const ball = state.balls[0];
  if (!ball) return 0;

  let phi = 0;
  if (state.possessionTeam === team) {
    const distAtk = Math.hypot(ball.x - atkHoop.x, ball.y - atkHoop.y);
    // Чем ближе мяч к чужому кольцу (вплоть до 90px), тем выше потенциал [0..0.7]
    const progress = 1 - Math.min(1, Math.max(0, (distAtk - 90) / 950));
    phi += 0.65 * progress;
    if (state.frontcourtEstablished) phi += 0.1;
  } else if (state.possessionTeam !== null) {
    // В защите: бонус за нахождение защитника между мячом и своим кольцом
    const ballDistDef = Math.hypot(ball.x - defHoop.x, ball.y - defHoop.y);
    let bestShell = 0;
    for (let i = 0; i < state.players.length; i++) {
      const p = state.players[i];
      if (p.team !== team || p.penaltyTimer > 0 || p.trippedTimer > 0) continue;
      const dToBall = Math.hypot(p.x - ball.x, p.y - ball.y);
      const dToDef = Math.hypot(p.x - defHoop.x, p.y - defHoop.y);
      if (dToDef <= ballDistDef + 28 && dToBall < 140) {
        const score = 1 - dToBall / 140;
        if (score > bestShell) bestShell = score;
      }
    }
    phi += 0.25 * bestShell - 0.35 * (1 - Math.min(1, ballDistDef / 950));
  }
  return phi;
}

/**
 * Трекер наград для многоагентного RL (MAPPO / Self-Play):
 * сочетает командный счёт с нулевой суммой, индивидуальный вклад (подбор/пас/перехват/блок/фол)
 * и потенциальное формирование награды (PBRS), не искажающее оптимальную политику.
 */
export class MatchRewardTracker {
  private prevScore: [number, number] = [0, 0];
  private prevViolations: [number, number] = [0, 0];
  private prevPotential: [number, number] = [0, 0];
  private prevPlayerStats = new Map<string, PlayerStatSnapshot>();
  private readonly cfg: RewardConfig;

  constructor(config: Partial<RewardConfig> = {}) {
    this.cfg = { ...DEFAULT_REWARD_CONFIG, ...config };
  }

  public reset(state: MatchState): void {
    this.prevScore = [state.score[0], state.score[1]];
    this.prevViolations = [state.violationsCount[0], state.violationsCount[1]];
    this.prevPotential = [computeTeamPotential(state, 0), computeTeamPotential(state, 1)];
    this.prevPlayerStats.clear();
    for (const p of [...state.players, ...state.bench]) {
      this.prevPlayerStats.set(p.id, {
        points: p.points,
        rebounds: p.rebounds,
        assists: p.assists,
        steals: p.steals,
        blocks: p.blocks,
        fouls: p.fouls,
        ownGoals: p.ownGoals,
      });
    }
  }

  /**
   * Вычисляет награды для всех 10 активных игроков на площадке за прошедший шаг среды.
   */
  public computeStepRewards(state: MatchState, outRewards?: Float32Array): Float32Array {
    const rewards = outRewards ?? new Float32Array(state.players.length);
    const dScore0 = state.score[0] - this.prevScore[0];
    const dScore1 = state.score[1] - this.prevScore[1];
    const dViol0 = state.violationsCount[0] - this.prevViolations[0];
    const dViol1 = state.violationsCount[1] - this.prevViolations[1];

    const nextPhi0 = state.finished ? 0 : computeTeamPotential(state, 0);
    const nextPhi1 = state.finished ? 0 : computeTeamPotential(state, 1);

    const pbrs0 = this.cfg.shapingScale * (this.cfg.gamma * nextPhi0 - this.prevPotential[0]);
    const pbrs1 = this.cfg.shapingScale * (this.cfg.gamma * nextPhi1 - this.prevPotential[1]);

    const teamBase0 =
      (dScore0 - dScore1) * this.cfg.teamPointWeight - dViol0 * this.cfg.teamViolationPenalty + pbrs0;
    const teamBase1 =
      (dScore1 - dScore0) * this.cfg.teamPointWeight - dViol1 * this.cfg.teamViolationPenalty + pbrs1;

    for (let i = 0; i < state.players.length; i++) {
      const p = state.players[i];
      const prev = this.prevPlayerStats.get(p.id);
      let indiv = 0;
      if (prev) {
        indiv += Math.max(0, p.points - prev.points) * this.cfg.individualPointBonus;
        indiv += Math.max(0, p.assists - prev.assists) * this.cfg.assistBonus;
        indiv += Math.max(0, p.rebounds - prev.rebounds) * this.cfg.reboundBonus;
        indiv += Math.max(0, p.steals - prev.steals) * this.cfg.stealBonus;
        indiv += Math.max(0, p.blocks - prev.blocks) * this.cfg.blockBonus;
        const dFouls = Math.max(0, p.fouls - prev.fouls);
        if (dFouls > 0) {
          indiv -= dFouls * this.cfg.personalFoulPenalty;
          if (p.fouls >= 5) indiv -= 0.45;
        }
        const dOwn = Math.max(0, p.ownGoals - prev.ownGoals);
        if (dOwn > 0) {
          indiv -= dOwn * 1.5;
        }
      }

      rewards[i] = (p.team === 0 ? teamBase0 : teamBase1) + indiv;
    }

    // Обновляем снимки для всех игроков (включая тех, кто мог сесть на скамейку в этом тике)
    this.prevScore[0] = state.score[0];
    this.prevScore[1] = state.score[1];
    this.prevViolations[0] = state.violationsCount[0];
    this.prevViolations[1] = state.violationsCount[1];
    this.prevPotential[0] = nextPhi0;
    this.prevPotential[1] = nextPhi1;

    for (const p of [...state.players, ...state.bench]) {
      let snap = this.prevPlayerStats.get(p.id);
      if (!snap) {
        snap = {
          points: p.points,
          rebounds: p.rebounds,
          assists: p.assists,
          steals: p.steals,
          blocks: p.blocks,
          fouls: p.fouls,
          ownGoals: p.ownGoals,
        };
        this.prevPlayerStats.set(p.id, snap);
      } else {
        snap.points = p.points;
        snap.rebounds = p.rebounds;
        snap.assists = p.assists;
        snap.steals = p.steals;
        snap.blocks = p.blocks;
        snap.fouls = p.fouls;
        snap.ownGoals = p.ownGoals;
      }
    }

    return rewards;
  }
}
