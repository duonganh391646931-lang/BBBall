import { stepPlayerAI } from '../ai';
import { CreateMatchOptions, PlayerDecisionDriver, createMatch, stepMatch } from '../sim';
import { MatchModifierId, MatchState, PlayerEntity, TeamTactic } from '../types';
import {
  applyAgentAction,
  computeActionMask,
  extractTeacherMoveVector,
  wrapTeacherCallbacks,
} from './actions';
import { encodeObservation } from './encoder';
import { MatchRewardTracker, RewardConfig } from './rewards';
import {
  AgentAction,
  DiscreteAction,
  NUM_DISCRETE_ACTIONS,
  OBS_DIM,
} from './schema';

export type ControlledTeamsMode = 'both' | 0 | 1;

export interface HeadlessEnvConfig {
  /** Чья команда управляется внешним RL-агентом: 'both' (10 агентов Self-Play) или 0 / 1 (5 агентов против эвристики) */
  controlledTeams?: ControlledTeamsMode;
  /** Число физических тиков (60 Гц) на один шаг принятия решений RL (по умолчанию 4 -> 15 Гц) */
  frameSkip?: number;
  /** Фиксированный шаг физики в секундах (по умолчанию 1 / 60) */
  physicsDt?: number;
  /** Длительность четверти в секундах (для быстрых эпизодов обучения можно ставить 30–45с) */
  quarterDuration?: number;
  /** Максимальное число шагов RL до принудительного усечения эпизода (truncation) */
  maxEpisodeSteps?: number;
  /** Активные модификаторы матча (пусто для чистого баскетбола на ранних стадиях curriculum) */
  modifiers?: MatchModifierId[];
  /** Фиксированные тактики команд или случайный выбор при reset */
  tactics?: [TeamTactic, TeamTactic];
  /** Случайная рандомизация черт/составов при каждом reset (Domain Randomization) */
  randomizeRosters?: boolean;
  /** Случайная рандомизация тактик при reset */
  randomizeTactics?: boolean;
  /** Настройки функции награды */
  rewardConfig?: Partial<RewardConfig>;
  /** Начальное зерно ГПСЧ */
  seed?: number;
}

export interface EnvStepInfo {
  stepCount: number;
  quarter: number;
  quarterTimeRemaining: number;
  score: [number, number];
  possessionTeam: 0 | 1 | null;
  shotClock: number;
  truncated: boolean;
  winner: 0 | 1 | null;
  violations: [number, number];
}

export interface EnvResetResult {
  /** Плоский буфер наблюдений [numAgents * OBS_DIM] */
  obs: Float32Array;
  /** Плоский буфер масок действий [numAgents * NUM_DISCRETE_ACTIONS] */
  masks: Uint8Array;
  info: EnvStepInfo;
}

export interface EnvStepResult extends EnvResetResult {
  /** Награды агентов [numAgents] за прошедшие frameSkip тиков */
  rewards: Float32Array;
  /** Признак завершения матча (или достижения maxEpisodeSteps) */
  done: boolean;
}

export interface TeacherStepRecord extends EnvStepResult {
  /** Записанные эталонные действия эвристического учителя для каждого агента */
  teacherActions: AgentAction[];
}

const ALL_TACTICS: readonly TeamTactic[] = ['balanced', 'blitz', 'perimeter', 'dirty'];

/**
 * Быстрый детерминированный ГПСЧ Mulberry32 для воспроизводимых эпизодов RL.
 */
export function createSeededRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Безголовая (Headless) векторно-совместимая среда «БалбесБола» в стиле Gymnasium / PettingZoo ParallelEnv.
 * Полностью изолирована от DOM, Canvas и WebAudio; не создаёт мусора в куче на горячем цикле шага.
 */
export class HeadlessBasketballEnv {
  public readonly controlledTeams: ControlledTeamsMode;
  public readonly numAgents: number;
  public readonly frameSkip: number;
  public readonly physicsDt: number;
  public quarterDuration: number;
  public readonly maxEpisodeSteps: number;

  private modifiers: MatchModifierId[];
  private baseTactics?: [TeamTactic, TeamTactic];
  private readonly randomizeRosters: boolean;
  private randomizeTactics: boolean;
  private rng: () => number;
  private episodeSeedCounter: number;

  public state!: MatchState;
  private readonly rewardTracker: MatchRewardTracker;

  private readonly obsBuf: Float32Array;
  private readonly maskBuf: Uint8Array;
  private readonly rewardsBuf: Float32Array;
  private readonly allPlayerRewardsBuf = new Float32Array(10);
  private readonly agentPlayers: PlayerEntity[] = [];
  private readonly playerIdToAgentIdx = new Map<string, number>();
  private readonly teacherActionsBuf: AgentAction[];
  private readonly scratchTeacherSingle: AgentAction = { discrete: 0, moveX: 0, moveY: 0 };

  private stepCount = 0;

  constructor(config: HeadlessEnvConfig = {}) {
    this.controlledTeams = config.controlledTeams ?? 'both';
    this.numAgents = this.controlledTeams === 'both' ? 10 : 5;
    this.frameSkip = Math.max(1, config.frameSkip ?? 4);
    this.physicsDt = config.physicsDt ?? 1 / 60;
    this.quarterDuration = config.quarterDuration ?? 45;
    this.maxEpisodeSteps = config.maxEpisodeSteps ?? 4500;
    this.modifiers = config.modifiers ? [...config.modifiers] : [];
    this.baseTactics = config.tactics;
    this.randomizeRosters = config.randomizeRosters ?? true;
    this.randomizeTactics = config.randomizeTactics ?? false;
    this.episodeSeedCounter = (config.seed ?? 42) >>> 0;
    this.rng = createSeededRng(this.episodeSeedCounter);

    this.rewardTracker = new MatchRewardTracker(config.rewardConfig);
    this.obsBuf = new Float32Array(this.numAgents * OBS_DIM);
    this.maskBuf = new Uint8Array(this.numAgents * NUM_DISCRETE_ACTIONS);
    this.rewardsBuf = new Float32Array(this.numAgents);
    this.teacherActionsBuf = Array.from({ length: this.numAgents }, () => ({
      discrete: DiscreteAction.NOOP_MOVE,
      moveX: 0,
      moveY: 0,
    }));

    this.reset(this.episodeSeedCounter);
  }

  public setCurriculum(options: {
    modifiers?: MatchModifierId[];
    quarterDuration?: number;
    randomizeTactics?: boolean;
  }): void {
    if (options.modifiers !== undefined) {
      this.modifiers = [...options.modifiers];
    }
    if (options.quarterDuration !== undefined) {
      this.quarterDuration = options.quarterDuration;
    }
    if (options.randomizeTactics !== undefined) {
      this.randomizeTactics = options.randomizeTactics;
    }
  }

  private refreshAgentOrdering(): void {
    this.agentPlayers.length = 0;
    this.playerIdToAgentIdx.clear();

    const candidates =
      this.controlledTeams === 'both'
        ? [...this.state.players]
        : this.state.players.filter((p) => p.team === this.controlledTeams);

    candidates.sort((a, b) => a.team * 10 + a.slotIndex - (b.team * 10 + b.slotIndex));
    for (let i = 0; i < candidates.length && i < this.numAgents; i++) {
      this.agentPlayers.push(candidates[i]);
      this.playerIdToAgentIdx.set(candidates[i].id, i);
    }
  }

  private fillObsAndMasks(): void {
    this.refreshAgentOrdering();
    for (let i = 0; i < this.agentPlayers.length; i++) {
      const p = this.agentPlayers[i];
      encodeObservation(this.state, p, this.obsBuf, i * OBS_DIM);
      computeActionMask(this.state, p, this.maskBuf, i * NUM_DISCRETE_ACTIONS);
    }
  }

  private buildInfo(truncated: boolean): EnvStepInfo {
    let winner: 0 | 1 | null = null;
    if (this.state.finished) {
      winner = this.state.score[0] >= this.state.score[1] ? 0 : 1;
    }
    return {
      stepCount: this.stepCount,
      quarter: this.state.quarter,
      quarterTimeRemaining: this.state.quarterTimeRemaining,
      score: [this.state.score[0], this.state.score[1]],
      possessionTeam: this.state.possessionTeam,
      shotClock: this.state.shotClock,
      truncated,
      winner,
      violations: [this.state.violationsCount[0], this.state.violationsCount[1]],
    };
  }

  public reset(seed?: number): EnvResetResult {
    if (seed !== undefined) {
      this.episodeSeedCounter = seed >>> 0;
    } else {
      this.episodeSeedCounter = (this.episodeSeedCounter + 1) >>> 0;
    }
    this.rng = createSeededRng(this.episodeSeedCounter);
    this.stepCount = 0;

    const tactics: [TeamTactic, TeamTactic] = this.baseTactics
      ? [this.baseTactics[0], this.baseTactics[1]]
      : this.randomizeTactics
      ? [
          ALL_TACTICS[Math.floor(this.rng() * ALL_TACTICS.length)],
          ALL_TACTICS[Math.floor(this.rng() * ALL_TACTICS.length)],
        ]
      : ['balanced', 'balanced'];

    const teamControllers: CreateMatchOptions['teamControllers'] =
      this.controlledTeams === 'both'
        ? ['external', 'external']
        : this.controlledTeams === 0
        ? ['external', 'heuristic']
        : ['heuristic', 'external'];

    const opts: CreateMatchOptions = {
      mode: 'exhibition',
      quarterDuration: this.quarterDuration,
      modifiers: this.modifiers,
      tactics,
      teamControllers,
      headless: true,
      rng: this.rng,
    };

    if (!this.randomizeRosters && this.state) {
      // Переиспользуем сгенерированные ростеры, если рандомизация отключена
      opts.rosterA = [...this.state.players.filter((p) => p.team === 0), ...this.state.bench.filter((p) => p.team === 0)];
      opts.rosterB = [...this.state.players.filter((p) => p.team === 1), ...this.state.bench.filter((p) => p.team === 1)];
    }

    this.state = createMatch(opts);
    this.rewardTracker.reset(this.state);
    this.fillObsAndMasks();

    return {
      obs: this.obsBuf,
      masks: this.maskBuf,
      info: this.buildInfo(false),
    };
  }

  /**
   * Выполняет шаг среды (frameSkip тиков физики) с действиями внешних RL-агентов.
   */
  public step(actions: readonly AgentAction[]): EnvStepResult {
    this.stepCount++;
    this.refreshAgentOrdering();

    for (let sub = 0; sub < this.frameSkip; sub++) {
      if (this.state.finished) break;

      const isFirstSubStep = sub === 0;
      const driver: PlayerDecisionDriver = (st, player, dt, stepRng, callbacks) => {
        const agentIdx = this.playerIdToAgentIdx.get(player.id);
        if (agentIdx !== undefined) {
          if (isFirstSubStep) {
            const act = actions[agentIdx];
            if (act) {
              applyAgentAction(
                st,
                player,
                act,
                this.maskBuf,
                callbacks,
                agentIdx * NUM_DISCRETE_ACTIONS
              );
            }
          }
          // На промежуточных тиках (sub > 0) агент продолжает движение к выбранному (targetX, targetY)
        } else {
          stepPlayerAI(st, player, dt, stepRng, callbacks);
        }
      };

      stepMatch(this.state, this.physicsDt, this.rng, driver);
    }

    // Вычисляем суммарную награду за весь макро-шаг (frameSkip тиков)
    this.rewardTracker.computeStepRewards(this.state, this.allPlayerRewardsBuf);
    this.refreshAgentOrdering();
    for (let i = 0; i < this.numAgents; i++) {
      const p = this.agentPlayers[i];
      const pIdx = p ? this.state.players.indexOf(p) : -1;
      this.rewardsBuf[i] = pIdx >= 0 ? this.allPlayerRewardsBuf[pIdx] : 0;
    }

    this.fillObsAndMasks();

    const truncated = !this.state.finished && this.stepCount >= this.maxEpisodeSteps;
    const done = this.state.finished || truncated;

    return {
      obs: this.obsBuf,
      masks: this.maskBuf,
      rewards: this.rewardsBuf,
      done,
      info: this.buildInfo(truncated),
    };
  }

  /**
   * Шаг сбора траекторий эвристического учителя для имитационного обучения (Behavior Cloning).
   * Возвращает действия учителя, согласованные с текущими obs и masks до начала шага!
   */
  public stepTeacher(): TeacherStepRecord {
    this.stepCount++;
    this.refreshAgentOrdering();

    for (let i = 0; i < this.numAgents; i++) {
      this.teacherActionsBuf[i].discrete = DiscreteAction.NOOP_MOVE;
      this.teacherActionsBuf[i].moveX = 0;
      this.teacherActionsBuf[i].moveY = 0;
    }

    for (let sub = 0; sub < this.frameSkip; sub++) {
      if (this.state.finished) break;

      const isFirstSubStep = sub === 0;
      const driver: PlayerDecisionDriver = (st, player, dt, stepRng, callbacks) => {
        const agentIdx = this.playerIdToAgentIdx.get(player.id);
        if (agentIdx !== undefined) {
          const wrapped = wrapTeacherCallbacks(
            st,
            player,
            callbacks,
            this.scratchTeacherSingle
          );
          stepPlayerAI(st, player, dt, stepRng, wrapped);

          const outAct = this.teacherActionsBuf[agentIdx];
          if (isFirstSubStep) {
            extractTeacherMoveVector(player, outAct);
          }
          const candDisc = this.scratchTeacherSingle.discrete;
          if (
            outAct.discrete === DiscreteAction.NOOP_MOVE &&
            candDisc !== DiscreteAction.NOOP_MOVE &&
            this.maskBuf[agentIdx * NUM_DISCRETE_ACTIONS + candDisc] === 1
          ) {
            outAct.discrete = candDisc;
          }
        } else {
          stepPlayerAI(st, player, dt, stepRng, callbacks);
        }
      };

      stepMatch(this.state, this.physicsDt, this.rng, driver);
    }

    this.rewardTracker.computeStepRewards(this.state, this.allPlayerRewardsBuf);
    this.refreshAgentOrdering();
    for (let i = 0; i < this.numAgents; i++) {
      const p = this.agentPlayers[i];
      const pIdx = p ? this.state.players.indexOf(p) : -1;
      this.rewardsBuf[i] = pIdx >= 0 ? this.allPlayerRewardsBuf[pIdx] : 0;
    }

    this.fillObsAndMasks();

    const truncated = !this.state.finished && this.stepCount >= this.maxEpisodeSteps;
    const done = this.state.finished || truncated;

    return {
      obs: this.obsBuf,
      masks: this.maskBuf,
      rewards: this.rewardsBuf,
      done,
      info: this.buildInfo(truncated),
      teacherActions: this.teacherActionsBuf,
    };
  }
}
