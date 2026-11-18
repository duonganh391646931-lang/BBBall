export type TraitRarity = 'common' | 'rare' | 'unique';

export interface PlayerStats {
  speed: number;      // Скорость перемещения (8..100)
  shooting: number;   // Точность броска (8..100)
  stamina: number;    // Выносливость (8..100)
  steal: number;      // Перехват мяча (8..100)
  jump: number;       // Высота прыжка (8..100)
  teamwork: number;   // Командная игра (0..100)
  dribble: number;    // Дриблинг / контроль мяча (8..100)
  iq: number;         // Баскетбольный ИИ / мозжечок (8..100)
  deviance: number;   // Девиантность / грязная игра (0..100)
}

export interface TraitDefinition {
  id: string;
  name: string;
  rarity: TraitRarity;
  rarityLabel: string;
  weight: number;
  description: string;
  shortTag: string;
  modifyStats?: (stats: PlayerStats) => PlayerStats;
}

export type PositionRole = 'PG' | 'SG' | 'SF' | 'PF' | 'C';

export const POSITION_LABELS: Record<PositionRole, string> = {
  PG: 'РЗ',
  SG: 'АЗ',
  SF: 'ЛФ',
  PF: 'ТФ',
  C: 'Ц',
};

export type HairStyle = 'afro' | 'mohawk' | 'headband' | 'spiky' | 'bald' | 'curly';

export type ShotStyle = 'standard' | 'floater' | 'hook';

export type AIActionState =
  | 'tipoff'
  | 'inbound_pass'
  | 'inbound_wait'
  | 'free_throw'
  | 'drive'
  | 'pump_fake'
  | 'ankle_breaker'
  | 'step_back'
  | 'eurostep'
  | 'floater'
  | 'post_up'
  | 'handoff'
  | 'shoot_windup'
  | 'messi_windup'
  | 'pass'
  | 'receive_pass'
  | 'cut_open'
  | 'relocate_arc'
  | 'backdoor_cut'
  | 'give_and_go'
  | 'screen_set'
  | 'offball_screen'
  | 'roll_to_rim'
  | 'pick_and_pop'
  | 'fast_break'
  | 'rebound_position'
  | 'safety_back'
  | 'chase_ball'
  | 'guard_carrier'
  | 'closeout'
  | 'take_charge'
  | 'chasedown_block'
  | 'intentional_foul'
  | 'help_defense'
  | 'mark_man'
  | 'rim_protect'
  | 'intercept_air'
  | 'bread_mode'
  | 'dirty_play'
  | 'tripped_down'
  | 'penalty_bench'
  | 'team_bench'
  | 'celebrate';

export interface PlayerEntity {
  id: string;
  name: string;
  jerseyNumber: number;
  team: 0 | 1; // 0 = Хозяева (атакуют ПРАВОЕ кольцо x=1030, защищают ЛЕВОЕ x=90), 1 = Гости (атакуют ЛЕВОЕ x=90)
  slotIndex: number; // 0..4
  role: PositionRole;

  // Appearance
  skinTone: string;
  hairColor: string;
  hairStyle: HairStyle;
  bodyScale: number;

  // Stats & Traits
  baseStats: PlayerStats;
  effectiveStats: PlayerStats;
  traitIds: string[];

  // Spatial 2.5D coordinates (strictly rendered top-down: x, y on floor, z = altitude)
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  heading: number; // radians, direction player faces

  // Dynamic gameplay state
  energy: number; // 0..100
  aiAction: AIActionState;
  aiThought: string;
  targetX: number;
  targetY: number;
  decisionTimer: number;
  actionCooldown: number;
  stealCooldown: number;
  pickupLockout: number;
  windupTimer: number;
  windupTargetHoop: 0 | 1 | null;
  windupBallId: number | null;
  windupShotStyle?: ShotStyle;
  schizoTimer: number;
  schizoBreadMode: boolean;
  cutTimer: number;
  screenTargetId: string | null;

  // Balanced cooldowns, basketball rules & foul/bench state
  superJumpCd: number;         // Кулдаун супер-прыжка Кузнечика (3.5с)
  dirtyTrickCd: number;        // Кулдаун пакостей (6с)
  refSuspicion: number;        // Накопленная подозрительность судьи за четверть (0..5)
  trippedTimer: number;        // Валяется после подножки
  penaltyTimer: number;        // Отбывает 8с на скамейке штрафников
  fouls: number;               // Персональные фолы (при 5 заменяется на запасного)
  isReserveSub: boolean;
  onCourt: boolean;            // true = играет в пятёрке на паркете, false = сидит на скамейке запасных
  startedOnBench: boolean;     // Начинал ли матч в статусе запасного игрока (условие «Последней надежды»)
  subbedInOnce: boolean;       // Выходил ли со скамейки на игровое поле в этом матче
  lastHopeAnnounced?: boolean; // Был ли уже показан спецэффект выхода «Последней надежды»
  benchSlot: number;           // Место на командной скамейке запасных (0..2)
  disqualified?: boolean;      // Удалён до конца матча за 5 персональных фолов
  paintTimer: number;          // Таймер правила 3 секунд в чужой краске
  closelyGuardedTimer: number; // Таймер правила 5 секунд под плотной опекой без движения
  airborneWithBall: boolean;   // Прыгнул с мячом в руках (приземление с мячом = пробежка!)
  caughtInAir?: boolean;       // Поймал мяч уже находясь в воздухе (приземление разрешено правилами!)
  hasStoppedDribble: boolean;  // Остановил ведение (повторный старт ведения = двойное ведение!)
  dribbleLiveTimer?: number;   // Сколько секунд игрок уже вёл мяч в текущем касании
  prevStepX?: number;          // Координата до шага физики (для расчёта реального смещения в борьбе)
  prevStepY?: number;

  // Память выбора стороны обхода (гистерезис против кручения между равными путями)
  dodgeSide: -1 | 1;
  dodgeCommitTimer: number;
  driveCommitX?: number;
  driveCommitY?: number;
  driveLockTimer?: number;
  markTargetId?: string | null;

  // Продвинутый ИИ балбесов (ложный замах, кроссовер, степ-бэк, евростеп, подстановка под фол, горячая рука и симуляция)
  pumpFakeTimer?: number;
  pumpFakeCd?: number;
  bittenByFakeTimer?: number;
  crossoverCd?: number;
  stepBackCd?: number;
  eurostepCd?: number;
  chargePlantTimer?: number;
  handoffHoldTimer?: number;
  handoffCd?: number;
  saveOobCd?: number;
  ballWatchTimer?: number;
  popSpotX?: number;
  popSpotY?: number;
  hotStreak: number; // >= 3 = «В ОГНЕ!», <= -3 = «Холодные руки»
  flopCd?: number;   // Кулдаун актёрской симуляции («Актёр Оскара»)

  // Animation
  stridePhase: number;

  // Match box-score stats
  points: number;
  rebounds: number;
  fgMade: number;
  fgAttempted: number;
  threeMade: number;
  threeAttempted: number;
  freeThrowsMade: number;
  ownGoals: number;
  assists: number;
  steals: number;
  blocks: number;
  dunks: number;
  dirtyPlays: number;
}

export type ShotType = 'none' | 'tipoff' | 'normal' | 'messi' | 'dunk' | 'pass' | 'panic' | 'kicked';

export type CoachPlayCall = 'iso_star' | 'pick_and_roll' | 'alley_oop' | 'feed_sniper';

export interface ActivePlayCallState {
  team: 0 | 1;
  play: CoachPlayCall;
  timer: number;
}

export interface ShotChartEvent {
  x: number;
  y: number;
  team: 0 | 1;
  made: boolean;
  isThree: boolean;
  shooterName: string;
  quarter: number;
}

export interface BallTrailPoint {
  x: number;
  y: number;
  z: number;
  alpha: number;
  isMessi: boolean;
}

export interface BallState {
  id: number;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  holderId: string | null;
  lastTouchTeam: 0 | 1 | null;
  lastShooterId: string | null;
  lastPasserId: string | null;
  passReceiverId: string | null;
  shotType: ShotType;
  shotStyle?: ShotStyle;
  targetHoop: 0 | 1 | null;
  shotOriginX: number;
  shotOriginY: number;
  isGuaranteedSwish: boolean;
  isTipoffBall: boolean;
  wasShotAttempt?: boolean;
  shotChartRecorded?: boolean;
  rimBounceCount: number;
  spinAngle: number;
  trail: BallTrailPoint[];
  trailHead: number;
}

export interface RefereeEntity {
  x: number;
  y: number;
  vx: number;
  vy: number;
  gazeAngle: number; // куда смотрит судья (спереди 80%, сзади 50%)
  whistleTimer: number;
  lastCallText: string | null;
  callsMade: number;
  missedFouls: number;
}

export type MatchModifierId =
  | 'two_balls'
  | 'ice_court'
  | 'low_gravity'
  | 'paint_trampolines'
  | 'explosive_ball';

export interface MatchModifierDef {
  id: MatchModifierId;
  name: string;
  shortLabel: string;
  description: string;
}

export type TeamTactic = 'balanced' | 'blitz' | 'perimeter' | 'dirty';

export interface PooledParticle {
  active: boolean;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  color: string;
  size: number;
  age: number;
  maxAge: number;
}

export interface TacticalCallout {
  active: boolean;
  x: number;
  y: number;
  text: string;
  color: string;
  age: number;
  duration: number;
}

export interface PlayEvent {
  id: number;
  quarterLabel: string;
  timeFormatted: string;
  text: string;
  team: 0 | 1 | 'ref' | 'neutral';
  highlight: boolean;
}

export interface MatchSummarySnapshot {
  teamNames: [string, string];
  teamColors: [string, string];
  score: [number, number];
  quarterScores: Array<[number, number]>;
  winner: 0 | 1;
  mvpName: string;
  mvpTeam: 0 | 1;
  mvpPoints: number;
  mvpRebounds: number;
  mvpAssists: number;
  mvpTraits: string[];
  totalSteals: number;
  totalBlocks: number;
  totalFouls: [number, number];
  refCalls: number;
  refMissed: number;
  messiGoals: number;
  ownGoals: number;
}

export interface ClubTeam {
  id: string;
  name: string;
  primaryColor: string;
  secondaryColor: string;
  isUserClub: boolean;
  roster: PlayerEntity[];
}

export interface BracketMatchup {
  id: string;
  round: 0 | 1 | 2; // 0 = 1/4 финала, 1 = 1/2 финала, 2 = Финал
  teamA: ClubTeam | null;
  teamB: ClubTeam | null;
  scoreA: number | null;
  scoreB: number | null;
  winnerId: string | null;
}

export interface TournamentState {
  active: boolean;
  phase: 'draft' | 'bracket' | 'reinforce' | 'champion';
  userClub: ClubTeam;
  aiClubs: ClubTeam[];
  draftPool: PlayerEntity[];
  reinforcePool: PlayerEntity[];
  currentRound: 0 | 1 | 2;
  matches: BracketMatchup[];
  championClub: ClubTeam | null;
}

export interface InboundPlayState {
  team: 0 | 1;
  ballId: number;
  passerId: string;
  x: number;
  y: number;
  timer: number; // 5 секунд на ввод мяча из-за линии!
  reason: string;
  isAfterScore?: boolean; // После забитого мяча разрешено двигаться вдоль лицевой, а соперник садится в защиту
}

export interface FreeThrowSequenceState {
  shooterId: string;
  shootingTeam: 0 | 1;
  targetHoop: 0 | 1;
  ballId: number;
  shotsRemaining: number;
  totalShots: number;
  phase: 'lineup' | 'shooting' | 'in_flight';
  phaseTimer: number;
  retainPossessionTeam: 0 | 1 | null; // при неспортивном/техническом фоле команда сохраняет владение после штрафных!
}

export interface MatchState {
  mode: 'exhibition' | 'tournament';
  teamNames: [string, string];
  teamColors: [string, string];
  players: PlayerEntity[];     // 10 активных игроков на площадке (по 5 от каждой команды)
  bench: PlayerEntity[];       // Запасные игроки на скамейках обеих команд (по 3 на команду)
  subCooldown: [number, number]; // Кулдаун плановых тренерских замен для каждой команды
  substitutionsCount: number;  // Общее число проведённых замен за матч
  timeoutsRemaining: [number, number]; // Остаток тренерских тайм-аутов у команд
  scoringRun: { team: 0 | 1 | null; points: number }; // Безответный рывок очков подряд
  activePlayCall: ActivePlayCallState | null; // Активная комбинация с планшета тренера
  playCallCooldown: [number, number]; // Перезарядка заказа комбинации тренером
  shotChart: ShotChartEvent[]; // Карта точек бросков за матч
  balls: BallState[];
  referee: RefereeEntity;
  score: [number, number];
  quarter: number; // 1..4, 5+ = OT
  quarterDuration: number; // seconds per quarter
  quarterTimeRemaining: number;
  quarterScores: Array<[number, number]>;
  shotClock: number;
  teamFouls: [number, number]; // командные фолы в текущей четверти (>=5 = пробивные)
  violationsCount: [number, number]; // нарушения правил (аут, пробежка, зона, 8с, 3с, 5с)
  possessionTeam: 0 | 1 | null;
  possessionArrow: 0 | 1; // стрелка поочерёдного владения FIBA при спорном мяче
  possessionTimer: number; // сколько секунд длится текущее владение (для быстрого отрыва)
  frontcourtEstablished: boolean; // выведен ли мяч в передовую зону (для правила зоны и 8 секунд)
  backcourtTimer: number; // отсчёт 8 секунд на вывод мяча со своей половины
  shotTouchedRim: boolean; // коснулся ли бросок дужки кольца (для сброса до 14с на подборе в атаке)
  shotClockExpiredInFlight: boolean; // истекли ли 24с во время полёта мяча к кольцу
  heldBallTieTimer: number; // таймер взаимного зажима мяча (спорный мяч в борьбе)
  inbound: InboundPlayState | null; // активный ввод мяча из-за боковой/лицевой линии
  freeThrow: FreeThrowSequenceState | null; // активная серия штрафных бросков
  aiCoachCooldown: number; // кулдаун адаптации тактики ИИ-тренером соперника
  tactics: [TeamTactic, TeamTactic];
  modifiers: MatchModifierId[];

  celebrationTimer: number;
  quarterBreakTimer: number;
  lastBannerText: string | null;
  lastBannerColor: string;
  hitstopTimer: number;
  cameraShake: number;
  netSwayLeft: number;
  netSwayRight: number;

  messiGoalsCount: number;
  ownGoalsCount: number;
  finished: boolean;
  summary: MatchSummarySnapshot | null;

  selectedPlayerId: string | null;

  // Zero-GC pools
  particles: PooledParticle[];
  particleCursor: number;
  callouts: TacticalCallout[];
  calloutCursor: number;

  events: PlayEvent[];
  nextEventId: number;
}

export type GameScreenState = 'title' | 'playing' | 'paused' | 'gameover';
