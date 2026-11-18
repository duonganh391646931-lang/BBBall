export const OBS_DIM = 244;
export const NUM_DISCRETE_ACTIONS = 23;
export const CONT_ACTION_DIM = 2;
export const MOVE_TARGET_RADIUS = 140;

export const DiscreteAction = {
  NOOP_MOVE: 0,
  SHOOT_STANDARD: 1,
  SHOOT_FLOATER: 2,
  SHOOT_HOOK: 3,
  DUNK: 4,
  PASS_MATE_0: 5,
  PASS_MATE_1: 6,
  PASS_MATE_2: 7,
  PASS_MATE_3: 8,
  ALLEY_OOP_MATE_0: 9,
  ALLEY_OOP_MATE_1: 10,
  ALLEY_OOP_MATE_2: 11,
  ALLEY_OOP_MATE_3: 12,
  PUMP_FAKE: 13,
  STEP_BACK: 14,
  EUROSTEP: 15,
  CROSSOVER: 16,
  SCREEN_SET: 17,
  TAKE_CHARGE: 18,
  JUMP_CONTEST_REBOUND: 19,
  STEAL_PRESS: 20,
  DIRTY_TRICK: 21,
  INTENTIONAL_FOUL: 22,
} as const;

export type DiscreteActionId = (typeof DiscreteAction)[keyof typeof DiscreteAction];

export const DISCRETE_ACTION_NAMES: readonly string[] = [
  'NOOP_MOVE',
  'SHOOT_STANDARD',
  'SHOOT_FLOATER',
  'SHOOT_HOOK',
  'DUNK',
  'PASS_MATE_0',
  'PASS_MATE_1',
  'PASS_MATE_2',
  'PASS_MATE_3',
  'ALLEY_OOP_MATE_0',
  'ALLEY_OOP_MATE_1',
  'ALLEY_OOP_MATE_2',
  'ALLEY_OOP_MATE_3',
  'PUMP_FAKE',
  'STEP_BACK',
  'EUROSTEP',
  'CROSSOVER',
  'SCREEN_SET',
  'TAKE_CHARGE',
  'JUMP_CONTEST_REBOUND',
  'STEAL_PRESS',
  'DIRTY_TRICK',
  'INTENTIONAL_FOUL',
];

export interface AgentAction {
  /** Дискретное тактическое действие из DiscreteAction (0..22) */
  discrete: number;
  /** Продольное смещение цели в канонической системе команды [-1..1] (+1 = к чужому кольцу) */
  moveX: number;
  /** Поперечное смещение цели [-1..1] (-1 = к верхнему борту, +1 = к нижнему борту) */
  moveY: number;
}

export interface LinearLayerWeights {
  inFeatures: number;
  outFeatures: number;
  /** Плоский массив весов [outFeatures * inFeatures] (row-major, как в PyTorch nn.Linear.weight) */
  weight: number[];
  /** Вектор смещений [outFeatures] */
  bias: number[];
}

export interface LayerNormWeights {
  features: number;
  weight: number[];
  bias: number[];
  eps?: number;
}

export interface ResidualBlockWeights {
  fc1: LinearLayerWeights;
  ln1: LayerNormWeights;
  fc2: LinearLayerWeights;
  ln2: LayerNormWeights;
}

export interface PolicyWeightsJSON {
  formatVersion: 1;
  modelName: string;
  trainedSteps: number;
  obsDim: number;
  discreteActions: number;
  contActionDim: number;
  hiddenDim: number;
  inputLinear: LinearLayerWeights;
  inputNorm: LayerNormWeights;
  blocks: ResidualBlockWeights[];
  discreteHead: LinearLayerWeights;
  moveMeanHead: LinearLayerWeights;
}
