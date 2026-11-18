import {
  AgentAction,
  CONT_ACTION_DIM,
  DiscreteAction,
  LayerNormWeights,
  LinearLayerWeights,
  NUM_DISCRETE_ACTIONS,
  OBS_DIM,
  PolicyWeightsJSON,
  ResidualBlockWeights,
} from './schema';
import { DEFAULT_EMBEDDED_POLICY_WEIGHTS } from './defaultTrainedWeights';

interface CompiledLinear {
  inFeatures: number;
  outFeatures: number;
  weight: Float32Array;
  bias: Float32Array;
}

interface CompiledLayerNorm {
  features: number;
  weight: Float32Array;
  bias: Float32Array;
  eps: number;
}

interface CompiledResBlock {
  fc1: CompiledLinear;
  ln1: CompiledLayerNorm;
  fc2: CompiledLinear;
  ln2: CompiledLayerNorm;
}

function compileLinear(src: LinearLayerWeights): CompiledLinear {
  if (src.weight.length !== src.inFeatures * src.outFeatures) {
    throw new Error(
      `Linear weight size mismatch: got ${src.weight.length}, expected ${src.outFeatures}x${src.inFeatures}`
    );
  }
  if (src.bias.length !== src.outFeatures) {
    throw new Error(
      `Linear bias size mismatch: got ${src.bias.length}, expected ${src.outFeatures}`
    );
  }
  return {
    inFeatures: src.inFeatures,
    outFeatures: src.outFeatures,
    weight: new Float32Array(src.weight),
    bias: new Float32Array(src.bias),
  };
}

function compileLayerNorm(src: LayerNormWeights): CompiledLayerNorm {
  return {
    features: src.features,
    weight: new Float32Array(src.weight),
    bias: new Float32Array(src.bias),
    eps: src.eps ?? 1e-5,
  };
}

function siluInPlace(vec: Float32Array): void {
  for (let i = 0; i < vec.length; i++) {
    const x = vec[i];
    vec[i] = x / (1 + Math.exp(-x));
  }
}

function linearForward(layer: CompiledLinear, input: Float32Array, out: Float32Array): void {
  const { inFeatures, outFeatures, weight, bias } = layer;
  for (let o = 0; o < outFeatures; o++) {
    let sum = bias[o];
    const rowOffset = o * inFeatures;
    for (let i = 0; i < inFeatures; i++) {
      sum += weight[rowOffset + i] * input[i];
    }
    out[o] = sum;
  }
}

function layerNormInPlace(ln: CompiledLayerNorm, vec: Float32Array): void {
  const n = ln.features;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += vec[i];
  mean /= n;

  let varSum = 0;
  for (let i = 0; i < n; i++) {
    const d = vec[i] - mean;
    varSum += d * d;
  }
  const invStd = 1 / Math.sqrt(varSum / n + ln.eps);
  for (let i = 0; i < n; i++) {
    vec[i] = (vec[i] - mean) * invStd * ln.weight[i] + ln.bias[i];
  }
}

/**
 * Вычисляет температуру выбора действия по характеристике «ИИ» (iq: 8..100):
 *  - Звезда (iq = 95) -> tau = 0.25 (почти жадный гроссмейстерский выбор лучшего хода)
 *  - Середняк (iq = 55) -> tau = 0.90 (живая вариативная игра)
 *  - Балбес (iq = 15) -> tau = 1.52 (широкий разброс решений и органичные чудачества в рамках правил)
 */
export function computeIqTemperature(iq: number): number {
  const norm = Math.max(0, Math.min(1.15, iq / 100));
  return Math.max(0.22, Math.min(1.65, 1.75 - 1.52 * norm));
}

/**
 * Быстрый zero-GC движок инференса нейросети в браузере и Node.js на чистых Float32Array.
 * Послойно идентичен PyTorch-модели BasketballActorCritic в training/model.py (точность паритета < 1e-5).
 */
export class NeuralPolicy {
  public readonly modelName: string;
  public readonly trainedSteps: number;
  public readonly hiddenDim: number;

  private readonly inputLinear: CompiledLinear;
  private readonly inputNorm: CompiledLayerNorm;
  private readonly blocks: CompiledResBlock[];
  private readonly discreteHead: CompiledLinear;
  private readonly moveMeanHead: CompiledLinear;

  // Переиспользуемые буферы без аллокаций в игровом цикле 60 FPS
  private readonly bufX: Float32Array;
  private readonly bufH1: Float32Array;
  private readonly bufH2: Float32Array;
  private readonly bufLogits: Float32Array;
  private readonly bufProbs: Float32Array;
  private readonly bufMove: Float32Array;

  constructor(json: PolicyWeightsJSON) {
    if (json.obsDim !== OBS_DIM) {
      throw new Error(`Unsupported obsDim=${json.obsDim}, expected ${OBS_DIM}`);
    }
    if (json.discreteActions !== NUM_DISCRETE_ACTIONS) {
      throw new Error(
        `Unsupported discreteActions=${json.discreteActions}, expected ${NUM_DISCRETE_ACTIONS}`
      );
    }
    if (json.contActionDim !== CONT_ACTION_DIM) {
      throw new Error(`Unsupported contActionDim=${json.contActionDim}, expected ${CONT_ACTION_DIM}`);
    }

    this.modelName = json.modelName;
    this.trainedSteps = json.trainedSteps;
    this.hiddenDim = json.hiddenDim;

    this.inputLinear = compileLinear(json.inputLinear);
    this.inputNorm = compileLayerNorm(json.inputNorm);
    this.blocks = json.blocks.map((b: ResidualBlockWeights) => ({
      fc1: compileLinear(b.fc1),
      ln1: compileLayerNorm(b.ln1),
      fc2: compileLinear(b.fc2),
      ln2: compileLayerNorm(b.ln2),
    }));
    this.discreteHead = compileLinear(json.discreteHead);
    this.moveMeanHead = compileLinear(json.moveMeanHead);

    this.bufX = new Float32Array(this.hiddenDim);
    this.bufH1 = new Float32Array(this.hiddenDim);
    this.bufH2 = new Float32Array(this.hiddenDim);
    this.bufLogits = new Float32Array(NUM_DISCRETE_ACTIONS);
    this.bufProbs = new Float32Array(NUM_DISCRETE_ACTIONS);
    this.bufMove = new Float32Array(CONT_ACTION_DIM);
  }

  /**
   * Возвращает сырые логиты и вектор движения (для автоматического теста паритета PyTorch <-> TS до 1e-5).
   */
  public forwardRaw(obs: Float32Array): { logits: Float32Array; moveMean: Float32Array } {
    linearForward(this.inputLinear, obs, this.bufX);
    layerNormInPlace(this.inputNorm, this.bufX);
    siluInPlace(this.bufX);

    for (let b = 0; b < this.blocks.length; b++) {
      const blk = this.blocks[b];
      linearForward(blk.fc1, this.bufX, this.bufH1);
      layerNormInPlace(blk.ln1, this.bufH1);
      siluInPlace(this.bufH1);

      linearForward(blk.fc2, this.bufH1, this.bufH2);
      layerNormInPlace(blk.ln2, this.bufH2);

      for (let i = 0; i < this.hiddenDim; i++) {
        const s = this.bufX[i] + this.bufH2[i];
        this.bufX[i] = s / (1 + Math.exp(-s));
      }
    }

    linearForward(this.discreteHead, this.bufX, this.bufLogits);
    linearForward(this.moveMeanHead, this.bufX, this.bufMove);
    this.bufMove[0] = Math.tanh(this.bufMove[0]);
    this.bufMove[1] = Math.tanh(this.bufMove[1]);
    return { logits: this.bufLogits, moveMean: this.bufMove };
  }

  /**
   * Прямой проход актора с масштабированием температурой стата «ИИ» и последующим наложением маски правил.
   */
  public act(
    obs: Float32Array,
    mask: Uint8Array,
    deterministic = true,
    rng: () => number = Math.random,
    out?: AgentAction,
    temperature = 1.0
  ): AgentAction {
    this.forwardRaw(obs);

    let chosenDiscrete: number = DiscreteAction.NOOP_MOVE;
    if (deterministic || temperature <= 0.05) {
      let bestLogit = -Infinity;
      for (let a = 0; a < NUM_DISCRETE_ACTIONS; a++) {
        if (mask[a] !== 0 && this.bufLogits[a] > bestLogit) {
          bestLogit = this.bufLogits[a];
          chosenDiscrete = a;
        }
      }
    } else {
      const invTemp = 1 / Math.max(0.1, temperature);
      let maxScaled = -Infinity;
      for (let a = 0; a < NUM_DISCRETE_ACTIONS; a++) {
        if (mask[a] !== 0) {
          const scaled = this.bufLogits[a] * invTemp;
          if (scaled > maxScaled) maxScaled = scaled;
        }
      }
      let sumExp = 0;
      for (let a = 0; a < NUM_DISCRETE_ACTIONS; a++) {
        if (mask[a] !== 0) {
          const e = Math.exp(this.bufLogits[a] * invTemp - maxScaled);
          this.bufProbs[a] = e;
          sumExp += e;
        } else {
          this.bufProbs[a] = 0;
        }
      }
      if (sumExp > 0) {
        let r = rng() * sumExp;
        for (let a = 0; a < NUM_DISCRETE_ACTIONS; a++) {
          if (mask[a] === 0) continue;
          r -= this.bufProbs[a];
          if (r <= 0) {
            chosenDiscrete = a;
            break;
          }
        }
      }
    }

    const moveX = this.bufMove[0];
    const moveY = this.bufMove[1];

    if (out) {
      out.discrete = chosenDiscrete;
      out.moveX = moveX;
      out.moveY = moveY;
      return out;
    }
    return { discrete: chosenDiscrete, moveX, moveY };
  }
}

export function createUntrainedPolicyWeights(
  hiddenDim = 192,
  numBlocks = 2,
  seed = 2026
): PolicyWeightsJSON {
  let s = seed >>> 0;
  const nextRand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
  const randNormal = (scale: number) => {
    const u1 = Math.max(1e-7, nextRand());
    const u2 = nextRand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
  };
  const makeLinear = (inF: number, outF: number, gain = 1.0): LinearLayerWeights => {
    const std = gain * Math.sqrt(2 / (inF + outF));
    const weight = new Array<number>(outF * inF);
    for (let i = 0; i < weight.length; i++) weight[i] = Number(randNormal(std).toFixed(6));
    const bias = new Array<number>(outF).fill(0);
    return { inFeatures: inF, outFeatures: outF, weight, bias };
  };
  const makeLN = (features: number): LayerNormWeights => ({
    features,
    weight: new Array<number>(features).fill(1),
    bias: new Array<number>(features).fill(0),
    eps: 1e-5,
  });

  const blocks: ResidualBlockWeights[] = Array.from({ length: numBlocks }, () => ({
    fc1: makeLinear(hiddenDim, hiddenDim, 1.0),
    ln1: makeLN(hiddenDim),
    fc2: makeLinear(hiddenDim, hiddenDim, 0.25),
    ln2: makeLN(hiddenDim),
  }));

  const discreteHead = makeLinear(hiddenDim, NUM_DISCRETE_ACTIONS, 0.05);
  discreteHead.bias[DiscreteAction.NOOP_MOVE] = 1.0;

  const moveMeanHead = makeLinear(hiddenDim, CONT_ACTION_DIM, 0.25);
  moveMeanHead.bias[0] = 0.25;

  return {
    formatVersion: 1,
    modelName: `init-${hiddenDim}x${numBlocks}`,
    trainedSteps: 0,
    obsDim: OBS_DIM,
    discreteActions: NUM_DISCRETE_ACTIONS,
    contActionDim: CONT_ACTION_DIM,
    hiddenDim,
    inputLinear: makeLinear(OBS_DIM, hiddenDim, 1.0),
    inputNorm: makeLN(hiddenDim),
    blocks,
    discreteHead,
    moveMeanHead,
  };
}

let activeNeuralPolicy: NeuralPolicy | null = null;

export function getActiveNeuralPolicy(): NeuralPolicy {
  if (!activeNeuralPolicy) {
    if (
      DEFAULT_EMBEDDED_POLICY_WEIGHTS &&
      DEFAULT_EMBEDDED_POLICY_WEIGHTS.obsDim === OBS_DIM &&
      DEFAULT_EMBEDDED_POLICY_WEIGHTS.discreteActions === NUM_DISCRETE_ACTIONS
    ) {
      activeNeuralPolicy = new NeuralPolicy(DEFAULT_EMBEDDED_POLICY_WEIGHTS);
    } else {
      activeNeuralPolicy = new NeuralPolicy(createUntrainedPolicyWeights(192, 2, 2026));
    }
  }
  return activeNeuralPolicy;
}

export function setActiveNeuralPolicy(policy: NeuralPolicy): void {
  activeNeuralPolicy = policy;
}

export function loadActiveNeuralPolicyFromJson(json: PolicyWeightsJSON): NeuralPolicy {
  const compiled = new NeuralPolicy(json);
  activeNeuralPolicy = compiled;
  return compiled;
}
