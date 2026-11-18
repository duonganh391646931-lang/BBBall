import { AIDecisionCallbacks } from '../ai';
import {
  COURT_BOTTOM,
  COURT_LEFT,
  COURT_RIGHT,
  COURT_TOP,
  MID_X,
  getAttackingHoop,
  getDefendingHoop,
  getHoopPosition,
  isInFrontcourt,
} from '../physics';
import { MatchState, PlayerEntity, ShotStyle } from '../types';
import { fromCanonicalDx, getOrderedOpponents, getOrderedTeammates, toCanonicalVx } from './encoder';
import {
  AgentAction,
  DiscreteAction,
  MOVE_TARGET_RADIUS,
  NUM_DISCRETE_ACTIONS,
} from './schema';

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Вычисляет бинарную маску допустимых дискретных действий (1 = разрешено, 0 = запрещено правилами/состоянием).
 * Гарантируется sum(mask) >= 1 в любом состоянии игры.
 */
export function computeActionMask(
  state: MatchState,
  player: PlayerEntity,
  out?: Uint8Array,
  outOffset = 0
): Uint8Array {
  const mask = out ?? new Uint8Array(NUM_DISCRETE_ACTIONS);
  for (let i = 0; i < NUM_DISCRETE_ACTIONS; i++) {
    mask[outOffset + i] = 0;
  }
  mask[outOffset + DiscreteAction.NOOP_MOVE] = 1;

  if (
    player.penaltyTimer > 0 ||
    player.trippedTimer > 0 ||
    player.windupTimer > 0 ||
    state.freeThrow !== null
  ) {
    return mask;
  }

  const isInboundPasser = state.inbound?.passerId === player.id;
  const heldBall = state.balls.find((b) => b.holderId === player.id) ?? null;
  const mates = getOrderedTeammates(state, player);
  const opps = getOrderedOpponents(state, player);
  const nearestGroundedOpp =
    opps.find((o) => o.penaltyTimer <= 0 && o.trippedTimer <= 0 && o.z < 18) ?? null;
  const nearestOppDist = nearestGroundedOpp
    ? Math.hypot(nearestGroundedOpp.x - player.x, nearestGroundedOpp.y - player.y)
    : 999;

  const atkHoopIdx = getAttackingHoop(player.team);
  const atkHoop = getHoopPosition(atkHoopIdx);
  const defHoop = getHoopPosition(getDefendingHoop(player.team));
  const distToHoop = Math.hypot(player.x - atkHoop.x, player.y - atkHoop.y);
  const inFront = isInFrontcourt(player.team, player.x);
  const canPassByTrait = !player.traitIds.includes('egoisto') || isInboundPasser;

  if (heldBall) {
    // Передачи и навесы алли-уп партнёрам [0..3]
    if (canPassByTrait) {
      for (let mIdx = 0; mIdx < 4; mIdx++) {
        const m = mates[mIdx];
        if (m && m.penaltyTimer <= 0 && m.trippedTimer <= 0 && !m.schizoBreadMode) {
          mask[outOffset + DiscreteAction.PASS_MATE_0 + mIdx] = 1;
          const mDistHoop = Math.hypot(m.x - atkHoop.x, m.y - atkHoop.y);
          if (
            !isInboundPasser &&
            inFront &&
            mDistHoop <= 170 &&
            (m.effectiveStats.jump >= 68 ||
              m.traitIds.includes('gorilla_dunk') ||
              m.traitIds.includes('grasshopper') ||
              m.traitIds.includes('sixth_sense_alley'))
          ) {
            mask[outOffset + DiscreteAction.ALLEY_OOP_MATE_0 + mIdx] = 1;
          }
        }
      }
    }

    // При вводе мяча из-за линии разрешены только передачи; при остатке <= 3.6с маскируем ожидание
    if (isInboundPasser) {
      let hasPassOption = false;
      for (let mIdx = 0; mIdx < 4; mIdx++) {
        if (mask[outOffset + DiscreteAction.PASS_MATE_0 + mIdx] === 1) {
          hasPassOption = true;
          break;
        }
      }
      if (hasPassOption && state.inbound && state.inbound.timer <= 3.6) {
        mask[outOffset + DiscreteAction.NOOP_MOVE] = 0;
      }
      return mask;
    }

    const desperateBuzzer = state.quarterTimeRemaining <= 2.4 || state.shotClock <= 2.2;
    if ((inFront || desperateBuzzer) && player.actionCooldown <= 0) {
      mask[outOffset + DiscreteAction.SHOOT_STANDARD] = 1;

      if (!player.hasStoppedDribble && distToHoop >= 52 && distToHoop <= 185) {
        mask[outOffset + DiscreteAction.SHOOT_FLOATER] = 1;
      }
      if (distToHoop >= 40 && distToHoop <= 150) {
        mask[outOffset + DiscreteAction.SHOOT_HOOK] = 1;
      }
      if (
        distToHoop <= 96 &&
        player.z === 0 &&
        (player.effectiveStats.jump >= 60 ||
          player.traitIds.includes('gorilla_dunk') ||
          player.traitIds.includes('grasshopper'))
      ) {
        mask[outOffset + DiscreteAction.DUNK] = 1;
      }
    }

    if ((player.pumpFakeCd ?? 0) <= 0 && nearestOppDist <= 68) {
      mask[outOffset + DiscreteAction.PUMP_FAKE] = 1;
    }
    if (
      !player.hasStoppedDribble &&
      inFront &&
      (player.stepBackCd ?? 0) <= 0 &&
      player.actionCooldown <= 0 &&
      distToHoop >= 170 &&
      distToHoop <= 270 &&
      nearestOppDist <= 58
    ) {
      mask[outOffset + DiscreteAction.STEP_BACK] = 1;
    }
    if (
      !player.hasStoppedDribble &&
      inFront &&
      (player.eurostepCd ?? 0) <= 0 &&
      player.actionCooldown <= 0 &&
      distToHoop >= 68 &&
      distToHoop <= 182 &&
      nearestOppDist <= 62
    ) {
      mask[outOffset + DiscreteAction.EUROSTEP] = 1;
    }
    if (
      !player.hasStoppedDribble &&
      inFront &&
      (player.crossoverCd ?? 0) <= 0 &&
      player.actionCooldown <= 0 &&
      nearestOppDist <= 56
    ) {
      mask[outOffset + DiscreteAction.CROSSOVER] = 1;
    }
    return mask;
  }

  // Действия без мяча в руках (заслон, подстановка под фол в атаке, подбор/блок, отбор, пакость, тактический фол)
  if (!state.inbound) {
    // Заслон в позиционной атаке без мяча
    if (state.possessionTeam === player.team && inFront && player.z === 0) {
      const carrier = mates.find((m) => state.balls.some((b) => b.holderId === m.id));
      if (carrier && Math.hypot(carrier.x - player.x, carrier.y - player.y) <= 135) {
        mask[outOffset + DiscreteAction.SCREEN_SET] = 1;
      }
    }

    // Подстановка под фол в атаке (Take Charge) или флоп «Актёра Оскара» перед летящим с мячом соперником
    const oppTeam: 0 | 1 = player.team === 0 ? 1 : 0;
    const enemyCarrier = opps.find(
      (o) => o.penaltyTimer <= 0 && state.balls.some((b) => b.holderId === o.id)
    );
    if (state.possessionTeam === oppTeam && enemyCarrier && player.z === 0) {
      const dCarrier = Math.hypot(enemyCarrier.x - player.x, enemyCarrier.y - player.y);
      const dDefHoop = Math.hypot(player.x - defHoop.x, player.y - defHoop.y);
      if (dCarrier <= 82 && dDefHoop >= 72) {
        mask[outOffset + DiscreteAction.TAKE_CHARGE] = 1;
      }
    }

    const anyAirBall = state.balls.some(
      (b) => b.holderId === null && (b.z > 18 || b.isTipoffBall)
    );
    const oppShootingNear = opps.some(
      (o) => o.windupTimer > 0 && Math.hypot(o.x - player.x, o.y - player.y) <= 115
    );
    if (player.z === 0 && (anyAirBall || oppShootingNear)) {
      mask[outOffset + DiscreteAction.JUMP_CONTEST_REBOUND] = 1;
    }

    if (player.stealCooldown <= 0 && enemyCarrier) {
      const dCarrier = Math.hypot(enemyCarrier.x - player.x, enemyCarrier.y - player.y);
      if (dCarrier <= 48) {
        mask[outOffset + DiscreteAction.STEAL_PRESS] = 1;
      }

      // Тактический фол в концовке 4-й четверти / овертайма, когда команда уступает 1–6 очков
      const scoreDiff = state.score[player.team] - state.score[oppTeam];
      if (
        state.quarter >= 4 &&
        state.quarterTimeRemaining <= 20 &&
        scoreDiff >= -6 &&
        scoreDiff <= -1 &&
        player.fouls < 4 &&
        dCarrier <= 52
      ) {
        mask[outOffset + DiscreteAction.INTENTIONAL_FOUL] = 1;
      }
    }

    // Грязный приём за спиной судьи (при достаточной Девиантности или грязной тактике)
    const dirtyInclined =
      player.effectiveStats.deviance >= 26 ||
      state.tactics[player.team] === 'dirty' ||
      player.traitIds.includes('dirty_elbow');
    if (dirtyInclined && player.dirtyTrickCd <= 0 && player.fouls < 4) {
      const kickBall = state.balls.some(
        (b) => b.holderId === null && b.z < 34 && Math.hypot(b.x - player.x, b.y - player.y) < 38
      );
      if (kickBall || nearestOppDist <= 40) {
        mask[outOffset + DiscreteAction.DIRTY_TRICK] = 1;
      }
    }
  }

  return mask;
}

/**
 * Применяет выбранное нейросетью действие к игроку, обрабатывая средовые перехватчики черт
 * («Биполярный гений», «Стадо баранов») и формируя живые русские реплики «Мозга ИИ» + векторы [V].
 */
export function applyAgentAction(
  state: MatchState,
  player: PlayerEntity,
  action: AgentAction,
  mask: Uint8Array,
  callbacks: AIDecisionCallbacks,
  maskOffset = 0,
  dt = 1 / 15,
  rng: () => number = Math.random
): void {
  if (player.penaltyTimer > 0) {
    player.aiAction = 'penalty_bench';
    player.aiThought = 'Отбывает штраф на скамье';
    return;
  }
  if (player.trippedTimer > 0) {
    player.aiAction = 'tripped_down';
    player.aiThought = 'Валяется на паркете!';
    return;
  }
  if (state.freeThrow || player.windupTimer > 0) {
    return;
  }

  const isInboundPasser = state.inbound?.passerId === player.id;
  const heldBall = state.balls.find((b) => b.holderId === player.id) ?? null;

  // Средовой перехватчик непроизвольной черты «Биполярный гений»
  if (!state.inbound && player.traitIds.includes('schizo_tactician')) {
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

  // Средовой перехватчик непроизвольной черты «Стадо баранов» (без мяча бежит толпой за мячиком)
  if (!state.inbound && !heldBall && player.traitIds.includes('sheep_herd') && state.balls.length > 0) {
    const b = state.balls[player.slotIndex % state.balls.length];
    player.aiAction = 'chase_ball';
    player.screenTargetId = null;
    player.aiThought = 'Бежит за мячиком толпой!';
    player.targetX = clamp(b.x, COURT_LEFT + 22, COURT_RIGHT - 22);
    player.targetY = clamp(b.y, COURT_TOP + 22, COURT_BOTTOM - 22);
    return;
  }

  // Валидируем дискретное действие по маске
  let disc = action.discrete | 0;
  if (disc < 0 || disc >= NUM_DISCRETE_ACTIONS || mask[maskOffset + disc] === 0) {
    if (isInboundPasser) {
      disc = DiscreteAction.NOOP_MOVE;
      for (let mIdx = 0; mIdx < 4; mIdx++) {
        if (mask[maskOffset + DiscreteAction.PASS_MATE_0 + mIdx] === 1) {
          disc = DiscreteAction.PASS_MATE_0 + mIdx;
          break;
        }
      }
    } else {
      disc = DiscreteAction.NOOP_MOVE;
    }
  }

  const atkHoopIdx = getAttackingHoop(player.team);
  const atkHoop = getHoopPosition(atkHoopIdx);
  const defHoop = getHoopPosition(getDefendingHoop(player.team));

  // 1. Позиционные стойки без смещения (Заслон и Подстановка под фол в атаке)
  if (disc === DiscreteAction.SCREEN_SET && !heldBall && !isInboundPasser) {
    player.aiAction = 'screen_set';
    player.aiThought = 'Ставит жёсткий заслон партнёру';
    player.targetX = player.x;
    player.targetY = player.y;
    player.vx *= 0.25;
    player.vy *= 0.25;
    return;
  }

  if (disc === DiscreteAction.TAKE_CHARGE && !heldBall && !isInboundPasser) {
    player.aiAction = 'take_charge';
    player.chargePlantTimer = 0.55;
    player.aiThought = player.traitIds.includes('oscar_actor')
      ? 'Готовит картинное падение на «Оскар»!'
      : 'Встал как вкопанный под фол в атаке!';
    player.targetX = player.x;
    player.targetY = player.y;
    player.vx *= 0.2;
    player.vy *= 0.2;
    return;
  }

  // 2. Применяем непрерывный вектор движения и естественную русскую реплику «Мозга ИИ»
  player.screenTargetId = null;
  if (!isInboundPasser) {
    const moveX = clamp(action.moveX, -1, 1);
    const moveY = clamp(action.moveY, -1, 1);
    const worldDx = fromCanonicalDx(player.team, moveX * MOVE_TARGET_RADIUS);
    const worldDy = moveY * MOVE_TARGET_RADIUS;

    if (heldBall && player.hasStoppedDribble) {
      player.targetX = player.x;
      player.targetY = player.y;
    } else {
      let nextX = clamp(player.x + worldDx, COURT_LEFT + 24, COURT_RIGHT - 24);
      if (state.frontcourtEstablished && state.possessionTeam === player.team) {
        nextX = player.team === 0 ? Math.max(MID_X + 24, nextX) : Math.min(MID_X - 24, nextX);
      }
      const nextY = clamp(player.y + worldDy, COURT_TOP + 24, COURT_BOTTOM - 24);
      player.targetX = nextX;
      player.targetY = nextY;
    }

    if (heldBall) {
      const dAtk = Math.hypot(player.x - atkHoop.x, player.y - atkHoop.y);
      player.aiAction = !isInFrontcourt(player.team, player.x) ? 'fast_break' : 'drive';
      player.aiThought = player.hasStoppedDribble
        ? 'Держит опорную ногу, ищет пас'
        : dAtk < 165
        ? 'Давит в краску к кольцу!'
        : 'Выводит мяч на ударную позицию';
    } else if (state.possessionTeam === player.team) {
      const movingToRim = moveX > 0.45 && Math.hypot(player.x - atkHoop.x, player.y - atkHoop.y) < 280;
      player.aiAction = movingToRim ? 'backdoor_cut' : 'cut_open';
      player.aiThought = movingToRim
        ? 'Врывается за спину защитнику!'
        : 'Растягивает оборону на дуге';
    } else if (state.balls.some((b) => b.holderId === null && b.targetHoop !== null)) {
      player.aiAction = 'rebound_position';
      player.aiThought = 'Отсекает соперника на щите (бокс-аут)';
    } else if (state.balls.some((b) => b.holderId === null)) {
      player.aiAction = 'chase_ball';
      player.aiThought = 'Летит за ничейным мячом!';
    } else {
      const dDef = Math.hypot(player.x - defHoop.x, player.y - defHoop.y);
      player.aiAction = dDef < 135 ? 'rim_protect' : 'mark_man';
      player.aiThought = dDef < 135 ? 'Страхует краску под своим щитом' : 'Держит плотную стойку в защите';
    }
  } else if (state.inbound) {
    player.aiAction = 'inbound_pass';
    player.aiThought = 'Высматривает открытого под вбрасывание';
    player.targetX = state.inbound.x;
    player.targetY = state.inbound.y;
  }

  if (disc === DiscreteAction.NOOP_MOVE) {
    return;
  }

  const mates = getOrderedTeammates(state, player);
  const opps = getOrderedOpponents(state, player);
  const nearestGroundedOpp =
    opps.find((o) => o.penaltyTimer <= 0 && o.trippedTimer <= 0 && o.z < 18) ?? opps[0];

  if (heldBall) {
    if (disc === DiscreteAction.SHOOT_STANDARD) {
      player.aiThought = 'Бросает по кольцу в прыжке!';
      callbacks.onShoot(player, heldBall, atkHoopIdx, false, false, 'standard');
      return;
    }
    if (disc === DiscreteAction.SHOOT_FLOATER) {
      player.aiAction = 'floater';
      player.aiThought = 'Мягкий флоутер через руки больших!';
      callbacks.onShoot(player, heldBall, atkHoopIdx, false, false, 'floater');
      return;
    }
    if (disc === DiscreteAction.SHOOT_HOOK) {
      player.aiAction = 'post_up';
      player.aiThought = 'Классический крюк из лоу-поста!';
      callbacks.onShoot(player, heldBall, atkHoopIdx, false, false, 'hook');
      return;
    }
    if (disc === DiscreteAction.DUNK) {
      player.aiThought = 'Вколачивает сверху в кольцо!';
      callbacks.onShoot(player, heldBall, atkHoopIdx, true, false, 'standard');
      return;
    }
    if (disc >= DiscreteAction.PASS_MATE_0 && disc <= DiscreteAction.PASS_MATE_3) {
      const mate = mates[disc - DiscreteAction.PASS_MATE_0];
      if (mate) {
        player.aiAction = 'pass';
        player.aiThought = `Отдаёт передачу на ${mate.name}`;
        callbacks.onPass(player, heldBall, mate, false, false);
      }
      return;
    }
    if (disc >= DiscreteAction.ALLEY_OOP_MATE_0 && disc <= DiscreteAction.ALLEY_OOP_MATE_3) {
      const mate = mates[disc - DiscreteAction.ALLEY_OOP_MATE_0];
      if (mate) {
        player.aiAction = 'pass';
        player.aiThought = `Навешивает АЛЛИ-УП на ${mate.name}!`;
        callbacks.onPass(player, heldBall, mate, true, false);
      }
      return;
    }
    if (disc === DiscreteAction.PUMP_FAKE && nearestGroundedOpp && callbacks.onPumpFake) {
      player.aiAction = 'pump_fake';
      player.aiThought = 'Ложный замах — поднимает защитника в воздух!';
      callbacks.onPumpFake(player, nearestGroundedOpp);
      return;
    }
    if (disc === DiscreteAction.STEP_BACK && nearestGroundedOpp && callbacks.onStepBackMove) {
      player.aiAction = 'step_back';
      player.aiThought = 'Отпрыгивает в степ-бэк на дугу!';
      callbacks.onStepBackMove(player, nearestGroundedOpp, heldBall, atkHoopIdx);
      return;
    }
    if (disc === DiscreteAction.EUROSTEP && nearestGroundedOpp && callbacks.onEurostepMove) {
      player.aiAction = 'eurostep';
      player.aiThought = 'Уходит евростепом мимо защитника!';
      callbacks.onEurostepMove(player, nearestGroundedOpp, heldBall, atkHoopIdx);
      return;
    }
    if (disc === DiscreteAction.CROSSOVER && nearestGroundedOpp && callbacks.onCrossoverMove) {
      player.aiAction = 'ankle_breaker';
      player.aiThought = 'Раскачивает кроссовером на ведении!';
      callbacks.onCrossoverMove(player, nearestGroundedOpp, heldBall);
      return;
    }
    return;
  }

  if (disc === DiscreteAction.JUMP_CONTEST_REBOUND) {
    const useSuper =
      player.traitIds.includes('grasshopper') && player.superJumpCd <= 0 && player.energy >= 15;
    player.aiAction = 'intercept_air';
    player.aiThought = useSuper
      ? 'Взлетает пружиной «Кузнечика» за мячом!'
      : 'Выпрыгивает на блок и подбор!';
    callbacks.onJump(player, useSuper, player.aiThought);
    return;
  }

  if (disc === DiscreteAction.STEAL_PRESS) {
    const enemyCarrier = opps.find((o) => state.balls.some((b) => b.holderId === o.id));
    const carrierBall = enemyCarrier
      ? state.balls.find((b) => b.holderId === enemyCarrier.id)
      : undefined;
    if (enemyCarrier && carrierBall) {
      player.aiAction = 'guard_carrier';
      player.aiThought = 'Ныряет руками на чистый перехват!';
      callbacks.onAttemptSteal(player, enemyCarrier, carrierBall);
    }
    return;
  }

  if (disc === DiscreteAction.DIRTY_TRICK) {
    const kickableBall = state.balls.find(
      (b) => b.holderId === null && b.z < 34 && Math.hypot(b.x - player.x, b.y - player.y) < 38
    );
    if (kickableBall) {
      player.aiAction = 'dirty_play';
      player.aiThought = 'Исподтишка пинает мяч ногой!';
      callbacks.onAttemptDirtyTrick(player, 'kick_ball', undefined, kickableBall);
      return;
    }
    const victim = opps.find(
      (o) => o.penaltyTimer <= 0 && o.trippedTimer <= 0 && Math.hypot(o.x - player.x, o.y - player.y) <= 42
    );
    if (victim) {
      const oppHasBall = state.balls.some((b) => b.holderId === victim.id);
      const useTrip = player.fouls < 4 && (oppHasBall || player.effectiveStats.deviance >= 58);
      player.aiAction = 'dirty_play';
      player.aiThought = useTrip
        ? 'Ставит хитрую подножку за спиной судьи!'
        : 'Толкает локтем пока судья отвернулся!';
      callbacks.onAttemptDirtyTrick(player, useTrip ? 'trip' : 'shove', victim);
    }
    return;
  }

  if (disc === DiscreteAction.INTENTIONAL_FOUL && callbacks.onIntentionalFoul) {
    const enemyCarrier = opps.find((o) => state.balls.some((b) => b.holderId === o.id));
    if (enemyCarrier) {
      player.aiAction = 'intentional_foul';
      player.aiThought = 'Тактический фол ради остановки секундомера!';
      callbacks.onIntentionalFoul(player, enemyCarrier);
    }
  }
}

/**
 * Перехватчик решений эвристического учителя (stepPlayerAI) для генерации датасета
 * имитационного обучения (Behavior Cloning) на всех 23 дискретных действиях.
 */
export function wrapTeacherCallbacks(
  state: MatchState,
  player: PlayerEntity,
  inner: AIDecisionCallbacks,
  recordedOut: AgentAction
): AIDecisionCallbacks {
  recordedOut.discrete = DiscreteAction.NOOP_MOVE;
  recordedOut.moveX = 0;
  recordedOut.moveY = 0;

  const mates = getOrderedTeammates(state, player);

  return {
    onShoot: (shooter, ball, targetHoop, isDunk, isPanic, shotStyle: ShotStyle = 'standard') => {
      if (isDunk) recordedOut.discrete = DiscreteAction.DUNK;
      else if (shotStyle === 'floater') recordedOut.discrete = DiscreteAction.SHOOT_FLOATER;
      else if (shotStyle === 'hook') recordedOut.discrete = DiscreteAction.SHOOT_HOOK;
      else recordedOut.discrete = DiscreteAction.SHOOT_STANDARD;
      inner.onShoot(shooter, ball, targetHoop, isDunk, isPanic, shotStyle);
    },
    onPass: (passer, ball, receiver, isAlleyOop, isSafeHighLob) => {
      const mIdx = mates.findIndex((m) => m.id === receiver.id);
      if (mIdx >= 0 && mIdx < 4) {
        recordedOut.discrete = isAlleyOop
          ? DiscreteAction.ALLEY_OOP_MATE_0 + mIdx
          : DiscreteAction.PASS_MATE_0 + mIdx;
      }
      inner.onPass(passer, ball, receiver, isAlleyOop, isSafeHighLob);
    },
    onJump: (jumper, useSuperJump, reason) => {
      recordedOut.discrete = DiscreteAction.JUMP_CONTEST_REBOUND;
      inner.onJump(jumper, useSuperJump, reason);
    },
    onAttemptSteal: (defender, carrier, ball) => {
      recordedOut.discrete = DiscreteAction.STEAL_PRESS;
      inner.onAttemptSteal(defender, carrier, ball);
    },
    onAttemptDirtyTrick: (offender, kind, victim, ball) => {
      recordedOut.discrete = DiscreteAction.DIRTY_TRICK;
      inner.onAttemptDirtyTrick(offender, kind, victim, ball);
    },
    onPumpFake: (shooter, defender) => {
      recordedOut.discrete = DiscreteAction.PUMP_FAKE;
      inner.onPumpFake?.(shooter, defender);
    },
    onGhostPumpFake: (shooter) => {
      inner.onGhostPumpFake?.(shooter);
    },
    onCrossoverMove: (carrier, defender, ball) => {
      recordedOut.discrete = DiscreteAction.CROSSOVER;
      inner.onCrossoverMove?.(carrier, defender, ball);
    },
    onStepBackMove: (carrier, defender, ball, targetHoop) => {
      recordedOut.discrete = DiscreteAction.STEP_BACK;
      inner.onStepBackMove?.(carrier, defender, ball, targetHoop);
    },
    onEurostepMove: (carrier, defender, ball, targetHoop) => {
      recordedOut.discrete = DiscreteAction.EUROSTEP;
      inner.onEurostepMove?.(carrier, defender, ball, targetHoop);
    },
    onHandoffMove: (giver, receiver, ball) => {
      const mIdx = mates.findIndex((m) => m.id === receiver.id);
      if (mIdx >= 0 && mIdx < 4) {
        recordedOut.discrete = DiscreteAction.PASS_MATE_0 + mIdx;
      }
      inner.onHandoffMove?.(giver, receiver, ball);
    },
    onIntentionalFoul: (defender, carrier) => {
      recordedOut.discrete = DiscreteAction.INTENTIONAL_FOUL;
      inner.onIntentionalFoul?.(defender, carrier);
    },
  };
}

export function extractTeacherMoveVector(player: PlayerEntity, out: AgentAction): void {
  if (player.aiAction === 'screen_set' || player.aiAction === 'offball_screen') {
    if (out.discrete === DiscreteAction.NOOP_MOVE) {
      out.discrete = DiscreteAction.SCREEN_SET;
    }
  } else if (player.aiAction === 'take_charge') {
    if (out.discrete === DiscreteAction.NOOP_MOVE) {
      out.discrete = DiscreteAction.TAKE_CHARGE;
    }
  }
  const rawDx = player.targetX - player.x;
  const rawDy = player.targetY - player.y;
  const canonDx = toCanonicalVx(player.team, rawDx);
  out.moveX = clamp(canonDx / MOVE_TARGET_RADIUS, -1, 1);
  out.moveY = clamp(rawDy / MOVE_TARGET_RADIUS, -1, 1);
}
