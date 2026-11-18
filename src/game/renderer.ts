import {
  COURT_BOTTOM,
  COURT_H,
  COURT_LEFT,
  COURT_RIGHT,
  COURT_TOP,
  COURT_W,
  HOOP_LEFT_X,
  HOOP_RIGHT_X,
  HOOP_Y,
  MID_X,
  PENALTY_BENCH_POS,
  RIM_RADIUS,
  TEAM_BENCH_BOX,
  THREE_PT_RADIUS,
  TRAMPOLINE_PADS,
  getHoopPosition,
} from './physics';
import { isLastHopeBuffActive } from './traits';
import { BallState, MatchState, PlayerEntity, POSITION_LABELS } from './types';

export interface CourtTransform {
  scale: number;
  offsetX: number;
  offsetY: number;
}

export function computeCourtTransform(canvasWidth: number, canvasHeight: number): CourtTransform {
  const padX = 28;
  const padTop = 58;
  const padBottom = 158;
  const availW = Math.max(320, canvasWidth - padX * 2);
  const availH = Math.max(220, canvasHeight - padTop - padBottom);
  const scale = Math.min(availW / COURT_W, availH / COURT_H);
  const offsetX = (canvasWidth - COURT_W * scale) * 0.5;
  const offsetY = padTop + (availH - COURT_H * scale) * 0.5;
  return { scale, offsetX, offsetY };
}

export function screenToWorld(
  clientX: number,
  clientY: number,
  rect: DOMRect,
  transform: CourtTransform
): { x: number; y: number } {
  const localX = clientX - rect.left;
  const localY = clientY - rect.top;
  return {
    x: (localX - transform.offsetX) / transform.scale,
    y: (localY - transform.offsetY) / transform.scale,
  };
}

export function renderMatchTopDown(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  state: MatchState,
  showTacticalOverlay = true
): CourtTransform {
  ctx.clearRect(0, 0, width, height);

  // Фон арены вокруг площадки
  ctx.fillStyle = '#141210';
  ctx.fillRect(0, 0, width, height);

  const tr = computeCourtTransform(width, height);
  ctx.save();

  const shakeMag = state.cameraShake * state.cameraShake * 9;
  const shakeX = shakeMag > 0.1 ? (Math.random() - 0.5) * shakeMag : 0;
  const shakeY = shakeMag > 0.1 ? (Math.random() - 0.5) * shakeMag : 0;

  ctx.translate(tr.offsetX + shakeX, tr.offsetY + shakeY);
  ctx.scale(tr.scale, tr.scale);

  drawCourtSurface(ctx, state);
  drawTeamBenches(ctx, state);
  drawPenaltyBenches(ctx, state);
  if (state.modifiers.includes('paint_trampolines')) {
    drawTrampolines(ctx);
  }
  drawTacticalVectors(ctx, state, showTacticalOverlay);
  drawReferee(ctx, state);

  // Сортируем игроков по высоте z, чтобы высоко прыгнувший «Кузнечик» рисовался поверх остальных!
  const sortedPlayers = [...state.players].sort((a, b) => a.z - b.z);
  for (const p of sortedPlayers) {
    drawTopDownPlayer(ctx, p, state);
  }

  for (const ball of state.balls) {
    drawTopDownBall(ctx, ball);
  }

  drawHoopsOverhead(ctx, state);
  drawParticles(ctx, state);
  drawCallouts(ctx, state);

  ctx.restore();
  return tr;
}

function drawCourtSurface(ctx: CanvasRenderingContext2D, state: MatchState): void {
  const cw = COURT_RIGHT - COURT_LEFT;
  const ch = COURT_BOTTOM - COURT_TOP;
  const isIce = state.modifiers.includes('ice_court');

  // Тень под настилом паркета
  ctx.fillStyle = '#0b0a09';
  ctx.fillRect(COURT_LEFT - 12, COURT_TOP - 12, cw + 24, ch + 24);

  // Бортик вокруг паркета
  ctx.fillStyle = '#221e1b';
  ctx.fillRect(COURT_LEFT - 8, COURT_TOP - 8, cw + 16, ch + 16);

  // Покрытие площадки: кленовый паркет или ледяной каток («Гололёд»)
  const plankCount = 22;
  const plankH = ch / plankCount;
  for (let i = 0; i < plankCount; i++) {
    if (isIce) {
      ctx.fillStyle = i % 2 === 0 ? '#bae6fd' : '#a5f3fc';
    } else {
      ctx.fillStyle = i % 2 === 0 ? '#d89b55' : '#cf9049';
    }
    ctx.fillRect(COURT_LEFT, COURT_TOP + i * plankH, cw, plankH + 0.5);
  }

  if (isIce) {
    // Ледяные блики по диагонали
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.42)';
    ctx.lineWidth = 8;
    for (let x = COURT_LEFT + 60; x < COURT_RIGHT; x += 140) {
      ctx.beginPath();
      ctx.moveTo(x, COURT_TOP + 10);
      ctx.lineTo(x - 75, COURT_BOTTOM - 10);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Трёхсекундные зоны (краска команд)
  const keyW = 190;
  const keyH = 164;
  ctx.save();
  ctx.globalAlpha = isIce ? 0.24 : 0.34;
  ctx.fillStyle = state.teamColors[0];
  ctx.fillRect(COURT_LEFT, HOOP_Y - keyH / 2, keyW, keyH);
  ctx.fillStyle = state.teamColors[1];
  ctx.fillRect(COURT_RIGHT - keyW, HOOP_Y - keyH / 2, keyW, keyH);
  ctx.restore();

  // Тактическая разметка паркета (строгий вид сверху)
  ctx.strokeStyle = isIce ? '#0f172a' : '#faf4e8';
  ctx.lineWidth = 3;

  // Периметр и центральная линия
  ctx.strokeRect(COURT_LEFT, COURT_TOP, cw, ch);
  ctx.beginPath();
  ctx.moveTo(MID_X, COURT_TOP);
  ctx.lineTo(MID_X, COURT_BOTTOM);
  ctx.stroke();

  // Центральный круг
  ctx.beginPath();
  ctx.arc(MID_X, HOOP_Y, 66, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(MID_X, HOOP_Y, 20, 0, Math.PI * 2);
  ctx.stroke();

  // Штрафные прямоугольники и полукруги
  ctx.strokeRect(COURT_LEFT, HOOP_Y - keyH / 2, keyW, keyH);
  ctx.beginPath();
  ctx.arc(COURT_LEFT + keyW, HOOP_Y, 60, -Math.PI / 2, Math.PI / 2);
  ctx.stroke();

  ctx.strokeRect(COURT_RIGHT - keyW, HOOP_Y - keyH / 2, keyW, keyH);
  ctx.beginPath();
  ctx.arc(COURT_RIGHT - keyW, HOOP_Y, 60, Math.PI / 2, (Math.PI * 3) / 2);
  ctx.stroke();

  // Трёхочковые дуги (слева и справа)
  ctx.beginPath();
  ctx.arc(HOOP_LEFT_X, HOOP_Y, THREE_PT_RADIUS, -1.22, 1.22);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(COURT_LEFT, HOOP_Y - Math.sin(1.22) * THREE_PT_RADIUS);
  ctx.lineTo(HOOP_LEFT_X + Math.cos(1.22) * THREE_PT_RADIUS, HOOP_Y - Math.sin(1.22) * THREE_PT_RADIUS);
  ctx.moveTo(COURT_LEFT, HOOP_Y + Math.sin(1.22) * THREE_PT_RADIUS);
  ctx.lineTo(HOOP_LEFT_X + Math.cos(1.22) * THREE_PT_RADIUS, HOOP_Y + Math.sin(1.22) * THREE_PT_RADIUS);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(HOOP_RIGHT_X, HOOP_Y, THREE_PT_RADIUS, Math.PI - 1.22, Math.PI + 1.22);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(COURT_RIGHT, HOOP_Y - Math.sin(1.22) * THREE_PT_RADIUS);
  ctx.lineTo(
    HOOP_RIGHT_X - Math.cos(1.22) * THREE_PT_RADIUS,
    HOOP_Y - Math.sin(1.22) * THREE_PT_RADIUS
  );
  ctx.moveTo(COURT_RIGHT, HOOP_Y + Math.sin(1.22) * THREE_PT_RADIUS);
  ctx.lineTo(
    HOOP_RIGHT_X - Math.cos(1.22) * THREE_PT_RADIUS,
    HOOP_Y + Math.sin(1.22) * THREE_PT_RADIUS
  );
  ctx.stroke();
}

function drawTeamBenches(ctx: CanvasRenderingContext2D, state: MatchState): void {
  for (const team of [0, 1] as const) {
    const box = TEAM_BENCH_BOX[team];
    ctx.save();
    ctx.fillStyle = '#24201d';
    ctx.strokeStyle = state.teamColors[team];
    ctx.lineWidth = 1.6;
    ctx.fillRect(box.x, box.y, box.width, box.height);
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // Спинка командной скамейки запасных сверху
    ctx.fillStyle = state.teamColors[team];
    ctx.globalAlpha = 0.35;
    ctx.fillRect(box.x, box.y, box.width, 4);
    ctx.globalAlpha = 1;

    ctx.font = '700 9.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#d6d3d1';
    ctx.textAlign = team === 0 ? 'left' : 'right';
    const labelX = team === 0 ? box.x + box.width + 7 : box.x - 7;
    ctx.fillText('ЗАПАС', labelX, box.y + 19);
    ctx.restore();
  }

  for (const bp of state.bench) {
    const isSelected = bp.id === state.selectedPlayerId;
    const teamColor = state.teamColors[bp.team];
    const hasLastHope = bp.traitIds.includes('last_hope') && bp.startedOnBench;

    ctx.save();
    if (isSelected) {
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(bp.x, bp.y, 15, 0, Math.PI * 2);
      ctx.stroke();
    } else if (hasLastHope && !bp.disqualified) {
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.7)';
      ctx.lineWidth = 1.6;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.arc(bp.x, bp.y, 13.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.globalAlpha = bp.disqualified ? 0.42 : 0.96;
    ctx.translate(bp.x, bp.y);

    // Плечи запасного (сидит лицом вниз к паркету)
    ctx.fillStyle = teamColor;
    ctx.strokeStyle = '#141210';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(0, -1, 10.5, 7.2, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Макушка головы
    ctx.fillStyle = bp.skinTone;
    ctx.beginPath();
    ctx.arc(0, 1.5, 5.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // Номер / маркер над головой запасного и мини-полоска сил
    ctx.save();
    ctx.font = '700 8.5px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    if (bp.disqualified) {
      ctx.fillStyle = '#fca5a5';
      ctx.fillText('5Ф', bp.x, bp.y - 10);
    } else if (hasLastHope) {
      ctx.fillStyle = '#fde047';
      ctx.fillText(`★#${bp.jerseyNumber}`, bp.x, bp.y - 10);
    } else {
      ctx.fillStyle = '#e7e5e4';
      ctx.fillText(`#${bp.jerseyNumber}`, bp.x, bp.y - 10);
    }

    if (!bp.disqualified) {
      const bw = 20;
      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      ctx.fillRect(bp.x - bw / 2, bp.y + 10, bw, 2.5);
      ctx.fillStyle = bp.energy > 65 ? '#34d399' : bp.energy > 35 ? '#fbbf24' : '#ef4444';
      ctx.fillRect(bp.x - bw / 2, bp.y + 10, (bw * Math.max(0, bp.energy)) / 100, 2.5);
    }
    ctx.restore();
  }
}

function drawPenaltyBenches(ctx: CanvasRenderingContext2D, state: MatchState): void {
  for (const team of [0, 1] as const) {
    const pos = PENALTY_BENCH_POS[team];
    ctx.save();
    ctx.fillStyle = '#26221f';
    ctx.strokeStyle = '#57534e';
    ctx.lineWidth = 1.5;
    ctx.fillRect(pos.x - 52, pos.y - 16, 104, 32);
    ctx.strokeRect(pos.x - 52, pos.y - 16, 104, 32);

    const benched = state.players.find((p) => p.team === team && p.penaltyTimer > 0);
    ctx.font = '600 9.5px Manrope, sans-serif';
    ctx.textAlign = team === 0 ? 'right' : 'left';
    ctx.fillStyle = benched ? '#fca5a5' : '#a8a29e';
    const labelX = team === 0 ? pos.x - 58 : pos.x + 58;
    ctx.fillText(
      benched ? `ШТРАФ ${benched.penaltyTimer.toFixed(1)}с` : 'ШТРАФНИКИ',
      labelX,
      pos.y + 4
    );
    ctx.restore();
  }

  // Судейский столик в центре верхней бровки со стрелкой поочерёдного владения FIBA или активной комбинацией тренера
  ctx.save();
  ctx.fillStyle = '#26221f';
  ctx.strokeStyle = state.activePlayCall ? '#fbbf24' : '#57534e';
  ctx.lineWidth = state.activePlayCall ? 2 : 1.5;
  ctx.fillRect(MID_X - 78, 12, 156, 32);
  ctx.strokeRect(MID_X - 78, 12, 156, 32);
  ctx.font = '700 9.5px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  if (state.activePlayCall) {
    const playNames = {
      iso_star: 'ИЗОЛЯЦИЯ',
      pick_and_roll: 'ПИК-Н-РОЛЛ',
      alley_oop: 'АЛЛИ-УП',
      feed_sniper: 'СНАЙПЕР',
    } as const;
    ctx.fillStyle = '#fde047';
    ctx.fillText(
      `${playNames[state.activePlayCall.play]} ${Math.ceil(state.activePlayCall.timer)}с`,
      MID_X,
      32
    );
  } else {
    ctx.fillStyle = state.teamColors[state.possessionArrow];
    const arrowStr = state.possessionArrow === 0 ? '◄ ВЛАДЕНИЕ' : 'ВЛАДЕНИЕ ►';
    ctx.fillText(arrowStr, MID_X, 32);
  }
  ctx.restore();
}

function drawTrampolines(ctx: CanvasRenderingContext2D): void {
  for (const pad of TRAMPOLINE_PADS) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(pad.x, pad.y, pad.radius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.28)';
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#34d399';
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(pad.x, pad.y, pad.radius * 0.55, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
}

function drawTacticalArrow(
  ctx: CanvasRenderingContext2D,
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  color: string,
  dashed = true,
  width = 1.6
): void {
  const dx = toX - fromX;
  const dy = toY - fromY;
  const dist = Math.hypot(dx, dy);
  if (dist < 18) return;

  const ang = Math.atan2(dy, dx);
  ctx.save();
  if (dashed) ctx.setLineDash([5, 5]);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();

  ctx.setLineDash([]);
  ctx.fillStyle = color;
  const headLen = 7;
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - Math.cos(ang - 0.45) * headLen, toY - Math.sin(ang - 0.45) * headLen);
  ctx.lineTo(toX - Math.cos(ang + 0.45) * headLen, toY - Math.sin(ang + 0.45) * headLen);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawTacticalVectors(
  ctx: CanvasRenderingContext2D,
  state: MatchState,
  showTacticalOverlay: boolean
): void {
  // 0. Дуга 5-секундного ввода мяча из-за линии (защитникам вход запрещён!) и индикатор штрафного
  if (state.inbound) {
    const inb = state.inbound;
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = inb.timer <= 1.8 ? 'rgba(248, 113, 113, 0.85)' : 'rgba(251, 191, 36, 0.7)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(inb.x, inb.y, 68, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = '700 11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = inb.timer <= 1.8 ? '#fca5a5' : '#fde68a';
    const tagY = inb.y < COURT_TOP ? inb.y - 20 : inb.y > COURT_BOTTOM ? inb.y + 28 : inb.y - 22;
    ctx.fillText(`ВВОД ${Math.max(0, inb.timer).toFixed(1)}с`, inb.x, tagY);
    ctx.restore();
  }

  if (state.freeThrow) {
    const ft = state.freeThrow;
    const shooter = state.players.find((pl) => pl.id === ft.shooterId);
    if (shooter) {
      const shotIdx = ft.totalShots - ft.shotsRemaining + 1;
      ctx.save();
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.75)';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(shooter.x, shooter.y, 24, 0, Math.PI * 2);
      ctx.stroke();
      ctx.font = '700 11px Oswald, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillStyle = '#fde68a';
      ctx.fillText(`ШТРАФНОЙ ${shotIdx}/${ft.totalShots}`, shooter.x, shooter.y - 26);
      ctx.restore();
    }
  }

  // 1. Классическое тренерское Т-обозначение заслона («Пик-н-ролл») и тактические стрелки рывков
  for (const p of state.players) {
    if (p.penaltyTimer > 0 || p.trippedTimer > 0) continue;

    if (p.aiAction === 'screen_set' || p.aiAction === 'offball_screen') {
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.heading);
      ctx.strokeStyle =
        p.aiAction === 'offball_screen'
          ? 'rgba(251, 146, 60, 0.92)'
          : 'rgba(254, 240, 138, 0.9)';
      ctx.lineWidth = 2.6;
      // Ножка и поперечная планка заслона (┤)
      ctx.beginPath();
      ctx.moveTo(11, 0);
      ctx.lineTo(18, 0);
      ctx.moveTo(18, -11);
      ctx.lineTo(18, 11);
      ctx.stroke();
      ctx.restore();
    } else if (p.aiAction === 'pump_fake') {
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(253, 224, 71, 0.9)';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 21, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    } else if (p.aiAction === 'take_charge') {
      // Дуга неподвижной защитной стойки «Под фол в атаке» перед грудью защитника
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.heading);
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.92)';
      ctx.lineWidth = 2.8;
      ctx.beginPath();
      ctx.arc(0, 0, 19, -0.85, 0.85);
      ctx.stroke();
      ctx.restore();
    } else if (showTacticalOverlay) {
      if (p.aiAction === 'post_up') {
        ctx.save();
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.72)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 18, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      } else if (
        p.aiAction === 'roll_to_rim' ||
        p.aiAction === 'backdoor_cut' ||
        p.aiAction === 'give_and_go' ||
        p.aiAction === 'fast_break' ||
        p.aiAction === 'handoff'
      ) {
        drawTacticalArrow(ctx, p.x, p.y, p.targetX, p.targetY, 'rgba(253, 224, 71, 0.68)', true, 1.8);
      } else if (
        p.aiAction === 'pick_and_pop' ||
        p.aiAction === 'relocate_arc' ||
        p.aiAction === 'step_back' ||
        p.aiAction === 'floater'
      ) {
        drawTacticalArrow(ctx, p.x, p.y, p.targetX, p.targetY, 'rgba(249, 115, 22, 0.78)', true, 1.8);
      } else if (p.aiAction === 'drive' || p.aiAction === 'ankle_breaker' || p.aiAction === 'eurostep') {
        drawTacticalArrow(ctx, p.x, p.y, p.targetX, p.targetY, 'rgba(255, 255, 255, 0.56)', false, 1.8);
      } else if (p.aiAction === 'intentional_foul') {
        drawTacticalArrow(ctx, p.x, p.y, p.targetX, p.targetY, 'rgba(248, 113, 113, 0.82)', true, 1.8);
      } else if (
        p.aiAction === 'help_defense' ||
        p.aiAction === 'intercept_air' ||
        p.aiAction === 'closeout' ||
        p.aiAction === 'chasedown_block'
      ) {
        drawTacticalArrow(ctx, p.x, p.y, p.targetX, p.targetY, 'rgba(56, 189, 248, 0.68)', true, 1.6);
      }
    }
  }

  // 2. Вектор цели выбранного игрока (всегда виден для инспекции мозга ИИ)
  const selected = state.players.find((p) => p.id === state.selectedPlayerId);
  if (selected && selected.penaltyTimer <= 0) {
    const isDef =
      selected.aiAction === 'guard_carrier' ||
      selected.aiAction === 'closeout' ||
      selected.aiAction === 'take_charge' ||
      selected.aiAction === 'chasedown_block' ||
      selected.aiAction === 'intentional_foul' ||
      selected.aiAction === 'help_defense' ||
      selected.aiAction === 'mark_man' ||
      selected.aiAction === 'rim_protect' ||
      selected.aiAction === 'intercept_air';
    const selColor = isDef ? 'rgba(56, 189, 248, 0.82)' : 'rgba(251, 191, 36, 0.85)';
    drawTacticalArrow(ctx, selected.x, selected.y, selected.targetX, selected.targetY, selColor, true, 2.0);
  }

  // 3. Проекция полёта мяча (броски в кольцо и острые передачи на ход)
  for (const ball of state.balls) {
    if (ball.holderId === null && ball.targetHoop !== null) {
      const hoop = getHoopPosition(ball.targetHoop);
      ctx.save();
      ctx.setLineDash(ball.shotType === 'messi' ? [8, 6] : [4, 6]);
      ctx.strokeStyle =
        ball.shotType === 'messi' ? 'rgba(251, 191, 36, 0.85)' : 'rgba(255, 255, 255, 0.32)';
      ctx.lineWidth = ball.shotType === 'messi' ? 2.8 : 1.5;
      ctx.beginPath();
      ctx.moveTo(ball.x, ball.y);
      ctx.lineTo(hoop.x, hoop.y);
      ctx.stroke();
      ctx.restore();
    } else if (showTacticalOverlay && ball.holderId === null && ball.shotType === 'pass' && ball.passReceiverId) {
      const rec = state.players.find((pl) => pl.id === ball.passReceiverId);
      if (rec) {
        ctx.save();
        ctx.setLineDash([3, 5]);
        ctx.strokeStyle = 'rgba(125, 211, 252, 0.65)';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(ball.x, ball.y);
        ctx.lineTo(rec.targetX, rec.targetY);
        ctx.stroke();
        ctx.restore();
      }
    }
  }
}

function drawReferee(ctx: CanvasRenderingContext2D, state: MatchState): void {
  const ref = state.referee;
  ctx.save();

  // Конус зрения Судьи (спереди 80% обнаружения фолов!)
  ctx.beginPath();
  ctx.moveTo(ref.x, ref.y);
  ctx.arc(ref.x, ref.y, 175, ref.gazeAngle - 0.72, ref.gazeAngle + 0.72);
  ctx.closePath();
  ctx.fillStyle =
    ref.whistleTimer > 0 ? 'rgba(239, 68, 68, 0.18)' : 'rgba(250, 204, 21, 0.08)';
  ctx.fill();

  // Фигурка Судьи сверху (полосатая чёрно-белая рубашка)
  ctx.translate(ref.x, ref.y);
  ctx.rotate(ref.gazeAngle);

  // Тень
  ctx.fillStyle = 'rgba(0,0,0,0.4)';
  ctx.beginPath();
  ctx.arc(2, 3, 12, 0, Math.PI * 2);
  ctx.fill();

  // Плечи в полоску
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(0, 0, 10, 14, 0, 0, Math.PI * 2);
  ctx.clip();
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(-14, -16, 28, 32);
  ctx.fillStyle = '#0f172a';
  for (let sy = -14; sy <= 14; sy += 6) {
    ctx.fillRect(-14, sy, 28, 3);
  }
  ctx.restore();
  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.ellipse(0, 0, 10, 14, 0, 0, Math.PI * 2);
  ctx.stroke();

  // Голова и кепка судьи
  ctx.fillStyle = '#f0b27a';
  ctx.beginPath();
  ctx.arc(2, 0, 6.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#1e293b';
  ctx.beginPath();
  ctx.arc(0, 0, 6.2, Math.PI * 0.5, Math.PI * 1.5);
  ctx.fill();

  ctx.restore();

  // Вспышка свистка судьи
  if (ref.whistleTimer > 0) {
    ctx.save();
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2.5;
    const r = 16 + (1.15 - ref.whistleTimer) * 22;
    ctx.beginPath();
    ctx.arc(ref.x, ref.y, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.font = '700 11px Oswald, sans-serif';
    ctx.fillStyle = '#fca5a5';
    ctx.textAlign = 'center';
    ctx.fillText('СВИСТОК!', ref.x, ref.y + 26);
    ctx.restore();
  }
}

function drawTopDownPlayer(
  ctx: CanvasRenderingContext2D,
  p: PlayerEntity,
  state: MatchState
): void {
  const isSelected = p.id === state.selectedPlayerId;
  const teamColor = state.teamColors[p.team];
  const altitudeScale = 1 + p.z / 210;
  const totalScale = p.bodyScale * altitudeScale;

  const lastHopeActive = isLastHopeBuffActive(p);

  // 0. Огненно-золотая аура «Последней надежды» или пламя куража «В ОГНЕ!» (hotStreak >= 3)
  if (lastHopeActive) {
    const pulse = Math.sin(state.quarterTimeRemaining * 9) * 2.2;
    ctx.save();
    ctx.fillStyle = 'rgba(251, 191, 36, 0.16)';
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.9)';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.arc(p.x, p.y, (21.5 + pulse) * p.bodyScale, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  } else if (p.hotStreak >= 3) {
    const pulse = Math.sin(state.quarterTimeRemaining * 11) * 1.8;
    ctx.save();
    ctx.fillStyle = 'rgba(249, 115, 22, 0.18)';
    ctx.strokeStyle = 'rgba(251, 146, 60, 0.92)';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.arc(p.x, p.y, (19.5 + pulse) * p.bodyScale, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  // 1. Напольное кольцо выбора и индикатор правила 3 секунд в краске
  if (isSelected) {
    ctx.save();
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 20 * p.bodyScale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  if (p.paintTimer >= 1.4) {
    const frac = Math.min(1, p.paintTimer / 3.0);
    ctx.save();
    ctx.strokeStyle = p.paintTimer >= 2.3 ? '#ef4444' : '#f59e0b';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 17 * p.bodyScale, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // Пружинные кольца «Кузнечика» в высоком прыжке
  if (p.traitIds.includes('grasshopper') && p.z > 25) {
    ctx.save();
    ctx.strokeStyle = 'rgba(74, 222, 128, 0.55)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 14 + p.z * 0.08, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // Тень на паркете (смещается при взлёте вверх z)
  const shadowRadius = Math.max(7, 13 * p.bodyScale * (1 - Math.min(0.45, p.z / 320)));
  ctx.save();
  ctx.fillStyle = 'rgba(0, 0, 0, 0.36)';
  ctx.beginPath();
  ctx.arc(p.x + p.z * 0.09, p.y + p.z * 0.12 + 2, shadowRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // 2. Сам чувачок сверху (масштабируется к камере при высоком прыжке!)
  ctx.save();
  ctx.translate(p.x, p.y);
  ctx.scale(totalScale, totalScale);

  if (p.trippedTimer > 0) {
    // Распластался звездой на паркете после подножки!
    ctx.rotate(p.slotIndex);
    ctx.fillStyle = teamColor;
    ctx.fillRect(-13, -8, 26, 16);
    ctx.fillStyle = p.skinTone;
    ctx.beginPath();
    ctx.arc(12, 0, 6.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fca5a5';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(12, -10, 5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    return;
  }

  ctx.rotate(p.heading);

  // Кроссовки сверху (шагают вперёд-назад при беге!)
  const stride = Math.sin(p.stridePhase) * 6.5;
  ctx.fillStyle = '#f8fafc';
  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.ellipse(stride, -8.5, 5.2, 3, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(-stride, 8.5, 5.2, 3, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Кисти рук (вытянуты вперёд при ведении мяча или замахе)
  const reach = p.windupTimer > 0 ? 12 : 8.5;
  ctx.fillStyle = p.skinTone;
  ctx.beginPath();
  ctx.arc(reach, -9.5, 3.8, 0, Math.PI * 2);
  ctx.arc(reach, 9.5, 3.8, 0, Math.PI * 2);
  ctx.fill();

  // Плечи / майка команды (овал поперёк направления движения)
  ctx.fillStyle = teamColor;
  ctx.strokeStyle = '#141210';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.ellipse(0, 0, 9.5, 13.5, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Круглая макушка головы сверху
  ctx.fillStyle = p.skinTone;
  ctx.beginPath();
  ctx.arc(2.5, 0, 6.8, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Причёска / повязка сверху
  if (p.hairStyle === 'afro') {
    ctx.fillStyle = p.hairColor;
    ctx.beginPath();
    ctx.arc(1.2, 0, 7.6, 0, Math.PI * 2);
    ctx.fill();
  } else if (p.hairStyle === 'headband') {
    ctx.fillStyle = p.hairColor;
    ctx.beginPath();
    ctx.arc(1.2, 0, 6.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(2.2, 0, 6.4, -1.1, 1.1);
    ctx.stroke();
  } else if (p.hairStyle === 'mohawk') {
    ctx.fillStyle = p.hairColor;
    ctx.fillRect(-5, -2.3, 13, 4.6);
  } else if (p.hairStyle !== 'bald') {
    ctx.fillStyle = p.hairColor;
    ctx.beginPath();
    ctx.arc(0.8, 0, 6.2, 0.5, Math.PI * 2 - 0.5);
    ctx.fill();
  }

  ctx.restore();

  // 3. Номер и роль чувачка (всегда вертикально читаемые, не вращаются вверх ногами!)
  ctx.save();
  ctx.font = '700 10px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = 'rgba(0,0,0,0.85)';
  ctx.lineWidth = 2.6;
  const labelY = p.y - 17 * totalScale;
  const badgeText = `${POSITION_LABELS[p.role]} #${p.jerseyNumber}`;
  ctx.strokeText(badgeText, p.x, labelY);
  ctx.fillText(badgeText, p.x, labelY);

  // Полоска выносливости под ногами (компактная)
  const barW = 24;
  const barX = p.x - barW / 2;
  const barY = p.y + 16 * totalScale;
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(barX, barY, barW, 3);
  ctx.fillStyle = p.energy > 45 ? '#34d399' : p.energy > 22 ? '#fbbf24' : '#ef4444';
  ctx.fillRect(barX, barY, (barW * Math.max(0, p.energy)) / 100, 3);

  // Иконки особых состояний (Последняя надежда / Кураж / Холод / Месси / Кузнечик)
  if (lastHopeActive) {
    ctx.fillStyle = '#fde047';
    ctx.font = '700 9px Oswald, sans-serif';
    ctx.fillText('★ПН', p.x + 19, p.y + 4);
  } else if (p.hotStreak >= 3) {
    ctx.fillStyle = '#fb923c';
    ctx.font = '700 9px Oswald, sans-serif';
    ctx.fillText(`🔥${p.hotStreak}`, p.x + 19, p.y + 4);
  } else if (p.hotStreak <= -3) {
    ctx.fillStyle = '#7dd3fc';
    ctx.font = '700 9px Oswald, sans-serif';
    ctx.fillText('❄', p.x + 17, p.y + 4);
  } else if (p.traitIds.includes('messi')) {
    ctx.fillStyle = '#fbbf24';
    ctx.font = '700 9px Oswald, sans-serif';
    ctx.fillText('★M', p.x + 17, p.y + 4);
  } else if (p.traitIds.includes('grasshopper')) {
    ctx.fillStyle = '#4ade80';
    ctx.font = '700 9px Oswald, sans-serif';
    ctx.fillText('▲К', p.x + 17, p.y + 4);
  }
  ctx.restore();
}

function drawTopDownBall(ctx: CanvasRenderingContext2D, ball: BallState): void {
  // Шлейф полёта мяча (золотой для удара «Месси»!)
  if (ball.holderId === null) {
    for (let i = 0; i < ball.trail.length; i++) {
      const idx = (ball.trailHead + i) % ball.trail.length;
      const pt = ball.trail[idx];
      if (pt.alpha <= 0.05) continue;
      const frac = (i + 1) / ball.trail.length;
      ctx.save();
      ctx.globalAlpha = frac * (pt.isMessi ? 0.65 : 0.25);
      ctx.fillStyle = pt.isMessi ? '#fbbf24' : '#fdba74';
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, (3 + pt.z * 0.025) * frac, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  // Тень мяча строго в точке (ball.x, ball.y) на паркете — позволяет точно видеть траекторию!
  ctx.save();
  ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
  const shadowR = Math.max(4, 7.5 * (1 - Math.min(0.55, ball.z / 280)));
  ctx.beginPath();
  ctx.arc(ball.x, ball.y, shadowR, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Сам мяч увеличивается к камере по мере подъёма вверх z (смещён совсем чуть-чуть вверх от тени)
  const scale = 1 + ball.z / 155;
  const r = 7.4 * scale;
  const drawX = ball.x;
  const drawY = ball.y - ball.z * 0.08;

  ctx.save();
  ctx.translate(drawX, drawY);
  ctx.rotate(ball.spinAngle);

  ctx.fillStyle = ball.shotType === 'messi' ? '#f59e0b' : '#ea580c';
  ctx.strokeStyle = '#1c1917';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Полоски баскетбольного мяча
  ctx.beginPath();
  ctx.moveTo(-r, 0);
  ctx.lineTo(r, 0);
  ctx.moveTo(0, -r);
  ctx.lineTo(0, r);
  ctx.strokeStyle = 'rgba(28, 25, 23, 0.72)';
  ctx.lineWidth = 1.1;
  ctx.stroke();

  ctx.restore();

  // Высотомер для очень высоко летящего мяча (чтобы было видно высоту дуги Месси)
  if (ball.holderId === null && ball.z > 105) {
    ctx.save();
    ctx.font = '700 9px "JetBrains Mono", monospace';
    ctx.fillStyle = ball.shotType === 'messi' ? '#fde047' : '#e2e8f0';
    ctx.textAlign = 'center';
    ctx.fillText(`↕${Math.round(ball.z)}`, drawX, drawY - r - 4);
    ctx.restore();
  }
}

function drawHoopsOverhead(ctx: CanvasRenderingContext2D, state: MatchState): void {
  const hoops: Array<{ x: number; side: 'left' | 'right'; sway: number }> = [
    { x: HOOP_LEFT_X, side: 'left', sway: state.netSwayLeft },
    { x: HOOP_RIGHT_X, side: 'right', sway: state.netSwayRight },
  ];

  for (const h of hoops) {
    ctx.save();
    const boardX = h.side === 'left' ? h.x - RIM_RADIUS - 4 : h.x + RIM_RADIUS + 4;

    // Сетка внутри кольца (пульсирует при попадании)
    const netR = RIM_RADIUS * (0.72 + h.sway * 0.35);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.65)';
    ctx.lineWidth = 1.4;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(h.x, HOOP_Y, netR, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Металлическая дужка кольца
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 3.4;
    ctx.beginPath();
    ctx.arc(h.x, HOOP_Y, RIM_RADIUS, 0, Math.PI * 2);
    ctx.stroke();

    // Баскетбольный щит сверху (толстая белая планка с оранжевым креплением)
    ctx.fillStyle = '#f8fafc';
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 2;
    ctx.fillRect(boardX - 3.5, HOOP_Y - 34, 7, 68);
    ctx.strokeRect(boardX - 3.5, HOOP_Y - 34, 7, 68);

    ctx.restore();
  }
}

function drawParticles(ctx: CanvasRenderingContext2D, state: MatchState): void {
  for (const p of state.particles) {
    if (!p.active) continue;
    const alpha = Math.max(0, 1 - p.age / p.maxAge);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(p.x, p.y - p.z * 0.06, p.size * (1 + p.z / 240), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

function drawCallouts(ctx: CanvasRenderingContext2D, state: MatchState): void {
  for (const c of state.callouts) {
    if (!c.active) continue;
    const progress = c.age / c.duration;
    const alpha = progress < 0.75 ? 1 : Math.max(0, (1 - progress) / 0.25);
    const lift = progress * 18;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = '700 12px Oswald, sans-serif';
    ctx.textAlign = 'center';
    ctx.strokeStyle = 'rgba(12, 10, 9, 0.92)';
    ctx.lineWidth = 3.5;
    ctx.strokeText(c.text, c.x, c.y - lift);
    ctx.fillStyle = c.color;
    ctx.fillText(c.text, c.x, c.y - lift);
    ctx.restore();
  }
}
