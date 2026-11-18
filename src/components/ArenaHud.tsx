import React from 'react';
import { formatClock, getQuarterLabel } from '../game/sim';
import {
  MATCH_MODIFIERS,
  TRAIT_BY_ID,
  getRarityColor,
  isLastHopeBuffActive,
} from '../game/traits';
import {
  AIActionState,
  CoachPlayCall,
  MatchModifierId,
  MatchState,
  POSITION_LABELS,
  PlayerEntity,
  TeamTactic,
} from '../game/types';

interface ArenaHudProps {
  match: MatchState;
  simSpeed: number;
  muted: boolean;
  showTacticalOverlay: boolean;
  hasActiveTournament: boolean;
  onChangeSpeed: (spd: number) => void;
  onToggleModifier: (mod: MatchModifierId) => void;
  onToggleTacticalOverlay: () => void;
  onChangeTactic: (team: 0 | 1, tactic: TeamTactic) => void;
  onCallTimeout: (team: 0 | 1) => void;
  onTriggerPlayCall: (team: 0 | 1, play: CoachPlayCall) => void;
  onSelectPlayer: (playerId: string) => void;
  onSubstitutePlayer: (playerId: string) => void;
  onOpenSandboxForPlayer: (playerId: string) => void;
  onOpenBoxScore: () => void;
  onOpenTournament: () => void;
  onRerollMatch: () => void;
  onToggleMute: () => void;
  onPause: () => void;
}

const COACH_PLAY_OPTIONS: Array<{ id: CoachPlayCall; key: string; label: string; hint: string }> = [
  { id: 'iso_star', key: '1', label: 'Изоляция', hint: 'Изоляция 1-в-1 для лидера атаки [1]' },
  { id: 'pick_and_roll', key: '2', label: 'Пик-н-ролл', hint: 'Заслон на мяче и рывок под кольцо [2]' },
  { id: 'alley_oop', key: '3', label: 'Алли-уп', hint: 'Навес над кольцом на прыгучего партнёра [3]' },
  { id: 'feed_sniper', key: '4', label: 'Снайпер', hint: 'Вывод лучшего стрелка на открытую дугу [4]' },
];

const TACTIC_OPTIONS: Array<{ value: TeamTactic; label: string }> = [
  { value: 'balanced', label: 'Баланс' },
  { value: 'blitz', label: 'Прессинг' },
  { value: 'perimeter', label: 'Трёшки' },
  { value: 'dirty', label: 'Грязная игра' },
];

const TACTIC_SHORT: Record<TeamTactic, string> = {
  balanced: 'Баланс',
  blitz: 'Прессинг',
  perimeter: 'Трёшки',
  dirty: 'Грязь',
};

const AI_ACTION_LABELS: Record<AIActionState, string> = {
  tipoff: 'Вбрасывание',
  inbound_pass: 'Ввод из аута',
  inbound_wait: 'Под ввод',
  free_throw: 'Штрафной',
  drive: 'Проход',
  pump_fake: 'Ложный замах',
  ankle_breaker: 'Кроссовер',
  step_back: 'Степ-бэк',
  eurostep: 'Евростеп',
  floater: 'Флоутер',
  post_up: 'Игра в посте',
  handoff: 'Из рук в руки',
  shoot_windup: 'Бросок',
  messi_windup: 'Удар Месси',
  pass: 'Передача',
  receive_pass: 'Приём паса',
  cut_open: 'Спейсинг',
  relocate_arc: 'Смещение по дуге',
  backdoor_cut: 'Рывок за спину',
  give_and_go: 'Дай и беги',
  screen_set: 'Заслон',
  offball_screen: 'Заслон снайперу',
  roll_to_rim: 'Пик-н-ролл',
  pick_and_pop: 'Пик-н-поп',
  fast_break: 'Быстрый отрыв',
  rebound_position: 'Бокс-аут',
  safety_back: 'Страховка тыла',
  chase_ball: 'Подбор',
  guard_carrier: 'Опека мяча',
  closeout: 'Клоузаут на дугу',
  take_charge: 'Под фол в атаке',
  chasedown_block: 'Блок вдогонку',
  intentional_foul: 'Тактический фол',
  help_defense: 'Подстраховка',
  mark_man: 'Персоналка',
  rim_protect: 'Защита кольца',
  intercept_air: 'Перехват',
  bread_mode: 'Хлебушек',
  dirty_play: 'Пакость',
  tripped_down: 'Сбит с ног',
  penalty_bench: 'Штрафник',
  team_bench: 'В запасе',
  celebrate: 'Празднует',
};

export const ArenaHud: React.FC<ArenaHudProps> = ({
  match,
  simSpeed,
  muted,
  showTacticalOverlay,
  hasActiveTournament,
  onChangeSpeed,
  onToggleModifier,
  onToggleTacticalOverlay,
  onChangeTactic,
  onCallTimeout,
  onTriggerPlayCall,
  onSelectPlayer,
  onSubstitutePlayer,
  onOpenSandboxForPlayer,
  onOpenBoxScore,
  onOpenTournament,
  onRerollMatch,
  onToggleMute,
  onPause,
}) => {
  const allPlayers: PlayerEntity[] = [...match.players, ...match.bench];
  const selectedPlayer: PlayerEntity | undefined =
    allPlayers.find((p) => p.id === match.selectedPlayerId) ?? match.players[0];

  const selectAdjacentPlayer = (delta: number) => {
    if (allPlayers.length === 0) return;
    const curIdx = Math.max(0, allPlayers.findIndex((p) => p.id === selectedPlayer?.id));
    const nextIdx = (curIdx + delta + allPlayers.length) % allPlayers.length;
    onSelectPlayer(allPlayers[nextIdx].id);
  };

  const canSubstituteSelected = selectedPlayer
    ? selectedPlayer.onCourt
      ? match.bench.some(
          (b) => b.team === selectedPlayer.team && !b.disqualified && b.fouls < 5
        )
      : !selectedPlayer.disqualified && selectedPlayer.fouls < 5
    : false;
  const selectedLastHopeActive = selectedPlayer
    ? isLastHopeBuffActive(selectedPlayer)
    : false;

  return (
    <div className="fixed inset-0 z-10 pointer-events-none flex flex-col justify-between">
      {/* ВЕРХНЯЯ ПЕРИМЕТРАЛЬНАЯ ПАНЕЛЬ ТАБЛО */}
      <div className="w-full px-2.5 pt-2 flex items-center justify-between gap-2 max-w-full">
        {/* Левый кластер: тактика команды и модификаторы матча */}
        <div className="pointer-events-auto flex items-center gap-1 bg-[#1c1917] border border-[#3b342e] rounded-lg px-2 py-1 shadow-md shrink-0">
          <label className="flex items-center gap-1 text-[11px] text-[#a8a29e] whitespace-nowrap">
            <span className="hidden 2xl:inline">Тактика:</span>
            <select
              aria-label="Тактика первой команды"
              value={match.tactics[0]}
              onChange={(e) => onChangeTactic(0, e.target.value as TeamTactic)}
              className="bg-[#26221f] text-[#f5f2eb] text-[11px] font-semibold px-1.5 py-1 rounded border border-[#443c35] cursor-pointer focus:outline-none focus:border-[#fbbf24]"
            >
              {TACTIC_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            title="Показывать тактические линии, рывки и заслоны ИИ [T]"
            onClick={(e) => {
              e.currentTarget.blur();
              onToggleTacticalOverlay();
            }}
            className={`px-1.5 py-1 text-[11px] font-semibold rounded transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
              showTacticalOverlay
                ? 'bg-[#292524] border border-[#fbbf24]/60 text-[#fde68a]'
                : 'bg-[#26221f] text-[#a8a29e] hover:text-[#f5f2eb]'
            }`}
          >
            Схемы
          </button>

          <div className="hidden lg:flex items-center gap-1" role="group" aria-label="Модификаторы матча">
            {MATCH_MODIFIERS.map((mod) => {
              const active = match.modifiers.includes(mod.id);
              return (
                <button
                  key={mod.id}
                  type="button"
                  title={mod.description}
                  onClick={(e) => {
                    e.currentTarget.blur();
                    onToggleModifier(mod.id);
                  }}
                  className={`px-1.5 py-1 text-[11px] font-semibold rounded transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                    active
                      ? 'bg-[#ea580c] text-[#fafaf9]'
                      : 'bg-[#26221f] text-[#a8a29e] hover:text-[#f5f2eb]'
                  }`}
                >
                  {mod.shortLabel}
                </button>
              );
            })}
          </div>
        </div>

        {/* Центральное табло арены */}
        <div className="pointer-events-auto flex items-center gap-2.5 sm:gap-3.5 bg-[#1c1917] border border-[#443c35] rounded-lg px-3 py-1 shadow-lg shrink-0">
          <div className="flex items-center gap-2">
            <span
              className="w-2.5 h-2.5 rounded-sm shrink-0"
              style={{ backgroundColor: match.teamColors[0] }}
              aria-hidden="true"
            />
            <div className="text-right">
              <div className="text-xs font-bold text-[#f5f2eb] truncate max-w-[110px] xl:max-w-[140px]">
                {match.teamNames[0]}
              </div>
              <div className="text-[10px] text-[#a8a29e] font-mono-num">
                Фолы:{' '}
                <span className={match.teamFouls[0] >= 5 ? 'text-[#f87171] font-bold' : ''}>
                  {match.teamFouls[0]}
                </span>{' '}
                · {TACTIC_SHORT[match.tactics[0]]}
              </div>
            </div>
            <span className="font-display text-2xl font-bold text-[#fafaf9] tabular-nums min-w-[28px] text-center">
              {match.score[0]}
            </span>
          </div>

          <div className="flex flex-col items-center px-2 border-x border-[#3b342e]">
            <div className="flex items-center gap-1 text-[11px] font-bold tracking-wider text-[#fbbf24]">
              <span
                title="Стрелка поочерёдного владения FIBA"
                className={match.possessionArrow === 0 ? 'text-[#fbbf24]' : 'text-[#57534e]'}
              >
                ◄
              </span>
              <span>{getQuarterLabel(match.quarter)}</span>
              <span aria-hidden="true">·</span>
              <span className="font-mono-num text-[#fafaf9] text-xs">
                {formatClock(match.quarterTimeRemaining)}
              </span>
              <span
                title="Стрелка поочерёдного владения FIBA"
                className={match.possessionArrow === 1 ? 'text-[#fbbf24]' : 'text-[#57534e]'}
              >
                ►
              </span>
            </div>
            <div className="text-[10px] font-mono-num text-[#a8a29e] flex items-center gap-1.5">
              <span>
                24с:{' '}
                <span className={match.shotClock <= 5 ? 'text-[#f87171] font-bold' : 'text-[#e7e5e4]'}>
                  {Math.ceil(match.shotClock)}
                </span>
              </span>
              {match.possessionTeam !== null &&
                !match.frontcourtEstablished &&
                !match.inbound &&
                !match.freeThrow &&
                match.balls.length === 1 && (
                  <span title="Правило 8 секунд на вывод мяча со своей половины">
                    · 8с:{' '}
                    <span
                      className={
                        8 - match.backcourtTimer <= 2.5 ? 'text-[#f87171] font-bold' : 'text-[#fde68a]'
                      }
                    >
                      {Math.max(0, Math.ceil(8 - match.backcourtTimer))}
                    </span>
                  </span>
                )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="font-display text-2xl font-bold text-[#fafaf9] tabular-nums min-w-[28px] text-center">
              {match.score[1]}
            </span>
            <div className="text-left">
              <div className="text-xs font-bold text-[#f5f2eb] truncate max-w-[110px] xl:max-w-[140px]">
                {match.teamNames[1]}
              </div>
              <div className="text-[10px] text-[#a8a29e] font-mono-num">
                Фолы:{' '}
                <span className={match.teamFouls[1] >= 5 ? 'text-[#f87171] font-bold' : ''}>
                  {match.teamFouls[1]}
                </span>{' '}
                · {TACTIC_SHORT[match.tactics[1]]}
              </div>
            </div>
            <span
              className="w-2.5 h-2.5 rounded-sm shrink-0"
              style={{ backgroundColor: match.teamColors[1] }}
              aria-hidden="true"
            />
          </div>
        </div>

        {/* Правый кластер: скорость симуляции, песочница, турнир, звук, пауза */}
        <div className="pointer-events-auto flex items-center gap-1 bg-[#1c1917] border border-[#3b342e] rounded-lg px-1.5 py-1 shadow-md shrink-0">
          <div className="flex items-center bg-[#26221f] rounded p-0.5" role="group" aria-label="Скорость игры">
            {[0.5, 1, 2, 4].map((spd) => (
              <button
                key={spd}
                type="button"
                onClick={(e) => {
                  e.currentTarget.blur();
                  onChangeSpeed(spd);
                }}
                className={`px-1.5 py-1 text-[11px] font-mono-num font-bold rounded transition-colors cursor-pointer whitespace-nowrap ${
                  simSpeed === spd
                    ? 'bg-[#ea580c] text-[#fafaf9]'
                    : 'text-[#a8a29e] hover:text-[#f5f2eb]'
                }`}
              >
                {spd}×
              </button>
            ))}
          </div>

          <button
            type="button"
            title="Статистика игроков и карта бросков [Tab]"
            onClick={(e) => {
              e.currentTarget.blur();
              onOpenBoxScore();
            }}
            className="px-2 py-1.5 bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] text-xs font-semibold text-[#f5f2eb] rounded cursor-pointer whitespace-nowrap shrink-0"
          >
            Протокол
          </button>

          <button
            type="button"
            onClick={(e) => {
              e.currentTarget.blur();
              if (selectedPlayer) onOpenSandboxForPlayer(selectedPlayer.id);
            }}
            className="px-2 py-1.5 bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] text-xs font-semibold text-[#f5f2eb] rounded cursor-pointer whitespace-nowrap shrink-0"
          >
            Песочница
          </button>

          <button
            type="button"
            onClick={(e) => {
              e.currentTarget.blur();
              onOpenTournament();
            }}
            className={`px-2 py-1.5 border text-xs font-semibold rounded cursor-pointer whitespace-nowrap shrink-0 ${
              hasActiveTournament
                ? 'bg-[#292524] border-[#fbbf24] text-[#fbbf24]'
                : 'bg-[#26221f] hover:bg-[#322d29] border-[#443c35] text-[#d6d3d1]'
            }`}
          >
            Турнир
          </button>

          {match.mode === 'exhibition' && (
            <button
              type="button"
              aria-label="Пересобрать составы"
              title="Новые случайные составы [R]"
              onClick={(e) => {
                e.currentTarget.blur();
                onRerollMatch();
              }}
              className="w-8 h-8 flex items-center justify-center bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] rounded text-[#e7e5e4] cursor-pointer shrink-0"
            >
              <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.2">
                <path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
                <path d="M21 3v5h-5" />
                <path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
                <path d="M3 21v-5h5" />
              </svg>
            </button>
          )}

          <button
            type="button"
            aria-label={muted ? 'Включить звук' : 'Выключить звук'}
            title="Звук [M]"
            onClick={(e) => {
              e.currentTarget.blur();
              onToggleMute();
            }}
            className="w-8 h-8 flex items-center justify-center bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] rounded text-[#e7e5e4] cursor-pointer shrink-0"
          >
            {muted ? (
              <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.2">
                <path d="M11 5L6 9H2v6h4l5 4V5z" />
                <line x1="23" y1="9" x2="17" y2="15" />
                <line x1="17" y1="9" x2="23" y2="15" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.2">
                <path d="M11 5L6 9H2v6h4l5 4V5z" />
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
              </svg>
            )}
          </button>

          <button
            type="button"
            aria-label="Пауза"
            title="Пауза [P / Esc]"
            onClick={(e) => {
              e.currentTarget.blur();
              onPause();
            }}
            className="w-8 h-8 flex items-center justify-center bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] rounded text-[#e7e5e4] cursor-pointer shrink-0"
          >
            <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          </button>
        </div>
      </div>

      {/* ВСПЛЫВАЮЩИЙ БАННЕР СОБЫТИЯ (ГОЛ / АВТОГОЛ МЕССИ / ОВЕРТАЙМ) */}
      {match.lastBannerText && (
        <div className="self-center mt-2 px-5 py-1.5 bg-[#1c1917] border-2 rounded-lg shadow-xl" style={{ borderColor: match.lastBannerColor }}>
          <span className="font-display text-base sm:text-lg tracking-wide text-[#fafaf9]">
            {match.lastBannerText}
          </span>
        </div>
      )}

      {/* НИЖНЯЯ КОМПАКТНАЯ ПАНЕЛЬ ДОСЬЕ ИГРОКА И СУДЕЙСКОГО ПРОТОКОЛА */}
      <div className="w-full px-3 pb-2.5 flex flex-col md:flex-row items-stretch md:items-end justify-between gap-2.5">
        {/* Левый док: Планшет тренера + карточка выбранного чувачка */}
        {selectedPlayer && (
          <div className="flex flex-col gap-1.5 max-w-2xl w-full">
            {/* Компактный тренерский планшет: комбинации [1-4] и минутный тайм-аут */}
            <div className="pointer-events-auto flex items-center justify-between gap-2 bg-[#1c1917] border border-[#3b342e] rounded-lg px-2.5 py-1 shadow-md overflow-x-auto">
              <div className="flex items-center gap-1 shrink-0" role="group" aria-label="Комбинации тренера">
                <span className="text-[11px] font-semibold text-[#a8a29e] mr-1 whitespace-nowrap">
                  Планшет:
                </span>
                {COACH_PLAY_OPTIONS.map((opt) => {
                  const isActive =
                    match.activePlayCall?.team === 0 && match.activePlayCall.play === opt.id;
                  const onCd = match.playCallCooldown[0] > 0 && !isActive;
                  return (
                    <button
                      key={opt.id}
                      type="button"
                      disabled={match.playCallCooldown[0] > 0}
                      title={opt.hint}
                      onClick={(e) => {
                        e.currentTarget.blur();
                        onTriggerPlayCall(0, opt.id);
                      }}
                      className={`px-2 py-0.5 text-[11px] font-semibold rounded transition-colors whitespace-nowrap shrink-0 ${
                        isActive
                          ? 'bg-[#ea580c] text-[#fafaf9] cursor-default'
                          : onCd
                          ? 'bg-[#211d1a] text-[#78716c] cursor-not-allowed'
                          : 'bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] text-[#e7e5e4] cursor-pointer'
                      }`}
                    >
                      <span className="font-mono-num text-[#fbbf24] mr-1">{opt.key}</span>{' '}
                      {opt.label}
                    </button>
                  );
                })}
                {match.playCallCooldown[0] > 0 && (
                  <span className="text-[10px] font-mono-num text-[#a8a29e] ml-1 whitespace-nowrap">
                    КД {Math.ceil(match.playCallCooldown[0])}с
                  </span>
                )}
              </div>

              <button
                type="button"
                disabled={match.timeoutsRemaining[0] <= 0}
                title="Взять тренерский тайм-аут: +22% сил пятёрке, сбить рывок и кураж соперника [O]"
                onClick={(e) => {
                  e.currentTarget.blur();
                  onCallTimeout(0);
                }}
                className={`px-2.5 py-0.5 text-[11px] font-mono-num font-semibold rounded border whitespace-nowrap shrink-0 transition-colors ${
                  match.timeoutsRemaining[0] > 0
                    ? 'bg-[#26221f] hover:bg-[#322d29] border-[#fbbf24]/60 text-[#fde68a] cursor-pointer'
                    : 'bg-[#191614] border-[#292524] text-[#57534e] cursor-not-allowed'
                }`}
              >
                Тайм-аут ({match.timeoutsRemaining[0]})
              </button>
            </div>

          <div className="pointer-events-auto bg-[#1c1917] border border-[#3b342e] rounded-lg px-3.5 py-2.5 shadow-lg w-full">
            <div className="flex items-center justify-between gap-2.5 mb-1.5">
              <div className="flex items-center gap-2 min-w-0">
                <button
                  type="button"
                  aria-label="Предыдущий игрок"
                  onClick={() => selectAdjacentPlayer(-1)}
                  className="w-6 h-6 flex items-center justify-center bg-[#26221f] hover:bg-[#322d29] rounded text-xs text-[#d6d3d1] cursor-pointer shrink-0"
                >
                  ‹
                </button>
                <span
                  className="w-2.5 h-2.5 rounded-sm shrink-0"
                  style={{ backgroundColor: match.teamColors[selectedPlayer.team] }}
                  aria-hidden="true"
                />
                <span className="font-bold text-sm text-[#fafaf9] truncate max-w-[175px] shrink-0">
                  {selectedPlayer.name}
                </span>
                {selectedLastHopeActive ? (
                  <span className="text-[11px] font-bold text-[#fde047] shrink-0">
                    ★ ПН
                  </span>
                ) : !selectedPlayer.onCourt ? (
                  <span className="text-[11px] font-semibold text-[#a8a29e] shrink-0">
                    {selectedPlayer.disqualified ? '· 5Ф' : '· Запас'}
                  </span>
                ) : null}
                {selectedPlayer.hotStreak >= 3 && (
                  <span className="text-[11px] font-bold text-[#fb923c] shrink-0">
                    · В ОГНЕ
                  </span>
                )}
                {selectedPlayer.hotStreak <= -3 && (
                  <span className="text-[11px] font-bold text-[#7dd3fc] shrink-0">
                    · ХОЛОД
                  </span>
                )}
                <span className="text-xs text-[#a8a29e] font-mono-num whitespace-nowrap hidden sm:inline">
                  {POSITION_LABELS[selectedPlayer.role]} #{selectedPlayer.jerseyNumber} · Очк:{selectedPlayer.points} · Пдб:{selectedPlayer.rebounds} · Прд:{selectedPlayer.assists} · Ф:{selectedPlayer.fouls} · {Math.round(selectedPlayer.energy)}%
                </span>
                <button
                  type="button"
                  aria-label="Следующий игрок"
                  onClick={() => selectAdjacentPlayer(1)}
                  className="w-6 h-6 flex items-center justify-center bg-[#26221f] hover:bg-[#322d29] rounded text-xs text-[#d6d3d1] cursor-pointer shrink-0"
                >
                  ›
                </button>
              </div>

              <div className="flex items-center gap-1.5 shrink-0">
                {canSubstituteSelected && (
                  <button
                    type="button"
                    title={
                      selectedPlayer.onCourt
                        ? 'Заменить игрока на свежего запасного со скамейки'
                        : 'Выпустить этого запасного игрока на площадку!'
                    }
                    onClick={(e) => {
                      e.currentTarget.blur();
                      onSubstitutePlayer(selectedPlayer.id);
                    }}
                    className="px-2 py-1 bg-[#26221f] hover:bg-[#322d29] border border-[#38bdf8]/60 text-[11px] font-semibold text-[#7dd3fc] rounded cursor-pointer whitespace-nowrap"
                  >
                    {selectedPlayer.onCourt ? '⇄ Заменить' : '⇄ На поле'}
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => onOpenSandboxForPlayer(selectedPlayer.id)}
                  className="px-2.5 py-1 bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] text-[11px] font-semibold text-[#fbbf24] rounded cursor-pointer whitespace-nowrap"
                >
                  Черты и статы
                </button>
              </div>
            </div>

            {/* Мысль ИИ и черты без капсульных пилюль */}
            <div className="text-xs text-[#d6d3d1] mb-1.5 truncate">
              <span className="text-[#a8a29e]">Мозг ИИ · </span>
              <span className="font-semibold text-[#fde68a]">
                {AI_ACTION_LABELS[selectedPlayer.aiAction]}:
              </span>{' '}
              {selectedPlayer.aiThought}
            </div>

            <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs mb-1.5">
              {selectedPlayer.traitIds.length === 0 ? (
                <span className="text-[#78716c]">Без особых черт</span>
              ) : (
                selectedPlayer.traitIds.map((tid, idx) => {
                  const def = TRAIT_BY_ID[tid];
                  if (!def) return null;
                  return (
                    <React.Fragment key={tid}>
                      {idx > 0 && (
                        <span className="text-[#57534e]" aria-hidden="true">
                          ·
                        </span>
                      )}
                      <span title={def.description} className="font-semibold" style={{ color: getRarityColor(def.rarity) }}>
                        {def.name} [{def.rarityLabel}]
                      </span>
                    </React.Fragment>
                  );
                })
              )}
            </div>

            {/* Компактная строка ключевых характеристик */}
            <div className="grid grid-cols-5 sm:grid-cols-9 gap-1.5 text-[11px] font-mono-num text-[#a8a29e] border-t border-[#292524] pt-1.5">
              <span>СКР <strong className="text-[#f5f2eb]">{selectedPlayer.effectiveStats.speed}</strong></span>
              <span>БРС <strong className="text-[#f5f2eb]">{selectedPlayer.effectiveStats.shooting}</strong></span>
              <span>ВЫН <strong className="text-[#f5f2eb]">{selectedPlayer.effectiveStats.stamina}</strong></span>
              <span>ПРХ <strong className="text-[#f5f2eb]">{selectedPlayer.effectiveStats.steal}</strong></span>
              <span>ПРЖ <strong className="text-[#f5f2eb]">{selectedPlayer.effectiveStats.jump}</strong></span>
              <span>КОМ <strong className="text-[#f5f2eb]">{selectedPlayer.effectiveStats.teamwork}</strong></span>
              <span>ДРБ <strong className="text-[#f5f2eb]">{selectedPlayer.effectiveStats.dribble}</strong></span>
              <span>ИИ <strong className="text-[#f5f2eb]">{selectedPlayer.effectiveStats.iq}</strong></span>
              <span title="Девиантность (склонность к подножкам и пинкам)">
                ДЕВ <strong className={selectedPlayer.effectiveStats.deviance >= 65 ? 'text-[#f87171]' : 'text-[#f5f2eb]'}>{selectedPlayer.effectiveStats.deviance}</strong>
              </span>
            </div>
          </div>
          </div>
        )}

        {/* Правый компактный лог судьи и ярких моментов */}
        <div className="pointer-events-auto hidden sm:block bg-[#1c1917] border border-[#3b342e] rounded-lg px-3 py-2 shadow-lg w-96 lg:w-[440px]">
          <div className="flex items-center justify-between gap-2 text-[11px] text-[#a8a29e] mb-1 whitespace-nowrap">
            <span className="font-semibold text-[#d6d3d1]">Протокол Судьи</span>
            <span className="font-mono-num">
              Свистки: {match.referee.callsMade} · Наруш: {match.violationsCount[0] + match.violationsCount[1]} · Проспал: {match.referee.missedFouls}
            </span>
          </div>
          <div className="space-y-1 max-h-16 overflow-y-auto text-xs">
            {match.events.slice(0, 3).map((ev) => (
              <div
                key={ev.id}
                className={`truncate ${
                  ev.team === 'ref'
                    ? 'text-[#fca5a5]'
                    : ev.highlight
                    ? 'text-[#fde68a] font-semibold'
                    : 'text-[#d6d3d1]'
                }`}
              >
                <span className="font-mono-num text-[#78716c] mr-1.5">
                  {ev.quarterLabel} {ev.timeFormatted}
                </span>
                {ev.text}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
