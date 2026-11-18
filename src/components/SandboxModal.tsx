import React, { useState } from 'react';
import {
  MATCH_MODIFIERS,
  TRAIT_CATALOG,
  getPlayerOverall,
  getRarityColor,
  isLastHopeBuffActive,
} from '../game/traits';
import {
  MatchModifierId,
  MatchState,
  POSITION_LABELS,
  PlayerStats,
  TraitRarity,
} from '../game/types';

interface SandboxModalProps {
  match: MatchState;
  initialPlayerId: string | null;
  onUpdatePlayer: (playerId: string, traitIds: string[], baseStats: PlayerStats) => void;
  onSubstitutePlayer: (playerId: string) => void;
  onToggleModifier: (modId: MatchModifierId) => void;
  onChangeQuarterDuration: (sec: number) => void;
  onRerollAllPlayers: () => void;
  onCloseAndPlay: () => void;
  onClose: () => void;
}

const STAT_META: Array<{ key: keyof PlayerStats; label: string }> = [
  { key: 'speed', label: 'Скорость' },
  { key: 'shooting', label: 'Точность броска' },
  { key: 'stamina', label: 'Выносливость' },
  { key: 'steal', label: 'Перехват мяча' },
  { key: 'jump', label: 'Высота прыжка' },
  { key: 'teamwork', label: 'Командная игра' },
  { key: 'dribble', label: 'Дриблинг' },
  { key: 'iq', label: 'Баскетбольный ИИ' },
  { key: 'deviance', label: 'Девиантность (грязь)' },
];

export const SandboxModal: React.FC<SandboxModalProps> = ({
  match,
  initialPlayerId,
  onUpdatePlayer,
  onSubstitutePlayer,
  onToggleModifier,
  onChangeQuarterDuration,
  onRerollAllPlayers,
  onCloseAndPlay,
  onClose,
}) => {
  const allRoster = [...match.players, ...match.bench];
  const [activePlayerId, setActivePlayerId] = useState<string>(
    initialPlayerId ?? match.players[0]?.id ?? ''
  );
  const [rarityFilter, setRarityFilter] = useState<'all' | TraitRarity>('all');

  const activePlayer =
    allRoster.find((p) => p.id === activePlayerId) ?? match.players[0];

  const filteredTraits = TRAIT_CATALOG.filter((t) =>
    rarityFilter === 'all' ? true : t.rarity === rarityFilter
  );

  const handleToggleTrait = (traitId: string) => {
    if (!activePlayer) return;
    const has = activePlayer.traitIds.includes(traitId);
    let nextTraits: string[];
    if (has) {
      nextTraits = activePlayer.traitIds.filter((id) => id !== traitId);
    } else {
      // Убираем взаимоисключающий рост, если выбран противоположный
      let cleaned = [...activePlayer.traitIds];
      if (traitId === 'lanky_beanpole') {
        cleaned = cleaned.filter((id) => id !== 'short_spark');
      } else if (traitId === 'short_spark') {
        cleaned = cleaned.filter((id) => id !== 'lanky_beanpole');
      }
      nextTraits = [...cleaned, traitId];
    }
    onUpdatePlayer(activePlayer.id, nextTraits, activePlayer.baseStats);
  };

  const handleStatChange = (key: keyof PlayerStats, val: number) => {
    if (!activePlayer) return;
    const nextBase: PlayerStats = {
      ...activePlayer.baseStats,
      [key]: val,
    };
    onUpdatePlayer(activePlayer.id, activePlayer.traitIds, nextBase);
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="sandbox-modal-title"
      className="fixed inset-0 z-40 flex items-center justify-center bg-[#0c0a09]/85 p-3 sm:p-5 overflow-y-auto"
    >
      <div className="bg-[#1c1917] border border-[#443c35] rounded-xl w-full max-w-6xl max-h-[92vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Шапка окна песочницы */}
        <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 border-b border-[#3b342e] bg-[#141210]">
          <div>
            <h2 id="sandbox-modal-title" className="font-display text-2xl text-[#fafaf9]">
              ПЕСОЧНИЦА ЧЕРТ, СОСТАВОВ И МОДИФИКАТОРОВ
            </h2>
            <p className="text-xs text-[#a8a29e]">
              Выдавайте любые черты вручную, крутите характеристики (включая Девиантность) и включайте безумные режимы площадки.
            </p>
          </div>

          <div className="flex items-center gap-2">
            {match.mode === 'exhibition' && (
              <button
                type="button"
                onClick={onRerollAllPlayers}
                className="px-3.5 py-2 bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] text-xs font-semibold text-[#f5f2eb] rounded-lg cursor-pointer whitespace-nowrap"
              >
                Пересобрать всех 16 балбесов
              </button>
            )}
            <button
              type="button"
              onClick={onCloseAndPlay}
              className="px-4 py-2 bg-[#ea580c] hover:bg-[#f97316] text-xs font-semibold text-[#fafaf9] rounded-lg cursor-pointer whitespace-nowrap"
            >
              На паркет →
            </button>
            <button
              type="button"
              aria-label="Закрыть песочницу"
              onClick={onClose}
              className="w-8 h-8 flex items-center justify-center bg-[#26221f] hover:bg-[#322d29] rounded-lg text-sm text-[#a8a29e] hover:text-[#fafaf9] cursor-pointer"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Полоса модификаторов матча и длины четверти */}
        <div className="px-5 py-2.5 bg-[#201d1a] border-b border-[#3b342e] flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs font-semibold text-[#a8a29e] mr-1">Модификаторы:</span>
            {MATCH_MODIFIERS.map((mod) => {
              const isOn = match.modifiers.includes(mod.id);
              return (
                <button
                  key={mod.id}
                  type="button"
                  title={mod.description}
                  onClick={() => onToggleModifier(mod.id)}
                  className={`px-3 py-1 text-xs font-semibold rounded-md border transition-colors cursor-pointer whitespace-nowrap ${
                    isOn
                      ? 'bg-[#ea580c] border-[#fb923c] text-[#fafaf9]'
                      : 'bg-[#141210] border-[#3b342e] text-[#a8a29e] hover:text-[#f5f2eb]'
                  }`}
                >
                  {mod.name}
                </button>
              );
            })}
          </div>

          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-[#a8a29e]">Четверть:</span>
            {[30, 45, 60].map((sec) => (
              <button
                key={sec}
                type="button"
                onClick={() => onChangeQuarterDuration(sec)}
                className={`px-2.5 py-1 font-mono-num font-semibold rounded cursor-pointer ${
                  match.quarterDuration === sec
                    ? 'bg-[#3b342e] text-[#fbbf24]'
                    : 'bg-[#141210] text-[#a8a29e] hover:text-[#f5f2eb]'
                }`}
              >
                {sec}с
              </button>
            ))}
          </div>
        </div>

        {/* Основная рабочая область: Список 10 игроков | Характеристики | Каталог 27 черт */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 overflow-y-auto divide-y lg:divide-y-0 lg:divide-x divide-[#3b342e]">
          {/* Колонка 1: Выбор чувачка (На поле 5 + Запасные 3 для каждой команды) */}
          <div className="lg:col-span-3 p-4 overflow-y-auto space-y-4">
            {[0, 1].map((teamIdx) => {
              const teamCourt = match.players.filter((p) => p.team === teamIdx);
              const teamBench = match.bench.filter((p) => p.team === teamIdx);
              return (
                <div key={teamIdx}>
                  <div className="flex items-center gap-2 mb-1.5">
                    <span
                      className="w-2.5 h-2.5 rounded-sm"
                      style={{ backgroundColor: match.teamColors[teamIdx] }}
                      aria-hidden="true"
                    />
                    <span className="text-xs font-bold text-[#e7e5e4] truncate">
                      {match.teamNames[teamIdx]}
                    </span>
                  </div>

                  <div className="text-[10px] font-bold uppercase tracking-wider text-[#a8a29e] mb-1">
                    На площадке (5)
                  </div>
                  <div className="space-y-1 mb-2.5">
                    {teamCourt.map((p) => {
                      const selected = p.id === activePlayer?.id;
                      const lhOn = isLastHopeBuffActive(p);
                      return (
                        <button
                          key={p.id}
                          type="button"
                          onClick={() => setActivePlayerId(p.id)}
                          className={`w-full text-left px-3 py-1.5 rounded-lg border transition-colors cursor-pointer ${
                            selected
                              ? 'bg-[#292524] border-[#fbbf24] text-[#fafaf9]'
                              : 'bg-[#141210] border-[#292524] text-[#d6d3d1] hover:border-[#443c35]'
                          }`}
                        >
                          <div className="flex items-center justify-between text-xs font-semibold">
                            <span className="truncate">
                              {POSITION_LABELS[p.role]} · {p.name}
                            </span>
                            <span className="font-mono-num text-[#fbbf24] ml-1">
                              {lhOn ? `★${getPlayerOverall(p)}` : getPlayerOverall(p)}
                            </span>
                          </div>
                          <div className="text-[11px] text-[#a8a29e] truncate mt-0.5">
                            Черты: {p.traitIds.length} · Силы: {Math.round(p.energy)}%
                          </div>
                        </button>
                      );
                    })}
                  </div>

                  <div className="text-[10px] font-bold uppercase tracking-wider text-[#7dd3fc] mb-1">
                    Скамейка запасных ({teamBench.length})
                  </div>
                  <div className="space-y-1">
                    {teamBench.map((p) => {
                      const selected = p.id === activePlayer?.id;
                      const isJoker = p.traitIds.includes('last_hope') && p.startedOnBench;
                      return (
                        <button
                          key={p.id}
                          type="button"
                          onClick={() => setActivePlayerId(p.id)}
                          className={`w-full text-left px-3 py-1.5 rounded-lg border transition-colors cursor-pointer ${
                            selected
                              ? 'bg-[#292524] border-[#38bdf8] text-[#fafaf9]'
                              : 'bg-[#171513] border-[#292524] text-[#d6d3d1] hover:border-[#443c35]'
                          }`}
                        >
                          <div className="flex items-center justify-between text-xs font-semibold">
                            <span className="truncate">
                              [ЗАПАС] {POSITION_LABELS[p.role]} · {p.name}
                            </span>
                            <span className="font-mono-num text-[#7dd3fc] ml-1">
                              {getPlayerOverall(p)}
                            </span>
                          </div>
                          <div className="text-[11px] text-[#a8a29e] truncate mt-0.5">
                            {p.disqualified
                              ? 'Удалён (5 фолов)'
                              : isJoker
                              ? '★ Готов к «Последней надежде»!'
                              : `Черты: ${p.traitIds.length} · Силы: ${Math.round(p.energy)}%`}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Колонка 2: Ползунки 9 характеристик выбранного игрока */}
          {activePlayer && (
            <div className="lg:col-span-4 p-4 overflow-y-auto">
              <div className="mb-3 pb-2.5 border-b border-[#292524]">
                <div className="flex items-center justify-between gap-2">
                  <div className="text-base font-bold text-[#fafaf9] truncate">
                    {activePlayer.name}{' '}
                    <span className="text-xs font-mono-num text-[#a8a29e]">
                      ({POSITION_LABELS[activePlayer.role]} #{activePlayer.jerseyNumber})
                    </span>
                  </div>
                  {!activePlayer.disqualified && (
                    <button
                      type="button"
                      onClick={() => onSubstitutePlayer(activePlayer.id)}
                      className="px-2.5 py-1 bg-[#26221f] hover:bg-[#322d29] border border-[#38bdf8]/60 text-[11px] font-semibold text-[#7dd3fc] rounded cursor-pointer whitespace-nowrap shrink-0"
                    >
                      {activePlayer.onCourt ? '⇄ В запас' : '⇄ Выпустить на поле'}
                    </button>
                  )}
                </div>
                <div className="text-xs text-[#a8a29e] mt-1">
                  {isLastHopeBuffActive(activePlayer) ? (
                    <span className="text-[#fde047] font-semibold">
                      ★ «Последняя надежда» АКТИВНА: вышел со скамейки и получил огромные баффы!
                    </span>
                  ) : activePlayer.onCourt ? (
                    <span>
                      Статус: <strong className="text-[#4ade80]">На поле</strong>
                      {activePlayer.startedOnBench ? ' (вышел из запаса)' : ' (стартовая пятёрка)'}
                    </span>
                  ) : (
                    <span>
                      Статус: <strong className="text-[#7dd3fc]">Запасной на скамейке</strong> — выдайте «Последнюю надежду» и выпустите на поле!
                    </span>
                  )}
                </div>
              </div>

              <div className="space-y-2.5">
                {STAT_META.map(({ key, label }) => {
                  const baseVal = activePlayer.baseStats[key];
                  const effVal = activePlayer.effectiveStats[key];
                  const diff = effVal - baseVal;
                  return (
                    <div key={key}>
                      <div className="flex items-center justify-between text-xs mb-0.5">
                        <label htmlFor={`stat-${key}`} className="text-[#d6d3d1] font-medium">
                          {label}
                        </label>
                        <span className="font-mono-num text-[#fafaf9]">
                          {effVal}
                          {diff !== 0 && (
                            <span
                              className={`ml-1 text-[11px] ${
                                diff > 0 ? 'text-[#4ade80]' : 'text-[#f87171]'
                              }`}
                            >
                              ({diff > 0 ? `+${diff}` : diff})
                            </span>
                          )}
                        </span>
                      </div>
                      <input
                        id={`stat-${key}`}
                        type="range"
                        min={key === 'teamwork' || key === 'deviance' ? 0 : 10}
                        max={100}
                        value={baseVal}
                        onChange={(e) => handleStatChange(key, Number(e.target.value))}
                        className="w-full accent-[#ea580c] cursor-pointer h-1.5 bg-[#292524] rounded-lg"
                      />
                    </div>
                  );
                })}
              </div>

              <div className="mt-4 pt-3 border-t border-[#292524] flex items-center justify-between">
                <span className="text-xs text-[#a8a29e]">
                  Активных черт: <strong className="text-[#fafaf9] font-mono-num">{activePlayer.traitIds.length}</strong>
                </span>
                <button
                  type="button"
                  onClick={() => onUpdatePlayer(activePlayer.id, [], activePlayer.baseStats)}
                  className="text-xs text-[#f87171] hover:underline cursor-pointer"
                >
                  Очистить все черты игрока
                </button>
              </div>
            </div>
          )}

          {/* Колонка 3: Ручная выдача любых черт из каталога (27 черт) */}
          {activePlayer && (
            <div className="lg:col-span-5 p-4 overflow-y-auto flex flex-col">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <span className="text-sm font-bold text-[#fafaf9]">
                  Каталог черт ({filteredTraits.length})
                </span>
                <div className="flex items-center gap-1 bg-[#141210] p-1 rounded-lg border border-[#292524]">
                  {(
                    [
                      { id: 'all', label: 'Все' },
                      { id: 'unique', label: 'Уникальные' },
                      { id: 'rare', label: 'Редкие' },
                      { id: 'common', label: 'Обычные' },
                    ] as const
                  ).map((tab) => (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setRarityFilter(tab.id)}
                      className={`px-2 py-1 text-[11px] font-semibold rounded cursor-pointer whitespace-nowrap ${
                        rarityFilter === tab.id
                          ? 'bg-[#292524] text-[#fafaf9]'
                          : 'text-[#a8a29e] hover:text-[#e7e5e4]'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-2 overflow-y-auto pr-1 max-h-[52vh]">
                {filteredTraits.map((trait) => {
                  const equipped = activePlayer.traitIds.includes(trait.id);
                  return (
                    <button
                      key={trait.id}
                      type="button"
                      onClick={() => handleToggleTrait(trait.id)}
                      className={`w-full text-left p-3 rounded-lg border transition-colors cursor-pointer ${
                        equipped
                          ? 'bg-[#292524] border-[#ea580c]'
                          : 'bg-[#141210] border-[#292524] hover:border-[#443c35]'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <span
                          className="text-xs font-bold"
                          style={{ color: getRarityColor(trait.rarity) }}
                        >
                          {trait.name} [{trait.rarityLabel}]
                        </span>
                        <span className="text-[11px] font-semibold text-[#e7e5e4]">
                          {equipped ? '✓ Выдана' : '+ Выдать'}
                        </span>
                      </div>
                      <p className="text-xs text-[#a8a29e] leading-snug">
                        {trait.description}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
