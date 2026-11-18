import React, { useState } from 'react';
import {
  CLUB_PRESETS,
  TRAIT_BY_ID,
  getPlayerOverall,
  getRarityColor,
} from '../game/traits';
import { assignRolesToRoster, findUserCurrentMatchup } from '../game/tournament';
import { POSITION_LABELS, PlayerEntity, TournamentState } from '../game/types';

interface TournamentModalProps {
  tournament: TournamentState;
  onUpdateTournament: (next: TournamentState) => void;
  onResetNewTournament: (clubName: string, clubColor: string) => void;
  onLaunchBracketMatch: (userRoster: PlayerEntity[], opponentRoster: PlayerEntity[], oppName: string, oppColor: string) => void;
  onClose: () => void;
}

export const TournamentModal: React.FC<TournamentModalProps> = ({
  tournament,
  onUpdateTournament,
  onResetNewTournament,
  onLaunchBracketMatch,
  onClose,
}) => {
  const [clubNameInput, setClubNameInput] = useState(tournament.userClub.name);
  const [clubColorInput, setClubColorInput] = useState(tournament.userClub.primaryColor);
  const [draftError, setDraftError] = useState<string | null>(null);
  const [selectedReplaceSlot, setSelectedReplaceSlot] = useState<number>(0);

  const draftedIds = new Set(tournament.userClub.roster.map((p) => p.id));

  const draftedIndexMap = new Map(
    tournament.userClub.roster.map((p, idx) => [p.id, idx])
  );

  const toggleDraftCandidate = (candidate: PlayerEntity) => {
    setDraftError(null);
    const current = tournament.userClub.roster;
    let nextRoster: PlayerEntity[];
    if (draftedIds.has(candidate.id)) {
      nextRoster = current.filter((p) => p.id !== candidate.id);
    } else {
      if (current.length >= 8) {
        setDraftError('В заявке клуба уже 8 игроков (5 старт + 3 запас)! Снимите кого-нибудь для замены.');
        return;
      }
      nextRoster = [...current, candidate];
    }

    const updatedClub = {
      ...tournament.userClub,
      name: clubNameInput.trim() || 'Оранжевые Метеоры',
      primaryColor: clubColorInput,
      roster: assignRolesToRoster(nextRoster),
    };

    const updatedMatches = tournament.matches.map((m) =>
      m.teamA?.isUserClub ? { ...m, teamA: updatedClub } : m
    );

    onUpdateTournament({
      ...tournament,
      userClub: updatedClub,
      matches: updatedMatches,
    });
  };

  const handleAutoDraftEight = () => {
    setDraftError(null);
    // Ставим кандидатов с «Последней надеждой» в слоты запасных (6–8), чтобы бафф срабатывал на 100%!
    const nonLastHope = tournament.draftPool.filter((p) => !p.traitIds.includes('last_hope'));
    const withLastHope = tournament.draftPool.filter((p) => p.traitIds.includes('last_hope'));
    const ordered = [...nonLastHope.slice(0, 5), ...withLastHope, ...nonLastHope.slice(5)].slice(0, 8);
    const fullRoster = assignRolesToRoster(ordered);
    const updatedClub = {
      ...tournament.userClub,
      name: clubNameInput.trim() || 'Оранжевые Метеоры',
      primaryColor: clubColorInput,
      roster: fullRoster,
    };
    const updatedMatches = tournament.matches.map((m) =>
      m.teamA?.isUserClub ? { ...m, teamA: updatedClub } : m
    );
    onUpdateTournament({
      ...tournament,
      userClub: updatedClub,
      matches: updatedMatches,
    });
  };

  const handleConfirmDraft = () => {
    const trimmedName = clubNameInput.trim();
    if (!trimmedName) {
      setDraftError('Введите название вашего баскетбольного клуба.');
      return;
    }
    if (tournament.userClub.roster.length < 5) {
      setDraftError(
        `Наберите минимум 5 игроков в старт (и до 3 в запас; сейчас выбрано: ${tournament.userClub.roster.length}/8).`
      );
      return;
    }
    const updatedClub = {
      ...tournament.userClub,
      name: trimmedName,
      primaryColor: clubColorInput,
      roster: assignRolesToRoster(tournament.userClub.roster),
    };
    const updatedMatches = tournament.matches.map((m) =>
      m.teamA?.isUserClub ? { ...m, teamA: updatedClub } : m
    );
    onUpdateTournament({
      ...tournament,
      phase: 'bracket',
      userClub: updatedClub,
      matches: updatedMatches,
    });
  };

  const handleReinforcePick = (newcomer: PlayerEntity | null) => {
    let nextRoster = [...tournament.userClub.roster];
    if (newcomer) {
      nextRoster[selectedReplaceSlot] = {
        ...newcomer,
        slotIndex: selectedReplaceSlot,
        role: nextRoster[selectedReplaceSlot].role,
        team: 0,
      };
    }
    const updatedClub = {
      ...tournament.userClub,
      roster: nextRoster,
    };
    const updatedMatches = tournament.matches.map((m) => {
      if (m.winnerId === null && m.teamA?.isUserClub) return { ...m, teamA: updatedClub };
      if (m.winnerId === null && m.teamB?.isUserClub) return { ...m, teamB: updatedClub };
      return m;
    });
    onUpdateTournament({
      ...tournament,
      phase: 'bracket',
      userClub: updatedClub,
      matches: updatedMatches,
    });
  };

  const currentMatchup = findUserCurrentMatchup(tournament);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="tournament-modal-title"
      className="fixed inset-0 z-40 flex items-center justify-center bg-[#0c0a09]/85 p-3 sm:p-5 overflow-y-auto"
    >
      <div className="bg-[#1c1917] border border-[#443c35] rounded-xl w-full max-w-6xl max-h-[92vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Верхняя шапка менеджера */}
        <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 bg-[#141210] border-b border-[#3b342e]">
          <div>
            <h2 id="tournament-modal-title" className="font-display text-2xl text-[#fafaf9]">
              КУБОК БАЛБЕСОВ · КЛУБНЫЙ МЕНЕДЖЕР И ДРАФТ (5 + 3 ЗАПАСНЫХ)
            </h2>
            <p className="text-xs text-[#a8a29e]">
              Наберите 5 игроков основы + 3 запасных (слоты 6–8 идеальны для черты «Последняя надежда»!) и пройдите плей-офф из 8 клубов.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => onResetNewTournament(clubNameInput || 'Оранжевые Метеоры', clubColorInput)}
              className="px-3 py-1.5 bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] text-xs font-semibold text-[#d6d3d1] rounded-lg cursor-pointer"
            >
              Новый турнир
            </button>
            <button
              type="button"
              aria-label="Закрыть окно турнира"
              onClick={onClose}
              className="w-8 h-8 flex items-center justify-center bg-[#26221f] hover:bg-[#322d29] rounded-lg text-sm text-[#a8a29e] hover:text-[#fafaf9] cursor-pointer"
            >
              ✕
            </button>
          </div>
        </div>

        {/* ЭТАП 1: ДРАФТ СТАРТОВОЙ ПЯТЁРКИ */}
        {tournament.phase === 'draft' && (
          <div className="p-5 overflow-y-auto space-y-4">
            <div className="flex flex-wrap items-end justify-between gap-4 bg-[#141210] border border-[#292524] rounded-lg p-4">
              <div className="flex flex-wrap items-center gap-4">
                <div>
                  <label htmlFor="club-name-input" className="block text-xs text-[#a8a29e] mb-1">
                    Название вашего клуба
                  </label>
                  <input
                    id="club-name-input"
                    type="text"
                    value={clubNameInput}
                    onChange={(e) => {
                      setClubNameInput(e.target.value);
                      setDraftError(null);
                    }}
                    className="bg-[#26221f] border border-[#443c35] rounded px-3 py-1.5 text-sm font-semibold text-[#fafaf9] w-56 focus:outline-none focus:border-[#fbbf24]"
                  />
                </div>

                <div>
                  <span className="block text-xs text-[#a8a29e] mb-1">Цвет формы</span>
                  <div className="flex items-center gap-1.5">
                    {CLUB_PRESETS.slice(0, 6).map((preset) => (
                      <button
                        key={preset.primary}
                        type="button"
                        aria-label={`Цвет ${preset.name}`}
                        onClick={() => setClubColorInput(preset.primary)}
                        className={`w-7 h-7 rounded border-2 cursor-pointer ${
                          clubColorInput === preset.primary ? 'border-[#fafaf9] scale-110' : 'border-transparent'
                        }`}
                        style={{ backgroundColor: preset.primary }}
                      />
                    ))}
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2.5">
                <span className="text-xs font-mono-num text-[#d6d3d1]">
                  Заявка (5 старт + 3 запас):{' '}
                  <strong className="text-[#fbbf24]">{tournament.userClub.roster.length}/8</strong>
                </span>
                <button
                  type="button"
                  onClick={handleAutoDraftEight}
                  className="px-3.5 py-2 bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] text-xs font-semibold text-[#f5f2eb] rounded-lg cursor-pointer"
                >
                  Авто-выбор 8 игроков
                </button>
                <button
                  type="button"
                  onClick={handleConfirmDraft}
                  className="px-5 py-2 bg-[#ea580c] hover:bg-[#f97316] text-xs font-bold text-[#fafaf9] rounded-lg cursor-pointer"
                >
                  Утвердить состав и открыть сетку →
                </button>
              </div>
            </div>

            {draftError && (
              <div role="alert" className="px-4 py-2.5 bg-[#450a0a] border border-[#dc2626] rounded-lg text-xs text-[#fecaca]">
                {draftError}
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {tournament.draftPool.map((cand) => {
                const picked = draftedIds.has(cand.id);
                const rosterIdx = draftedIndexMap.get(cand.id);
                const isBenchPick = rosterIdx !== undefined && rosterIdx >= 5;
                return (
                  <button
                    key={cand.id}
                    type="button"
                    onClick={() => toggleDraftCandidate(cand)}
                    className={`text-left p-3.5 rounded-lg border transition-all cursor-pointer flex flex-col justify-between ${
                      picked
                        ? isBenchPick
                          ? 'bg-[#292524] border-[#38bdf8] shadow-md'
                          : 'bg-[#292524] border-[#ea580c] shadow-md'
                        : 'bg-[#141210] border-[#292524] hover:border-[#443c35]'
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between gap-1 mb-1">
                        <span className="text-xs font-mono-num text-[#a8a29e]">
                          {POSITION_LABELS[cand.role]} · #{cand.jerseyNumber}
                        </span>
                        <span className="text-xs font-mono-num font-bold text-[#fbbf24]">
                          Рейт {getPlayerOverall(cand)}
                        </span>
                      </div>
                      <div className="font-bold text-sm text-[#fafaf9] mb-2 truncate">
                        {cand.name}
                      </div>

                      {/* Черты кандидата */}
                      <div className="space-y-1 mb-3">
                        {cand.traitIds.map((tid) => {
                          const def = TRAIT_BY_ID[tid];
                          if (!def) return null;
                          return (
                            <div key={tid} className="text-[11px] leading-tight">
                              <span className="font-semibold" style={{ color: getRarityColor(def.rarity) }}>
                                {def.name} [{def.rarityLabel}]
                              </span>
                              <span className="text-[#78716c]"> — {def.shortTag}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div>
                      <div className="grid grid-cols-3 gap-1 text-[10px] font-mono-num text-[#a8a29e] border-t border-[#292524] pt-2 mb-2">
                        <span>СКР {cand.effectiveStats.speed}</span>
                        <span>БРС {cand.effectiveStats.shooting}</span>
                        <span>ПРЖ {cand.effectiveStats.jump}</span>
                        <span>ПРХ {cand.effectiveStats.steal}</span>
                        <span>КОМ {cand.effectiveStats.teamwork}</span>
                        <span>ДЕВ {cand.effectiveStats.deviance}</span>
                      </div>
                      <div
                        className={`text-center py-1 rounded text-xs font-semibold ${
                          picked
                            ? isBenchPick
                              ? 'bg-[#0284c7] text-[#fafaf9]'
                              : 'bg-[#ea580c] text-[#fafaf9]'
                            : 'bg-[#26221f] text-[#d6d3d1]'
                        }`}
                      >
                        {picked
                          ? isBenchPick
                            ? `✓ Запасной (#${(rosterIdx ?? 5) + 1})`
                            : `✓ Старт (#${(rosterIdx ?? 0) + 1})`
                          : '+ Взять в клуб'}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* ЭТАП 2: МИНИ-ДРАФТ УСИЛЕНИЯ МЕЖДУ РАУНДАМИ ПЛЕЙ-ОФФ */}
        {tournament.phase === 'reinforce' && (
          <div className="p-6 overflow-y-auto space-y-5">
            <div>
              <h3 className="font-display text-2xl text-[#fbbf24] mb-1">
                ПОБЕДА В РАУНДЕ! СКАУТЫ ПРИВЕЛИ УСИЛЕНИЕ
              </h3>
              <p className="text-sm text-[#d6d3d1]">
                Выберите слот в вашей пятёрке для замены и подпишите одного из свободных агентов — или оставьте победный состав без изменений.
              </p>
            </div>

            <div>
              <div className="text-xs font-semibold text-[#a8a29e] mb-2">
                1. Кого из текущего состава готовы заменить:
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-4 gap-2.5">
                {tournament.userClub.roster.map((p, idx) => (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => setSelectedReplaceSlot(idx)}
                    className={`p-3 rounded-lg border text-left cursor-pointer ${
                      selectedReplaceSlot === idx
                        ? 'bg-[#292524] border-[#fbbf24]'
                        : 'bg-[#141210] border-[#292524]'
                    }`}
                  >
                    <div className="text-xs font-bold text-[#fafaf9] truncate">
                      {idx >= 5 ? '[ЗАПАС] ' : ''}
                      {POSITION_LABELS[p.role]} · {p.name}
                    </div>
                    <div className="text-[11px] font-mono-num text-[#a8a29e] mt-0.5">
                      Рейтинг: {getPlayerOverall(p)}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="text-xs font-semibold text-[#a8a29e] mb-2">
                2. Выберите новичка на эту позицию:
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {tournament.reinforcePool.map((cand) => (
                  <div
                    key={cand.id}
                    className="bg-[#141210] border border-[#3b342e] rounded-lg p-4 flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-bold text-sm text-[#fafaf9]">{cand.name}</span>
                        <span className="text-xs font-mono-num font-bold text-[#fbbf24]">
                          Рейт {getPlayerOverall(cand)}
                        </span>
                      </div>
                      <div className="space-y-1 my-2">
                        {cand.traitIds.map((tid) => {
                          const def = TRAIT_BY_ID[tid];
                          return def ? (
                            <div key={tid} className="text-xs" style={{ color: getRarityColor(def.rarity) }}>
                              {def.name} [{def.rarityLabel}]
                            </div>
                          ) : null;
                        })}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleReinforcePick(cand)}
                      className="mt-3 w-full py-2 bg-[#ea580c] hover:bg-[#f97316] text-xs font-semibold text-[#fafaf9] rounded cursor-pointer"
                    >
                      Подписать вместо слота #{selectedReplaceSlot + 1}
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => handleReinforcePick(null)}
                className="px-5 py-2.5 bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] text-xs font-semibold text-[#f5f2eb] rounded-lg cursor-pointer"
              >
                Оставить состав без замен →
              </button>
            </div>
          </div>
        )}

        {/* ЭТАП 3 и 4: ТУРНИРНАЯ СЕТКА ПЛЕЙ-ОФФ (1/4 -> 1/2 -> ФИНАЛ) И ЧЕМПИОН */}
        {(tournament.phase === 'bracket' || tournament.phase === 'champion') && (
          <div className="p-5 overflow-y-auto space-y-5">
            {tournament.phase === 'champion' && tournament.championClub && (
              <div className="bg-[#141210] border-2 border-[#fbbf24] rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
                <div>
                  <div className="text-xs font-bold text-[#fbbf24] mb-1">ИТОГ КУБКА БАЛБЕСОВ</div>
                  <h3 className="font-display text-3xl text-[#fafaf9]">
                    ЧЕМПИОН КУБКА: {tournament.championClub.name}!
                  </h3>
                  <p className="text-sm text-[#a8a29e] mt-1">
                    {tournament.championClub.isUserClub
                      ? 'Ваша безумная пятёрка взяла главный трофей!'
                      : 'Ваш клуб уступил в плей-офф, но вы можете собрать новый драфт и взять реванш!'}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => onResetNewTournament(tournament.userClub.name, tournament.userClub.primaryColor)}
                  className="px-5 py-3 bg-[#ea580c] hover:bg-[#f97316] text-sm font-bold text-[#fafaf9] rounded-lg cursor-pointer"
                >
                  Начать новый сезон кубка
                </button>
              </div>
            )}

            {/* 3 колонки плей-офф: Четвертьфиналы, Полуфиналы, Финал */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {(
                [
                  { round: 0, title: '01. Четвертьфиналы (1/4)' },
                  { round: 1, title: '02. Полуфиналы (1/2)' },
                  { round: 2, title: '03. Гранд-Финал Кубка' },
                ] as const
              ).map(({ round, title }) => (
                <div key={round} className="space-y-3">
                  <h3 className="text-sm font-bold text-[#e7e5e4] pb-1.5 border-b border-[#292524]">
                    {title}
                  </h3>
                  {tournament.matches
                    .filter((m) => m.round === round)
                    .map((m) => {
                      const involvesUser = m.teamA?.isUserClub || m.teamB?.isUserClub;
                      return (
                        <div
                          key={m.id}
                          className={`rounded-lg border p-3 space-y-2 ${
                            involvesUser
                              ? 'bg-[#26221f] border-[#ea580c]'
                              : 'bg-[#141210] border-[#292524]'
                          }`}
                        >
                          {[
                            { team: m.teamA, score: m.scoreA },
                            { team: m.teamB, score: m.scoreB },
                          ].map((row, rIdx) => {
                            const isWin = m.winnerId && row.team?.id === m.winnerId;
                            return (
                              <div
                                key={rIdx}
                                className="flex items-center justify-between text-xs"
                              >
                                <div className="flex items-center gap-2 min-w-0">
                                  <span
                                    className="w-2.5 h-2.5 rounded-sm shrink-0"
                                    style={{
                                      backgroundColor: row.team?.primaryColor ?? '#44403c',
                                    }}
                                  />
                                  <span
                                    className={`truncate ${
                                      isWin
                                        ? 'font-bold text-[#fafaf9]'
                                        : row.team
                                        ? 'text-[#d6d3d1]'
                                        : 'text-[#57534e]'
                                    }`}
                                  >
                                    {row.team ? row.team.name : 'Ожидание соперника'}
                                  </span>
                                </div>
                                <span className="font-mono-num font-bold text-[#fafaf9] ml-2">
                                  {row.score !== null ? row.score : '—'}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      );
                    })}
                </div>
              ))}
            </div>

            {/* Кнопка запуска текущего матча клуба игрока */}
            {tournament.phase === 'bracket' && currentMatchup && currentMatchup.teamA && currentMatchup.teamB && (
              <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-[#292524]">
                <div className="text-sm text-[#d6d3d1]">
                  Следующий матч вашего клуба:{' '}
                  <strong className="text-[#fafaf9]">
                    {currentMatchup.teamA.name} vs {currentMatchup.teamB.name}
                  </strong>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    const opp = currentMatchup.teamA!.isUserClub
                      ? currentMatchup.teamB!
                      : currentMatchup.teamA!;
                    onLaunchBracketMatch(
                      tournament.userClub.roster,
                      opp.roster,
                      opp.name,
                      opp.primaryColor
                    );
                  }}
                  className="px-6 py-3 bg-[#ea580c] hover:bg-[#f97316] font-display text-lg tracking-wide text-[#fafaf9] rounded-lg cursor-pointer"
                >
                  Играть матч 5×5 →
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
