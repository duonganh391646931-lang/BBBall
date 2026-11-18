import React from 'react';
import { TRAIT_BY_ID, getRarityColor } from '../game/traits';
import { MatchState } from '../game/types';

interface StatusModalProps {
  mode: 'paused' | 'gameover';
  match: MatchState;
  isTournamentMatch: boolean;
  onResume: () => void;
  onOpenBoxScore: () => void;
  onPlayAgainOrAdvance: () => void;
  onRerollNewMatch: () => void;
  onReturnToTitle: () => void;
}

export const StatusModal: React.FC<StatusModalProps> = ({
  mode,
  match,
  isTournamentMatch,
  onResume,
  onOpenBoxScore,
  onPlayAgainOrAdvance,
  onRerollNewMatch,
  onReturnToTitle,
}) => {
  if (mode === 'paused') {
    return (
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="pause-modal-title"
        className="fixed inset-0 z-30 flex items-center justify-center bg-[#0c0a09]/80 px-4"
      >
        <div className="bg-[#1c1917] border border-[#443c35] rounded-xl p-6 max-w-md w-full shadow-2xl">
          <h2 id="pause-modal-title" className="font-display text-3xl text-[#fafaf9] mb-2">
            ТАЙМ-АУТ (ПАУЗА)
          </h2>
          <p className="text-sm text-[#a8a29e] mb-6">
            Матч приостановлен. Текущий счёт{' '}
            <span className="font-mono-num font-bold text-[#fafaf9]">
              {match.score[0]} : {match.score[1]}
            </span>
            .
          </p>

          <div className="flex flex-col gap-2.5">
            <button
              type="button"
              onClick={onResume}
              className="w-full py-3 px-4 bg-[#ea580c] hover:bg-[#f97316] text-[#fafaf9] font-semibold rounded-lg cursor-pointer transition-colors"
            >
              Продолжить матч
            </button>
            <button
              type="button"
              onClick={onOpenBoxScore}
              className="w-full py-2.5 px-4 bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] text-[#f5f2eb] text-sm font-semibold rounded-lg cursor-pointer transition-colors"
            >
              Протокол матча и карта бросков
            </button>
            {!isTournamentMatch && (
              <button
                type="button"
                onClick={onRerollNewMatch}
                className="w-full py-2.5 px-4 bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] text-[#f5f2eb] text-sm font-semibold rounded-lg cursor-pointer transition-colors"
              >
                Новый матч со свежими балбесами
              </button>
            )}
            <button
              type="button"
              onClick={onReturnToTitle}
              className="w-full py-2.5 px-4 bg-[#141210] hover:bg-[#26221f] border border-[#3b342e] text-[#a8a29e] hover:text-[#f5f2eb] text-sm font-semibold rounded-lg cursor-pointer transition-colors"
            >
              На главный экран
            </button>
          </div>
        </div>
      </div>
    );
  }

  const summary = match.summary;
  if (!summary) return null;

  const winnerName = summary.teamNames[summary.winner];
  const winnerColor = summary.teamColors[summary.winner];

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="gameover-modal-title"
      className="fixed inset-0 z-30 flex items-center justify-center bg-[#0c0a09]/85 px-4 py-6 overflow-y-auto"
    >
      <div className="bg-[#1c1917] border border-[#443c35] rounded-xl p-6 max-w-xl w-full shadow-2xl">
        <div className="text-xs font-semibold text-[#a8a29e] mb-1">
          ФИНАЛЬНЫЙ ПРОТОКОЛ МАТЧА 5×5
        </div>
        <h2
          id="gameover-modal-title"
          className="font-display text-3xl text-[#fafaf9] mb-4"
        >
          ПОБЕДА: <span style={{ color: winnerColor }}>{winnerName}</span>
        </h2>

        {/* Табло по четвертям */}
        <div className="bg-[#141210] border border-[#3b342e] rounded-lg p-4 mb-4">
          <div className="flex items-center justify-between mb-3">
            <span className="font-bold text-base text-[#f5f2eb]">{summary.teamNames[0]}</span>
            <span className="font-display text-4xl font-bold text-[#fafaf9] tabular-nums">
              {summary.score[0]} : {summary.score[1]}
            </span>
            <span className="font-bold text-base text-[#f5f2eb]">{summary.teamNames[1]}</span>
          </div>

          <div className="flex items-center justify-center gap-4 text-xs font-mono-num text-[#a8a29e] border-t border-[#292524] pt-2">
            {summary.quarterScores.map((qs, idx) => (
              <span key={idx}>
                {idx < 4 ? `Q${idx + 1}` : `OT${idx - 3}`}:{' '}
                <strong className="text-[#e7e5e4]">
                  {qs[0]}-{qs[1]}
                </strong>
              </span>
            ))}
          </div>
        </div>

        {/* Три главные плашки статистики */}
        <div className="grid grid-cols-3 gap-3 mb-5">
          <div className="bg-[#26221f] border border-[#3b342e] rounded-lg p-3">
            <div className="text-[11px] text-[#a8a29e] mb-1">MVP МАТЧА</div>
            <div className="font-bold text-sm text-[#fafaf9] truncate">{summary.mvpName}</div>
            <div className="text-xs font-mono-num text-[#fbbf24] mt-0.5">
              {summary.mvpPoints} очк · {summary.mvpRebounds} пдб · {summary.mvpAssists} прд
            </div>
            <div className="text-[11px] text-[#a8a29e] truncate mt-0.5">
              {summary.mvpTraits
                .map((id) => TRAIT_BY_ID[id]?.name)
                .filter(Boolean)
                .join(' · ')}
            </div>
          </div>

          <div className="bg-[#26221f] border border-[#3b342e] rounded-lg p-3">
            <div className="text-[11px] text-[#a8a29e] mb-1">ОТЧЁТ СУДЬИ</div>
            <div className="font-mono-num text-lg font-bold text-[#fafaf9]">
              {summary.refCalls} свист.
            </div>
            <div className="text-[11px] text-[#a8a29e] mt-0.5">
              Проспал пакостей: <strong className="text-[#a3e635] font-mono-num">{summary.refMissed}</strong>
            </div>
          </div>

          <div className="bg-[#26221f] border border-[#3b342e] rounded-lg p-3">
            <div className="text-[11px] text-[#a8a29e] mb-1">ХАОС И ПЕРЕХВАТЫ</div>
            <div className="font-mono-num text-lg font-bold text-[#fafaf9]">
              {summary.totalSteals + summary.totalBlocks} перехв.
            </div>
            <div className="text-[11px] text-[#a8a29e] mt-0.5">
              Автоголы: <strong className="text-[#f87171] font-mono-num">{summary.ownGoals}</strong> · Месси:{' '}
              <strong className="text-[#fbbf24] font-mono-num">{summary.messiGoals}</strong>
            </div>
          </div>
        </div>

        {/* Итоговые действия */}
        <div className="flex flex-wrap items-center justify-end gap-3">
          <button
            type="button"
            onClick={onReturnToTitle}
            className="px-4 py-2.5 bg-[#141210] hover:bg-[#26221f] border border-[#3b342e] text-sm font-semibold text-[#a8a29e] hover:text-[#f5f2eb] rounded-lg cursor-pointer"
          >
            В главное меню
          </button>

          <button
            type="button"
            onClick={onOpenBoxScore}
            className="px-4 py-2.5 bg-[#26221f] hover:bg-[#322d29] border border-[#fbbf24]/55 text-sm font-semibold text-[#fde68a] rounded-lg cursor-pointer"
          >
            Протокол и карта бросков
          </button>

          {!isTournamentMatch && (
            <button
              type="button"
              onClick={onRerollNewMatch}
              className="px-4 py-2.5 bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] text-sm font-semibold text-[#f5f2eb] rounded-lg cursor-pointer"
            >
              Сгенерировать новые команды
            </button>
          )}

          <button
            type="button"
            data-start="true"
            onClick={onPlayAgainOrAdvance}
            className="px-6 py-2.5 bg-[#ea580c] hover:bg-[#f97316] text-sm font-semibold text-[#fafaf9] rounded-lg cursor-pointer"
          >
            {isTournamentMatch ? 'К турнирной сетке кубка →' : 'Реванш этими составами'}
          </button>
        </div>
      </div>
    </div>
  );
};
