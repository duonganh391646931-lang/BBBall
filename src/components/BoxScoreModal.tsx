import React, { useState } from 'react';
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
  RIM_RADIUS,
  THREE_PT_RADIUS,
} from '../game/physics';
import { formatClock, getQuarterLabel } from '../game/sim';
import { TRAIT_BY_ID, getRarityColor, isLastHopeBuffActive } from '../game/traits';
import { MatchState, POSITION_LABELS, PlayerEntity } from '../game/types';

interface BoxScoreModalProps {
  match: MatchState;
  onSelectPlayer: (playerId: string) => void;
  onClose: () => void;
}

type ShotFilter = 'all' | 0 | 1;

function formatMadeAttempted(made: number, att: number): string {
  return `${made}/${att}`;
}

function formatPct(made: number, att: number): string {
  if (att <= 0) return '0%';
  return `${Math.round((made / att) * 100)}%`;
}

export const BoxScoreModal: React.FC<BoxScoreModalProps> = ({
  match,
  onSelectPlayer,
  onClose,
}) => {
  const [shotFilter, setShotFilter] = useState<ShotFilter>('all');

  const getTeamRoster = (team: 0 | 1): PlayerEntity[] => {
    const onCourt = match.players
      .filter((p) => p.team === team)
      .sort((a, b) => a.slotIndex - b.slotIndex);
    const onBench = match.bench
      .filter((p) => p.team === team)
      .sort((a, b) => a.benchSlot - b.benchSlot);
    return [...onCourt, ...onBench];
  };

  const team0Roster = getTeamRoster(0);
  const team1Roster = getTeamRoster(1);

  const computeTeamTotals = (roster: PlayerEntity[]) => {
    return roster.reduce(
      (acc, p) => {
        acc.points += p.points;
        acc.rebounds += p.rebounds;
        acc.assists += p.assists;
        acc.steals += p.steals;
        acc.blocks += p.blocks;
        acc.fgMade += p.fgMade;
        acc.fgAttempted += p.fgAttempted;
        acc.threeMade += p.threeMade;
        acc.threeAttempted += p.threeAttempted;
        acc.ftMade += p.freeThrowsMade;
        acc.fouls += p.fouls;
        return acc;
      },
      {
        points: 0,
        rebounds: 0,
        assists: 0,
        steals: 0,
        blocks: 0,
        fgMade: 0,
        fgAttempted: 0,
        threeMade: 0,
        threeAttempted: 0,
        ftMade: 0,
        fouls: 0,
      }
    );
  };

  const totals0 = computeTeamTotals(team0Roster);
  const totals1 = computeTeamTotals(team1Roster);

  const visibleShots =
    shotFilter === 'all'
      ? match.shotChart
      : match.shotChart.filter((s) => s.team === shotFilter);

  const madeCount = visibleShots.filter((s) => s.made).length;
  const threeArcY = Math.sin(1.22) * THREE_PT_RADIUS;
  const threeArcDx = Math.cos(1.22) * THREE_PT_RADIUS;

  const renderTeamTable = (
    team: 0 | 1,
    roster: PlayerEntity[],
    totals: ReturnType<typeof computeTeamTotals>
  ) => (
    <div className="bg-[#141210] border border-[#3b342e] rounded-lg overflow-hidden">
      <div className="px-3.5 py-2.5 bg-[#211d1a] border-b border-[#3b342e] flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="w-3 h-3 rounded-sm shrink-0"
            style={{ backgroundColor: match.teamColors[team] }}
            aria-hidden="true"
          />
          <h3 className="font-display text-lg text-[#fafaf9] truncate">
            {match.teamNames[team]}
          </h3>
        </div>
        <div className="text-xs font-mono-num text-[#a8a29e] whitespace-nowrap">
          С игры: <strong className="text-[#f5f2eb]">{formatPct(totals.fgMade, totals.fgAttempted)}</strong> · 3-очк:{' '}
          <strong className="text-[#f5f2eb]">{formatPct(totals.threeMade, totals.threeAttempted)}</strong> · ТА:{' '}
          <strong className="text-[#fbbf24]">{match.timeoutsRemaining[team]}</strong>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="border-b border-[#292524] text-[#a8a29e] font-semibold">
              <th className="py-2 pl-3 pr-2">Игрок</th>
              <th className="py-2 px-1.5 text-right font-mono-num" title="Набранные очки">ОЧК</th>
              <th className="py-2 px-1.5 text-right font-mono-num" title="Подборы">ПДБ</th>
              <th className="py-2 px-1.5 text-right font-mono-num" title="Результативные передачи">ПРД</th>
              <th className="py-2 px-1.5 text-right font-mono-num" title="Перехваты">ПРХ</th>
              <th className="py-2 px-1.5 text-right font-mono-num" title="Блок-шоты">БЛК</th>
              <th className="py-2 px-1.5 text-right font-mono-num" title="Броски с игры (точные/всего)">БР</th>
              <th className="py-2 px-1.5 text-right font-mono-num" title="Трёхочковые броски">3-ОЧК</th>
              <th className="py-2 px-1.5 text-right font-mono-num" title="Персональные фолы">ФОЛ</th>
              <th className="py-2 pl-1.5 pr-3 text-right font-mono-num" title="Запас сил">СИЛЫ</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#221e1b]">
            {roster.map((p) => {
              const lastHope = isLastHopeBuffActive(p);
              const isSelected = p.id === match.selectedPlayerId;
              return (
                <tr
                  key={p.id}
                  onClick={() => onSelectPlayer(p.id)}
                  className={`cursor-pointer transition-colors ${
                    isSelected
                      ? 'bg-[#292524]'
                      : p.onCourt
                      ? 'hover:bg-[#1c1917]'
                      : 'bg-[#12100e] text-[#a8a29e] hover:bg-[#1c1917]'
                  }`}
                >
                  <td className="py-1.5 pl-3 pr-2 max-w-[210px]">
                    <div className="flex items-center gap-1.5 whitespace-nowrap overflow-hidden">
                      <span className="font-mono-num text-[11px] text-[#a8a29e] shrink-0">
                        {POSITION_LABELS[p.role]} #{p.jerseyNumber}
                      </span>
                      <span
                        className={`font-semibold truncate ${
                          p.disqualified
                            ? 'line-through text-[#f87171]'
                            : p.onCourt
                            ? 'text-[#fafaf9]'
                            : 'text-[#d6d3d1]'
                        }`}
                      >
                        {p.name}
                      </span>
                      {!p.onCourt && (
                        <span className="text-[10px] text-[#78716c] shrink-0">
                          {p.disqualified ? '· 5Ф' : '· Зап'}
                        </span>
                      )}
                      {lastHope && (
                        <span className="text-[10px] font-bold text-[#fde047] shrink-0" title="Последняя надежда активна">
                          ★ПН
                        </span>
                      )}
                      {p.hotStreak >= 3 && (
                        <span className="text-[10px] font-bold text-[#fb923c] shrink-0" title={`Серия точных бросков: ${p.hotStreak}`}>
                          В огне
                        </span>
                      )}
                      {p.hotStreak <= -3 && (
                        <span className="text-[10px] font-bold text-[#7dd3fc] shrink-0" title={`Серия промахов: ${Math.abs(p.hotStreak)}`}>
                          Холод
                        </span>
                      )}
                    </div>
                    {p.traitIds.length > 0 && (
                      <div className="text-[10px] truncate mt-0.5">
                        {p.traitIds.map((tid, idx) => {
                          const t = TRAIT_BY_ID[tid];
                          if (!t) return null;
                          return (
                            <React.Fragment key={tid}>
                              {idx > 0 && <span className="text-[#57534e]"> · </span>}
                              <span style={{ color: getRarityColor(t.rarity) }}>{t.name}</span>
                            </React.Fragment>
                          );
                        })}
                      </div>
                    )}
                  </td>
                  <td className="py-2 px-1.5 text-right font-mono-num font-bold text-[#fafaf9]">
                    {p.points}
                  </td>
                  <td className="py-2 px-1.5 text-right font-mono-num text-[#e7e5e4]">
                    {p.rebounds}
                  </td>
                  <td className="py-2 px-1.5 text-right font-mono-num text-[#e7e5e4]">
                    {p.assists}
                  </td>
                  <td className="py-2 px-1.5 text-right font-mono-num text-[#e7e5e4]">
                    {p.steals}
                  </td>
                  <td className="py-2 px-1.5 text-right font-mono-num text-[#e7e5e4]">
                    {p.blocks}
                  </td>
                  <td className="py-2 px-1.5 text-right font-mono-num text-[#d6d3d1] whitespace-nowrap">
                    {formatMadeAttempted(p.fgMade, p.fgAttempted)}
                  </td>
                  <td className="py-2 px-1.5 text-right font-mono-num text-[#d6d3d1] whitespace-nowrap">
                    {formatMadeAttempted(p.threeMade, p.threeAttempted)}
                  </td>
                  <td
                    className={`py-2 px-1.5 text-right font-mono-num ${
                      p.fouls >= 4 ? 'text-[#f87171] font-bold' : 'text-[#a8a29e]'
                    }`}
                  >
                    {p.fouls}
                  </td>
                  <td className="py-2 pl-1.5 pr-3 text-right font-mono-num text-[#a8a29e]">
                    {Math.round(p.energy)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr className="border-t border-[#3b342e] bg-[#1c1917] font-mono-num font-bold text-[#fafaf9]">
              <td className="py-2 pl-3 pr-2 font-sans text-xs text-[#d6d3d1]">Итого команда</td>
              <td className="py-2 px-1.5 text-right text-[#fbbf24]">{totals.points}</td>
              <td className="py-2 px-1.5 text-right">{totals.rebounds}</td>
              <td className="py-2 px-1.5 text-right">{totals.assists}</td>
              <td className="py-2 px-1.5 text-right">{totals.steals}</td>
              <td className="py-2 px-1.5 text-right">{totals.blocks}</td>
              <td className="py-2 px-1.5 text-right whitespace-nowrap">
                {formatMadeAttempted(totals.fgMade, totals.fgAttempted)}
              </td>
              <td className="py-2 px-1.5 text-right whitespace-nowrap">
                {formatMadeAttempted(totals.threeMade, totals.threeAttempted)}
              </td>
              <td className="py-2 px-1.5 text-right">{totals.fouls}</td>
              <td className="py-2 pl-1.5 pr-3 text-right text-[#78716c]">—</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="boxscore-modal-title"
      className="fixed inset-0 z-40 flex items-center justify-center bg-[#0c0a09]/88 px-4 py-5 overflow-y-auto"
    >
      <div className="bg-[#1c1917] border border-[#443c35] rounded-xl p-5 max-w-6xl w-full shadow-2xl max-h-[92vh] overflow-y-auto">
        {/* Шапка протокола */}
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3.5 mb-4 border-b border-[#3b342e]">
          <div>
            <h2 id="boxscore-modal-title" className="font-display text-2xl text-[#fafaf9]">
              ПРОТОКОЛ МАТЧА И КАРТА БРОСКОВ
            </h2>
            <p className="text-xs text-[#a8a29e] font-mono-num mt-0.5">
              {getQuarterLabel(match.quarter)} · {formatClock(match.quarterTimeRemaining)} · Счёт{' '}
              <strong className="text-[#fafaf9]">
                {match.score[0]} : {match.score[1]}
              </strong>
              {match.quarterScores.length > 0 && (
                <span>
                  {' · По четвертям: '}
                  {match.quarterScores
                    .map((qs, idx) => `${idx < 4 ? `Q${idx + 1}` : `OT${idx - 3}`} ${qs[0]}-${qs[1]}`)
                    .join(' / ')}
                </span>
              )}
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 bg-[#ea580c] hover:bg-[#f97316] text-xs font-semibold text-[#fafaf9] rounded-lg cursor-pointer whitespace-nowrap shrink-0"
          >
            Закрыть протокол [Tab / Esc]
          </button>
        </div>

        {/* Верхний блок: Карта точек бросков сверху */}
        <div className="bg-[#141210] border border-[#3b342e] rounded-lg p-3 mb-3.5">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
            <div className="flex items-center gap-3 text-xs">
              <span className="font-semibold text-[#fafaf9]">Карта бросков сверху</span>
              <span className="text-[#a8a29e] font-mono-num">
                Попадания: <strong className="text-[#34d399]">{madeCount}</strong> из{' '}
                <strong className="text-[#f5f2eb]">{visibleShots.length}</strong> (
                {formatPct(madeCount, visibleShots.length)})
              </span>
              <span className="text-[#a8a29e] hidden sm:inline">
                ● точный · ✕ промах
              </span>
            </div>

            <div className="flex items-center gap-1 bg-[#211d1a] p-1 rounded-lg border border-[#3b342e]">
              <button
                type="button"
                onClick={() => setShotFilter('all')}
                className={`px-2.5 py-1 text-xs font-semibold rounded cursor-pointer whitespace-nowrap transition-colors ${
                  shotFilter === 'all'
                    ? 'bg-[#ea580c] text-[#fafaf9]'
                    : 'text-[#a8a29e] hover:text-[#fafaf9]'
                }`}
              >
                Все броски
              </button>
              <button
                type="button"
                onClick={() => setShotFilter(0)}
                className={`px-2.5 py-1 text-xs font-semibold rounded cursor-pointer whitespace-nowrap transition-colors ${
                  shotFilter === 0
                    ? 'bg-[#292524] text-[#fafaf9] border border-[#fbbf24]/60'
                    : 'text-[#a8a29e] hover:text-[#fafaf9]'
                }`}
              >
                {match.teamNames[0]}
              </button>
              <button
                type="button"
                onClick={() => setShotFilter(1)}
                className={`px-2.5 py-1 text-xs font-semibold rounded cursor-pointer whitespace-nowrap transition-colors ${
                  shotFilter === 1
                    ? 'bg-[#292524] text-[#fafaf9] border border-[#fbbf24]/60'
                    : 'text-[#a8a29e] hover:text-[#fafaf9]'
                }`}
              >
                {match.teamNames[1]}
              </button>
            </div>
          </div>

          <div className="w-full max-w-xl mx-auto">
            <svg
              viewBox={`0 0 ${COURT_W} ${COURT_H}`}
              className="w-full h-auto rounded border border-[#292524] bg-[#191614]"
              role="img"
              aria-label="Карта точек бросков на площадке"
            >
              {/* Паркет */}
              <rect
                x={COURT_LEFT}
                y={COURT_TOP}
                width={COURT_RIGHT - COURT_LEFT}
                height={COURT_BOTTOM - COURT_TOP}
                fill="#26201b"
                stroke="#78716c"
                strokeWidth="3"
              />
              {/* Краска команд */}
              <rect
                x={COURT_LEFT}
                y={HOOP_Y - 82}
                width={190}
                height={164}
                fill={match.teamColors[0]}
                fillOpacity="0.14"
                stroke="#78716c"
                strokeWidth="2"
              />
              <rect
                x={COURT_RIGHT - 190}
                y={HOOP_Y - 82}
                width={190}
                height={164}
                fill={match.teamColors[1]}
                fillOpacity="0.14"
                stroke="#78716c"
                strokeWidth="2"
              />
              {/* Центральная линия и круг */}
              <line
                x1={MID_X}
                y1={COURT_TOP}
                x2={MID_X}
                y2={COURT_BOTTOM}
                stroke="#78716c"
                strokeWidth="2"
              />
              <circle cx={MID_X} cy={HOOP_Y} r={66} fill="none" stroke="#78716c" strokeWidth="2" />
              {/* Трёхочковые дуги */}
              <path
                d={`M ${COURT_LEFT} ${HOOP_Y - threeArcY} L ${HOOP_LEFT_X + threeArcDx} ${HOOP_Y - threeArcY} A ${THREE_PT_RADIUS} ${THREE_PT_RADIUS} 0 0 1 ${HOOP_LEFT_X + threeArcDx} ${HOOP_Y + threeArcY} L ${COURT_LEFT} ${HOOP_Y + threeArcY}`}
                fill="none"
                stroke="#78716c"
                strokeWidth="2"
              />
              <path
                d={`M ${COURT_RIGHT} ${HOOP_Y - threeArcY} L ${HOOP_RIGHT_X - threeArcDx} ${HOOP_Y - threeArcY} A ${THREE_PT_RADIUS} ${THREE_PT_RADIUS} 0 0 0 ${HOOP_RIGHT_X - threeArcDx} ${HOOP_Y + threeArcY} L ${COURT_RIGHT} ${HOOP_Y + threeArcY}`}
                fill="none"
                stroke="#78716c"
                strokeWidth="2"
              />
              {/* Кольца */}
              <circle cx={HOOP_LEFT_X} cy={HOOP_Y} r={RIM_RADIUS} fill="none" stroke="#f97316" strokeWidth="3.5" />
              <circle cx={HOOP_RIGHT_X} cy={HOOP_Y} r={RIM_RADIUS} fill="none" stroke="#f97316" strokeWidth="3.5" />

              {/* Точки бросков */}
              {visibleShots.map((s, idx) =>
                s.made ? (
                  <circle
                    key={idx}
                    cx={s.x}
                    cy={s.y}
                    r={s.isThree ? 7.5 : 6}
                    fill={match.teamColors[s.team]}
                    stroke={s.isThree ? '#fde047' : '#fafaf9'}
                    strokeWidth={s.isThree ? 2.2 : 1.5}
                  >
                    <title>{`${s.shooterName}: точный ${s.isThree ? '3-очковый' : '2-очковый'} (Q${s.quarter})`}</title>
                  </circle>
                ) : (
                  <g key={idx} transform={`translate(${s.x}, ${s.y})`}>
                    <title>{`${s.shooterName}: промах ${s.isThree ? 'из-за дуги' : 'с игры'} (Q${s.quarter})`}</title>
                    <line x1="-5" y1="-5" x2="5" y2="5" stroke="#f87171" strokeWidth="2.2" />
                    <line x1="-5" y1="5" x2="5" y2="-5" stroke="#f87171" strokeWidth="2.2" />
                  </g>
                )
              )}

              {visibleShots.length === 0 && (
                <text
                  x={MID_X}
                  y={HOOP_Y + 5}
                  textAnchor="middle"
                  fill="#a8a29e"
                  fontSize="18"
                  fontFamily="Manrope, sans-serif"
                >
                  Пока нет выполненных бросков с игры
                </text>
              )}
            </svg>
          </div>
        </div>

        {/* Таблицы составов обеих команд (старт + скамейка) */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          {renderTeamTable(0, team0Roster, totals0)}
          {renderTeamTable(1, team1Roster, totals1)}
        </div>
      </div>
    </div>
  );
};
