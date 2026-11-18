import React from 'react';
import { TRAIT_CATALOG } from '../game/traits';

interface TitleOverlayProps {
  onStartQuickMatch: () => void;
  onOpenTournament: () => void;
  onOpenSandbox: () => void;
}

export const TitleOverlay: React.FC<TitleOverlayProps> = ({
  onStartQuickMatch,
  onOpenTournament,
  onOpenSandbox,
}) => {
  return (
    <div
      className="fixed inset-0 z-20 flex flex-col justify-between pointer-events-none"
      style={{
        background:
          'linear-gradient(90deg, rgba(16,14,12,0.95) 0%, rgba(16,14,12,0.88) 38%, rgba(16,14,12,0.22) 72%, rgba(16,14,12,0.08) 100%)',
      }}
    >
      <div className="flex-1 flex items-center px-8 sm:px-14 lg:px-20">
        <div className="max-w-lg pointer-events-auto">
          {/* Кастомная векторная эмблема арены БАЛБЕСБОЛ */}
          <svg
            viewBox="0 0 560 118"
            className="w-full max-w-[490px] h-auto mb-5"
            role="img"
            aria-label="БалбесБол — Тактический ИИ Баскетбол 5 на 5"
          >
            {/* Щит-основание логотипа */}
            <path
              d="M12 16 L96 8 L96 96 L54 112 L12 96 Z"
              fill="#1c1917"
              stroke="#ea580c"
              strokeWidth="3.5"
            />
            {/* Баскетбольный мяч со схемой тактики сверху */}
            <circle cx="54" cy="56" r="28" fill="#ea580c" stroke="#141210" strokeWidth="3" />
            <path
              d="M26 56 H82 M54 28 V84 M35 36 Q54 56 35 76 M73 36 Q54 56 73 76"
              fill="none"
              stroke="#141210"
              strokeWidth="2.4"
            />
            {/* Свисток / корона тактики сверху щита */}
            <path d="M40 20 L54 12 L68 20" fill="none" stroke="#fbbf24" strokeWidth="3" />

            {/* Типографика эмблемы */}
            <text
              x="114"
              y="66"
              fill="#f5f2eb"
              fontFamily="Oswald, sans-serif"
              fontWeight="700"
              fontSize="58"
              letterSpacing="1.5"
            >
              БАЛБЕСБОЛ
            </text>
            <line x1="116" y1="78" x2="538" y2="78" stroke="#3b342e" strokeWidth="2" />
            <text
              x="116"
              y="99"
              fill="#d6d3d1"
              fontFamily="Manrope, sans-serif"
              fontWeight="700"
              fontSize="11.5"
              letterSpacing="1.1"
            >
              ТАКТИЧЕСКИЙ 5×5 СИМУЛЯТОР ИИ · ДРАФТ · ПЕСОЧНИЦА
            </text>
          </svg>

          <p className="text-[#d6d3d1] text-base leading-relaxed mb-7 max-w-md">
            Шестнадцать процедурных чувачков со случайными характеристиками, запасными на скамейке и
            безумными чертами рубятся 5 на 5 под присмотром зрячего судьи.
          </p>

          <div className="flex flex-col sm:flex-row flex-wrap gap-3">
            <button
              type="button"
              data-start="true"
              onClick={(e) => {
                e.currentTarget.blur();
                onStartQuickMatch();
              }}
              className="px-6 py-3.5 bg-[#ea580c] hover:bg-[#f97316] active:translate-y-0.5 text-[#fafaf9] font-display text-lg tracking-wide rounded-lg transition-all cursor-pointer whitespace-nowrap shrink-0 shadow-lg focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#fbbf24]"
            >
              Быстрый матч 5×5
            </button>

            <button
              type="button"
              onClick={(e) => {
                e.currentTarget.blur();
                onOpenTournament();
              }}
              className="px-5 py-3.5 bg-[#26221f] hover:bg-[#322d29] border border-[#443c35] text-[#f5f2eb] font-semibold text-sm rounded-lg transition-colors cursor-pointer whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-[#fbbf24]"
            >
              Кубковый Турнир и Драфт
            </button>

            <button
              type="button"
              onClick={(e) => {
                e.currentTarget.blur();
                onOpenSandbox();
              }}
              className="px-5 py-3.5 bg-[#1c1917] hover:bg-[#292524] border border-[#3b342e] text-[#d6d3d1] font-semibold text-sm rounded-lg transition-colors cursor-pointer whitespace-nowrap shrink-0 focus-visible:outline-2 focus-visible:outline-[#fbbf24]"
            >
              Песочница черт ({TRAIT_CATALOG.length})
            </button>
          </div>
        </div>
      </div>

      {/* Нижняя плоская строка управления без рамок и плашек */}
      <div className="px-8 sm:px-14 lg:px-20 pb-5 text-xs text-[#a8a29e]">
        Клик по игроку — досье ИИ и черты · Пробел / Enter — старт · P / Esc — пауза · R — новые составы · M — звук
      </div>
    </div>
  );
};
