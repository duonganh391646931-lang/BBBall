import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ArenaHud } from './components/ArenaHud';
import { BoxScoreModal } from './components/BoxScoreModal';
import { SandboxModal } from './components/SandboxModal';
import { StatusModal } from './components/StatusModal';
import { TitleOverlay } from './components/TitleOverlay';
import { TournamentModal } from './components/TournamentModal';
import { CourtTransform, renderMatchTopDown, screenToWorld } from './game/renderer';
import {
  callTeamTimeout,
  createMatch,
  requestPlayerSubstitution,
  stepMatch,
  syncModifierBalls,
  triggerCoachPlayCall,
  updatePlayerTraitsAndStatsInSandbox,
} from './game/sim';
import { sound } from './game/sound';
import {
  createInitialTournament,
  recordTournamentUserMatchResult,
} from './game/tournament';
import {
  CoachPlayCall,
  GameScreenState,
  MatchModifierId,
  MatchState,
  PlayerEntity,
  PlayerStats,
  TeamTactic,
  TournamentState,
} from './game/types';

function detectInitialScreenState(): GameScreenState {
  if (typeof window === 'undefined') return 'title';
  const params = new URLSearchParams(window.location.search);
  return params.get('autostart') === '1' ? 'playing' : 'title';
}

export default function App() {
  const [screenState, setScreenState] = useState<GameScreenState>(detectInitialScreenState);
  const [simSpeed, setSimSpeed] = useState<number>(1);
  const [muted, setMuted] = useState<boolean>(false);
  const [showTacticalOverlay, setShowTacticalOverlay] = useState<boolean>(true);
  const [sandboxOpen, setSandboxOpen] = useState<boolean>(false);
  const [sandboxFocusPlayerId, setSandboxFocusPlayerId] = useState<string | null>(null);
  const [tournamentOpen, setTournamentOpen] = useState<boolean>(false);
  const [boxScoreOpen, setBoxScoreOpen] = useState<boolean>(false);
  const [tournament, setTournament] = useState<TournamentState>(() =>
    createInitialTournament()
  );

  // Живое состояние матча хранится в ref для 60 FPS без пересоздания Canvas, а HUD получает срез ~12 Гц
  const matchRef = useRef<MatchState>(createMatch({ quarterDuration: 45 }));
  const [hudSnapshot, setHudSnapshot] = useState<MatchState>(() => matchRef.current);

  const screenStateRef = useRef<GameScreenState>(screenState);
  const simSpeedRef = useRef<number>(simSpeed);
  const showTacticalOverlayRef = useRef<boolean>(showTacticalOverlay);
  const modalOpenRef = useRef<boolean>(false);
  const canvasDirtyRef = useRef<boolean>(true);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastTransformRef = useRef<CourtTransform>({ scale: 1, offsetX: 0, offsetY: 0 });
  const tournamentRef = useRef<TournamentState>(tournament);

  useEffect(() => {
    screenStateRef.current = screenState;
    document.body.dataset.state = screenState;
    canvasDirtyRef.current = true;
  }, [screenState]);

  useEffect(() => {
    simSpeedRef.current = simSpeed;
  }, [simSpeed]);

  useEffect(() => {
    showTacticalOverlayRef.current = showTacticalOverlay;
    canvasDirtyRef.current = true;
  }, [showTacticalOverlay]);

  useEffect(() => {
    modalOpenRef.current = sandboxOpen || tournamentOpen || boxScoreOpen;
    canvasDirtyRef.current = true;
  }, [sandboxOpen, tournamentOpen, boxScoreOpen]);

  useEffect(() => {
    tournamentRef.current = tournament;
  }, [tournament]);

  const forceHudSync = useCallback(() => {
    canvasDirtyRef.current = true;
    setHudSnapshot({
      ...matchRef.current,
      players: [...matchRef.current.players],
      bench: [...matchRef.current.bench],
      balls: [...matchRef.current.balls],
      score: [...matchRef.current.score] as [number, number],
      teamFouls: [...matchRef.current.teamFouls] as [number, number],
      timeoutsRemaining: [...matchRef.current.timeoutsRemaining] as [number, number],
      playCallCooldown: [...matchRef.current.playCallCooldown] as [number, number],
      activePlayCall: matchRef.current.activePlayCall
        ? { ...matchRef.current.activePlayCall }
        : null,
      shotChart: [...matchRef.current.shotChart],
      modifiers: [...matchRef.current.modifiers],
      tactics: [...matchRef.current.tactics] as [TeamTactic, TeamTactic],
      events: [...matchRef.current.events],
    });
  }, []);

  const startFreshExhibition = useCallback(() => {
    sound.unlock();
    const prev = matchRef.current;
    matchRef.current = createMatch({
      mode: 'exhibition',
      quarterDuration: prev.quarterDuration,
      modifiers: prev.modifiers,
      tactics: prev.tactics,
    });
    setSandboxOpen(false);
    setTournamentOpen(false);
    setBoxScoreOpen(false);
    setScreenState('playing');
    forceHudSync();
  }, [forceHudSync]);

  // Главный 60 FPS игровой цикл (смонтирован один раз, не пересоздаёт canvas!)
  useEffect(() => {
    let rafId = 0;
    let timerId = 0;
    let cancelled = false;
    let lastTime = performance.now();
    let lastHudKey = '';
    let hudSyncAccum = 0;

    const resizeCanvas = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = window.innerWidth;
      const h = window.innerHeight;
      if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
        canvas.width = Math.floor(w * dpr);
        canvas.height = Math.floor(h * dpr);
        canvasDirtyRef.current = true;
      }
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const frame = (now: number) => {
      if (cancelled) return;
      const rawDt = Math.min((now - lastTime) / 1000, 0.05);
      lastTime = now;

      const curScreen = screenStateRef.current;
      const m = matchRef.current;

      // На титульном экране крутится живая фоновая демонстрация 5x5, в игре — основной матч!
      const shouldStep =
        !modalOpenRef.current &&
        (curScreen === 'playing' || curScreen === 'title');

      if (shouldStep) {
        const speedMult = curScreen === 'title' ? 1 : simSpeedRef.current;
        // Разбиваем большие ускорения (2x, 4x) на подшаги для точной баллистики и перехватов Кузнечика
        const subSteps = speedMult >= 4 ? 4 : speedMult >= 2 ? 2 : 1;
        const stepDt = (rawDt * speedMult) / subSteps;

        for (let s = 0; s < subSteps; s++) {
          stepMatch(m, stepDt);
          if (m.finished) {
            if (curScreen === 'title') {
              // На заставке перезапускаем демо-разминку бесшовно
              matchRef.current = createMatch({ quarterDuration: 45 });
            } else {
              // Если шёл турнирный матч — записываем результат в сетку кубка!
              if (m.mode === 'tournament') {
                const nextTour = recordTournamentUserMatchResult(
                  tournamentRef.current,
                  m.score[0],
                  m.score[1]
                );
                setTournament(nextTour);
              }
              setScreenState('gameover');
              forceHudSync();
            }
            break;
          }
        }
      }

      // Отрисовка строгого тактического вида сверху (не тратим CPU впустую, когда открыто модальное окно или пауза)
      if (shouldStep || canvasDirtyRef.current) {
        canvasDirtyRef.current = false;
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          if (ctx) {
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            ctx.save();
            ctx.scale(dpr, dpr);
            lastTransformRef.current = renderMatchTopDown(
              ctx,
              window.innerWidth,
              window.innerHeight,
              matchRef.current,
              showTacticalOverlayRef.current
            );
            ctx.restore();
          }
        }
      }

      // Обновляем React HUD не чаще 3 раз в секунду при реальном изменении данных
      hudSyncAccum += rawDt;
      if (hudSyncAccum >= 0.33 && screenStateRef.current === 'playing' && !modalOpenRef.current) {
        hudSyncAccum = 0;
        const curM = matchRef.current;
        const sel =
          curM.players.find((p) => p.id === curM.selectedPlayerId) ??
          curM.bench.find((p) => p.id === curM.selectedPlayerId);
        const hudKey = `${curM.quarter}|${Math.ceil(curM.quarterTimeRemaining)}|${Math.ceil(curM.shotClock)}|${curM.score[0]}:${curM.score[1]}|${curM.timeoutsRemaining[0]}:${curM.timeoutsRemaining[1]}|${Math.ceil(curM.playCallCooldown[0])}|${curM.activePlayCall?.play ?? ''}|${curM.substitutionsCount}|${curM.nextEventId}|${curM.lastBannerText ?? ''}|${sel?.id ?? ''}:${sel?.aiAction ?? ''}:${sel?.fouls ?? 0}:${sel?.hotStreak ?? 0}:${Math.round((sel?.energy ?? 0) / 5)}`;
        if (hudKey !== lastHudKey) {
          lastHudKey = hudKey;
          forceHudSync();
        }
      }

      timerId = window.setTimeout(() => {
        if (!cancelled) {
          rafId = requestAnimationFrame(frame);
        }
      }, 16);
    };

    rafId = requestAnimationFrame(frame);
    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [forceHudSync]);

  // Горячие клавиши управления оболочкой игры
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

      if (screenStateRef.current === 'title' && !modalOpenRef.current) {
        if (e.code === 'Space' || e.code === 'Enter') {
          e.preventDefault();
          startFreshExhibition();
          return;
        }
      }

      if (e.code === 'Tab' && screenStateRef.current !== 'title') {
        e.preventDefault();
        forceHudSync();
        setBoxScoreOpen((prev) => !prev);
        return;
      }

      if (e.code === 'Escape' || e.code === 'KeyP') {
        if (boxScoreOpen) {
          setBoxScoreOpen(false);
          return;
        }
        if (sandboxOpen) {
          setSandboxOpen(false);
          return;
        }
        if (tournamentOpen) {
          setTournamentOpen(false);
          return;
        }
        if (screenStateRef.current === 'playing') {
          setScreenState('paused');
        } else if (screenStateRef.current === 'paused') {
          setScreenState('playing');
        }
      } else if (e.code === 'KeyM') {
        setMuted((prev) => {
          const next = !prev;
          sound.muted = next;
          return next;
        });
      } else if (e.code === 'KeyT' && !modalOpenRef.current) {
        setShowTacticalOverlay((prev) => !prev);
      } else if (
        e.code === 'KeyO' &&
        screenStateRef.current === 'playing' &&
        !modalOpenRef.current
      ) {
        if (callTeamTimeout(matchRef.current, 0)) {
          forceHudSync();
        }
      } else if (
        (e.code === 'Digit1' || e.code === 'Digit2' || e.code === 'Digit3' || e.code === 'Digit4') &&
        screenStateRef.current === 'playing' &&
        !modalOpenRef.current
      ) {
        const playMap: Record<string, CoachPlayCall> = {
          Digit1: 'iso_star',
          Digit2: 'pick_and_roll',
          Digit3: 'alley_oop',
          Digit4: 'feed_sniper',
        };
        const chosen = playMap[e.code];
        if (chosen && triggerCoachPlayCall(matchRef.current, 0, chosen)) {
          forceHudSync();
        }
      } else if (
        e.code === 'KeyR' &&
        screenStateRef.current === 'playing' &&
        !modalOpenRef.current &&
        matchRef.current.mode === 'exhibition'
      ) {
        startFreshExhibition();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [sandboxOpen, tournamentOpen, boxScoreOpen, forceHudSync, startFreshExhibition]);

  // Выбор чувачка кликом прямо по тактическому паркету
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    sound.unlock();
    if (screenState !== 'playing' || sandboxOpen || tournamentOpen || boxScoreOpen) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const world = screenToWorld(e.clientX, e.clientY, rect, lastTransformRef.current);

    let closest: PlayerEntity | null = null;
    let bestDist = 42;
    for (const p of [...matchRef.current.players, ...matchRef.current.bench]) {
      const d = Math.hypot(p.x - world.x, p.y - world.y);
      if (d < bestDist) {
        bestDist = d;
        closest = p;
      }
    }
    if (closest) {
      matchRef.current.selectedPlayerId = closest.id;
      forceHudSync();
    }
  };

  const handleSubstitutePlayer = (playerId: string) => {
    requestPlayerSubstitution(matchRef.current, playerId);
    forceHudSync();
  };

  const handleToggleModifier = (modId: MatchModifierId) => {
    const m = matchRef.current;
    if (m.modifiers.includes(modId)) {
      m.modifiers = m.modifiers.filter((id) => id !== modId);
    } else {
      m.modifiers = [...m.modifiers, modId];
    }
    syncModifierBalls(m);
    forceHudSync();
  };

  const handleUpdatePlayerInSandbox = (
    playerId: string,
    traitIds: string[],
    baseStats: PlayerStats
  ) => {
    updatePlayerTraitsAndStatsInSandbox(matchRef.current, playerId, traitIds, baseStats);
    forceHudSync();
  };

  const handleLaunchTournamentMatch = (
    userRoster: PlayerEntity[],
    oppRoster: PlayerEntity[],
    oppName: string,
    oppColor: string
  ) => {
    sound.unlock();
    matchRef.current = createMatch({
      mode: 'tournament',
      teamNames: [tournament.userClub.name, oppName],
      teamColors: [tournament.userClub.primaryColor, oppColor],
      rosterA: userRoster,
      rosterB: oppRoster,
      quarterDuration: matchRef.current.quarterDuration,
      modifiers: matchRef.current.modifiers,
      tactics: [matchRef.current.tactics[0], 'balanced'],
    });
    setTournamentOpen(false);
    setSandboxOpen(false);
    setScreenState('playing');
    forceHudSync();
  };

  return (
    <main className="relative w-screen h-screen overflow-hidden bg-[#141210]">
      {/* Изолированный холст тактической площадки 5x5 */}
      <canvas
        ref={canvasRef}
        data-testid="basketball-court-canvas"
        onClick={handleCanvasClick}
        className="block w-full h-full cursor-pointer"
      />

      {screenState === 'title' && (
        <TitleOverlay
          onStartQuickMatch={startFreshExhibition}
          onOpenTournament={() => {
            sound.unlock();
            setTournamentOpen(true);
          }}
          onOpenSandbox={() => {
            sound.unlock();
            setSandboxFocusPlayerId(matchRef.current.players[0]?.id ?? null);
            setSandboxOpen(true);
          }}
        />
      )}

      {screenState !== 'title' && (
        <ArenaHud
          match={hudSnapshot}
          simSpeed={simSpeed}
          muted={muted}
          showTacticalOverlay={showTacticalOverlay}
          hasActiveTournament={tournament.userClub.roster.length >= 5}
          onChangeSpeed={setSimSpeed}
          onToggleModifier={handleToggleModifier}
          onToggleTacticalOverlay={() => setShowTacticalOverlay((prev) => !prev)}
          onChangeTactic={(team, tactic) => {
            matchRef.current.tactics[team] = tactic;
            forceHudSync();
          }}
          onCallTimeout={(team) => {
            sound.unlock();
            if (callTeamTimeout(matchRef.current, team)) {
              forceHudSync();
            }
          }}
          onTriggerPlayCall={(team, play) => {
            sound.unlock();
            if (triggerCoachPlayCall(matchRef.current, team, play)) {
              forceHudSync();
            }
          }}
          onSelectPlayer={(pid) => {
            matchRef.current.selectedPlayerId = pid;
            forceHudSync();
          }}
          onSubstitutePlayer={handleSubstitutePlayer}
          onOpenSandboxForPlayer={(pid) => {
            setSandboxFocusPlayerId(pid);
            setSandboxOpen(true);
          }}
          onOpenBoxScore={() => {
            forceHudSync();
            setBoxScoreOpen(true);
          }}
          onOpenTournament={() => setTournamentOpen(true)}
          onRerollMatch={startFreshExhibition}
          onToggleMute={() => {
            setMuted((prev) => {
              const next = !prev;
              sound.muted = next;
              return next;
            });
          }}
          onPause={() => setScreenState('paused')}
        />
      )}

      {(screenState === 'paused' || screenState === 'gameover') &&
        !sandboxOpen &&
        !tournamentOpen &&
        !boxScoreOpen && (
          <StatusModal
            mode={screenState}
            match={hudSnapshot}
            isTournamentMatch={hudSnapshot.mode === 'tournament'}
            onResume={() => setScreenState('playing')}
            onOpenBoxScore={() => {
              forceHudSync();
              setBoxScoreOpen(true);
            }}
            onPlayAgainOrAdvance={() => {
              if (hudSnapshot.mode === 'tournament') {
                setScreenState('playing');
                setTournamentOpen(true);
              } else {
                const prev = matchRef.current;
                matchRef.current = createMatch({
                  mode: 'exhibition',
                  teamNames: prev.teamNames,
                  teamColors: prev.teamColors,
                  rosterA: [
                    ...prev.players.filter((p) => p.team === 0),
                    ...prev.bench.filter((p) => p.team === 0),
                  ],
                  rosterB: [
                    ...prev.players.filter((p) => p.team === 1),
                    ...prev.bench.filter((p) => p.team === 1),
                  ],
                  quarterDuration: prev.quarterDuration,
                  modifiers: prev.modifiers,
                  tactics: prev.tactics,
                });
                setScreenState('playing');
                forceHudSync();
              }
            }}
            onRerollNewMatch={startFreshExhibition}
            onReturnToTitle={() => {
              matchRef.current = createMatch({ quarterDuration: 45 });
              setScreenState('title');
              forceHudSync();
            }}
          />
        )}

      {sandboxOpen && (
        <SandboxModal
          match={hudSnapshot}
          initialPlayerId={sandboxFocusPlayerId}
          onUpdatePlayer={handleUpdatePlayerInSandbox}
          onSubstitutePlayer={handleSubstitutePlayer}
          onToggleModifier={handleToggleModifier}
          onChangeQuarterDuration={(sec) => {
            matchRef.current.quarterDuration = sec;
            if (matchRef.current.quarter === 1 && matchRef.current.score[0] === 0 && matchRef.current.score[1] === 0) {
              matchRef.current.quarterTimeRemaining = sec;
            }
            forceHudSync();
          }}
          onRerollAllPlayers={startFreshExhibition}
          onCloseAndPlay={() => {
            setSandboxOpen(false);
            if (screenState === 'title') {
              setScreenState('playing');
            }
            forceHudSync();
          }}
          onClose={() => setSandboxOpen(false)}
        />
      )}

      {tournamentOpen && (
        <TournamentModal
          tournament={tournament}
          onUpdateTournament={setTournament}
          onResetNewTournament={(name, color) =>
            setTournament(createInitialTournament(name, color))
          }
          onLaunchBracketMatch={handleLaunchTournamentMatch}
          onClose={() => setTournamentOpen(false)}
        />
      )}

      {boxScoreOpen && (
        <BoxScoreModal
          match={hudSnapshot}
          onSelectPlayer={(pid) => {
            matchRef.current.selectedPlayerId = pid;
            forceHudSync();
          }}
          onClose={() => setBoxScoreOpen(false)}
        />
      )}
    </main>
  );
}
