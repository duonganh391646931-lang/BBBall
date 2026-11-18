import {
  CLUB_PRESETS,
  generateRandomPlayer,
  getPlayerOverall,
} from './traits';
import {
  BracketMatchup,
  ClubTeam,
  PlayerEntity,
  PositionRole,
  TournamentState,
} from './types';

const ROLES_ORDER: PositionRole[] = ['PG', 'SG', 'SF', 'PF', 'C'];

export function createInitialTournament(
  clubName = 'Оранжевые Метеоры',
  clubColor = '#ea580c',
  rng: () => number = Math.random
): TournamentState {
  const usedNames = new Set<string>();

  // Генерируем 12 разнообразных кандидатов на стартовый драфт (на старт 5 + скамейку запасных 3)
  const draftPool: PlayerEntity[] = Array.from({ length: 12 }, (_, idx) => {
    const p = generateRandomPlayer(0, idx % 5, usedNames, rng, idx >= 5);
    p.role = ROLES_ORDER[idx % 5];
    return p;
  });

  const userClub: ClubTeam = {
    id: 'club_user',
    name: clubName,
    primaryColor: clubColor,
    secondaryColor: '#fed7aa',
    isUserClub: true,
    roster: [],
  };

  // 7 ИИ-клубов для сетки на 8 команд (у каждого 5 игроков старта + 3 запасных)
  const availablePresets = CLUB_PRESETS.filter((c) => c.name !== clubName);
  const aiClubs: ClubTeam[] = Array.from({ length: 7 }, (_, cIdx) => {
    const preset = availablePresets[cIdx % availablePresets.length];
    const starters = Array.from({ length: 5 }, (__, slot) =>
      generateRandomPlayer(1, slot, usedNames, rng, false)
    );
    const subs = Array.from({ length: 3 }, (__, bSlot) =>
      generateRandomPlayer(1, bSlot, usedNames, rng, true)
    );
    return {
      id: `club_ai_${cIdx}`,
      name: preset.name,
      primaryColor: preset.primary,
      secondaryColor: preset.secondary,
      isUserClub: false,
      roster: [...starters, ...subs],
    };
  });

  const allClubs: ClubTeam[] = [userClub, ...aiClubs];
  const matches: BracketMatchup[] = [
    // Раунд 0: Четвертьфиналы (4 пары)
    {
      id: 'qf_0',
      round: 0,
      teamA: allClubs[0],
      teamB: allClubs[1],
      scoreA: null,
      scoreB: null,
      winnerId: null,
    },
    {
      id: 'qf_1',
      round: 0,
      teamA: allClubs[2],
      teamB: allClubs[3],
      scoreA: null,
      scoreB: null,
      winnerId: null,
    },
    {
      id: 'qf_2',
      round: 0,
      teamA: allClubs[4],
      teamB: allClubs[5],
      scoreA: null,
      scoreB: null,
      winnerId: null,
    },
    {
      id: 'qf_3',
      round: 0,
      teamA: allClubs[6],
      teamB: allClubs[7],
      scoreA: null,
      scoreB: null,
      winnerId: null,
    },
    // Раунд 1: Полуфиналы (2 пары)
    {
      id: 'sf_0',
      round: 1,
      teamA: null,
      teamB: null,
      scoreA: null,
      scoreB: null,
      winnerId: null,
    },
    {
      id: 'sf_1',
      round: 1,
      teamA: null,
      teamB: null,
      scoreA: null,
      scoreB: null,
      winnerId: null,
    },
    // Раунд 2: Финал
    {
      id: 'final_0',
      round: 2,
      teamA: null,
      teamB: null,
      scoreA: null,
      scoreB: null,
      winnerId: null,
    },
  ];

  return {
    active: true,
    phase: 'draft',
    userClub,
    aiClubs,
    draftPool,
    reinforcePool: [],
    currentRound: 0,
    matches,
    championClub: null,
  };
}

export function assignRolesToRoster(roster: PlayerEntity[]): PlayerEntity[] {
  const benchRoles: PositionRole[] = ['PG', 'SF', 'C'];
  return roster.slice(0, 8).map((p, idx) => {
    const isBench = idx >= 5;
    return {
      ...p,
      slotIndex: isBench ? idx - 5 : idx,
      benchSlot: isBench ? idx - 5 : 0,
      role: isBench ? benchRoles[(idx - 5) % benchRoles.length] : ROLES_ORDER[idx],
      onCourt: !isBench,
      startedOnBench: isBench,
      subbedInOnce: false,
      team: 0,
    };
  });
}

function simulateParallelAIMatchScore(
  teamA: ClubTeam,
  teamB: ClubTeam,
  rng: () => number
): { scoreA: number; scoreB: number; winner: ClubTeam } {
  const avgA =
    teamA.roster.reduce((s, p) => s + getPlayerOverall(p), 0) / Math.max(1, teamA.roster.length);
  const avgB =
    teamB.roster.reduce((s, p) => s + getPlayerOverall(p), 0) / Math.max(1, teamB.roster.length);

  let scoreA = Math.round(34 + (avgA - 50) * 0.45 + rng() * 22);
  let scoreB = Math.round(34 + (avgB - 50) * 0.45 + rng() * 22);
  if (scoreA === scoreB) {
    if (rng() < 0.5) scoreA += 2;
    else scoreB += 2;
  }
  return {
    scoreA,
    scoreB,
    winner: scoreA > scoreB ? teamA : teamB,
  };
}

export function findUserCurrentMatchup(t: TournamentState): BracketMatchup | null {
  return (
    t.matches.find(
      (m) =>
        m.round === t.currentRound &&
        m.winnerId === null &&
        (m.teamA?.isUserClub || m.teamB?.isUserClub)
    ) ?? null
  );
}

export function recordTournamentUserMatchResult(
  t: TournamentState,
  userScore: number,
  oppScore: number,
  rng: () => number = Math.random
): TournamentState {
  const next: TournamentState = {
    ...t,
    matches: t.matches.map((m) => ({ ...m })),
  };

  const roundMatches = next.matches.filter((m) => m.round === next.currentRound);
  const winners: ClubTeam[] = [];

  for (const m of roundMatches) {
    if (!m.teamA || !m.teamB) continue;
    if (m.teamA.isUserClub || m.teamB.isUserClub) {
      const userIsA = m.teamA.isUserClub;
      m.scoreA = userIsA ? userScore : oppScore;
      m.scoreB = userIsA ? oppScore : userScore;
      const winClub = m.scoreA >= m.scoreB ? m.teamA : m.teamB;
      m.winnerId = winClub.id;
      winners.push(winClub);
    } else {
      const sim = simulateParallelAIMatchScore(m.teamA, m.teamB, rng);
      m.scoreA = sim.scoreA;
      m.scoreB = sim.scoreB;
      m.winnerId = sim.winner.id;
      winners.push(sim.winner);
    }
  }

  const userWonRound = winners.some((w) => w.isUserClub);

  if (next.currentRound === 0) {
    // Заполняем полуфиналы
    const sf0 = next.matches.find((m) => m.id === 'sf_0')!;
    const sf1 = next.matches.find((m) => m.id === 'sf_1')!;
    sf0.teamA = winners[0] ?? null;
    sf0.teamB = winners[1] ?? null;
    sf1.teamA = winners[2] ?? null;
    sf1.teamB = winners[3] ?? null;
    next.currentRound = 1;
  } else if (next.currentRound === 1) {
    const fin = next.matches.find((m) => m.id === 'final_0')!;
    fin.teamA = winners[0] ?? null;
    fin.teamB = winners[1] ?? null;
    next.currentRound = 2;
  } else {
    next.championClub = winners[0] ?? null;
    next.phase = 'champion';
    return next;
  }

  if (userWonRound) {
    // Генерируем 3 свежих кандидата для усиления состава между стадиями
    const used = new Set(next.userClub.roster.map((p) => p.name));
    next.reinforcePool = Array.from({ length: 3 }, (_, idx) =>
      generateRandomPlayer(0, idx, used, rng)
    );
    next.phase = 'reinforce';
  } else {
    // Если клуб игрока вылетел — доигрываем оставшиеся раунды ИИ и определяем чемпиона
    while (next.phase !== 'champion') {
      const rem = next.matches.filter((m) => m.round === next.currentRound);
      const rWinners: ClubTeam[] = [];
      for (const m of rem) {
        if (!m.teamA || !m.teamB) continue;
        const sim = simulateParallelAIMatchScore(m.teamA, m.teamB, rng);
        m.scoreA = sim.scoreA;
        m.scoreB = sim.scoreB;
        m.winnerId = sim.winner.id;
        rWinners.push(sim.winner);
      }
      if (next.currentRound === 1) {
        const fin = next.matches.find((m) => m.id === 'final_0')!;
        fin.teamA = rWinners[0] ?? null;
        fin.teamB = rWinners[1] ?? null;
        next.currentRound = 2;
      } else {
        next.championClub = rWinners[0] ?? null;
        next.phase = 'champion';
      }
    }
  }

  return next;
}
