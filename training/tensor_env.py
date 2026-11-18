"""
Полностью тензорная 2.5D-среда «БалбесБола» на PyTorch (TensorBasketballArena).
Обсчитывает N параллельных баскетбольных площадок 5x5 прямо в видеопамяти GPU (CUDA) или на CPU
без пересылки кадров между процессами, соблюдая физический и размерностный паритет с src/game/sim.ts:
  - Площадка 1180x620, кольца (96, 320, 95) и (1084, 320, 95), дуга 248px, гравитация 540/275.
  - Правила 24с / 8с / 5с ввода / 3с в краске, конус зрения Судьи (80% спереди / 50% за спиной + подозрительность).
  - 9 характеристик, 31 слот черт, 5 модификаторов (включая «Два мяча»), заслоны, евростеп против подстановки (Take Charge).
  - 3-ступенчатый Curriculum Learning + встроенный тензорный учитель для быстрого разогрева.
"""

from __future__ import annotations

import math
import torch

from model import CONT_ACTION_DIM, NUM_DISCRETE_ACTIONS, OBS_DIM

COURT_W = 1180.0
COURT_H = 620.0
COURT_LEFT = 50.0
COURT_RIGHT = 1130.0
COURT_TOP = 64.0
COURT_BOTTOM = 576.0
MID_X = 590.0
HOOP_Y = 320.0
HOOP_LEFT_X = 96.0
HOOP_RIGHT_X = 1084.0
RIM_Z = 95.0
THREE_PT_RADIUS = 248.0
MOVE_TARGET_RADIUS = 140.0

# Индексы ключевых черт (совпадают с порядком TRAIT_CATALOG в src/game/traits.ts)
TRAIT_MESSI = 0
TRAIT_GRASSHOPPER = 1
TRAIT_EGOISTO = 2
TRAIT_MAGNET = 3
TRAIT_SCHIZO = 4
TRAIT_LAST_HOPE = 6
TRAIT_GORILLA_DUNK = 8
TRAIT_STEALTH = 10
TRAIT_DIRTY_ELBOW = 14
TRAIT_SHEEP_HERD = 19
TRAIT_OSCAR_ACTOR = 25


class TensorBasketballArena:
    """
    Векторизованная тензорная арена 5x5 (10 агентов на площадку).
    Поддерживает 3 ступени учебной программы (Curriculum Stage 1, 2, 3).
    """

    def __init__(
        self,
        num_envs: int = 1536,
        device: torch.device | str = "cpu",
        frame_skip: int = 4,
        quarter_duration: float = 45.0,
        curriculum_stage: int = 1,
        seed: int = 42,
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_agents_per_env = 10
        self.total_agents = self.num_envs * 10
        self.device = torch.device(device)
        self.frame_skip = int(frame_skip)
        self.dt = 1.0 / 60.0
        self.quarter_duration = float(quarter_duration)
        self.stage = int(curriculum_stage)

        torch.manual_seed(seed)

        N = self.num_envs
        # Команды и роли слотов [0..9]: 0..4 = Team 0 (PG..C), 5..9 = Team 1 (PG..C)
        self.agent_team = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], device=self.device).unsqueeze(0).expand(N, 10)
        self.agent_slot = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], device=self.device).unsqueeze(0).expand(N, 10)
        self.role_onehot = torch.eye(5, device=self.device)[self.agent_slot]  # [N, 10, 5]

        # Тензоры состояния игроков [N, 10]
        self.px = torch.zeros((N, 10), device=self.device)
        self.py = torch.zeros((N, 10), device=self.device)
        self.pz = torch.zeros((N, 10), device=self.device)
        self.pvx = torch.zeros((N, 10), device=self.device)
        self.pvy = torch.zeros((N, 10), device=self.device)
        self.pvz = torch.zeros((N, 10), device=self.device)
        self.heading = torch.zeros((N, 10), device=self.device)
        self.target_x = torch.zeros((N, 10), device=self.device)
        self.target_y = torch.zeros((N, 10), device=self.device)

        self.energy = torch.full((N, 10), 100.0, device=self.device)
        self.fouls = torch.zeros((N, 10), device=self.device)
        self.hot_streak = torch.zeros((N, 10), device=self.device)
        self.paint_timer = torch.zeros((N, 10), device=self.device)
        self.guarded_timer = torch.zeros((N, 10), device=self.device)
        self.steal_cd = torch.zeros((N, 10), device=self.device)
        self.super_jump_cd = torch.zeros((N, 10), device=self.device)
        self.skill_cd = torch.zeros((N, 10), device=self.device)
        self.dirty_cd = torch.zeros((N, 10), device=self.device)
        self.tripped_timer = torch.zeros((N, 10), device=self.device)
        self.penalty_timer = torch.zeros((N, 10), device=self.device)
        self.bitten_timer = torch.zeros((N, 10), device=self.device)
        self.stopped_dribble = torch.zeros((N, 10), dtype=torch.bool, device=self.device)
        self.screening = torch.zeros((N, 10), dtype=torch.bool, device=self.device)
        self.taking_charge = torch.zeros((N, 10), dtype=torch.bool, device=self.device)
        self.eurostepping = torch.zeros((N, 10), dtype=torch.bool, device=self.device)
        self.ref_suspicion = torch.zeros((N, 10), device=self.device)
        self.schizo_bread = torch.zeros((N, 10), dtype=torch.bool, device=self.device)
        self.schizo_timer = torch.full((N, 10), 10.0, device=self.device)
        self.last_hope = torch.zeros((N, 10), dtype=torch.bool, device=self.device)

        # 9 характеристик [N, 10, 9] (speed, shooting, stamina, steal, jump, teamwork, dribble, iq, deviance)
        self.stats = torch.zeros((N, 10, 9), device=self.device)
        # 31 слот черт [N, 10, 31]
        self.traits = torch.zeros((N, 10, 31), device=self.device)

        # Состояние до 2 мячей [N, 2]
        self.bx = torch.zeros((N, 2), device=self.device)
        self.by = torch.zeros((N, 2), device=self.device)
        self.bz = torch.zeros((N, 2), device=self.device)
        self.bvx = torch.zeros((N, 2), device=self.device)
        self.bvy = torch.zeros((N, 2), device=self.device)
        self.bvz = torch.zeros((N, 2), device=self.device)
        self.b_holder = torch.full((N, 2), -1, dtype=torch.long, device=self.device)  # -1 = ничейный, 0..9 = индекс игрока
        self.b_mode = torch.zeros((N, 2), dtype=torch.long, device=self.device)  # 0=loose/held, 1=pass, 2=shot, 3=tipoff
        self.b_receiver = torch.full((N, 2), -1, dtype=torch.long, device=self.device)
        self.b_target_hoop = torch.full((N, 2), -1, dtype=torch.long, device=self.device)
        self.b_flight_timer = torch.zeros((N, 2), device=self.device)
        self.b_make_prob = torch.zeros((N, 2), device=self.device)
        self.b_is_three = torch.zeros((N, 2), dtype=torch.bool, device=self.device)
        self.b_shooter = torch.full((N, 2), -1, dtype=torch.long, device=self.device)
        self.b_passer = torch.full((N, 2), -1, dtype=torch.long, device=self.device)
        self.b_active = torch.zeros((N, 2), dtype=torch.bool, device=self.device)

        # Глобальное состояние площадок [N]
        self.quarter = torch.ones((N,), dtype=torch.long, device=self.device)
        self.clock = torch.full((N,), self.quarter_duration, device=self.device)
        self.shot_clock = torch.full((N,), 24.0, device=self.device)
        self.backcourt_timer = torch.zeros((N,), device=self.device)
        self.poss_team = torch.full((N,), -1, dtype=torch.long, device=self.device)
        self.frontcourt = torch.zeros((N,), dtype=torch.bool, device=self.device)
        self.score = torch.zeros((N, 2), device=self.device)
        self.team_fouls = torch.zeros((N, 2), device=self.device)
        self.inbound_team = torch.full((N,), -1, dtype=torch.long, device=self.device)
        self.inbound_passer = torch.full((N,), -1, dtype=torch.long, device=self.device)
        self.inbound_timer = torch.zeros((N,), device=self.device)
        self.shot_touched_rim = torch.zeros((N,), dtype=torch.bool, device=self.device)
        self.dirty_bonus_used = torch.zeros((N,), dtype=torch.bool, device=self.device)

        # Судья [N]
        self.ref_x = torch.full((N,), MID_X, device=self.device)
        self.ref_y = torch.full((N,), COURT_BOTTOM + 16.0, device=self.device)
        self.ref_gaze = torch.full((N,), -math.pi * 0.5, device=self.device)

        # Тактика, комбинация планшета и 5 модификаторов [N]
        self.tactics = torch.zeros((N, 2), dtype=torch.long, device=self.device)
        self.play_call = torch.zeros((N, 2), dtype=torch.long, device=self.device)
        self.modifiers = torch.zeros((N, 5), device=self.device)
        self.prev_phi = torch.zeros((N, 2), device=self.device)

        # Предрасчитанные таблицы индексов партнёров [10, 4]
        mates_list = []
        for i in range(10):
            base = 0 if i < 5 else 5
            mates_list.append([m for m in range(base, base + 5) if m != i])
        self.mate_indices = torch.tensor(mates_list, dtype=torch.long, device=self.device)  # [10, 4]

        self.reset_all()

    def set_stage(self, stage: int) -> None:
        self.stage = max(1, min(3, int(stage)))
        self.reset_all()

    def _reset_indices(self, env_mask: torch.Tensor) -> None:
        if not env_mask.any():
            return
        idx = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
        M = int(idx.numel())

        # Базовая расстановка 5x5 перед вбрасыванием
        t0_x = torch.tensor([360.0, 420.0, 450.0, 500.0, 555.0], device=self.device)
        t1_x = COURT_W - t0_x
        base_x = torch.cat([t0_x, t1_x], dim=0).unsqueeze(0).expand(M, 10)
        base_y = torch.tensor(
            [320.0, 185.0, 455.0, 245.0, 320.0, 320.0, 185.0, 455.0, 245.0, 320.0],
            device=self.device,
        ).unsqueeze(0).expand(M, 10)

        self.px[idx] = base_x + (torch.rand((M, 10), device=self.device) - 0.5) * 18.0
        self.py[idx] = base_y + (torch.rand((M, 10), device=self.device) - 0.5) * 18.0
        self.pz[idx] = 0.0
        self.pvx[idx] = 0.0
        self.pvy[idx] = 0.0
        self.pvz[idx] = 0.0
        self.heading[idx, :5] = 0.0
        self.heading[idx, 5:] = math.pi
        self.target_x[idx] = self.px[idx]
        self.target_y[idx] = self.py[idx]

        self.energy[idx] = 100.0
        self.fouls[idx] = 0.0
        self.hot_streak[idx] = 0.0
        self.paint_timer[idx] = 0.0
        self.guarded_timer[idx] = 0.0
        self.steal_cd[idx] = 0.0
        self.super_jump_cd[idx] = 0.0
        self.skill_cd[idx] = 0.0
        self.dirty_cd[idx] = 0.0
        self.tripped_timer[idx] = 0.0
        self.penalty_timer[idx] = 0.0
        self.bitten_timer[idx] = 0.0
        self.stopped_dribble[idx] = False
        self.screening[idx] = False
        self.taking_charge[idx] = False
        self.eurostepping[idx] = False
        self.ref_suspicion[idx] = 0.0
        self.schizo_bread[idx] = False
        self.schizo_timer[idx] = 8.0 + torch.rand((M, 10), device=self.device) * 4.0

        # Генерация 9 характеристик с широкой дисперсией (от неуклюжих балбесов 20 до звёзд 98)
        rand_stats = 22.0 + torch.rand((M, 10, 9), device=self.device) * 76.0
        if self.stage == 1:
            rand_stats[..., 8] = 5.0  # На Ступени 1 Девиантность подавлена
        self.stats[idx] = rand_stats

        # Черты и модификаторы включаются на Ступени 3
        self.traits[idx] = 0.0
        self.last_hope[idx] = False
        self.modifiers[idx] = 0.0
        if self.stage >= 3:
            trait_pick = torch.randint(0, 31, (M, 10, 2), device=self.device)
            self.traits[idx] = torch.zeros((M, 10, 31), device=self.device).scatter_(
                2, trait_pick, (torch.rand((M, 10, 2), device=self.device) < 0.75).float()
            )
            # Эгоисто обнуляет командность
            has_ego = self.traits[idx, :, TRAIT_EGOISTO] > 0.5
            self.stats[idx, :, 5] = torch.where(has_ego, torch.zeros_like(self.stats[idx, :, 5]), self.stats[idx, :, 5])
            self.last_hope[idx] = (self.traits[idx, :, TRAIT_LAST_HOPE] > 0.5) & (
                torch.rand((M, 10), device=self.device) < 0.25
            )
            # В 35% матчей включаем 1 случайный модификатор (включая two_balls = 0)
            mod_active = torch.rand((M,), device=self.device) < 0.35
            mod_choice = torch.randint(0, 5, (M,), device=self.device)
            mod_oh = torch.nn.functional.one_hot(mod_choice, num_classes=5).float()
            self.modifiers[idx] = torch.where(mod_active.unsqueeze(-1), mod_oh, torch.zeros_like(mod_oh))

        # Спорный мяч в центре (и второй мяч, если активен two_balls)
        two_balls = self.modifiers[idx, 0] > 0.5
        self.bx[idx, 0] = MID_X
        self.by[idx, 0] = HOOP_Y
        self.bz[idx, 0] = 65.0
        self.bvx[idx, 0] = (torch.rand((M,), device=self.device) - 0.5) * 40.0
        self.bvy[idx, 0] = (torch.rand((M,), device=self.device) - 0.5) * 40.0
        self.bvz[idx, 0] = 165.0
        self.b_holder[idx, 0] = -1
        self.b_mode[idx, 0] = 3
        self.b_receiver[idx, 0] = -1
        self.b_target_hoop[idx, 0] = -1
        self.b_flight_timer[idx, 0] = 0.0
        self.b_active[idx, 0] = True

        self.bx[idx, 1] = MID_X
        self.by[idx, 1] = HOOP_Y - 75.0
        self.bz[idx, 1] = 45.0
        self.bvx[idx, 1] = 0.0
        self.bvy[idx, 1] = 0.0
        self.bvz[idx, 1] = 80.0
        self.b_holder[idx, 1] = -1
        self.b_mode[idx, 1] = 0
        self.b_receiver[idx, 1] = -1
        self.b_target_hoop[idx, 1] = -1
        self.b_flight_timer[idx, 1] = 0.0
        self.b_active[idx, 1] = two_balls

        self.quarter[idx] = 1
        self.clock[idx] = self.quarter_duration
        self.shot_clock[idx] = 24.0
        self.backcourt_timer[idx] = 0.0
        self.poss_team[idx] = -1
        self.frontcourt[idx] = False
        self.score[idx] = 0.0
        self.team_fouls[idx] = 0.0
        self.inbound_team[idx] = -1
        self.inbound_passer[idx] = -1
        self.inbound_timer[idx] = 0.0
        self.shot_touched_rim[idx] = False
        self.dirty_bonus_used[idx] = False

        self.ref_x[idx] = MID_X
        self.ref_y[idx] = COURT_BOTTOM + 16.0
        self.tactics[idx] = torch.randint(0, 4, (M, 2), device=self.device)
        self.play_call[idx] = torch.randint(0, 5, (M, 2), device=self.device)
        self.prev_phi[idx] = self._compute_potential()[idx]

    def reset_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_mask = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self._reset_indices(all_mask)
        return self.get_obs_and_masks()

    def _ref_detect_prob(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Возвращает (gaze_cos, detect_chance) размера [N, 10] по формуле computeRefereeDetectionChance."""
        dx = self.px - self.ref_x.unsqueeze(1)
        dy = self.py - self.ref_y.unsqueeze(1)
        dist = torch.clamp(torch.hypot(dx, dy), min=1e-3)
        gaze_cos = (torch.cos(self.ref_gaze).unsqueeze(1) * dx + torch.sin(self.ref_gaze).unsqueeze(1) * dy) / dist
        base_prob = torch.where(gaze_cos >= 0.0, 0.80, 0.50)
        stealth = (self.traits[:, :, TRAIT_STEALTH] > 0.5).float() * 0.45
        susp = self.ref_suspicion * 0.10
        chance = torch.clamp(base_prob - stealth + susp, 0.05, 0.98)
        if self.stage == 1:
            chance = torch.full_like(chance, 0.85)
        return gaze_cos, chance

    def _compute_potential(self) -> torch.Tensor:
        """Потенциал состояния Phi(s) [N, 2] для PBRS-формирования награды."""
        phi = torch.zeros((self.num_envs, 2), device=self.device)
        bx0 = self.bx[:, 0]
        by0 = self.by[:, 0]
        d_hoop1 = torch.hypot(bx0 - HOOP_RIGHT_X, by0 - HOOP_Y)
        d_hoop0 = torch.hypot(bx0 - HOOP_LEFT_X, by0 - HOOP_Y)

        prog0 = 1.0 - torch.clamp((d_hoop1 - 90.0) / 950.0, 0.0, 1.0)
        prog1 = 1.0 - torch.clamp((d_hoop0 - 90.0) / 950.0, 0.0, 1.0)

        p0_has = self.poss_team == 0
        p1_has = self.poss_team == 1
        phi[:, 0] = torch.where(p0_has, 0.65 * prog0 + 0.10 * self.frontcourt.float(), -0.30 * prog1)
        phi[:, 1] = torch.where(p1_has, 0.65 * prog1 + 0.10 * self.frontcourt.float(), -0.30 * prog0)
        return phi

    def _start_inbound(self, env_mask: torch.Tensor, recv_team: torch.Tensor) -> None:
        if not env_mask.any():
            return
        idx = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
        rt = recv_team[idx]
        passer_slot = torch.where(rt == 0, torch.zeros_like(rt), torch.full_like(rt, 5))
        inb_x = torch.where(rt == 0, torch.full((idx.numel(),), COURT_LEFT - 14.0, device=self.device), torch.full((idx.numel(),), COURT_RIGHT + 14.0, device=self.device))

        self.inbound_team[idx] = rt
        self.inbound_passer[idx] = passer_slot
        self.inbound_timer[idx] = 5.0
        self.poss_team[idx] = rt
        self.frontcourt[idx] = False
        self.shot_clock[idx] = 24.0
        self.backcourt_timer[idx] = 0.0
        self.dirty_bonus_used[idx] = False

        self.px[idx, passer_slot] = inb_x
        self.py[idx, passer_slot] = HOOP_Y
        self.pvx[idx, passer_slot] = 0.0
        self.pvy[idx, passer_slot] = 0.0

        # Гарантируем открытого разыгрывающего под приём паса (слот + 1)
        guard_slot = passer_slot + 1
        guard_x = torch.where(rt == 0, torch.full_like(inb_x, COURT_LEFT + 135.0), torch.full_like(inb_x, COURT_RIGHT - 135.0))
        self.px[idx, guard_slot] = guard_x
        self.py[idx, guard_slot] = HOOP_Y - 85.0

        self.b_holder[idx, 0] = passer_slot
        self.b_mode[idx, 0] = 0
        self.b_target_hoop[idx, 0] = -1
        self.b_receiver[idx, 0] = -1
        self.bx[idx, 0] = inb_x
        self.by[idx, 0] = HOOP_Y
        self.bz[idx, 0] = 26.0
        self.bvx[idx, 0] = 0.0
        self.bvy[idx, 0] = 0.0
        self.bvz[idx, 0] = 0.0

    def get_obs_and_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Формирует плоские тензоры (obs [N*10, 244], masks [N*10, 23]) в канонических зеркальных координатах."""
        N = self.num_envs
        dev = self.device
        team = self.agent_team  # [N, 10]
        is_t0 = team == 0
        sign = torch.where(is_t0, 1.0, -1.0)

        canon_px = torch.where(is_t0, self.px, COURT_W - self.px)
        canon_pvx = self.pvx * sign
        atk_hoop_x = torch.where(is_t0, HOOP_RIGHT_X, HOOP_LEFT_X)
        def_hoop_x = torch.where(is_t0, HOOP_LEFT_X, HOOP_RIGHT_X)
        dist_atk = torch.hypot(self.px - atk_hoop_x, self.py - HOOP_Y)

        agent_ids = torch.arange(10, device=dev).unsqueeze(0).expand(N, 10)
        held0 = self.b_holder[:, 0:1] == agent_ids
        held1 = self.b_active[:, 1:2] & (self.b_holder[:, 1:2] == agent_ids)
        has_ball = held0 | held1
        is_inb_passer = self.inbound_passer.unsqueeze(1) == agent_ids

        # 1. Глобальный блок (18)
        gaze_cos, ref_chance = self._ref_detect_prob()
        score_my = torch.gather(self.score, 1, team)
        score_opp = torch.gather(self.score, 1, 1 - team)
        fouls_my = torch.gather(self.team_fouls, 1, team)
        fouls_opp = torch.gather(self.team_fouls, 1, 1 - team)
        poss_my = (self.poss_team.unsqueeze(1) == team).float()
        poss_opp = (self.poss_team.unsqueeze(1) == (1 - team)).float()
        inb_my = (self.inbound_team.unsqueeze(1) == team).float()
        inb_opp = (self.inbound_team.unsqueeze(1) == (1 - team)).float()
        ref_canon_x = torch.where(is_t0, self.ref_x.unsqueeze(1), COURT_W - self.ref_x.unsqueeze(1))

        obs_global = torch.stack(
            [
                torch.clamp(self.quarter.float() / 5.0, 0.0, 1.0).unsqueeze(1).expand(N, 10),
                torch.clamp(self.clock / self.quarter_duration, 0.0, 1.0).unsqueeze(1).expand(N, 10),
                torch.clamp(self.shot_clock / 24.0, 0.0, 1.0).unsqueeze(1).expand(N, 10),
                torch.clamp(self.backcourt_timer / 8.0, 0.0, 1.0).unsqueeze(1).expand(N, 10),
                torch.clamp((score_my - score_opp) / 30.0, -1.0, 1.0),
                poss_my,
                poss_opp,
                (self.frontcourt.unsqueeze(1) & (poss_my > 0.5)).float(),
                torch.clamp(fouls_my / 5.0, 0.0, 1.0),
                torch.clamp(fouls_opp / 5.0, 0.0, 1.0),
                inb_my,
                inb_opp,
                torch.where(self.inbound_team >= 0, torch.clamp(self.inbound_timer / 5.0, 0.0, 1.0), torch.zeros_like(self.clock)).unsqueeze(1).expand(N, 10),
                (ref_canon_x - MID_X) / (COURT_W * 0.5),
                ((self.ref_y - HOOP_Y) / (COURT_H * 0.5)).unsqueeze(1).expand(N, 10),
                torch.clamp(gaze_cos, -1.0, 1.0),
                torch.clamp(self.ref_suspicion / 5.0, 0.0, 1.0),
                ref_chance,
            ],
            dim=-1,
        )  # [N, 10, 18]

        # 2. Самосознание (69)
        in_atk_paint = (torch.abs(self.px - atk_hoop_x) < 140.0) & (torch.abs(self.py - HOOP_Y) < 92.0)
        obs_kin_stat = torch.stack(
            [
                torch.clamp((canon_px - MID_X) / (COURT_W * 0.5), -1.2, 1.2),
                torch.clamp((self.py - HOOP_Y) / (COURT_H * 0.5), -1.2, 1.2),
                torch.clamp(self.pz / 250.0, 0.0, 1.5),
                torch.clamp(canon_pvx / 300.0, -1.5, 1.5),
                torch.clamp(self.pvy / 300.0, -1.5, 1.5),
                torch.clamp(self.pvz / 500.0, -1.5, 1.5),
                torch.cos(self.heading) * sign,
                torch.sin(self.heading),
                torch.clamp(dist_atk / 1100.0, 0.0, 1.2),
                (dist_atk >= THREE_PT_RADIUS).float(),
                has_ball.float(),
                self.stopped_dribble.float(),
                is_inb_passer.float(),
                in_atk_paint.float(),
                torch.clamp(self.paint_timer / 3.0, 0.0, 1.2),
                torch.clamp(self.guarded_timer / 5.0, 0.0, 1.2),
                torch.clamp(self.energy / 100.0, 0.0, 1.0),
                torch.clamp(self.fouls / 5.0, 0.0, 1.0),
                torch.clamp(self.hot_streak / 4.0, -1.0, 1.0),
                (self.steal_cd <= 0.0).float(),
                ((self.traits[:, :, TRAIT_GRASSHOPPER] > 0.5) & (self.super_jump_cd <= 0.0) & (self.energy >= 15.0)).float(),
                (self.skill_cd <= 0.0).float(),
                (self.skill_cd <= 0.0).float(),
                self.last_hope.float(),
            ],
            dim=-1,
        )  # [N, 10, 24]

        stat_scales = torch.tensor([115.0] * 8 + [100.0], device=dev)
        obs_stats = self.stats / stat_scales  # [N, 10, 9]
        obs_self = torch.cat([obs_kin_stat, self.role_onehot, obs_stats, self.traits], dim=-1)  # [N, 10, 69]

        # 3. Блок мячей: основной мяч (16) + второй мяч (8) = 24
        b0_cx = torch.where(is_t0, self.bx[:, 0:1], COURT_W - self.bx[:, 0:1])
        b0_y = self.by[:, 0:1].expand(N, 10)
        atk_idx = torch.where(is_t0, 1, 0)
        def_idx = 1 - atk_idx
        obs_ball1 = torch.stack(
            [
                torch.clamp((b0_cx - canon_px) / COURT_W, -1.2, 1.2),
                torch.clamp((b0_y - self.py) / COURT_H, -1.2, 1.2),
                torch.clamp((b0_cx - MID_X) / (COURT_W * 0.5), -1.2, 1.2),
                torch.clamp((b0_y - HOOP_Y) / (COURT_H * 0.5), -1.2, 1.2),
                torch.clamp(self.bz[:, 0:1] / 250.0, 0.0, 1.5).expand(N, 10),
                torch.clamp(self.bvx[:, 0:1] * sign / 600.0, -1.5, 1.5),
                torch.clamp(self.bvy[:, 0:1] / 600.0, -1.5, 1.5).expand(N, 10),
                torch.clamp(self.bvz[:, 0:1] / 600.0, -1.5, 1.5).expand(N, 10),
                ((self.b_holder[:, 0:1] < 0) & (self.b_mode[:, 0:1] == 0)).float().expand(N, 10),
                (self.b_mode[:, 0:1] == 1).float().expand(N, 10),
                (self.b_mode[:, 0:1] == 2).float().expand(N, 10),
                (self.b_receiver[:, 0:1] == agent_ids).float(),
                (self.b_target_hoop[:, 0:1] == atk_idx).float(),
                (self.b_target_hoop[:, 0:1] == def_idx).float(),
                self.shot_touched_rim.float().unsqueeze(1).expand(N, 10),
                (self.b_mode[:, 0:1] == 3).float().expand(N, 10),
            ],
            dim=-1,
        )

        b1_act = self.b_active[:, 1:2].float().expand(N, 10)
        b1_cx = torch.where(is_t0, self.bx[:, 1:2], COURT_W - self.bx[:, 1:2])
        b1_y = self.by[:, 1:2].expand(N, 10)
        obs_ball2 = (
            torch.stack(
                [
                    torch.ones((N, 10), device=dev),
                    torch.clamp((b1_cx - canon_px) / COURT_W, -1.2, 1.2),
                    torch.clamp((b1_y - self.py) / COURT_H, -1.2, 1.2),
                    torch.clamp(self.bz[:, 1:2] / 250.0, 0.0, 1.5).expand(N, 10),
                    torch.clamp(self.bvx[:, 1:2] * sign / 600.0, -1.5, 1.5),
                    torch.clamp(self.bvy[:, 1:2] / 600.0, -1.5, 1.5).expand(N, 10),
                    (self.b_target_hoop[:, 1:2] == atk_idx).float(),
                    (self.b_target_hoop[:, 1:2] == def_idx).float(),
                ],
                dim=-1,
            )
            * b1_act.unsqueeze(-1)
        )

        # Попарные расстояния [N, 10, 10]
        dx_all = self.px.unsqueeze(2) - self.px.unsqueeze(1)
        dy_all = self.py.unsqueeze(2) - self.py.unsqueeze(1)
        dist_all = torch.hypot(dx_all, dy_all)  # [N, 10, 10]

        # 4. Четыре партнёра [N, 10, 4, 14] -> [N, 10, 56]
        m_idx = self.mate_indices.unsqueeze(0).expand(N, 10, 4)  # [N, 10, 4]
        gather_2d = lambda src: torch.gather(src.unsqueeze(1).expand(N, 10, 10), 2, m_idx)
        m_px = gather_2d(self.px)
        m_py = gather_2d(self.py)
        m_pz = gather_2d(self.pz)
        m_pvx = gather_2d(self.pvx)
        m_pvy = gather_2d(self.pvy)
        m_has_ball = gather_2d(has_ball.float())
        m_active = gather_2d(((self.penalty_timer <= 0.0) & (self.tripped_timer <= 0.0) & (~self.schizo_bread)).float())
        m_shoot = gather_2d(self.stats[:, :, 1]) / 115.0
        m_jump = gather_2d(self.stats[:, :, 4]) / 115.0
        m_canon_x = torch.where(is_t0.unsqueeze(-1), m_px, COURT_W - m_px)
        m_dist_atk = torch.hypot(m_px - atk_hoop_x.unsqueeze(-1), m_py - HOOP_Y)

        # Минимальная дистанция защитника до каждого игрока [N, 10]
        same_team = team.unsqueeze(2) == team.unsqueeze(1)
        opp_dist_only = torch.where(same_team, torch.full_like(dist_all, 999.0), dist_all)
        min_def_per_player = opp_dist_only.min(dim=-1).values  # [N, 10]
        m_min_def = gather_2d(min_def_per_player)
        m_lane_risk = torch.clamp((140.0 - m_min_def) / 140.0, 0.0, 1.0)

        obs_mates = torch.stack(
            [
                m_active,
                torch.clamp((m_canon_x - canon_px.unsqueeze(-1)) / COURT_W, -1.2, 1.2),
                torch.clamp((m_py - self.py.unsqueeze(-1)) / COURT_H, -1.2, 1.2),
                torch.clamp((m_canon_x - MID_X) / (COURT_W * 0.5), -1.2, 1.2),
                torch.clamp((m_py - HOOP_Y) / (COURT_H * 0.5), -1.2, 1.2),
                torch.clamp(m_pz / 250.0, 0.0, 1.5),
                torch.clamp(m_pvx * sign.unsqueeze(-1) / 300.0, -1.5, 1.5),
                torch.clamp(m_pvy / 300.0, -1.5, 1.5),
                m_has_ball,
                torch.clamp(m_dist_atk / 1100.0, 0.0, 1.2),
                torch.clamp(m_min_def / 300.0, 0.0, 1.0),
                m_lane_risk,
                m_shoot,
                m_jump,
            ],
            dim=-1,
        ).reshape(N, 10, 56)

        # 5. Пять соперников, отсортированных по близости [N, 10, 5, 14] -> [N, 10, 70]
        o_idx = torch.topk(opp_dist_only, k=5, dim=-1, largest=False).indices  # [N, 10, 5]
        gather_opp = lambda src: torch.gather(src.unsqueeze(1).expand(N, 10, 10), 2, o_idx)
        o_px = gather_opp(self.px)
        o_py = gather_opp(self.py)
        o_pz = gather_opp(self.pz)
        o_pvx = gather_opp(self.pvx)
        o_pvy = gather_opp(self.pvy)
        o_has_ball = gather_opp(has_ball.float())
        o_active = gather_opp(((self.penalty_timer <= 0.0) & (self.tripped_timer <= 0.0)).float())
        o_bitten = gather_opp(((self.bitten_timer > 0.0) | (self.pz > 20.0)).float())
        o_shoot = gather_opp(self.stats[:, :, 1]) / 115.0
        o_steal = gather_opp(self.stats[:, :, 3]) / 115.0
        o_canon_x = torch.where(is_t0.unsqueeze(-1), o_px, COURT_W - o_px)
        o_dist_def = torch.hypot(o_px - def_hoop_x.unsqueeze(-1), o_py - HOOP_Y)

        obs_opps = torch.stack(
            [
                o_active,
                torch.clamp((o_canon_x - canon_px.unsqueeze(-1)) / COURT_W, -1.2, 1.2),
                torch.clamp((o_py - self.py.unsqueeze(-1)) / COURT_H, -1.2, 1.2),
                torch.clamp((o_canon_x - MID_X) / (COURT_W * 0.5), -1.2, 1.2),
                torch.clamp((o_py - HOOP_Y) / (COURT_H * 0.5), -1.2, 1.2),
                torch.clamp(o_pz / 250.0, 0.0, 1.5),
                torch.clamp(o_pvx * sign.unsqueeze(-1) / 300.0, -1.5, 1.5),
                torch.clamp(o_pvy / 300.0, -1.5, 1.5),
                o_has_ball,
                torch.zeros_like(o_active),
                o_bitten,
                torch.clamp(o_dist_def / 1100.0, 0.0, 1.2),
                o_shoot,
                o_steal,
            ],
            dim=-1,
        ).reshape(N, 10, 70)

        # 6. Тактика + комбинация + 5 модификаторов (7)
        tac_code = torch.gather(self.tactics, 1, team).float() / 3.0
        play_code = torch.gather(self.play_call, 1, team).float() * 0.25
        mods_exp = self.modifiers.unsqueeze(1).expand(N, 10, 5)
        obs_tail = torch.cat([tac_code.unsqueeze(-1), play_code.unsqueeze(-1), mods_exp], dim=-1)

        obs = torch.cat([obs_global, obs_self, obs_ball1, obs_ball2, obs_mates, obs_opps, obs_tail], dim=-1)
        assert obs.shape[-1] == OBS_DIM

        # Вычисление масок допустимых действий [N, 10, 23]
        masks = torch.zeros((N, 10, NUM_DISCRETE_ACTIONS), dtype=torch.uint8, device=dev)
        masks[:, :, 0] = 1
        can_act = (self.penalty_timer <= 0.0) & (self.tripped_timer <= 0.0)
        in_front = torch.where(is_t0, self.px > MID_X, self.px < MID_X)
        near_opp_d = torch.gather(opp_dist_only, 2, o_idx[:, :, 0:1]).squeeze(-1)

        # Действия с мячом
        can_pass = (self.traits[:, :, TRAIT_EGOISTO] < 0.5) | is_inb_passer
        for m_slot in range(4):
            m_ok = can_act & has_ball & can_pass & (m_active[:, :, m_slot] > 0.5)
            masks[:, :, 5 + m_slot] = m_ok.to(torch.uint8)
            ao_ok = m_ok & (~is_inb_passer) & in_front & (m_dist_atk[:, :, m_slot] <= 170.0) & (m_jump[:, :, m_slot] * 115.0 >= 68.0)
            masks[:, :, 9 + m_slot] = ao_ok.to(torch.uint8)

        # Запрет задержки ввода мяча при остатке <= 3.6с
        has_any_pass = masks[:, :, 5:9].sum(dim=-1) > 0
        urgent_inb = is_inb_passer & has_any_pass & (self.inbound_timer.unsqueeze(1) <= 3.6)
        masks[:, :, 0] = torch.where(urgent_inb, torch.zeros_like(masks[:, :, 0]), masks[:, :, 0])

        live_carrier = can_act & has_ball & (~is_inb_passer)
        can_shoot = live_carrier & (in_front | (self.shot_clock.unsqueeze(1) <= 2.2) | (self.clock.unsqueeze(1) <= 2.4))
        masks[:, :, 1] = can_shoot.to(torch.uint8)
        masks[:, :, 2] = (can_shoot & (~self.stopped_dribble) & (dist_atk >= 52.0) & (dist_atk <= 185.0)).to(torch.uint8)
        masks[:, :, 3] = (can_shoot & (dist_atk >= 40.0) & (dist_atk <= 150.0)).to(torch.uint8)
        masks[:, :, 4] = (can_shoot & (dist_atk <= 96.0) & (self.pz <= 1.0) & (self.stats[:, :, 4] >= 60.0)).to(torch.uint8)
        masks[:, :, 13] = (live_carrier & (self.skill_cd <= 0.0) & (near_opp_d <= 68.0)).to(torch.uint8)
        masks[:, :, 14] = (live_carrier & (~self.stopped_dribble) & in_front & (self.skill_cd <= 0.0) & (dist_atk >= 170.0) & (dist_atk <= 270.0) & (near_opp_d <= 58.0)).to(torch.uint8)
        masks[:, :, 15] = (live_carrier & (~self.stopped_dribble) & in_front & (self.skill_cd <= 0.0) & (dist_atk >= 68.0) & (dist_atk <= 182.0) & (near_opp_d <= 62.0)).to(torch.uint8)
        masks[:, :, 16] = (live_carrier & (~self.stopped_dribble) & in_front & (self.skill_cd <= 0.0) & (near_opp_d <= 56.0)).to(torch.uint8)

        # Действия без мяча
        off_ball = can_act & (~has_ball) & (self.inbound_team.unsqueeze(1) < 0)
        masks[:, :, 17] = (off_ball & (poss_my > 0.5) & in_front & (self.pz <= 1.0)).to(torch.uint8)
        masks[:, :, 18] = (off_ball & (poss_opp > 0.5) & (near_opp_d <= 80.0) & (self.pz <= 1.0)).to(torch.uint8)
        any_air = (self.bz[:, 0] > 18.0) | (self.b_mode[:, 0] == 3)
        masks[:, :, 19] = (off_ball & (self.pz <= 1.0) & any_air.unsqueeze(1)).to(torch.uint8)
        opp_carrier_near = (o_has_ball[:, :, 0] > 0.5) & (near_opp_d <= 48.0)
        masks[:, :, 20] = (off_ball & (self.steal_cd <= 0.0) & opp_carrier_near).to(torch.uint8)

        if self.stage >= 2:
            dirty_ok = off_ball & (self.dirty_cd <= 0.0) & (self.fouls < 4.0) & (self.stats[:, :, 8] >= 26.0) & (near_opp_d <= 42.0)
            masks[:, :, 21] = dirty_ok.to(torch.uint8)
            clutch_foul = off_ball & opp_carrier_near & (self.quarter.unsqueeze(1) >= 4) & (self.clock.unsqueeze(1) <= 18.0) & ((score_my - score_opp) < 0.0) & ((score_my - score_opp) >= -6.0) & (self.fouls < 4.0)
            masks[:, :, 22] = clutch_foul.to(torch.uint8)

        return obs.reshape(-1, OBS_DIM), masks.reshape(-1, NUM_DISCRETE_ACTIONS)

    @torch.no_grad()
    def teacher_act(self, obs_flat: torch.Tensor, masks_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Встроенный векторизованный учитель для Ступени 1 (Behavior Cloning прямо в тензорах):
        выводит мяч за 8с, пасует открытому партнёру, бросает по статам и держит защитную стойку между мячом и щитом.
        """
        N = self.num_envs
        masks = masks_flat.reshape(N, 10, NUM_DISCRETE_ACTIONS).bool()
        is_t0 = self.agent_team == 0
        sign = torch.where(is_t0, 1.0, -1.0)
        atk_x = torch.where(is_t0, HOOP_RIGHT_X, HOOP_LEFT_X)
        def_x = torch.where(is_t0, HOOP_LEFT_X, HOOP_RIGHT_X)

        agent_ids = torch.arange(10, device=self.device).unsqueeze(0).expand(N, 10)
        has_ball = self.b_holder[:, 0:1] == agent_ids
        poss_my = self.poss_team.unsqueeze(1) == self.agent_team
        dist_atk = torch.hypot(self.px - atk_x, self.py - HOOP_Y)

        # Целевая точка в мировых координатах
        # 1) С мячом: бежим к дуге/кольцу соперника
        aim_x = torch.where(is_t0, torch.full_like(self.px, HOOP_RIGHT_X - 115.0), torch.full_like(self.px, HOOP_LEFT_X + 115.0))
        aim_y = HOOP_Y + (self.agent_slot.float() - 2.0) * 38.0

        # 2) В атаке без мяча: расставляемся по ролям (спейсинг на дуге и в лоу-посте)
        sp_offset = torch.tensor([235.0, 225.0, 195.0, 115.0, 75.0], device=self.device)[self.agent_slot]
        sp_y = torch.tensor([320.0, 145.0, 495.0, 225.0, 395.0], device=self.device)[self.agent_slot]
        offball_x = torch.where(is_t0, HOOP_RIGHT_X - sp_offset, HOOP_LEFT_X + sp_offset)

        # 3) В защите: встаём строго на линии между визави/мячом и своим кольцом
        opp_pair = (self.agent_slot + torch.where(is_t0, 5, 0)) % 10
        man_x = torch.gather(self.px, 1, opp_pair)
        man_y = torch.gather(self.py, 1, opp_pair)
        def_shell_x = 0.68 * man_x + 0.32 * def_x
        def_shell_y = 0.72 * man_y + 0.28 * HOOP_Y

        loose = self.b_holder[:, 0:1] < 0
        tx = torch.where(has_ball, aim_x, torch.where(poss_my, offball_x, def_shell_x))
        ty = torch.where(has_ball, aim_y, torch.where(poss_my, sp_y, def_shell_y))
        tx = torch.where(loose, self.bx[:, 0:1].expand(N, 10), tx)
        ty = torch.where(loose, self.by[:, 0:1].expand(N, 10), ty)

        move_x = torch.clamp((tx - self.px) * sign / MOVE_TARGET_RADIUS, -1.0, 1.0)
        move_y = torch.clamp((ty - self.py) / MOVE_TARGET_RADIUS, -1.0, 1.0)
        move = torch.stack([move_x, move_y], dim=-1)

        disc = torch.zeros((N, 10), dtype=torch.long, device=self.device)
        # Ввод мяча из-за линии -> сразу пасуем первому открытому
        is_inb = self.inbound_passer.unsqueeze(1) == agent_ids
        disc = torch.where(is_inb & masks[:, :, 5], torch.full_like(disc, 5), disc)

        # Броски и данки в хорошей позиции
        good_range = (dist_atk < 145.0) | ((dist_atk < 265.0) & (self.stats[:, :, 1] >= 68.0)) | (self.shot_clock.unsqueeze(1) < 4.0)
        disc = torch.where(has_ball & (~is_inb) & good_range & masks[:, :, 1], torch.full_like(disc, 1), disc)
        disc = torch.where(has_ball & (~is_inb) & (dist_atk < 88.0) & masks[:, :, 4], torch.full_like(disc, 4), disc)

        # Передача вперёд в отрыв при выводе мяча
        in_back = ~torch.where(is_t0, self.px > MID_X, self.px < MID_X)
        disc = torch.where(has_ball & (~is_inb) & in_back & masks[:, :, 6] & (torch.rand_like(self.px) < 0.35), torch.full_like(disc, 6), disc)

        # Прыжок на спорном/подборе и чистый отбор
        disc = torch.where((disc == 0) & masks[:, :, 19] & (torch.hypot(self.px - self.bx[:, 0:1], self.py - self.by[:, 0:1]) < 65.0), torch.full_like(disc, 19), disc)
        disc = torch.where((disc == 0) & masks[:, :, 20] & (torch.rand_like(self.px) < 0.25), torch.full_like(disc, 20), disc)

        return disc.reshape(-1), move.reshape(-1, CONT_ACTION_DIM)

    @torch.no_grad()
    def step(
        self, disc_flat: torch.Tensor, move_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        N = self.num_envs
        dev = self.device
        disc = disc_flat.reshape(N, 10).long()
        move = torch.clamp(move_flat.reshape(N, 10, 2), -1.0, 1.0)

        team = self.agent_team
        is_t0 = team == 0
        sign = torch.where(is_t0, 1.0, -1.0)
        agent_ids = torch.arange(10, device=dev).unsqueeze(0).expand(N, 10)

        prev_score = self.score.clone()
        indiv_rew = torch.zeros((N, 10), device=dev)
        viol_team = torch.zeros((N, 2), device=dev)

        # Применяем решения агентов (1 раз перед циклом физических подшагов frame_skip)
        self.screening.zero_()
        self.taking_charge.zero_()
        self.eurostepping.zero_()

        is_inb_passer = self.inbound_passer.unsqueeze(1) == agent_ids
        has_b0 = self.b_holder[:, 0:1] == agent_ids

        # Установка целевой точки движения
        dx_w = move[:, :, 0] * sign * MOVE_TARGET_RADIUS
        dy_w = move[:, :, 1] * MOVE_TARGET_RADIUS
        nx = torch.clamp(self.px + dx_w, COURT_LEFT + 22.0, COURT_RIGHT - 22.0)
        ny = torch.clamp(self.py + dy_w, COURT_TOP + 22.0, COURT_BOTTOM - 22.0)
        fc_lock = self.frontcourt.unsqueeze(1) & (self.poss_team.unsqueeze(1) == team) & has_b0
        nx = torch.where(fc_lock & is_t0, torch.clamp(nx, min=MID_X + 22.0), nx)
        nx = torch.where(fc_lock & (~is_t0), torch.clamp(nx, max=MID_X - 22.0), nx)

        self.target_x = torch.where(has_b0 & self.stopped_dribble, self.px, nx)
        self.target_y = torch.where(has_b0 & self.stopped_dribble, self.py, ny)

        # Позиционные стойки: SCREEN_SET (17) и TAKE_CHARGE (18)
        can_stance = (~has_b0) & (~is_inb_passer) & (self.penalty_timer <= 0.0) & (self.tripped_timer <= 0.0)
        self.screening = can_stance & (disc == 17)
        self.taking_charge = can_stance & (disc == 18)
        self.target_x = torch.where(self.screening | self.taking_charge, self.px, self.target_x)
        self.target_y = torch.where(self.screening | self.taking_charge, self.py, self.target_y)

        # Дискретные действия игрока с основным мячом (слот 0)
        carrier_env, carrier_slot = torch.nonzero(has_b0, as_tuple=True)
        if carrier_env.numel() > 0:
            c_act = disc[carrier_env, carrier_slot]
            c_team = team[carrier_env, carrier_slot]
            c_x = self.px[carrier_env, carrier_slot]
            c_y = self.py[carrier_env, carrier_slot]
            atk_hoop_idx = 1 - c_team
            hoop_x = torch.where(atk_hoop_idx == 1, HOOP_RIGHT_X, HOOP_LEFT_X)
            d_hoop = torch.hypot(c_x - hoop_x, c_y - HOOP_Y)

            # Передачи (5..8) и Алли-упы (9..12)
            is_pass = (c_act >= 5) & (c_act <= 12)
            if is_pass.any():
                pe = carrier_env[is_pass]
                ps = carrier_slot[is_pass]
                pa = c_act[is_pass]
                m_local = (pa - 5) % 4
                recv_slot = self.mate_indices[ps, m_local]
                rx = self.px[pe, recv_slot]
                ry = self.py[pe, recv_slot]
                p_dist = torch.clamp(torch.hypot(rx - self.px[pe, ps], ry - self.py[pe, ps]), min=20.0)
                speed = 460.0
                self.b_holder[pe, 0] = -1
                self.b_mode[pe, 0] = 1
                self.b_receiver[pe, 0] = recv_slot
                self.b_passer[pe, 0] = ps
                self.bvx[pe, 0] = (rx - self.px[pe, ps]) / p_dist * speed
                self.bvy[pe, 0] = (ry - self.py[pe, ps]) / p_dist * speed
                self.bvz[pe, 0] = torch.where(pa >= 9, torch.full_like(p_dist, 210.0), torch.full_like(p_dist, 65.0))
                self.inbound_team[pe] = -1
                self.inbound_passer[pe] = -1
                self.inbound_timer[pe] = 0.0
                self.stopped_dribble[pe, ps] = False

            # Броски (1..4: standard, floater, hook, dunk)
            is_shot = (c_act >= 1) & (c_act <= 4) & (self.inbound_team[carrier_env] < 0)
            if is_shot.any():
                se = carrier_env[is_shot]
                ss = carrier_slot[is_shot]
                sa = c_act[is_shot]
                shx = hoop_x[is_shot]
                sd = torch.clamp(d_hoop[is_shot], min=15.0)
                shoot_stat = self.stats[se, ss, 1] / 100.0
                # Точность зависит от стата Бросок, дистанции, стиля и плотной опеки
                opp_s = torch.where(
                    (ss < 5).unsqueeze(1),
                    torch.arange(5, 10, device=dev).unsqueeze(0),
                    torch.arange(0, 5, device=dev).unsqueeze(0),
                )
                ox = self.px[se.unsqueeze(1), opp_s]
                oy = self.py[se.unsqueeze(1), opp_s]
                contest = (torch.hypot(ox - self.px[se, ss].unsqueeze(1), oy - self.py[se, ss].unsqueeze(1)) < 64.0).float().sum(dim=-1)
                style_bonus = torch.where(sa == 4, 0.28, torch.where((sa == 2) | (sa == 3), 0.08, 0.0))
                base_prob = 0.34 + 0.46 * shoot_stat + style_bonus - torch.clamp((sd - 110.0) / 760.0, 0.0, 0.42) - 0.12 * contest
                prob = torch.clamp(base_prob, 0.06, 0.95)

                flight_t = torch.clamp(sd / 430.0, 0.28, 1.35)
                self.b_holder[se, 0] = -1
                self.b_mode[se, 0] = 2
                self.b_target_hoop[se, 0] = atk_hoop_idx[is_shot]
                self.b_flight_timer[se, 0] = flight_t
                self.b_make_prob[se, 0] = prob
                self.b_is_three[se, 0] = (sa != 4) & (sd >= THREE_PT_RADIUS)
                self.b_shooter[se, 0] = ss
                self.bvx[se, 0] = (shx - self.px[se, ss]) / flight_t
                self.bvy[se, 0] = (HOOP_Y - self.py[se, ss]) / flight_t
                self.bvz[se, 0] = 215.0
                self.stopped_dribble[se, ss] = False

            # Финты (13..16): pump_fake, step_back, eurostep, crossover
            is_euro = c_act == 15
            if is_euro.any():
                self.eurostepping[carrier_env[is_euro], carrier_slot[is_euro]] = True
                self.skill_cd[carrier_env[is_euro], carrier_slot[is_euro]] = 3.5

            is_stepback = c_act == 14
            if is_stepback.any():
                sbe = carrier_env[is_stepback]
                sbs = carrier_slot[is_stepback]
                self.px[sbe, sbs] = torch.clamp(self.px[sbe, sbs] - sign[sbe, sbs] * 42.0, COURT_LEFT + 22.0, COURT_RIGHT - 22.0)
                self.skill_cd[sbe, sbs] = 4.0

        # Прыжки (19), чистые отборы (20) и грязные приёмы за спиной судьи (21)
        jumpers = (disc == 19) & (self.pz <= 1.0) & (~has_b0)
        if jumpers.any():
            jump_v = 195.0 + self.stats[:, :, 4] * 1.35 + (self.traits[:, :, TRAIT_GRASSHOPPER] > 0.5).float() * 95.0
            self.pvz = torch.where(jumpers, jump_v, self.pvz)
            self.super_jump_cd = torch.where(jumpers, torch.full_like(self.super_jump_cd, 3.5), self.super_jump_cd)

        # Отбор мяча (20)
        stealers = (disc == 20) & (self.steal_cd <= 0.0) & (~has_b0)
        if stealers.any():
            self.steal_cd = torch.where(stealers, torch.full_like(self.steal_cd, 1.3), self.steal_cd)
            st_e, st_s = torch.nonzero(stealers, as_tuple=True)
            bh = self.b_holder[st_e, 0]
            valid_c = (bh >= 0) & ((bh < 5) != (st_s < 5))
            if valid_c.any():
                ve = st_e[valid_c]
                vs = st_s[valid_c]
                vc = bh[valid_c]
                d_pair = torch.hypot(self.px[ve, vs] - self.px[ve, vc], self.py[ve, vs] - self.py[ve, vc])
                in_rng = d_pair <= 50.0
                if in_rng.any():
                    ve, vs, vc = ve[in_rng], vs[in_rng], vc[in_rng]
                    p_steal = torch.clamp(0.18 + (self.stats[ve, vs, 3] - self.stats[ve, vc, 6]) / 220.0, 0.06, 0.52)
                    roll = torch.rand_like(p_steal)
                    won = roll < p_steal
                    foul = (~won) & (roll > 0.80)
                    if won.any():
                        we, ws = ve[won], vs[won]
                        self.b_holder[we, 0] = ws
                        self.poss_team[we] = team[we, ws]
                        self.frontcourt[we] = False
                        self.shot_clock[we] = 24.0
                        self.backcourt_timer[we] = 0.0
                        indiv_rew[we, ws] += 0.38
                    if foul.any():
                        fe, fs = ve[foul], vs[foul]
                        self.fouls[fe, fs] += 1.0
                        ft = team[fe, fs]
                        self.team_fouls.index_put_(
                            (fe, ft), torch.ones((fe.numel(),), dtype=torch.float32, device=dev), accumulate=True
                        )
                        indiv_rew[fe, fs] -= 0.32 + (self.fouls[fe, fs] >= 5.0).float() * 0.45

        # Грязный приём (21) на Ступени 2+ с защитой от Reward Hacking (бонус <= 1 раз за владение при срыве атаки)
        if self.stage >= 2:
            dirty_act = (disc == 21) & (self.dirty_cd <= 0.0) & (~has_b0)
            if dirty_act.any():
                self.dirty_cd = torch.where(dirty_act, torch.full_like(self.dirty_cd, 4.0), self.dirty_cd)
                _, det_prob = self._ref_detect_prob()
                de, ds = torch.nonzero(dirty_act, as_tuple=True)
                bh = self.b_holder[de, 0]
                valid_v = (bh >= 0) & ((bh < 5) != (ds < 5))
                if valid_v.any():
                    de, ds, dv = de[valid_v], ds[valid_v], bh[valid_v]
                    close_v = torch.hypot(self.px[de, ds] - self.px[de, dv], self.py[de, ds] - self.py[de, dv]) <= 44.0
                    if close_v.any():
                        de, ds, dv = de[close_v], ds[close_v], dv[close_v]
                        caught = torch.rand((de.numel(),), device=dev) < det_prob[de, ds]
                        self.ref_suspicion[de, ds] = torch.clamp(self.ref_suspicion[de, ds] + 1.0, 0.0, 5.0)
                        if caught.any():
                            ce, cs = de[caught], ds[caught]
                            self.fouls[ce, cs] += 1.0
                            self.penalty_timer[ce, cs] = 8.0
                            indiv_rew[ce, cs] -= 0.55
                        uncaught = ~caught
                        if uncaught.any():
                            ue, us, uv = de[uncaught], ds[uncaught], dv[uncaught]
                            self.tripped_timer[ue, uv] = 1.2
                            self.b_holder[ue, 0] = -1
                            self.b_mode[ue, 0] = 0
                            can_bonus = ~self.dirty_bonus_used[ue]
                            dev_scale = (self.stats[ue, us, 8] / 100.0) * 0.10
                            indiv_rew[ue, us] += torch.where(can_bonus, dev_scale, torch.zeros_like(dev_scale))
                            self.dirty_bonus_used[ue] = True

        # Подшаги непрерывной 3D-физики (60 Гц * frame_skip)
        ice_mask = self.modifiers[:, 1:2] > 0.5
        low_grav = self.modifiers[:, 2] > 0.5
        grav = torch.where(low_grav, torch.full((N,), 275.0, device=dev), torch.full((N,), 540.0, device=dev))

        for _ in range(self.frame_skip):
            dt = self.dt
            self.steal_cd = torch.clamp(self.steal_cd - dt, min=0.0)
            self.super_jump_cd = torch.clamp(self.super_jump_cd - dt, min=0.0)
            self.skill_cd = torch.clamp(self.skill_cd - dt, min=0.0)
            self.dirty_cd = torch.clamp(self.dirty_cd - dt, min=0.0)
            self.tripped_timer = torch.clamp(self.tripped_timer - dt, min=0.0)
            self.penalty_timer = torch.clamp(self.penalty_timer - dt, min=0.0)
            self.bitten_timer = torch.clamp(self.bitten_timer - dt, min=0.0)

            # Перемещение игроков к (target_x, target_y)
            can_move = (self.penalty_timer <= 0.0) & (self.tripped_timer <= 0.0) & (self.inbound_passer.unsqueeze(1) != agent_ids)
            dx = self.target_x - self.px
            dy = self.target_y - self.py
            dist = torch.hypot(dx, dy)
            max_spd = (108.0 + (self.stats[:, :, 0] / 100.0) * 126.0) * torch.where(self.energy < 30.0, 0.75, 1.0)
            want_vx = torch.where(dist > 6.0, dx / torch.clamp(dist, min=1e-3) * max_spd, torch.zeros_like(dx))
            want_vy = torch.where(dist > 6.0, dy / torch.clamp(dist, min=1e-3) * max_spd, torch.zeros_like(dy))
            accel = torch.where(ice_mask, 0.08, 0.34)
            self.pvx = torch.where(can_move, self.pvx + (want_vx - self.pvx) * accel, torch.zeros_like(self.pvx))
            self.pvy = torch.where(can_move, self.pvy + (want_vy - self.pvy) * accel, torch.zeros_like(self.pvy))

            self.px = torch.clamp(self.px + self.pvx * dt, COURT_LEFT + 14.0, COURT_RIGHT - 14.0)
            self.py = torch.clamp(self.py + self.pvy * dt, COURT_TOP + 14.0, COURT_BOTTOM - 14.0)
            self.heading = torch.where(torch.hypot(self.pvx, self.pvy) > 12.0, torch.atan2(self.pvy, self.pvx), self.heading)

            # Вертикальная физика прыжков
            air = (self.pz > 0.0) | (self.pvz > 0.0)
            self.pz = torch.where(air, self.pz + self.pvz * dt, torch.zeros_like(self.pz))
            self.pvz = torch.where(air, self.pvz - grav.unsqueeze(1) * dt, torch.zeros_like(self.pvz))
            landed = self.pz <= 0.0
            self.pz = torch.where(landed, torch.zeros_like(self.pz), self.pz)
            self.pvz = torch.where(landed, torch.zeros_like(self.pvz), self.pvz)

            # Проверка Take Charge против таранящего дриблёра (контрится Евростепом!)
            has_b0_sub = self.b_holder[:, 0:1] == agent_ids
            ce, cs = torch.nonzero(has_b0_sub & (~self.eurostepping), as_tuple=True)
            if ce.numel() > 0:
                c_spd = torch.hypot(self.pvx[ce, cs], self.pvy[ce, cs])
                opp_slots = torch.where(
                    (cs < 5).unsqueeze(1),
                    torch.arange(5, 10, device=dev).unsqueeze(0),
                    torch.arange(0, 5, device=dev).unsqueeze(0),
                )
                chg_mask = self.taking_charge[ce.unsqueeze(1), opp_slots]
                chg_dist = torch.hypot(self.px[ce.unsqueeze(1), opp_slots] - self.px[ce, cs].unsqueeze(1), self.py[ce.unsqueeze(1), opp_slots] - self.py[ce, cs].unsqueeze(1))
                hit_charge = (c_spd.unsqueeze(1) >= 96.0) & chg_mask & (chg_dist < 28.0)
                if hit_charge.any():
                    he = ce[hit_charge.any(dim=-1)]
                    hs = cs[hit_charge.any(dim=-1)]
                    ht = team[he, hs]
                    viol_team.index_put_(
                        (he, ht), torch.ones((he.numel(),), dtype=torch.float32, device=dev), accumulate=True
                    )
                    indiv_rew[he, hs] -= 0.32
                    self._start_inbound(
                        torch.zeros((N,), dtype=torch.bool, device=dev).index_fill_(0, he, True),
                        1 - self.poss_team,
                    )

            # Физика основного мяча (слот 0)
            held_mask = self.b_holder[:, 0] >= 0
            if held_mask.any():
                he = torch.nonzero(held_mask, as_tuple=False).squeeze(-1)
                hs = self.b_holder[he, 0]
                self.bx[he, 0] = self.px[he, hs]
                self.by[he, 0] = self.py[he, hs]
                self.bz[he, 0] = self.pz[he, hs] + 24.0

            free_mask = ~held_mask
            if free_mask.any():
                self.bx[free_mask, 0] += self.bvx[free_mask, 0] * dt
                self.by[free_mask, 0] += self.bvy[free_mask, 0] * dt
                self.bz[free_mask, 0] = torch.clamp(self.bz[free_mask, 0] + self.bvz[free_mask, 0] * dt, min=0.0)
                self.bvz[free_mask, 0] -= grav[free_mask] * dt
                bounce = free_mask & (self.bz[:, 0] <= 0.0) & (self.bvz[:, 0] < 0.0)
                self.bvz[bounce, 0] = -self.bvz[bounce, 0] * 0.62
                self.bvx[bounce, 0] *= 0.88
                self.bvy[bounce, 0] *= 0.88

            # Завершение полёта броска (mode == 2)
            shot_fly = (self.b_holder[:, 0] < 0) & (self.b_mode[:, 0] == 2)
            if shot_fly.any():
                self.b_flight_timer[shot_fly, 0] -= dt
                arrived = shot_fly & (self.b_flight_timer[:, 0] <= 0.0)
                if arrived.any():
                    ae = torch.nonzero(arrived, as_tuple=False).squeeze(-1)
                    made = torch.rand((ae.numel(),), device=dev) < self.b_make_prob[ae, 0]
                    if made.any():
                        me = ae[made]
                        shooter_s = self.b_shooter[me, 0].clamp(min=0)
                        st_team = team[me, shooter_s]
                        pts = torch.where(self.b_is_three[me, 0], torch.full((me.numel(),), 3.0, device=dev), torch.full((me.numel(),), 2.0, device=dev))
                        self.score[me, st_team] += pts
                        indiv_rew[me, shooter_s] += pts * 0.25
                        passer_s = self.b_passer[me, 0]
                        valid_ast = (passer_s >= 0) & ((passer_s < 5) == (shooter_s < 5)) & (passer_s != shooter_s)
                        if valid_ast.any():
                            indiv_rew[me[valid_ast], passer_s[valid_ast]] += 0.35
                        inb_mask = torch.zeros((N,), dtype=torch.bool, device=dev).index_fill_(0, me, True)
                        next_recv = torch.zeros((N,), dtype=torch.long, device=dev)
                        next_recv[me] = 1 - st_team
                        self._start_inbound(inb_mask, next_recv)

                    missed = ~made
                    if missed.any():
                        misse = ae[missed]
                        self.b_mode[misse, 0] = 0
                        self.b_target_hoop[misse, 0] = -1
                        self.shot_touched_rim[misse] = True
                        self.shot_clock[misse] = torch.clamp(self.shot_clock[misse], min=14.0)
                        self.bz[misse, 0] = RIM_Z
                        self.bvx[misse, 0] = (torch.rand((misse.numel(),), device=dev) - 0.5) * 210.0
                        self.bvy[misse, 0] = (torch.rand((misse.numel(),), device=dev) - 0.5) * 210.0
                        self.bvz[misse, 0] = 120.0

            # Подбор / приём паса свободными игроками
            can_catch = (self.b_holder[:, 0] < 0) & (self.b_mode[:, 0] != 2)
            if can_catch.any():
                ce = torch.nonzero(can_catch, as_tuple=False).squeeze(-1)
                d_b = torch.hypot(self.px[ce] - self.bx[ce, 0:1], self.py[ce] - self.by[ce, 0:1])
                z_ok = (self.bz[ce, 0:1] - self.pz[ce]) <= 46.0
                active_p = (self.penalty_timer[ce] <= 0.0) & (self.tripped_timer[ce] <= 0.0)
                score_d = torch.where(z_ok & active_p, d_b, torch.full_like(d_b, 999.0))
                min_d, best_p = score_d.min(dim=-1)
                caught = min_d <= 32.0
                if caught.any():
                    ge = ce[caught]
                    gp = best_p[caught]
                    new_t = team[ge, gp]
                    was_rim = self.shot_touched_rim[ge]
                    if was_rim.any():
                        indiv_rew[ge[was_rim], gp[was_rim]] += 0.25
                        self.shot_touched_rim[ge] = False
                    changed = self.poss_team[ge] != new_t
                    self.b_holder[ge, 0] = gp
                    self.b_mode[ge, 0] = 0
                    self.b_receiver[ge, 0] = -1
                    self.poss_team[ge] = new_t
                    self.shot_clock[ge] = torch.where(changed, torch.full_like(self.shot_clock[ge], 24.0), self.shot_clock[ge])
                    self.frontcourt[ge] = torch.where(changed, torch.zeros_like(self.frontcourt[ge]), self.frontcourt[ge])
                    self.backcourt_timer[ge] = torch.where(changed, torch.zeros_like(self.backcourt_timer[ge]), self.backcourt_timer[ge])

            # Часы четверти, 24с, 8с, 5с ввода и 3с в краске
            live_clock = self.inbound_team < 0
            self.clock = torch.where(live_clock, self.clock - dt, self.clock)
            self.shot_clock = torch.where(live_clock & (self.poss_team >= 0), self.shot_clock - dt, self.shot_clock)

            # 3 секунды в трёхсекундной зоне атаки (FIBA/NBA)
            atk_hx = torch.where(is_t0, HOOP_RIGHT_X, HOOP_LEFT_X)
            in_paint_now = (
                live_clock.unsqueeze(1)
                & (self.poss_team.unsqueeze(1) == team)
                & (self.b_mode[:, 0:1] != 2)
                & (torch.abs(self.px - atk_hx) < 138.0)
                & (torch.abs(self.py - HOOP_Y) < 90.0)
            )
            self.paint_timer = torch.where(in_paint_now, self.paint_timer + dt, torch.zeros_like(self.paint_timer))
            v3_mask = (self.paint_timer >= 3.0).any(dim=-1)
            if v3_mask.any():
                v3e = torch.nonzero(v3_mask, as_tuple=False).squeeze(-1)
                v3t = self.poss_team[v3e].clamp(min=0)
                viol_team[v3e, v3t] += 1.0
                self.paint_timer[v3e] = 0.0
                self._start_inbound(v3_mask, 1 - self.poss_team)

            # 5 секунд на ввод мяча
            inb_active = self.inbound_team >= 0
            if inb_active.any():
                self.inbound_timer[inb_active] -= dt
                inb_viol = inb_active & (self.inbound_timer <= 0.0)
                if inb_viol.any():
                    ve = torch.nonzero(inb_viol, as_tuple=False).squeeze(-1)
                    vt = self.inbound_team[ve]
                    viol_team[ve, vt] += 1.0
                    self._start_inbound(inb_viol, 1 - self.inbound_team)

            # 8 секунд тыловой зоны и установление передовой зоны
            bh0 = self.b_holder[:, 0]
            live_held = (self.inbound_team < 0) & (bh0 >= 0)
            if live_held.any():
                le = torch.nonzero(live_held, as_tuple=False).squeeze(-1)
                ls = bh0[le]
                lt = team[le, ls]
                in_fc = torch.where(lt == 0, self.px[le, ls] > MID_X + 18.0, self.px[le, ls] < MID_X - 18.0)
                self.frontcourt[le] = self.frontcourt[le] | in_fc
                self.backcourt_timer[le] = torch.where(self.frontcourt[le], torch.zeros_like(self.backcourt_timer[le]), self.backcourt_timer[le] + dt)
                v8 = (~self.frontcourt[le]) & (self.backcourt_timer[le] >= 8.0)
                if v8.any():
                    v8e = le[v8]
                    viol_team[v8e, lt[v8]] += 1.0
                    m8 = torch.zeros((N,), dtype=torch.bool, device=dev).index_fill_(0, v8e, True)
                    self._start_inbound(m8, 1 - self.poss_team)

            # 24 секунды на атаку
            v24 = (self.inbound_team < 0) & (self.poss_team >= 0) & (self.shot_clock <= 0.0) & (self.b_mode[:, 0] != 2)
            if v24.any():
                v24e = torch.nonzero(v24, as_tuple=False).squeeze(-1)
                viol_team[v24e, self.poss_team[v24e]] += 1.0
                self._start_inbound(v24, 1 - self.poss_team)

            # Судья двигается вдоль бровки и сканирует площадку
            self.ref_x = torch.clamp(self.ref_x + (self.bx[:, 0] - self.ref_x) * 3.0 * dt, COURT_LEFT + 90.0, COURT_RIGHT - 90.0)
            target_ang = torch.atan2(self.by[:, 0] - self.ref_y, self.bx[:, 0] - self.ref_x)
            self.ref_gaze = target_ang + torch.sin(self.clock * 1.9) * 0.65

        # Завершение четверти / матча (4 четверти)
        q_end = self.clock <= 0.0
        done_envs = q_end & (self.quarter >= 4) & (self.score[:, 0] != self.score[:, 1])
        next_q = q_end & (~done_envs)
        if next_q.any():
            self.quarter[next_q] = torch.clamp(self.quarter[next_q] + 1, max=5)
            self.clock[next_q] = torch.where(self.quarter[next_q] <= 4, torch.full_like(self.clock[next_q], self.quarter_duration), torch.full_like(self.clock[next_q], 20.0))
            self.team_fouls[next_q] = 0.0

        # Вычисление командной награды + PBRS
        d_score = self.score - prev_score  # [N, 2]
        next_phi = torch.where(done_envs.unsqueeze(-1), torch.zeros_like(self.prev_phi), self._compute_potential())
        pbrs = 0.12 * (0.995 * next_phi - self.prev_phi)
        self.prev_phi = next_phi

        team_rew0 = (d_score[:, 0] - d_score[:, 1]) - 0.18 * viol_team[:, 0] + pbrs[:, 0]
        team_rew1 = (d_score[:, 1] - d_score[:, 0]) - 0.18 * viol_team[:, 1] + pbrs[:, 1]
        team_rew = torch.stack([team_rew0, team_rew1], dim=-1)  # [N, 2]
        rewards = torch.gather(team_rew, 1, team) + indiv_rew  # [N, 10]

        metrics = {
            "done_count": float(done_envs.sum().item()),
            "avg_total_points": float(self.score[done_envs].sum(dim=-1).mean().item()) if done_envs.any() else 0.0,
            "violations_step": float(viol_team.sum().item()),
        }

        if done_envs.any():
            self._reset_indices(done_envs)

        dones = done_envs.unsqueeze(1).expand(N, 10).reshape(-1).to(torch.uint8)
        obs_out, masks_out = self.get_obs_and_masks()
        return obs_out, masks_out, rewards.reshape(-1), dones, metrics
