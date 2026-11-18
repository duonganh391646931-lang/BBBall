import { getTeamBenchSeatPos } from './physics';
import {
  HairStyle,
  MatchModifierDef,
  PlayerEntity,
  PlayerStats,
  PositionRole,
  TraitDefinition,
  TraitRarity,
} from './types';

export function clampStat(v: number, min = 8, max = 100): number {
  return Math.round(Math.max(min, Math.min(max, v)));
}

export const MATCH_MODIFIERS: MatchModifierDef[] = [
  {
    id: 'two_balls',
    name: 'Два мяча',
    shortLabel: '2 Мяча',
    description: 'На площадке одновременно разыгрываются сразу два баскетбольных мяча!',
  },
  {
    id: 'ice_court',
    name: 'Гололёд',
    shortLabel: 'Гололёд',
    description: 'Паркет залит льдом: низкое трение, безумная инерция и заносы на виражах.',
  },
  {
    id: 'low_gravity',
    name: 'Лунная гравитация',
    shortLabel: 'Луна',
    description: 'Прыжки и навесы длятся почти вдвое дольше в условиях слабой гравитации.',
  },
  {
    id: 'paint_trampolines',
    name: 'Батуты в краске',
    shortLabel: 'Батуты',
    description: 'В трёхсекундных зонах лежат пружинные маты, подбрасывающие игроков к потолку.',
  },
  {
    id: 'explosive_ball',
    name: 'Взрывной мяч',
    shortLabel: 'Бабах',
    description: 'Попадания в кольцо вызывают ударную волну, разбрасывающую чувачков вокруг.',
  },
];

export const TRAIT_CATALOG: TraitDefinition[] = [
  // ==================== УНИКАЛЬНЫЕ ЧЕРТЫ [Уникальная] (низкий шанс ~8% пула) ====================
  {
    id: 'messi',
    name: 'Месси',
    rarity: 'unique',
    rarityLabel: 'Уникальная',
    weight: 2.1,
    shortTag: 'Навес в ближайшее кольцо (15% / 7%)',
    description:
      'При получении мяча в руки с шансом 15% на чужой половине и 7% на своей мгновенно закидывает идеальный навес через полполя в БЛИЖАЙШЕЕ кольцо — включая своё собственное!',
    modifyStats: (s) => ({
      ...s,
      shooting: clampStat(s.shooting * 1.2),
    }),
  },
  {
    id: 'grasshopper',
    name: 'Кузнечик',
    rarity: 'unique',
    rarityLabel: 'Уникальная',
    weight: 2.2,
    shortTag: 'Супер-прыжок (КД 3.5с) и перехват дуг',
    description:
      'Прыгает экстремально высоко (перезарядка пружин 3.5с) и перехватывает даже летящий под потолком мяч (в том числе от «Месси»), если стоит на пути его полёта.',
    modifyStats: (s) => ({
      ...s,
      jump: 100,
      steal: clampStat(s.steal * 1.2),
    }),
  },
  {
    id: 'egoisto',
    name: 'Эгоисто',
    rarity: 'unique',
    rarityLabel: 'Уникальная',
    weight: 1.9,
    shortTag: 'Пас = 0, бросок с отклонением',
    description:
      'Командная игра = 0: никогда никому не пасует. Владеет броском с отклонением через руки двух защитников (+25% дриблинг, +20% бросок).',
    modifyStats: (s) => ({
      ...s,
      teamwork: 0,
      dribble: clampStat(s.dribble * 1.25),
      shooting: clampStat(s.shooting * 1.2),
    }),
  },
  {
    id: 'magnet',
    name: 'Магнит в шортах',
    rarity: 'unique',
    rarityLabel: 'Уникальная',
    weight: 1.7,
    shortTag: 'Притягивает свободные мячи',
    description:
      'Бесхозные мячи в радиусе 150px физически искривляют курс и катятся прямо ему в руки (+35% перехват).',
    modifyStats: (s) => ({
      ...s,
      steal: clampStat(s.steal * 1.35),
    }),
  },
  {
    id: 'schizo_tactician',
    name: 'Биполярный гений',
    rarity: 'unique',
    rarityLabel: 'Уникальная',
    weight: 1.6,
    shortTag: 'То киберспорт, то хлебушек',
    description:
      'Каждые 10 секунд переключается между режимом «Леброн» (+35% ко всему) и режимом «Хлебушек» (уходит в угол считать ворон).',
  },
  {
    id: 'gorilla_dunk',
    name: 'Данк-Горилла',
    rarity: 'unique',
    rarityLabel: 'Уникальная',
    weight: 1.8,
    shortTag: 'Взрывной слэм-данк',
    description:
      'Вблизи чужого кольца взмывает с мощным слэм-данком, разбрасывая защитников ударной волной (+25% прыжок).',
    modifyStats: (s) => ({
      ...s,
      jump: clampStat(s.jump * 1.25),
      dribble: clampStat(s.dribble * 1.15),
    }),
  },
  {
    id: 'last_hope',
    name: 'Последняя надежда',
    rarity: 'unique',
    rarityLabel: 'Уникальная',
    weight: 2.0,
    shortTag: 'Джокер со скамейки: мега-бафф при выходе',
    description:
      'Если балбес начинает матч запасным и выходит со скамейки на игровое поле, то получает огромные баффы (+40% и минимум +22 ко всем игровым навыкам вплоть до 115, +18% к точности бросков и второе дыхание)!',
  },

  // ==================== РЕДКИЕ ЧЕРТЫ [Редкая] (средний шанс ~22% пула) ====================
  {
    id: 'crook',
    name: 'Жулик',
    rarity: 'rare',
    rarityLabel: 'Редкая',
    weight: 5.2,
    shortTag: 'Девиантность +45, мастер подножек',
    description:
      'Обожает грязную игру: резко повышает Девиантность (+45), при каждом удобном случае ставит подножки и пинает мяч ногой.',
    modifyStats: (s) => ({
      ...s,
      deviance: clampStat(s.deviance + 45, 0, 100),
      steal: clampStat(s.steal * 1.15),
    }),
  },
  {
    id: 'stealthy',
    name: 'Незаметный',
    rarity: 'rare',
    rarityLabel: 'Редкая',
    weight: 5.0,
    shortTag: '-45% шанс быть пойманным судьёй',
    description:
      'Невинное лицо: судья замечает его подножки и пинки ногой гораздо реже (-45% к базовой зоркости арбитра).',
    modifyStats: (s) => ({
      ...s,
      iq: clampStat(s.iq * 1.12),
    }),
  },
  {
    id: 'parking_sniper',
    name: 'Снайпер с парковки',
    rarity: 'rare',
    rarityLabel: 'Редкая',
    weight: 4.8,
    shortTag: 'Бьёт только из-за дуги',
    description:
      'Поливает трёхочковые издалека (+30% точность), но панически боится бросать из-под кольца и пятится за дугу.',
    modifyStats: (s) => ({
      ...s,
      shooting: clampStat(s.shooting * 1.3),
    }),
  },
  {
    id: 'pickpocket',
    name: 'Карманник',
    rarity: 'rare',
    rarityLabel: 'Редкая',
    weight: 4.6,
    shortTag: 'Чистый отбор мяча +45%',
    description:
      'Виртуоз легального перехвата (+45% перехват мяча без нарушения правил).',
    modifyStats: (s) => ({
      ...s,
      steal: clampStat(s.steal * 1.45),
    }),
  },
  {
    id: 'turbo_hamster',
    name: 'Турбо-хомяк',
    rarity: 'rare',
    rarityLabel: 'Редкая',
    weight: 4.5,
    shortTag: '+40% скорость, быстро выдыхается',
    description:
      'Носится по паркету как метеор (+40% скорость), но сжигает запас сил почти вдвое быстрее.',
    modifyStats: (s) => ({
      ...s,
      speed: clampStat(s.speed * 1.4),
      stamina: clampStat(s.stamina * 0.72),
    }),
  },
  {
    id: 'pass_telepath',
    name: 'Телепат паса',
    rarity: 'rare',
    rarityLabel: 'Редкая',
    weight: 4.4,
    shortTag: 'Идеальные передачи на ход',
    description:
      'Находит открытых партнёров через всю площадку (+45% командная игра, +25% ИИ).',
    modifyStats: (s) => ({
      ...s,
      teamwork: clampStat(s.teamwork * 1.45, 0, 100),
      iq: clampStat(s.iq * 1.25),
    }),
  },
  {
    id: 'barrier_blocker',
    name: 'Шлагбаум',
    rarity: 'rare',
    rarityLabel: 'Редкая',
    weight: 4.3,
    shortTag: 'Охотник за блок-шотами в краске',
    description:
      'Стережёт трёхсекундную зону и выпрыгивает накрывать чужие броски (+30% прыжок и перехват).',
    modifyStats: (s) => ({
      ...s,
      jump: clampStat(s.jump * 1.3),
      steal: clampStat(s.steal * 1.22),
    }),
  },
  {
    id: 'clutch_psycho',
    name: 'Адреналиновый псих',
    rarity: 'rare',
    rarityLabel: 'Редкая',
    weight: 4.0,
    shortTag: 'Усиливается, когда команда горит',
    description:
      'Когда его команда уступает в счёте, получает мощный буст к скорости и точности броска.',
    modifyStats: (s) => ({
      ...s,
      stamina: clampStat(s.stamina * 1.2),
    }),
  },
  {
    id: 'flopper',
    name: 'Актёр Оскара',
    rarity: 'rare',
    rarityLabel: 'Редкая',
    weight: 4.1,
    shortTag: 'Мастер симуляции фола в атаке',
    description:
      'При плотном контакте с бегущим соперником картинно падает на паркет и выпрашивает фол в нападении! Но если судья раскусит спектакль — накажет техническим штрафным.',
    modifyStats: (s) => ({
      ...s,
      iq: clampStat(s.iq * 1.15),
      deviance: clampStat(s.deviance + 18, 0, 100),
    }),
  },
  {
    id: 'sixth_sense_alley',
    name: 'Король Алли-упов',
    rarity: 'rare',
    rarityLabel: 'Редкая',
    weight: 4.2,
    shortTag: 'Парашюты и вколачивания в небе',
    description:
      'Обожает воздушные комбинации: раздаёт парашюты над кольцом и вколачивает Алли-уп одним касанием с огромной дистанции (+25% прыжок, +20% командная игра).',
    modifyStats: (s) => ({
      ...s,
      jump: clampStat(s.jump * 1.25),
      teamwork: clampStat(s.teamwork * 1.2, 0, 100),
    }),
  },

  // ==================== ОБЫЧНЫЕ ЧЕРТЫ [Обычная] (встречаются намного чаще, ~70% пула!) ====================
  {
    id: 'crooked_lame',
    name: 'Кривоногий и хромой',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 16,
    shortTag: 'Все характеристики -20%',
    description:
      'Все характеристики снижены на 20%. Ковыляет как умеет, зато с душой!',
    modifyStats: (s) => ({
      speed: clampStat(s.speed * 0.8),
      shooting: clampStat(s.shooting * 0.8),
      stamina: clampStat(s.stamina * 0.8),
      steal: clampStat(s.steal * 0.8),
      jump: clampStat(s.jump * 0.8),
      teamwork: clampStat(s.teamwork * 0.8, 0, 100),
      dribble: clampStat(s.dribble * 0.8),
      iq: clampStat(s.iq * 0.8),
      deviance: clampStat(s.deviance * 0.8, 0, 100),
    }),
  },
  {
    id: 'grabby_hands',
    name: 'Руки-загребуки',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 15,
    shortTag: 'Всегда забирает стартовый мяч',
    description:
      'Длинные загребущие лапы: гарантированно выигрывает стартовое вбрасывание мяча в начале каждой четверти и овертайма!',
    modifyStats: (s) => ({
      ...s,
      steal: clampStat(s.steal * 1.1),
    }),
  },
  {
    id: 'ball_hog',
    name: 'Жадина',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 14,
    shortTag: 'Командная игра -40%',
    description:
      'Неохотно расстаётся с мячом, предпочитая возиться в одиночку (-40% командная игра, +10% дриблинг).',
    modifyStats: (s) => ({
      ...s,
      teamwork: clampStat(s.teamwork * 0.6, 0, 100),
      dribble: clampStat(s.dribble * 1.1),
    }),
  },
  {
    id: 'heavy_breather',
    name: 'Одышечник',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 14,
    shortTag: 'Выносливость -35%, бросок +15%',
    description:
      'Быстро выдыхается на бегу (-35% выносливость), но неплохо кидает с места (+15% точность).',
    modifyStats: (s) => ({
      ...s,
      stamina: clampStat(s.stamina * 0.65),
      shooting: clampStat(s.shooting * 1.15),
    }),
  },
  {
    id: 'butterfingers',
    name: 'Руки-масло',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 14,
    shortTag: 'Роняет мяч под прессингом',
    description:
      'Скользкие ладошки: дриблинг -30%, перехват -20%, в суматохе может выронить мяч на пол.',
    modifyStats: (s) => ({
      ...s,
      dribble: clampStat(s.dribble * 0.7),
      steal: clampStat(s.steal * 0.8),
    }),
  },
  {
    id: 'panicker',
    name: 'Паникёр',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 14,
    shortTag: 'В панике швыряет мяч при опеке',
    description:
      'Если защитник подходит вплотную, пугается и мгновенно швыряет мяч наугад!',
    modifyStats: (s) => ({
      ...s,
      speed: clampStat(s.speed * 1.1),
      iq: clampStat(s.iq * 0.85),
    }),
  },
  {
    id: 'sheep_herd',
    name: 'Стадо баранов',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 13,
    shortTag: 'Носится толпой за мячом',
    description:
      'Забывает про позиционную схему и всей душой бежит туда, где сейчас мячик.',
    modifyStats: (s) => ({
      ...s,
      iq: clampStat(s.iq * 0.65),
      stamina: clampStat(s.stamina * 1.15),
    }),
  },
  {
    id: 'sleepy_fly',
    name: 'Сонная муха',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 13,
    shortTag: 'Медленно думает, много сил',
    description:
      'Долго принимает решения и не спешит (-18% скорость), зато почти не устаёт (+35% выносливость).',
    modifyStats: (s) => ({
      ...s,
      speed: clampStat(s.speed * 0.82),
      stamina: clampStat(s.stamina * 1.35),
      iq: clampStat(s.iq * 0.8),
    }),
  },
  {
    id: 'showboat',
    name: 'Понторез',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 13,
    shortTag: 'Долгий замах ради понта',
    description:
      'Перед броском крутит мяч на пальце (+15% точность, но даёт защитникам окно для блока).',
    modifyStats: (s) => ({
      ...s,
      shooting: clampStat(s.shooting * 1.15),
      dribble: clampStat(s.dribble * 1.1),
    }),
  },
  {
    id: 'lanky_beanpole',
    name: 'Дылда-каланча',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 13,
    shortTag: 'Крупный размах рук, медленный',
    description:
      'Широкие плечи и длинные руки (+15% перехват, +10% прыжок), но медленный разгон (-15% скорость).',
    modifyStats: (s) => ({
      ...s,
      steal: clampStat(s.steal * 1.15),
      jump: clampStat(s.jump * 1.1),
      speed: clampStat(s.speed * 0.85),
    }),
  },
  {
    id: 'short_spark',
    name: 'Коротышка-живчик',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 13,
    shortTag: 'Шустрый и юркий, низкий прыжок',
    description:
      'Шныряет между игроками (+22% скорость и дриблинг), но прыгает невысоко (-22% прыжок).',
    modifyStats: (s) => ({
      ...s,
      speed: clampStat(s.speed * 1.22),
      dribble: clampStat(s.dribble * 1.22),
      jump: clampStat(s.jump * 0.78),
    }),
  },
  {
    id: 'crab_scuttler',
    name: 'Суетливый краб',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 12,
    shortTag: 'Виляет на бегу, мажет броски',
    description:
      'Бегает зигзагами и сбивает прицел даже себе (+15% скорость, -18% точность броска).',
    modifyStats: (s) => ({
      ...s,
      speed: clampStat(s.speed * 1.15),
      shooting: clampStat(s.shooting * 0.82),
    }),
  },
  {
    id: 'kind_soul',
    name: 'Слишком добрый',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 12,
    shortTag: 'Предпочитает пас броску',
    description:
      'Всегда ищет открытого друга передачей (+35% командная игра, Девиантность -50%).',
    modifyStats: (s) => ({
      ...s,
      teamwork: clampStat(s.teamwork * 1.35, 0, 100),
      deviance: clampStat(s.deviance * 0.5, 0, 100),
    }),
  },
  {
    id: 'brick_magnet_board',
    name: 'Король щита',
    rarity: 'common',
    rarityLabel: 'Обычная',
    weight: 12.5,
    shortTag: 'Броски от щита и мощный подбор',
    description:
      'Любит классику старой школы: кладёт мячи от щита под углом 45° и уверенно забирает отскоки под кольцом (+20% прыжок, +15% перехват).',
    modifyStats: (s) => ({
      ...s,
      jump: clampStat(s.jump * 1.2),
      steal: clampStat(s.steal * 1.15),
    }),
  },
];

export const TRAIT_BY_ID: Record<string, TraitDefinition> = Object.fromEntries(
  TRAIT_CATALOG.map((t) => [t.id, t])
);

const FIRST_NAMES = [
  'Толик',
  'Жорик',
  'Санёк',
  'Димон',
  'Виталик',
  'Боря',
  'Гена',
  'Славик',
  'Паша',
  'Лёха',
  'Кузя',
  'Федя',
  'Русик',
  'Вадик',
  'Макар',
  'Тимоха',
  'Арсен',
  'Глеб',
  'Сёма',
  'Антоха',
  'Колян',
  'Миха',
  'Стасик',
  'Юрец',
];

const FUNNY_SURNAMES = [
  'Кирпичов',
  'Сквозняк',
  'Пельмень',
  'Пружинкин',
  'Косолапов',
  'Шальной',
  'Форточкин',
  'Батон',
  'Тапочкин',
  'Штангин',
  'Крендель',
  'Бубликов',
  'Суетолог',
  'Кедов',
  'Рикошет',
  'Валенок',
  'Пончик',
  'Свистунов',
  'Гвоздёв',
  'Чебурек',
  'Подкат',
  'Локотков',
  'Шнурков',
  'Тумбочкин',
];

export const CLUB_PRESETS = [
  { name: 'Оранжевые Метеоры', primary: '#ea580c', secondary: '#fed7aa' },
  { name: 'Лазурные Караси', primary: '#0284c7', secondary: '#bae6fd' },
  { name: 'Изумрудные Кабаны', primary: '#059669', secondary: '#a7f3d0' },
  { name: 'Багровые Кирпичи', primary: '#dc2626', secondary: '#fecaca' },
  { name: 'Золотые Чебуреки', primary: '#d97706', secondary: '#fde68a' },
  { name: 'Фиолетовые Утюги', primary: '#7c3aed', secondary: '#ddd6fe' },
  { name: 'Бирюзовые Шланги', primary: '#0d9488', secondary: '#99f6e4' },
  { name: 'Стальные Тапки', primary: '#475569', secondary: '#e2e8f0' },
];

const ROLES_ORDER: PositionRole[] = ['PG', 'SG', 'SF', 'PF', 'C'];
const SKIN_TONES = ['#f5cba7', '#e59866', '#f0b27a', '#ca8a5b', '#8d5524', '#fad7a0'];
const HAIR_COLORS = ['#1c1917', '#451a03', '#92400e', '#d97706', '#dc2626', '#e5e7eb', '#38bdf8'];
const HAIR_STYLES: HairStyle[] = ['afro', 'mohawk', 'headband', 'spiky', 'bald', 'curly'];

export function pickWeightedTrait(
  rng: () => number,
  excludeIds: Set<string>
): TraitDefinition {
  const candidates = TRAIT_CATALOG.filter((t) => !excludeIds.has(t.id));
  const totalWeight = candidates.reduce((acc, t) => acc + t.weight, 0);
  let roll = rng() * totalWeight;
  for (const trait of candidates) {
    roll -= trait.weight;
    if (roll <= 0) {
      return trait;
    }
  }
  return candidates[candidates.length - 1];
}

export function isLastHopeBuffActive(
  p: Pick<PlayerEntity, 'traitIds' | 'startedOnBench' | 'subbedInOnce' | 'onCourt'>
): boolean {
  return Boolean(
    p.traitIds.includes('last_hope') && p.startedOnBench && p.subbedInOnce && p.onCourt
  );
}

export function computeEffectiveStats(
  base: PlayerStats,
  traitIds: string[],
  lastHopeActive = false
): PlayerStats {
  let current: PlayerStats = { ...base };
  for (const id of traitIds) {
    const def = TRAIT_BY_ID[id];
    if (def?.modifyStats) {
      current = def.modifyStats(current);
    }
  }
  if (lastHopeActive && traitIds.includes('last_hope')) {
    const boost = (val: number) =>
      clampStat(Math.max(val * 1.4, val + 22), 8, 115);
    current = {
      ...current,
      speed: boost(current.speed),
      shooting: boost(current.shooting),
      stamina: clampStat(Math.max(current.stamina * 1.45, current.stamina + 24), 8, 115),
      steal: boost(current.steal),
      jump: boost(current.jump),
      teamwork: traitIds.includes('egoisto')
        ? 0
        : clampStat(Math.max(current.teamwork * 1.35, current.teamwork + 20), 0, 115),
      dribble: boost(current.dribble),
      iq: boost(current.iq),
    };
  }
  return current;
}

export function computeBodyScale(traitIds: string[]): number {
  let scale = 1.0;
  if (traitIds.includes('lanky_beanpole')) scale *= 1.18;
  if (traitIds.includes('short_spark')) scale *= 0.84;
  if (traitIds.includes('gorilla_dunk')) scale *= 1.1;
  return scale;
}

function randRange(rng: () => number, min: number, max: number): number {
  return Math.round(min + rng() * (max - min));
}

export function getHomeFormationCoords(team: 0 | 1, slotIndex: number): { x: number; y: number } {
  // 5v5 half-court tactical positions around mid-court before tipoff
  const offsets0 = [
    { x: 430, y: 310 }, // PG
    { x: 380, y: 175 }, // SG
    { x: 380, y: 445 }, // SF
    { x: 300, y: 235 }, // PF
    { x: 515, y: 310 }, // C (at center jump-ball circle)
  ];
  const offsets1 = [
    { x: 750, y: 310 }, // PG
    { x: 800, y: 175 }, // SG
    { x: 800, y: 445 }, // SF
    { x: 880, y: 235 }, // PF
    { x: 665, y: 310 }, // C (at center jump-ball circle)
  ];
  const arr = team === 0 ? offsets0 : offsets1;
  return arr[slotIndex % 5];
}

export function generateRandomPlayer(
  team: 0 | 1,
  slotIndex: number,
  usedNames: Set<string>,
  rng: () => number = Math.random,
  isBench = false
): PlayerEntity {
  let name = '';
  for (let tries = 0; tries < 16; tries++) {
    const fn = FIRST_NAMES[Math.floor(rng() * FIRST_NAMES.length)];
    const ln = FUNNY_SURNAMES[Math.floor(rng() * FUNNY_SURNAMES.length)];
    name = `${fn} ${ln}`;
    if (!usedNames.has(name)) break;
  }
  usedNames.add(name);

  const benchRoles: PositionRole[] = ['PG', 'SF', 'C'];
  const role = isBench ? benchRoles[slotIndex % benchRoles.length] : ROLES_ORDER[slotIndex % 5];

  const baseStats: PlayerStats = {
    speed: randRange(rng, 38, 86),
    shooting: randRange(rng, 35, 86),
    stamina: randRange(rng, 42, 88),
    steal: randRange(rng, 32, 84),
    jump: randRange(rng, 36, 86),
    teamwork: randRange(rng, 30, 88),
    dribble: randRange(rng, 36, 86),
    iq: randRange(rng, 32, 86),
    deviance: randRange(rng, 10, 72),
  };

  const traitCount = rng() < 0.58 ? 2 : 3;
  const chosenIds = new Set<string>();
  for (let i = 0; i < traitCount; i++) {
    const excluded = new Set(chosenIds);
    if (chosenIds.has('lanky_beanpole')) excluded.add('short_spark');
    if (chosenIds.has('short_spark')) excluded.add('lanky_beanpole');
    chosenIds.add(pickWeightedTrait(rng, excluded).id);
  }

  const traitIds = Array.from(chosenIds);
  const effectiveStats = computeEffectiveStats(baseStats, traitIds, false);
  const bodyScale = computeBodyScale(traitIds);
  const home = isBench
    ? getTeamBenchSeatPos(team, slotIndex, 3)
    : getHomeFormationCoords(team, slotIndex);

  return {
    id: `p_${team}_${isBench ? 'b' : 's'}${slotIndex}_${Math.floor(rng() * 1000000)}`,
    name,
    jerseyNumber: Math.floor(rng() * 99) + 1,
    team,
    slotIndex,
    role,
    skinTone: SKIN_TONES[Math.floor(rng() * SKIN_TONES.length)],
    hairColor: HAIR_COLORS[Math.floor(rng() * HAIR_COLORS.length)],
    hairStyle: HAIR_STYLES[Math.floor(rng() * HAIR_STYLES.length)],
    bodyScale,
    baseStats,
    effectiveStats,
    traitIds,
    x: home.x,
    y: home.y,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    heading: isBench ? Math.PI / 2 : team === 0 ? 0 : Math.PI,
    energy: 100,
    aiAction: isBench ? 'team_bench' : 'tipoff',
    aiThought: isBench ? 'Готов ворваться со скамейки!' : 'Ждёт спорного мяча',
    targetX: home.x,
    targetY: home.y,
    decisionTimer: 0,
    actionCooldown: 0,
    stealCooldown: 0,
    pickupLockout: 0,
    windupTimer: 0,
    windupTargetHoop: null,
    windupBallId: null,
    windupShotStyle: 'standard',
    schizoTimer: 5 + rng() * 5,
    schizoBreadMode: false,
    cutTimer: 0,
    screenTargetId: null,
    superJumpCd: 0,
    dirtyTrickCd: 3.5 + rng() * 3,
    refSuspicion: 0,
    trippedTimer: 0,
    penaltyTimer: 0,
    fouls: 0,
    isReserveSub: false,
    onCourt: !isBench,
    startedOnBench: isBench,
    subbedInOnce: false,
    lastHopeAnnounced: false,
    benchSlot: isBench ? slotIndex : 0,
    disqualified: false,
    paintTimer: 0,
    closelyGuardedTimer: 0,
    airborneWithBall: false,
    hasStoppedDribble: false,
    dribbleLiveTimer: 0,
    dodgeSide: slotIndex % 2 === 0 ? 1 : -1,
    dodgeCommitTimer: 0,
    driveLockTimer: 0,
    markTargetId: null,
    stepBackCd: 0,
    eurostepCd: 0,
    chargePlantTimer: 0,
    handoffHoldTimer: 0,
    handoffCd: 0,
    saveOobCd: 0,
    hotStreak: 0,
    flopCd: 0,
    stridePhase: rng() * Math.PI * 2,
    points: 0,
    rebounds: 0,
    fgMade: 0,
    fgAttempted: 0,
    threeMade: 0,
    threeAttempted: 0,
    freeThrowsMade: 0,
    ownGoals: 0,
    assists: 0,
    steals: 0,
    blocks: 0,
    dunks: 0,
    dirtyPlays: 0,
  };
}

export function createClumsyReservePlayer(replaced: PlayerEntity): PlayerEntity {
  const baseStats: PlayerStats = {
    speed: 42,
    shooting: 36,
    stamina: 55,
    steal: 34,
    jump: 38,
    teamwork: 52,
    dribble: 38,
    iq: 40,
    deviance: 12,
  };
  const traitIds = ['crooked_lame'];
  return {
    ...replaced,
    id: `${replaced.id}_sub`,
    name: `Стажёр (${replaced.role})`,
    jerseyNumber: 99,
    baseStats,
    effectiveStats: computeEffectiveStats(baseStats, traitIds, false),
    traitIds,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    energy: 100,
    fouls: 0,
    penaltyTimer: 0,
    trippedTimer: 0,
    refSuspicion: 0,
    cutTimer: 0,
    screenTargetId: null,
    paintTimer: 0,
    closelyGuardedTimer: 0,
    airborneWithBall: false,
    hasStoppedDribble: false,
    dribbleLiveTimer: 0,
    isReserveSub: true,
    onCourt: true,
    startedOnBench: true,
    subbedInOnce: true,
    disqualified: false,
    dodgeSide: replaced.slotIndex % 2 === 0 ? 1 : -1,
    dodgeCommitTimer: 0,
    driveLockTimer: 0,
    markTargetId: null,
    stepBackCd: 0,
    eurostepCd: 0,
    chargePlantTimer: 0,
    handoffHoldTimer: 0,
    handoffCd: 0,
    saveOobCd: 0,
    windupShotStyle: 'standard',
    hotStreak: 0,
    flopCd: 0,
    points: 0,
    rebounds: 0,
    fgMade: 0,
    fgAttempted: 0,
    threeMade: 0,
    threeAttempted: 0,
    freeThrowsMade: 0,
    ownGoals: 0,
    assists: 0,
    steals: 0,
    blocks: 0,
    dunks: 0,
    dirtyPlays: 0,
    aiThought: 'Вышел со скамейки!',
  };
}

export function getPlayerOverall(p: PlayerEntity): number {
  const s = p.effectiveStats;
  return Math.round(
    (s.speed + s.shooting + s.stamina + s.steal + s.jump + s.teamwork + s.dribble + s.iq) / 8
  );
}

export function getRarityColor(rarity: TraitRarity): string {
  if (rarity === 'unique') return '#fbbf24';
  if (rarity === 'rare') return '#38bdf8';
  return '#a8a29e';
}
