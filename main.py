"""Kaggriculture agent.

Architecture
------------
* Strategy layer   - decides land purchases, the target use of every tile
                     (animal / crop), what to buy and what to sell.
* Task layer       - from the live observation derives, per tile, the ordered
                     list of actions still due today (dig, build, plant, water,
                     feed, care, collect fertilizer, harvest, ...).
* Execution layer  - greedy claim-based dispatcher that moves the farmer and the
                     hired hands to tiles with due work, handles shed pickups /
                     drops, hires the right number of hands, and emits the market
                     order list (<= 10 orders per turn).

Only `agent(obs, config)` is required by Kaggle.
"""
import math

# ----------------------------------------------------------------------------
# Game constants (mirrors kaggriculture.py)
# ----------------------------------------------------------------------------
CROPS = {
    "WHEAT":      {"seed": 10, "fy": 2, "my": 4, "iv": 0, "mx": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "fy": 2, "my": 3, "iv": 0, "mx": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "fy": 8, "my": 8, "iv": 1, "mx": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "fy": 10, "my": 10, "iv": 2, "mx": 4, "ongoing": True},
    "MELON":      {"seed": 80, "fy": 10, "my": 12, "iv": 0, "mx": 6, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "st": "COOP", "fy": 4, "iv": 1, "mh": 4, "prod": "EGG"},
    "COW":   {"cost": 400, "st": "PASTURE", "fy": 8, "iv": 2, "mh": 6, "prod": "MILK"},
    "SHEEP": {"cost": 500, "st": "PASTURE", "fy": 6, "iv": 3, "mh": 6, "prod": "WOOL"},
}
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
MP = {
    "WHEAT":      (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT":     (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO":     (60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON":      (250, 300, "log", 0.20, "sq", 3.60),
    "EGG":        (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK":       (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL":       (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}
I0 = 10000
SHOPS = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def _shape(f, x, T):
    x = max(0.0, x)
    if f == "linear":
        return x
    if f == "sq":
        return x * x
    if f == "sqrt":
        return math.sqrt(x)
    if f == "log":
        return math.log(1.0 + x)
    if f == "log10":
        return math.log10(1.0 + x)
    if f == "hinge":
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def price_at(item, inv):
    base, T, bf, bt, af, at = MP[item]
    if inv < I0:
        p = base + bt * base / _shape(bf, T, T) * _shape(bf, I0 - inv, T)
    else:
        p = base - at * base / _shape(af, T, T) * _shape(af, inv - I0, T)
    return max(1, int(round(p)))


def sell_revenue(item, inv, n):
    """Revenue of selling n units alone starting from market inventory inv."""
    tot = 0
    for _ in range(n):
        p = price_at(item, inv)
        tot += p
        if p > 1:
            inv += 1
    return tot


def fib_cost(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def quad_of(x, y, half=5):
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def g(d, k, default=None):
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(k, default)
    return getattr(d, k, default)


# ----------------------------------------------------------------------------
# Default strategy parameters
# ----------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "wheat_harvest_age": 4,
    "melon_tiles": 0,            # melon tiles planted in the opening
    "melon_last_plant_day": 4,
    "max_hands": 14,
    "turns_per_unit": 20.0,     # planning capacity per unit (safety margin)
    "goose_buy_last_day": 22,
    "cow_buy_last_day": 18,
    "wheat_per_animal": 0.75,   # wheat tiles per animal
    "cash_reserve": 150,
    "land_last_day": 16,
    "sell_fert_min": 45,        # below this price fertilizer is used on wheat
    "fert_wheat": True,
    "use_cows": 0,
    "skip_water": True,
}


class FarmAgent:
    def __init__(self, params=None):
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.mem = {}
        self.last_step = -1

    # ------------------------------------------------------------------
    def reset(self):
        self.mem = {"targets": {}, "unit_target": {}}

    # ------------------------------------------------------------------
    def __call__(self, obs, config=None):
        step = g(obs, "step", 0) or 0
        if step <= self.last_step or not self.mem:
            self.reset()
        self.last_step = step
        try:
            return self.act(obs, config)
        except Exception as e:  # never crash on the ladder
            import traceback
            traceback.print_exc()
            return {"farmer": ["PASS"], "hands": [], "market": []}

    # ------------------------------------------------------------------
    def act(self, obs, config):
        P = self.p
        self.cfg_steps = int(g(config, "episodeSteps", 720) or 720)
        self.tpd = int(g(config, "turnsPerDay", 24) or 24)
        self.size = int(g(config, "boardSize", 10) or 10)
        self.shed_cap = int(g(config, "shedCapacity", 100) or 100)
        self.max_orders = int(g(config, "maxMarketOrdersPerTurn", 10) or 10)
        tpd = self.tpd
        step = g(obs, "step", 0) or 0
        self.step = step
        self.day = step // tpd
        self.hour = step % tpd
        self.last_act_step = self.cfg_steps - 2          # 718
        self.last_day = self.last_act_step // tpd         # 29
        self.last_hour = self.last_act_step % tpd         # 22
        self.last_refresh_day = (self.cfg_steps - 1) // tpd - 1  # 28: last end-of-day processed
        me_id = g(obs, "player", 0)
        self.me_id = me_id
        farms = g(obs, "farms")
        self.farm = farms[me_id]
        self.opp = farms[1 - me_id] if len(farms) > 1 else None
        self.priv = g(obs, "private", {}) or {}
        self.shed = dict(g(self.priv, "shed", {}) or {})
        self.seeds = dict(g(self.priv, "seeds", {}) or {})
        self.invs = [dict(i or {}) for i in (g(self.priv, "inventories", [{}]) or [{}])]
        self.market = g(obs, "market", {}) or {}
        self.minv = dict(g(self.market, "inventory", {}) or {})
        self.prices = dict(g(self.market, "prices", {}) or {})
        self.shops = list(g(g(obs, "town", {}) or {}, "unlocked_shops", []) or [])
        self.money = float(self.farm["money"])
        self.tiles = self.farm["tiles"]
        half = self.size // 2
        self.shed_tiles = [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]
        self.units = [tuple(self.farm["farmer"])] + [tuple(h) for h in self.farm["hands"]]
        while len(self.invs) < len(self.units):
            self.invs.append({})

        if self.hour == 0 or "day_plan" not in self.mem or self.mem.get("plan_day") != self.day:
            self.plan_day()

        market_orders = []
        unit_actions = self.dispatch()
        market_orders = self.market_orders(unit_actions)
        out = {"farmer": unit_actions[0], "hands": unit_actions[1:], "market": market_orders[: self.max_orders]}
        return out

    # ==================================================================
    # Strategy
    # ==================================================================
    def owned(self, x, y):
        return self.tiles[y][x] != "LOCKED"

    def count_animals(self):
        n = {"GOOSE": 0, "COW": 0, "SHEEP": 0}
        for y in range(self.size):
            for x in range(self.size):
                t = self.tiles[y][x]
                if isinstance(t, dict) and t.get("animal"):
                    n[t["animal"]] += 1
        return n

    def plan_day(self):
        """Decide per-tile targets for today."""
        self.mem["plan_day"] = self.day
        self.mem["day_plan"] = True
        self.mem["unit_target"] = {}
        self.mem["hired_today"] = 0

    def target_of(self, x, y):
        return self.mem["targets"].get((x, y))

    # ==================================================================
    # Tasks
    # ==================================================================
    # (the per-tile action logic is implemented in tile_tasks)
    def plant_allowed(self, crop):
        cd = CROPS[crop]
        # must be harvestable by last day
        return self.day + cd["fy"] <= self.last_day

    def tile_tasks(self, x, y):
        """Ordered list of (op, need) due on this tile today. need is an item that
        must be in the acting unit's inventory (or 'SEED:<crop>')."""
        t = self.tiles[y][x]
        if t == "LOCKED":
            return []
        tgt = self.target_of(x, y)
        day = self.day
        last = self.last_day
        tasks = []
        if t is None:
            if tgt in CROPS:
                if self.plant_allowed(tgt):
                    return [(("PLANT", tgt), "SEED:" + tgt), (("WATER",), None)]
            elif tgt in ANIMALS:
                st = ANIMALS[tgt]["st"]
                return [(("BUILD_" + st,), None), (("PLACE", tgt), tgt), (("FEED",), "WHEAT"), (("CARE",), None)]
            return []
        kind = t.get("kind")
        if kind == "WEED":
            if tgt is not None:
                return [(("DIG",), None)] + self._after_empty(tgt)
            return []
        if kind == "PLANT":
            return self._plant_tasks(t, tgt)
        if kind in ("COOP", "PASTURE"):
            if not t.get("animal"):
                if tgt in ANIMALS and ANIMALS[tgt]["st"] == kind and day <= self.last_refresh_day - 1:
                    return [(("PLACE", tgt), tgt), (("FEED",), "WHEAT"), (("CARE",), None)]
                if tgt is not None and tgt not in ANIMALS:
                    return [(("DIG",), None)] + self._after_empty(tgt)
                return []
            return self._animal_tasks(t)
        return []

    def _after_empty(self, tgt):
        if tgt in CROPS:
            if self.plant_allowed(tgt):
                return [(("PLANT", tgt), "SEED:" + tgt), (("WATER",), None)]
            return []
        if tgt in ANIMALS:
            st = ANIMALS[tgt]["st"]
            return [(("BUILD_" + st,), None), (("PLACE", tgt), tgt), (("FEED",), "WHEAT"), (("CARE",), None)]
        return []

    def harvest_age(self, crop):
        if crop == "WHEAT":
            return self.p["wheat_harvest_age"]
        if crop == "CARROT":
            return 3
        if crop == "MELON":
            return 10
        return CROPS[crop]["my"]

    def _plant_tasks(self, t, tgt):
        crop = t["crop"]
        cd = CROPS[crop]
        day = self.day
        age = day - t["planted_day"]
        tasks = []
        if not cd["ongoing"]:
            ws = (cd["my"] + 1) // 2
            ha = self.harvest_age(crop)
            harvest = False
            if age >= cd["fy"] and t["yield_units"] > 0:
                if age >= ha or age >= cd["my"] or day >= self.last_day:
                    harvest = True
                # harvest early if it cannot reach the target age before game end
                if day + (ha - age) > self.last_day and day == self.last_day:
                    harvest = True
            in_window = ws <= age <= cd["my"] and t["yield_units"] < cd["mx"]
            need_water = False
            if not t["watered_today"]:
                if in_window:
                    need_water = True
                elif not harvest and (t["consecutive_unwatered"] >= 1 or not self.p["skip_water"]):
                    need_water = True
            # fertilize wheat at age 1-2 if we have fertilizer to use
            if (crop == "WHEAT" and self.p["fert_wheat"] and self.mem.get("use_fert")
                    and t.get("fertilized_until_day", -1) < day and ws - 1 <= age <= ws
                    and day + 2 <= self.last_day):
                tasks.append((("FERTILIZE",), "FERTILIZER"))
            if need_water:
                tasks.append((("WATER",), None))
            if harvest:
                tasks.append((("HARVEST",), None))
                if tgt in CROPS and self.plant_allowed(tgt):
                    tasks.append((("PLANT", tgt), "SEED:" + tgt))
                    tasks.append((("WATER",), None))
            return tasks
        # ongoing
        units = t["yield_units"]
        prod_tonight = False
        dsf = day + 1 - t["planted_day"] - cd["fy"]
        if dsf >= 0 and dsf % cd["iv"] == 0 and dsf // cd["iv"] + 1 <= cd["mx"]:
            prod_tonight = True
        fert_active = t.get("fertilized_until_day", -1) >= day
        if not t["watered_today"] and (t["consecutive_unwatered"] >= 1 or (fert_active and prod_tonight)
                                       or not self.p["skip_water"]):
            tasks.append((("WATER",), None))
        if units > 0:
            done = t.get("max_lifespan_step", -1) >= 0
            if done or day >= self.last_day or (prod_tonight and units + 2 > cd["mx"]) or units >= 2:
                tasks.append((("HARVEST",), None))
        return tasks

    def _animal_tasks(self, t):
        a = ANIMALS[t["animal"]]
        day = self.day
        lrd = self.last_refresh_day
        tasks = []
        placed = t.get("placed_day", day)
        pending = t.get("pending_care_bonus", 0) or 0
        # production tonight?
        dsf = day + 1 - placed - a["fy"]
        prod_tonight = dsf >= 0 and dsf % a["iv"] == 0 and day <= lrd
        # next production day strictly after today
        nd = day + 1
        while True:
            d2 = nd + 1 - placed - a["fy"]
            if d2 >= 0 and d2 % a["iv"] == 0:
                break
            nd += 1
        if day <= lrd and not t.get("fed_today"):
            tasks.append((("FEED",), "WHEAT"))
        if day <= lrd and not t.get("cared_today") and nd <= lrd:
            tasks.append((("CARE",), None))
        if t.get("fertilizer_available"):
            tasks.append((("COLLECT_FERTILIZER",), None))
        units = t.get("yield_units", 0) or 0
        if units > 0:
            fed = t.get("fed_today") or day <= lrd
            nxt = (1 + (pending if fed else 0)) if prod_tonight else 0
            if units + nxt > a["mh"] or day >= self.last_day or (day == lrd and prod_tonight is False):
                tasks.append((("HARVEST",), None))
        return tasks

    # ==================================================================
    # Dispatcher
    # ==================================================================
    def dispatch(self):
        raise NotImplementedError

    def market_orders(self, unit_actions):
        raise NotImplementedError


_AGENT = {}


def agent(obs, config=None):
    pid = g(obs, "player", 0)
    if pid not in _AGENT:
        _AGENT[pid] = FarmAgent()
    return _AGENT[pid](obs, config)
