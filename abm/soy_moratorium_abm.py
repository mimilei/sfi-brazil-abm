#!/usr/bin/env python3
# abm_agri_autorun.py

from dataclasses import dataclass, field
from typing import List, Dict
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ======== CONFIG =========
# =========================
CONFIG = {
    "periods": 600,
    "seed": 321,

    # Output pricing (fixed over time)
    "demand_markup": 0.50,      # price_out = base_cost_per_land*(1+markup)
    "base_cost_per_land": 1.0,  # used as a cost proxy and in profit calc

    # Initialization (truncated normal land per farmer)
    "n_farmers": 200,
    "init_land_lo": 5.0,
    "init_land_hi": 15.0,
    "init_land_mean": 10.0,
    "init_land_sd": (10.0 - 5.0) / 3.0,

    # Inputs (per-period purchases; seeds/fert add temporary boost, not persistent)
    "upgrade_boost": 1.7,
    "fert_init_price": 0.4,
    "seed_init_price": 0.4,

    # NEW: Basic input (fixed price), required each period, no extra decay
    "basic_input_price": 0.20,   # price per land-unit per period

    # Supplier price reaction for seeds/fert (level-based via EMA)
    "supplier_step": 0.002,
    "supplier_ema_alpha": 0.20,

    # Productivity decay (linear; clamped by floor)
    "baseline_abs_decay": 0.00060,  # per period (always)
    "input_abs_decay": 0.00480,     # extra per period if seeds/fert used
    "base_productivity_floor": 0.50,

    # Deforestation
    "piece_size": 1.0,              # land chunk size when expanding
    "deforest_cap_fraction": 0.50,  # max new land per period relative to current land
}

# Optional CSV export (set to a filename to save the time series)
CSV_EXPORT = None   # e.g., "abm_run.csv"


# =========================
# ====== MODEL CORE =======
# =========================

def truncnorm(mean: float, sd: float, lo: float, hi: float) -> float:
    """Simple truncated normal sampler."""
    for _ in range(10000):
        x = random.gauss(mean, sd)
        if lo <= x <= hi:
            return x
    return max(lo, min(hi, x))

@dataclass
class LandParcel:
    area: float
    # per-period flags (reset each period)
    has_fertilizer_now: bool = False
    has_seeds_now: bool = False
    multiplier: float = 1.0  # structural multiplier (kept at 1.0 here)

@dataclass
class SupplierAdaptivePricing:
    """For seeds/fertiliser: price reacts to level of demand vs EMA."""
    name: str
    price: float
    step: float = 0.002
    ema_alpha: float = 0.20
    ema_qty: float = 0.0
    def observe_and_adjust(self, qty: float):
        self.ema_qty = (1 - self.ema_alpha) * self.ema_qty + self.ema_alpha * qty
        delta = qty - self.ema_qty
        eps = 1e-9
        if delta > eps:
            self.price += self.step
        elif delta < -eps:
            self.price = max(0.0, self.price - self.step)

@dataclass
class SupplierBasicFixed:
    """Basic input: fixed price, always required, no extra decay."""
    name: str
    price: float
    def observe_and_adjust(self, qty: float):
        pass  # fixed price, no reaction

@dataclass
class Farmer:
    id: int
    land: float
    base_productivity: float = 1.0
    base_productivity_floor: float = CONFIG["base_productivity_floor"]
    baseline_abs_decay_per_period: float = CONFIG["baseline_abs_decay"]
    input_abs_decay_per_period: float = CONFIG["input_abs_decay"]
    upgrade_boost: float = CONFIG["upgrade_boost"]
    parcels: List[LandParcel] = field(default_factory=list)
    active: bool = True

    def ensure_parcels(self):
        if not self.parcels:
            self.parcels = [LandParcel(area=self.land)]

    def clear_inputs(self):
        for p in self.parcels:
            p.has_fertilizer_now = False
            p.has_seeds_now = False

    def current_output(self) -> float:
        self.ensure_parcels()
        out = 0.0
        for p in self.parcels:
            boost = self.upgrade_boost if (p.has_fertilizer_now or p.has_seeds_now) else 1.0
            out += self.base_productivity * p.multiplier * boost * p.area
        return out

    def expected_profit(self, price_out: float, fert_price: float, seed_price: float,
                        base_cost_per_land: float, basic_price: float) -> float:
        """Expected profit given current per-period flags; basic input is always paid."""
        self.ensure_parcels()
        # revenue
        revenue = price_out * sum(
            self.base_productivity * p.multiplier *
            (self.upgrade_boost if (p.has_fertilizer_now or p.has_seeds_now) else 1.0) * p.area
            for p in self.parcels
        )
        # base cost with mild economies of scale
        scale = max(0.2, 1.0 - 0.05 * (self.land - 1.0))
        base_cost = base_cost_per_land * self.land * scale
        # basic input cost (required)
        basic_cost = basic_price * self.land
        # optional inputs
        fert_area = sum(p.area for p in self.parcels if p.has_fertilizer_now)
        seed_area = sum(p.area for p in self.parcels if p.has_seeds_now)
        input_cost = fert_price * fert_area + seed_price * seed_area
        return revenue - base_cost - basic_cost - input_cost

    def try_inputs_if_needed(self, price_out: float, fert_price: float, seed_price: float,
                             base_cost_per_land: float, target_profit: float, basic_price: float) -> bool:
        """Adopt at most one input (on largest parcel) if profit < target."""
        self.ensure_parcels()
        self.clear_inputs()  # evaluate baseline without inputs
        prof0 = self.expected_profit(price_out, fert_price, seed_price, base_cost_per_land, basic_price)
        if prof0 >= target_profit:
            return False

        p = max(self.parcels, key=lambda x: x.area)  # largest parcel

        best_gain = -1e18
        best = None

        # try fertiliser
        p.has_fertilizer_now = True
        pf = self.expected_profit(price_out, fert_price, seed_price, base_cost_per_land, basic_price)
        gain_f = pf - prof0
        p.has_fertilizer_now = False
        if gain_f > best_gain:
            best_gain, best = gain_f, ('fert', p)

        # try seeds
        p.has_seeds_now = True
        ps = self.expected_profit(price_out, fert_price, seed_price, base_cost_per_land, basic_price)
        gain_s = ps - prof0
        p.has_seeds_now = False
        if gain_s > best_gain:
            best_gain, best = gain_s, ('seed', p)

        if best and best_gain > 1e-9:
            kind, parc = best
            if kind == 'fert': parc.has_fertilizer_now = True
            else:               parc.has_seeds_now = True
            return True
        return False

    def deforest_until_target(self, price_out: float, fert_price: float, seed_price: float,
                              base_cost_per_land: float, target_profit: float,
                              piece_size: float, cap_fraction: float, basic_price: float) -> float:
        """If still below target, add land from forest up to cap_fraction of current land."""
        self.ensure_parcels()
        added = 0.0
        max_new = cap_fraction * self.land
        while added + 1e-9 < max_new:
            prof_now = self.expected_profit(price_out, fert_price, seed_price, base_cost_per_land, basic_price)
            if prof_now >= target_profit:
                break
            step = min(piece_size, max_new - added)
            self.land += step
            added += step
            # merge into an existing raw parcel if possible
            merged = False
            for p in self.parcels:
                if (not p.has_fertilizer_now) and (not p.has_seeds_now) and abs(p.multiplier - 1.0) < 1e-12:
                    p.area += step
                    merged = True
                    break
            if not merged:
                self.parcels.append(LandParcel(area=step))
        return added

    def end_of_period(self):
        # linear decay; extra if seeds/fert used anywhere
        any_inputs = any(p.has_fertilizer_now or p.has_seeds_now for p in self.parcels)
        dec = self.baseline_abs_decay_per_period + (self.input_abs_decay_per_period if any_inputs else 0.0)
        self.base_productivity = max(self.base_productivity_floor, self.base_productivity - dec)
        self.clear_inputs()  # reset for next period


@dataclass
class SimulationConfig:
    periods: int = CONFIG["periods"]
    seed: int = CONFIG["seed"]
    demand_markup: float = CONFIG["demand_markup"]
    base_cost_per_land: float = CONFIG["base_cost_per_land"]

    n_farmers: int = CONFIG["n_farmers"]
    init_land_lo: float = CONFIG["init_land_lo"]
    init_land_hi: float = CONFIG["init_land_hi"]
    init_land_mean: float = CONFIG["init_land_mean"]
    init_land_sd: float = CONFIG["init_land_sd"]

    fert_init_price: float = CONFIG["fert_init_price"]
    seed_init_price: float = CONFIG["seed_init_price"]
    supplier_step: float = CONFIG["supplier_step"]
    supplier_ema_alpha: float = CONFIG["supplier_ema_alpha"]

    upgrade_boost: float = CONFIG["upgrade_boost"]
    baseline_abs_decay: float = CONFIG["baseline_abs_decay"]
    input_abs_decay: float = CONFIG["input_abs_decay"]
    base_productivity_floor: float = CONFIG["base_productivity_floor"]

    piece_size: float = CONFIG["piece_size"]
    deforest_cap_fraction: float = CONFIG["deforest_cap_fraction"]

    # NEW: basic input (fixed)
    basic_input_price: float = CONFIG["basic_input_price"]

    # target fixed from t=0 = min initial profit (no margin)
    target_from_min_initial: bool = True
    target_profit: float = 0.0

@dataclass
class SimulationState:
    farmers: List[Farmer]
    fert_supplier: SupplierAdaptivePricing
    seed_supplier: SupplierAdaptivePricing
    basic_supplier: SupplierBasicFixed
    output_price: float
    arable_pool: float = 0.0

class AgriABM:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        random.seed(cfg.seed)

        # Farmers
        farmers: List[Farmer] = []
        for i in range(cfg.n_farmers):
            land = truncnorm(cfg.init_land_mean, cfg.init_land_sd, cfg.init_land_lo, cfg.init_land_hi)
            f = Farmer(
                id=i,
                land=land,
                base_productivity=1.0,
                base_productivity_floor=cfg.base_productivity_floor,
                baseline_abs_decay_per_period=cfg.baseline_abs_decay,
                input_abs_decay_per_period=cfg.input_abs_decay,
                upgrade_boost=cfg.upgrade_boost,
            )
            f.ensure_parcels()
            farmers.append(f)

        # Suppliers
        fert = SupplierAdaptivePricing("Fertilizer", cfg.fert_init_price, step=cfg.supplier_step, ema_alpha=cfg.supplier_ema_alpha)
        seed = SupplierAdaptivePricing("Seeds", cfg.seed_init_price, step=cfg.supplier_step, ema_alpha=cfg.supplier_ema_alpha)
        basic = SupplierBasicFixed("Basic", cfg.basic_input_price)

        # Fixed output price
        price_out = cfg.base_cost_per_land * (1.0 + cfg.demand_markup)

        self.state = SimulationState(
            farmers=farmers,
            fert_supplier=fert,
            seed_supplier=seed,
            basic_supplier=basic,
            output_price=price_out
        )

        # Target = min initial profit (no margin), computed once at t=0
        if self.cfg.target_from_min_initial:
            profits0 = [
                f.expected_profit(price_out, fert.price, seed.price, cfg.base_cost_per_land, basic.price)
                for f in farmers
            ]
            if profits0:
                self.cfg.target_profit = min(profits0)

    def step(self, t: int) -> Dict[str, float]:
        st, cfg = self.state, self.cfg
        price_out = st.output_price
        bp = st.basic_supplier.price  # basic input price (fixed)

        fert_q = 0.0
        seed_q = 0.0
        total_output = 0.0
        n_active = 0
        deforest_added = 0.0

        # Decisions and production
        for f in st.farmers:
            if not f.active:
                continue
            n_active += 1

            # Try inputs if needed (seeds/fert) — basic is always paid in profit calc
            _ = f.try_inputs_if_needed(price_out, st.fert_supplier.price, st.seed_supplier.price,
                                       cfg.base_cost_per_land, cfg.target_profit, bp)

            # If still below target, deforest up to cap
            prof_now = f.expected_profit(price_out, st.fert_supplier.price, st.seed_supplier.price,
                                         cfg.base_cost_per_land, bp)
            if prof_now < cfg.target_profit:
                deforest_added += f.deforest_until_target(price_out, st.fert_supplier.price, st.seed_supplier.price,
                                                          cfg.base_cost_per_land, cfg.target_profit,
                                                          cfg.piece_size, cfg.deforest_cap_fraction, bp)

            # Production (after decisions)
            out = f.current_output()
            total_output += out

            # Track per-period input areas for pricing/series
            for p in f.parcels:
                if p.has_fertilizer_now: fert_q += p.area
                if p.has_seeds_now:      seed_q += p.area

        # Suppliers update prices (basic stays fixed)
        st.fert_supplier.observe_and_adjust(fert_q)
        st.seed_supplier.observe_and_adjust(seed_q)
        st.basic_supplier.observe_and_adjust(sum(f.land for f in st.farmers if f.active))  # no-op

        # End-of-period decay + reset flags
        for f in st.farmers:
            if f.active:
                f.end_of_period()

        return {
            "t": t,
            "fert_price": st.fert_supplier.price,
            "seed_price": st.seed_supplier.price,
            "basic_price": st.basic_supplier.price,
            "fert_qty": fert_q,
            "seed_qty": seed_q,
            "total_output": total_output,
            "n_active": n_active,
            "deforestation": deforest_added,
            "target": cfg.target_profit,
            "price_out": st.output_price,
        }

    def run(self) -> List[Dict[str, float]]:
        return [self.step(t) for t in range(self.cfg.periods)]

def run_simulation(**kwargs) -> List[Dict[str, float]]:
    cfg = SimulationConfig(**kwargs)
    sim = AgriABM(cfg)
    return sim.run()


# =========================
# ======= AUTO-RUN ========
# =========================

history = run_simulation(
    periods=CONFIG["periods"],
    seed=CONFIG["seed"],
    demand_markup=CONFIG["demand_markup"],
    piece_size=CONFIG["piece_size"],
    deforest_cap_fraction=CONFIG["deforest_cap_fraction"],
)

df = pd.DataFrame(history)
print(df.head(10))

# --- Plots ---
plt.figure()
plt.plot(df["t"], df["fert_qty"], label="fert area")
plt.plot(df["t"], df["seed_qty"], label="seed area")
plt.title("Input Adoption Over Time")
plt.xlabel("t"); plt.ylabel("area"); plt.legend(); plt.show()

plt.figure()
plt.plot(df["t"], df["fert_price"], label="fert price")
plt.plot(df["t"], df["seed_price"], label="seed price")
plt.plot(df["t"], df["basic_price"], label="basic price", linestyle="--")
plt.title("Supplier Prices Over Time")
plt.xlabel("t"); plt.ylabel("price"); plt.legend(); plt.show()

plt.figure()
plt.plot(df["t"], df["n_active"])
plt.title("Active Farmers Over Time")
plt.xlabel("t"); plt.ylabel("count"); plt.show()

plt.figure()
plt.plot(df["t"], df["total_output"])
plt.title("Total Output Over Time")
plt.xlabel("t"); plt.ylabel("units"); plt.show()

plt.figure()
plt.plot(df["t"], df["deforestation"])
plt.title("Deforestation per Period")
plt.xlabel("t"); plt.ylabel("land added from forest"); plt.show()

# Land-weighted distribution of base productivity at the end
# (simulate again to access final state)
sim_cfg = SimulationConfig(periods=CONFIG["periods"], seed=CONFIG["seed"], demand_markup=CONFIG["demand_markup"])
sim = AgriABM(sim_cfg)
for _t in range(CONFIG["periods"]): sim.step(_t)
vals, wts = [], []
for f in sim.state.farmers:
    if f.active:
        vals.append(f.base_productivity); wts.append(f.land)
vals, wts = np.array(vals, float), np.array(wts, float)
wmean = np.average(vals, weights=wts) if wts.sum() > 0 else float('nan')
print(f"End weighted mean base productivity: {wmean:.3f}")

plt.figure()
plt.hist(vals, bins=30, weights=wts, edgecolor="black")
plt.title("Distribution of Base Productivity at End (Land-weighted)")
plt.xlabel("base productivity"); plt.ylabel("land-weighted frequency"); plt.show()

# Optional CSV export
if CSV_EXPORT:
    df.to_csv(CSV_EXPORT, index=False)
    print(f"Saved time series to {CSV_EXPORT}")
