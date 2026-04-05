"""
Compound Wing Optimizer

A tool to design the most aerodynamically efficient compound wing (a straight root
transitioning into a tapered tip) for a given cruise condition. It tries to maximize
the L/D ratio while making sure we still generate the required lift.

It uses Peter Sharpe's AeroSandbox library under the hood:
- NeuralFoil gives us quick 2D airfoil polars.
- The 3D Lifting Line Theory (LLT) solver calculates induced drag.
- IPOPT handles the actual optimization math.

I also added multithreading to speed things up, some fail-safes so the optimizer
doesn't crash if it gets stuck in an impossible design space, and CSV logging
so we can easily plot the iteration history later.
"""

import os
import multiprocessing
import pandas as pd

# import multiprocessing before numpy to prevent errors
threads = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads

print(f"Using {threads} processes for optimization")

import aerosandbox as asb  # noqa: E402
import aerosandbox.numpy as np  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# 1. USER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# Tweak these to match your aircraft's requirements.
# Just make sure the bounds actually leave enough room for a valid design

AIRFOIL_NAME = "e387"
MTOW = 15  # [kg]
VELOCITY = 35  # [m/s]
LIFT_REQ = MTOW * 9.81  # [N]

#Initial guesses for geometric parameters
ROOT_CHORD = 0.11  # [m]
INIT_HALF_SPAN = 0.7  # [m]
INIT_RECT_FRAC = 0.72  # fraction of total span that is rectangular in shape
INIT_TAPER_RATIO = 0.38  # (tip chord)/(root chord)
INIT_ALPHA = 3.5  # [degrees]

"""---------------------------------- BOUNDS ----------------------------------"""

# Be sure to check the bounds to make sure that the optimizer is able to find a feasible solution.
# Cross verify values and give sufficient breathing space for a valid optimization problem.

# Half span of the wing
BOUND_MIN_HALF_SPAN = 0.55
BOUND_MAX_HALF_SPAN = 1.5

# Rectangular section span as a fraction of half span
BOUND_MIN_RECT_FRAC = 0.62
BOUND_MAX_RECT_FRAC = 0.85

# Toggle to optimize taper ratio (Toggle used if there are too many optimization variables and if user wants to limit designs)
OPTIMIZE_TAPER_RATIO = True
BOUND_MIN_TAPER = 0.30
BOUND_MAX_TAPER = 0.55

# Alpha bounds set so that the optimizer does not run into a stall condition or use unreal angles of attack just to maximize lift
BOUND_MIN_ALPHA = 2.0
BOUND_MAX_ALPHA = 8.0

# Toggle to optimize root chord (Toggle used if there are too many optimization variables and if user wants to limit designs)
OPTIMIZE_ROOT_CHORD = True
BOUND_MIN_ROOT_CHORD = 0.11
BOUND_MAX_ROOT_CHORD = 0.20

MAX_ASPECT_RATIO = 15.0  # Used along with the root chord optimization for structural constraints

# ══════════════════════════════════════════════════════════════════════════════
# 2. 2D AIRFOIL PRE-COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
# We calculate the 2D polar just once at the root Reynolds number to save time.
# This gives us a safe limits for CL so the optimizer doesn't try pushing into a stall & actually operates in a low drag condition

airfoil = asb.Airfoil(AIRFOIL_NAME)
Re = 1.225 * VELOCITY * ROOT_CHORD / 0.0000171
alphas = np.linspace(-5, 15, 100)
nf = airfoil.get_aero_from_neuralfoil(Re=Re, alpha=alphas, n_crit=9.0, model_size="xxxlarge")
Cl = nf["CL"]
Cd = nf["CD"]

CRUISE_CL_MAX = float(np.max(Cl))
CRUISE_CL_MIN = 0.2

print(f"2D polar: CL_min={CRUISE_CL_MIN:.3f}  CL_max={CRUISE_CL_MAX:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PHASE 1 — BASELINE WING CHECK
# ══════════════════════════════════════════════════════════════════════════════
# Let's see how our initial guess performs before we let the optimizer modify it.

print("=" * 60)
print("   PHASE 1: BASELINE WING ANALYSIS")
print("=" * 60)

# Calculate baseline geometry (Always uses the constant ROOT_CHORD)
base_rect_span = INIT_HALF_SPAN * INIT_RECT_FRAC
base_tip_chord = ROOT_CHORD * INIT_TAPER_RATIO
base_x_le_tip = 0.25 * (ROOT_CHORD - base_tip_chord)  # ASSUMPTION - Quarter Chord line is kept at 0 sweep for stability
base_area = 2 * ((ROOT_CHORD * base_rect_span) + (0.5 * (ROOT_CHORD + base_tip_chord) * (INIT_HALF_SPAN - base_rect_span)))
base_AR = (2 * INIT_HALF_SPAN) ** 2 / base_area

baseline_wing = asb.Wing(
    name="Baseline Compound Wing",
    symmetric=True,
    xsecs=[
        asb.WingXSec(xyz_le=[0, 0, 0], chord=ROOT_CHORD, airfoil=asb.Airfoil("e387")),
        asb.WingXSec(xyz_le=[0, base_rect_span, 0], chord=ROOT_CHORD, airfoil=asb.Airfoil("e387")),
        asb.WingXSec(xyz_le=[base_x_le_tip, INIT_HALF_SPAN, 0], chord=base_tip_chord, airfoil=asb.Airfoil("e387")),
    ],
)

base_airplane = asb.Airplane(name="Baseline Aircraft", wings=[baseline_wing])
base_op_point = asb.OperatingPoint(velocity=VELOCITY, alpha=INIT_ALPHA)

base_llt = asb.NonlinearLiftingLine(airplane=base_airplane, op_point=base_op_point, spanwise_resolution=10)
base_aero = base_llt.run()
base_lift = 0.5 * 1.225 * VELOCITY**2 * base_area * base_aero["CL"]

print(f"  Full Span      : {2 * INIT_HALF_SPAN:.3f} m")
print(f"  Area           : {base_area:.4f} m²")
print(f"  Aspect Ratio   : {base_AR:.2f}")
print(f"  Alpha          : {INIT_ALPHA:.2f} °")
print(f"  CL             : {base_aero['CL']:.4f}")
print(f"  CD             : {base_aero['CD']:.6f}")
print(f"  L/D Ratio      : {base_aero['CL'] / base_aero['CD']:.2f}")
print(f"  Lift force     : {base_lift:.2f} N (Target: {LIFT_REQ:.2f} N)")
print("\n[INFO] Moving to Phase 2: Optimization...")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PHASE 2 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
# Time to define the variables we want IPOPT to tweak, and set up our constraints
# (like making sure the wing actually generates enough target lift).

opti = asb.Opti()
dynamic_pressure = 0.5 * 1.225 * VELOCITY**2

# Safety check (calculates minimum required area for the given lift force and Cl bounds) to avoid infeasible optimization problems
S_min_required = LIFT_REQ / (dynamic_pressure * CRUISE_CL_MAX)
effective_max_root_chord = BOUND_MAX_ROOT_CHORD if OPTIMIZE_ROOT_CHORD else ROOT_CHORD
max_possible_area = 2 * effective_max_root_chord * BOUND_MAX_HALF_SPAN
if max_possible_area < S_min_required:
    required_half_span = (S_min_required / (2 * effective_max_root_chord)) * 1.1
    print(f"[WARNING] Auto-expanding BOUND_MAX_HALF_SPAN → {required_half_span:.3f} m")
    BOUND_MAX_HALF_SPAN = required_half_span

# --- Optimization Variables ---
opti_half_span = opti.variable(
    init_guess=INIT_HALF_SPAN,
    lower_bound=BOUND_MIN_HALF_SPAN,
    upper_bound=BOUND_MAX_HALF_SPAN,
)

opti_rect_frac = opti.variable(
    init_guess=INIT_RECT_FRAC,
    lower_bound=BOUND_MIN_RECT_FRAC,
    upper_bound=BOUND_MAX_RECT_FRAC,
)

opti_alpha = opti.variable(
    init_guess=INIT_ALPHA, 
    lower_bound=BOUND_MIN_ALPHA, 
    upper_bound=BOUND_MAX_ALPHA
)

if OPTIMIZE_ROOT_CHORD:
    opti_root_chord = opti.variable(
        init_guess=ROOT_CHORD,
        lower_bound=BOUND_MIN_ROOT_CHORD,
        upper_bound=BOUND_MAX_ROOT_CHORD,
    )
else:
    opti_root_chord = ROOT_CHORD

if OPTIMIZE_TAPER_RATIO:
    opti_taper = opti.variable(
        init_guess=INIT_TAPER_RATIO,
        lower_bound=BOUND_MIN_TAPER,
        upper_bound=BOUND_MAX_TAPER,
    )
else:
    opti_taper = INIT_TAPER_RATIO

# ---------------------------------- Other Variables that be calculated ----------------------------------
opti_rect_span = opti_half_span * opti_rect_frac
opti_tip_chord = opti_root_chord * opti_taper
opti_x_le_tip = 0.25 * (opti_root_chord - opti_tip_chord)
opti_area = 2 * (
    (opti_root_chord * opti_rect_span)
    + (0.5 * (opti_root_chord + opti_tip_chord) * (opti_half_span - opti_rect_span))
)

# ------------------------------------------- Wing & aero ------------------------------------------------
# Intermediate wing that is used in between iterations

opt_wing = asb.Wing(
    name="Optimized Compound Wing",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, 0, 0], chord=opti_root_chord, airfoil=asb.Airfoil(AIRFOIL_NAME)
        ),
        asb.WingXSec(
            xyz_le=[0, opti_rect_span, 0],
            chord=opti_root_chord,
            airfoil=asb.Airfoil(AIRFOIL_NAME),
        ),
        asb.WingXSec(
            xyz_le=[opti_x_le_tip, opti_half_span, 0],
            chord=opti_tip_chord,
            airfoil=asb.Airfoil(AIRFOIL_NAME),
        ),
    ],
)
opt_airplane = asb.Airplane(name="Optimizer Aircraft", wings=[opt_wing])
opt_op_point = asb.OperatingPoint(velocity=VELOCITY, alpha=opti_alpha)
opt_llt = asb.NonlinearLiftingLine(
    airplane=opt_airplane, op_point=opt_op_point, opti=opti, spanwise_resolution=10
)
opt_aero = opt_llt.run()
CL = opt_aero["CL"]
CD = opt_aero["CD"]

# -------------------------- Constraints -------------------------------

current_lift = dynamic_pressure * opti_area * CL
opti_AR = (2 * opti_half_span) ** 2 / opti_area

# 1. Lift equality
opti.subject_to(current_lift / LIFT_REQ == 1)

# 2. CL band
opti.subject_to(CL <= CRUISE_CL_MAX)
opti.subject_to(CL >= CRUISE_CL_MIN)

# 3. Tip chord manufacturability floor
opti.subject_to(opti_tip_chord >= 0.044)

# 4. Aspect ratio hard cap
opti.subject_to(opti_AR <= MAX_ASPECT_RATIO)

# 5. Rect fraction
opti.subject_to(opti_rect_frac >= 0.62)


# --- Objective function ---
objective = CD / CL  # minimises 1/(L/D)
opti.minimize(objective)

# ══════════════════════════════════════════════════════════════════════════════
# 5. SOLVE & SAVE
# ══════════════════════════════════════════════════════════════════════════════

history = []


def opti_callback(i):
    try:
        row = {
            "iteration": i,
            "half_span": float(opti.debug.value(opti_half_span)),
            "root_chord": float(opti.debug.value(opti_root_chord))
            if OPTIMIZE_ROOT_CHORD
            else ROOT_CHORD,
            "rect_span": float(opti.debug.value(opti_rect_span)),
            "rect_frac": float(opti.debug.value(opti_rect_frac)),
            "taper_ratio": float(opti.debug.value(opti_taper))
            if OPTIMIZE_TAPER_RATIO
            else INIT_TAPER_RATIO,
            "alpha": float(opti.debug.value(opti_alpha)),
            "CL": float(opti.debug.value(CL)),
            "Lift": float(opti.debug.value(current_lift)),
            "L_D": float(opti.debug.value(CL / CD)),
            "area": float(opti.debug.value(opti_area)),
            "AR": float(opti.debug.value(opti_AR)),
        }
        history.append(row)
    except Exception:
        pass


try:
    sol = opti.solve(
        max_iter=1000,
        verbose=True,
        callback=opti_callback,
        options={  # tweaking options of IPOPT solver for more robustness
            "ipopt.mu_strategy": "adaptive",
            "ipopt.bound_push": 1e-6,
            "ipopt.bound_frac": 1e-6,
            "ipopt.acceptable_tol": 1e-6,
        },
    )

    # Extract results
    res_half_span = float(sol(opti_half_span))
    res_root_chord = float(sol(opti_root_chord)) if OPTIMIZE_ROOT_CHORD else ROOT_CHORD
    res_rect_frac = float(sol(opti_rect_frac))
    res_rect_span = res_half_span * res_rect_frac
    res_taper = float(sol(opti_taper))
    res_alpha = float(sol(opti_alpha))
    res_tip_chord = float(sol(opti_tip_chord))
    res_area = float(sol(opti_area))
    res_CL = float(sol(CL))
    res_CD = float(sol(CD))
    res_AR = (res_half_span * 2) ** 2 / res_area

    print("\n" + "=" * 60)
    print("   PHASE 2: OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"  Full Span      : {2 * res_half_span:.4f} m (Baseline: {2 * INIT_HALF_SPAN:.4f} m)")
    print(f"  Root Chord     : {res_root_chord:.4f} m (Optimized: {OPTIMIZE_ROOT_CHORD})")
    print(f"  Rect Span      : {res_rect_span:.4f} m (Baseline: {base_rect_span:.4f} m)")
    print(f"  Tapered Span   : {res_half_span - res_rect_span:.4f} m")
    print(f"  Tip Chord      : {res_tip_chord:.4f} m (Taper Ratio: {res_taper:.2f})")
    print(f"  Angle of Attack: {res_alpha:.2f} °")
    print("-" * 60)
    print(f"  Wing Area      : {res_area:.5f} m² (Baseline: {base_area:.5f} m²)")
    print(f"  Aspect Ratio   : {res_AR:.2f} (Max Allowed: {MAX_ASPECT_RATIO})")
    print("-" * 60)
    print(f"  CL             : {res_CL:.4f}")
    print(f"  CD             : {res_CD:.6f}")
    print(f"  L / D Ratio    : {res_CL / res_CD:.2f}")
    print("=" * 60)

except RuntimeError as e:
    print(f"\n[ERROR] Optimization failed: {e}")

finally:
    if history:
        pd.DataFrame(history).to_csv("compound_wing_opti_results.csv", index=False)
