"""
Compound Wing Optimizer

A tool to design the most aerodynamically efficient compound wing (a straight root
transitioning into a tapered tip) for a given cruise condition. It tries to maximize
the L/D ratio while making sure we still generate the required lift.

It uses Peter Sharpe's AeroSandbox library under the hood:
- NeuralFoil gives us quick 2D airfoil polars.
- The 3D Lifting Line Theory (LLT) solver calculates total drag.
- IPOPT handles the actual optimization math.

Key Features:
- Multithreading (Single threaded runs too slow)
- Some fail-safes so the optimizer doesn't crash if it gets stuck in an impossible design space
- CSV logging so we can easily plot the iteration history later.
- Variable Names are kept to suit intuition and are self explanatory in most cases

Scope for improvement: 
- No consideration for moments or stability derivatives is done, could be added in later
- 3D optimization of dihedral angles and washout (geometric twist at tip)
- Implement stuctural optimization and aeroelasticity
"""

import os
import multiprocessing
import pandas as pd

# import multiprocessing before numpy to prevent errors
threads = str(multiprocessing.cpu_count()-2)
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
# Make sure the bounds actually leave enough room for a valid design

PLOT_FLAGS = [1, 1, 1, 1, 1, 1]  # [Geometry, L/D, Lift, Drag, CL, CD]
BYPASS_OPTIMIZATION = False      # Set to True to skip optimization and just plot from existing CSV

AIRFOIL_NAME = "e387"
MTOW = 15                 # [kg]
VELOCITY = 35             # [m/s]
LIFT_REQ = MTOW * 9.81    # [N]

# Initial guesses for geometric parameters (from simple analytical analysis and hand calculations)
ROOT_CHORD = 0.25        # [m]
INIT_HALF_SPAN = 1.00   # [m]
INIT_RECT_FRAC = 0.85   # fraction of total span that is rectangular in shape
INIT_TAPER_RATIO = 0.4  
INIT_ALPHA = 3          # [degrees]

"""---------------------------------- BOUNDS ----------------------------------"""

# Be sure to check the bounds to make sure that the optimizer is able to find a feasible solution.
# Cross verify values and give sufficient breathing space for a valid optimization problem.

BOUND_MIN_HALF_SPAN = 0.60
BOUND_MAX_HALF_SPAN = 1.80

BOUND_MIN_RECT_FRAC = 0.5
BOUND_MAX_RECT_FRAC = 0.85

OPTIMIZE_TAPER_RATIO = False  # Keep variables limited - for optimization loop to run smoothly and find optimal solution
BOUND_MIN_TAPER = 0.30
BOUND_MAX_TAPER = 0.55

BOUND_MIN_ALPHA = 1.5
BOUND_MAX_ALPHA = 6  # Keep cruise alpha low — force optimizer to solve via geometry

OPTIMIZE_ROOT_CHORD = True
BOUND_MIN_ROOT_CHORD = 0.12
BOUND_MAX_ROOT_CHORD = 0.28

MAX_ASPECT_RATIO = 15.0  # Used along with the root chord optimization for structural constraints

# ══════════════════════════════════════════════════════════════════════════════
# 2. 2D AIRFOIL PRE-COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
# We calculate the 2D polar just once at the root Reynolds number to save time.
# Gives safe limits for CL so the optimizer doesn't push into stall & actually operates in low drag conditions

airfoil = asb.Airfoil(AIRFOIL_NAME)
Re = 1.225 * VELOCITY * ROOT_CHORD / 0.0000171
alphas = np.linspace(-5, 15, 100)
nf = airfoil.get_aero_from_neuralfoil(Re=Re, alpha=alphas, n_crit=9.0, model_size="xxxlarge")
Cl = nf["CL"]
Cd = nf["CD"]

CL_MAX_2D = float(np.max(Cl))
CRUISE_CL_MAX = 0.8 * CL_MAX_2D  # 0.8 CL_Max is mostly within linear zone of Cl-alpha polar
CRUISE_CL_MIN = 0.15  # Value of Cl at very low AoA for most cambered airfoils

# Also find the CL for best 2D L/D — useful reference 
Cl_over_Cd = Cl / Cd
best_2d_LD_idx = int(np.argmax(Cl_over_Cd))
BEST_2D_CL = float(Cl[best_2d_LD_idx])  # CL value corresponding to best 2D L/D
BEST_2D_LD = float(Cl_over_Cd[best_2d_LD_idx])  # Best 2D L/D ratio

print(f"2D polar: CL_max_2D={CL_MAX_2D:.3f}  CL_max_3D_limit={CRUISE_CL_MAX:.3f}")
print(f"2D best L/D = {BEST_2D_LD:.1f} at CL = {BEST_2D_CL:.2f} and alpha = {alphas[best_2d_LD_idx]:.1f} deg")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PHASE 1 — BASELINE WING CHECK
# ══════════════════════════════════════════════════════════════════════════════
# How our initial Wing performs before optimizer tweaks it

print("=" * 60)
print("   PHASE 1: BASELINE WING ANALYSIS")
print("=" * 60)

# Calculate baseline geometry
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
base_drag = 0.5 * 1.225 * VELOCITY**2 * base_area * base_aero["CD"]
base_LD = base_aero['CL'] / base_aero['CD']

print(f"  Full Span      : {2 * INIT_HALF_SPAN:.3f} m")
print(f"  Area           : {base_area:.4f} m²")
print(f"  Aspect Ratio   : {base_AR:.2f}")
print(f"  Alpha          : {INIT_ALPHA:.2f} °")
print(f"  CL             : {base_aero['CL']:.4f}")
print(f"  CD             : {base_aero['CD']:.6f}")
print(f"  L/D Ratio      : {base_LD:.2f}")
print(f"  Lift force     : {base_lift:.2f} N (Target: {LIFT_REQ:.2f} N)")
print(f"  Drag Force     : {base_drag:.2f} N")


'''======================== Fair comparison: what L/D does THIS geometry give at the required lift? ========================'''

# Calculate the Cl required in base config, so L/D is not artificially inflated by generating more lift
base_CL_required = LIFT_REQ / (0.5 * 1.225 * VELOCITY**2 * base_area)

# Check if base config can produce the required lift
if base_CL_required > CRUISE_CL_MAX:
    print(f"  [!] This geometry CANNOT produce {LIFT_REQ:.1f} N (needs CL={base_CL_required:.3f} > limit {CRUISE_CL_MAX:.3f})")
elif base_aero['CL'] < base_CL_required:
    print(f"  L/D for comparison: {base_LD:.2f}")  # Actual L/D of the wing (Checking if L/D is not artificially inflated)
else:
    eff_L_D = base_CL_required / base_aero["CD"]
    print(f"  Effective L/D for comparison: {eff_L_D:.2f}")  # Since this is fixed lift analysis, We want to see the drag force reduce


print("\n[INFO] Moving to Phase 2: Optimization...")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PHASE 2 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
# Define the variables we want IPOPT to tweak, and set up constraints
# (like making sure the wing actually generates enough target lift)

opti = asb.Opti()
dynamic_pressure = 0.5 * 1.225 * VELOCITY**2

# Safety check (calculates minimum required area for the given lift force and Cl bounds) 
# to avoid infeasible optimization problems - As a fix, it changes the bounds of max span to meet the requirement
S_min_required = LIFT_REQ / (dynamic_pressure * CRUISE_CL_MAX)
effective_max_root_chord = BOUND_MAX_ROOT_CHORD if OPTIMIZE_ROOT_CHORD else ROOT_CHORD
max_possible_area = 2 * effective_max_root_chord * BOUND_MAX_HALF_SPAN
if max_possible_area < S_min_required:
    required_half_span = (S_min_required / (2 * effective_max_root_chord)) * 1.1

    # Updates the max span
    print(f"[WARNING] Auto-expanding BOUND_MAX_HALF_SPAN → {required_half_span:.3f} m")
    BOUND_MAX_HALF_SPAN = required_half_span

# --- Defining Optimization Variables ---
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

# ---------------------------------- Other Variables that need to be calculated ----------------------------------
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
            airfoil=asb.Airfoil(AIRFOIL_NAME)
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

# 3. Tip chord manufacturability
opti.subject_to(opti_tip_chord >= 0.05)

# 4. Aspect ratio hard cap (structural reasons)
opti.subject_to(opti_AR <= MAX_ASPECT_RATIO)

# 5. Rect fraction
opti.subject_to(opti_rect_frac >= 0.5)  

# --- Objective function ---
objective = CD/CL  # minimises 1/(L/D)
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
            "CD": float(opti.debug.value(CD)),
            "L_D": float(opti.debug.value(CL / CD)),
            "Lift": float(opti.debug.value(current_lift)),
            "area": float(opti.debug.value(opti_area)),
            "AR": float(opti.debug.value(opti_AR)),
        }
        history.append(row)
    except Exception:
        pass


if BYPASS_OPTIMIZATION:
    print("\n[INFO] Bypassing Optimization. Reading from CSV directly...")
    if os.path.exists("wing_opti_history.csv"):
        df = pd.read_csv("wing_opti_history.csv")
    else:
        print("[ERROR] wing_opti_history.csv not found!")
        df = None
else:
    try:
        sol = opti.solve(
            max_iter=1000,
            verbose=False,  # Can turn off Verbose to keep terminal window clean (iterations still log in .csv file)
            callback=opti_callback,  # Logs iteration data
            options={
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
    
        # --- L/D improvement summary and comparison --- 
        opt_LD = res_CL / res_CD
        print(f"\n  {'COMPARISON':=^58}")
        print(f"  Baseline Effective L/D: {eff_L_D:.2f}")
        print(f"  Baseline lift: {base_lift:.1f} N vs target: {LIFT_REQ:.1f} N")
        print(f"  Optimized L/D (at target lift)   : {opt_LD:.2f}")
        if opt_LD > eff_L_D:
            print(f"  ✓ Optimizer IMPROVED L/D by {((opt_LD/eff_L_D)-1)*100:.2f}%")
        else:
            print(f"  ✗ Optimizer L/D is {((1-opt_LD/eff_L_D))*100:.2f}% lower than baseline")
        print(f"  {'':=^58}")
        
        # --- Write summary to .txt file ---
        with open("optimization_summary.txt", "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("   OPERATING POINT & AIRFOIL\n")
            f.write("=" * 60 + "\n")
            f.write(f"  Airfoil        : {AIRFOIL_NAME}\n")
            f.write(f"  Velocity       : {VELOCITY:.2f} m/s\n")
            f.write(f"  Target Lift    : {LIFT_REQ:.2f} N\n\n")
            f.write("=" * 60 + "\n")
            f.write("   BASELINE ANALYSIS RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"  Full Span      : {2 * INIT_HALF_SPAN:.3f} m\n")
            f.write(f"  Area           : {base_area:.4f} m²\n")
            f.write(f"  Aspect Ratio   : {base_AR:.2f}\n")
            f.write(f"  Alpha          : {INIT_ALPHA:.2f} °\n")
            f.write(f"  CL             : {base_aero['CL']:.4f}\n")
            f.write(f"  CD             : {base_aero['CD']:.6f}\n")
            f.write(f"  L/D Ratio      : {base_LD:.2f}\n")
            f.write(f"  Lift force     : {base_lift:.2f} N (Target: {LIFT_REQ:.2f} N)\n")
            f.write(f"  Drag Force     : {base_drag:.2f} N\n\n")

            if base_CL_required > CRUISE_CL_MAX:
                f.write(f"  [!] This geometry CANNOT produce {LIFT_REQ:.1f} N (needs CL={base_CL_required:.3f} > limit {CRUISE_CL_MAX:.3f})\n")
            elif base_aero['CL'] < base_CL_required:
                f.write(f"  L/D for comparison: {base_LD:.2f}\n")
            else:
                f.write(f"  Effective L/D for comparison: {eff_L_D:.2f}\n (Wing is overproducing lift to artificially inflate L/D)\n Calculate effective L/D using required lift force\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("   PHASE 2: OPTIMIZATION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"  Full Span      : {2 * res_half_span:.4f} m (Baseline: {2 * INIT_HALF_SPAN:.4f} m)\n")
            f.write(f"  Root Chord     : {res_root_chord:.4f} m (Optimized: {OPTIMIZE_ROOT_CHORD})\n")
            f.write(f"  Rect Span      : {res_rect_span:.4f} m (Baseline: {base_rect_span:.4f} m)\n")
            f.write(f"  Tapered Span   : {res_half_span - res_rect_span:.4f} m\n")
            f.write(f"  Tip Chord      : {res_tip_chord:.4f} m (Taper Ratio: {res_taper:.2f})\n")
            f.write(f"  Angle of Attack: {res_alpha:.2f} °\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Wing Area      : {res_area:.5f} m² (Baseline: {base_area:.5f} m²)\n")
            f.write(f"  Aspect Ratio   : {res_AR:.2f} (Max Allowed: {MAX_ASPECT_RATIO})\n")
            f.write("-" * 60 + "\n")
            f.write(f"  CL             : {res_CL:.4f}\n")
            f.write(f"  CD             : {res_CD:.6f}\n")
            f.write(f"  L / D Ratio    : {res_CL / res_CD:.2f}\n")
            f.write("=" * 60 + "\n")

            f.write(f"\n  {'COMPARISON':=^58}\n")
            f.write(f"  Baseline Effective L/D: {eff_L_D:.2f}\n")
            f.write(f"  Baseline lift: {base_lift:.1f} N vs target: {LIFT_REQ:.1f} N\n")
            f.write(f"  Optimized L/D (at target lift)   : {opt_LD:.2f}\n")
            if opt_LD > eff_L_D:
                f.write(f"  ✓ Optimizer IMPROVED L/D by {((opt_LD/eff_L_D)-1)*100:.2f}%\n")
            else:
                f.write(f"  ✗ Optimizer L/D is {((1-opt_LD/eff_L_D))*100:.2f}% lower than baseline\n")
            f.write(f"  {'':=^58}\n")
    
    except RuntimeError as e:
        print(f"\n[ERROR] Optimization failed: {e}")
    
    finally:
        if history:
            df = pd.DataFrame(history)
            df.to_csv("wing_opti_history.csv", index=False)
        else:
            df = None

# ============================================================
# 4. PLOTS
# ============================================================
if df is not None and not df.empty:
    if any(PLOT_FLAGS):
        import matplotlib.pyplot as plt
        
        iters = df["iteration"]
        
        # Plot 1: Baseline and Final Geometry Top View
        if PLOT_FLAGS[0]:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # quick helper to trace the wing perimeter
            def wing_pts(span, r_chord, rect_span, t_chord):
                y = [0, rect_span, span, span, rect_span, 0]
                x = [-r_chord/4, -r_chord/4, -t_chord/4, 3*t_chord/4, 3*r_chord/4, 3*r_chord/4]
                return x[::-1] + x, [-iy for iy in y[::-1]] + y
            
            # pull final dims from the dataframe
            f_span = df["half_span"].iloc[-1]
            f_root = df["root_chord"].iloc[-1]
            f_rect = df["rect_span"].iloc[-1]
            f_taper = df["taper_ratio"].iloc[-1]
            f_tip = f_root * f_taper
            
            # compute baseline dims
            b_rect = INIT_RECT_FRAC * INIT_HALF_SPAN
            b_tip = ROOT_CHORD * INIT_TAPER_RATIO
            b_taper_span = INIT_HALF_SPAN - b_rect
            b_area = 2 * (b_rect * ROOT_CHORD + 0.5 * (ROOT_CHORD + b_tip) * b_taper_span)
            b_AR = (2 * INIT_HALF_SPAN)**2 / b_area
            
            bx, by = wing_pts(INIT_HALF_SPAN, ROOT_CHORD, b_rect, b_tip)
            ox, oy = wing_pts(f_span, f_root, f_rect, f_tip)
            
            ax.plot(by, bx, '--', color='gray', label="Baseline", lw=2)
            ax.plot(oy, ox, '-', color='blue', label="Optimized", lw=2)
            ax.fill(oy, ox, alpha=0.1, color='blue')
            
            # text boxes on the plot
            b_text = f"""Baseline Dimensions:
Span: {2*INIT_HALF_SPAN:.2f} m
Root Chord: {ROOT_CHORD:.3f} m
Tip Chord: {b_tip:.3f} m
Taper Ratio: {INIT_TAPER_RATIO:.2f}
Area: {b_area:.2f} m²
Aspect Ratio: {b_AR:.2f}"""
            
            o_text = f"""Optimized Dimensions:
Span: {2*f_span:.2f} m
Root Chord: {f_root:.3f} m
Tip Chord: {f_tip:.3f} m
Taper Ratio: {f_taper:.2f}
Area: {df['area'].iloc[-1]:.2f} m²
Aspect Ratio: {df['AR'].iloc[-1]:.2f}"""

            ax.text(0.05, 0.95, b_text, transform=ax.transAxes, va='top', 
                    bbox=dict(facecolor='white', alpha=0.8, ec='gray'), fontsize=12)
            ax.text(0.05, 0.05, o_text, transform=ax.transAxes, va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, ec='gray'), fontsize=12)
            
            ax.set_xlabel("Spanwise Position y (m)", fontsize=14)
            ax.set_ylabel("Chordwise Position x (m)", fontsize=14)
            ax.set_title("Compound Wing Planform: Baseline vs Optimized", fontsize=16)
            ax.grid(True, ls='--', alpha=0.7)
            ax.legend(fontsize=12)
            ax.axis('equal')
            plt.tight_layout()
            
        # Plot 2: L/D history
        if PLOT_FLAGS[1]:
            plt.figure(figsize=(8, 6))
            plt.plot(iters, df["L_D"], lw=2, color='green', label="L/D Ratio")
            plt.xlabel("Iteration", fontsize=14)
            plt.ylabel("L/D Ratio", fontsize=14)
            plt.title("L/D Ratio Optimization History", fontsize=16)
            plt.grid(True, ls='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
        # Plot 3 & 4: Forces
        if any(PLOT_FLAGS[2:4]):
            n_plots = sum(PLOT_FLAGS[2:4])
            fig, axs = plt.subplots(n_plots, 1, figsize=(8, 5 * n_plots))
            fig.suptitle("Forces History", fontsize=16)
            
            # make it iterable if there's only 1 subplot
            if n_plots == 1: 
                axs = [axs]
            
            ax_idx = 0
            if PLOT_FLAGS[2]:
                axs[ax_idx].plot(iters, df["Lift"], lw=2, color='blue', label="Lift Force (N)")
                axs[ax_idx].axhline(LIFT_REQ, color='red', ls='--', lw=2, label=f"Target Lift ({LIFT_REQ:.1f} N)")
                axs[ax_idx].set_ylabel("Lift Force (N)", fontsize=14)
                ax_idx += 1
                
            if PLOT_FLAGS[3]:
                drag = df["Lift"] / df["L_D"]
                axs[ax_idx].plot(iters, drag, lw=2, color='red', label="Drag Force (N)")
                axs[ax_idx].set_ylabel("Drag Force (N)", fontsize=14)

            for ax in axs:
                ax.set_xlabel("Iteration", fontsize=14)
                ax.grid(True, ls='--', alpha=0.7)
                ax.legend(fontsize=12)
                
            plt.tight_layout()
            
        # Plot 5 & 6: Force Coefficients
        if any(PLOT_FLAGS[4:6]):
            n_plots = sum(PLOT_FLAGS[4:6])
            fig, axs = plt.subplots(n_plots, 1, figsize=(8, 5 * n_plots))
            fig.suptitle("Force Coefficients History", fontsize=16)
            
            if n_plots == 1: 
                axs = [axs]
            
            ax_idx = 0
            if PLOT_FLAGS[4]:
                axs[ax_idx].plot(iters, df["CL"], lw=2, color='purple', label="CL")
                axs[ax_idx].set_ylabel("Lift Coefficient (CL)", fontsize=14)
                ax_idx += 1
                
            if PLOT_FLAGS[5]:
                axs[ax_idx].plot(iters, df["CD"], lw=2, color='orange', label="CD")
                axs[ax_idx].set_ylabel("Drag Coefficient (CD)", fontsize=14)
                
            for ax in axs:
                ax.set_xlabel("Iteration", fontsize=14)
                ax.grid(True, ls='--', alpha=0.7)
                ax.legend(fontsize=12)
                
            disclaimer = "Disclaimer: Coefficient values can be misleading as the wing area changes per iteration. Check total forces for accurate physical representation."
            fig.text(0.5, 0.02, disclaimer, ha='center', va='bottom', 
                     fontsize=10, style='italic', color='dimgray', wrap=True)
            fig.tight_layout(rect=[0, 0.08, 1, 1])
            
        plt.show()
