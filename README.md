# ✈️ Compound Wing Optimizer


A robust aerodynamic optimization tool for designing efficient **compound wings** (wings with a straight root section that tapers off towards the tip). The goal of this script is to automatically find the wing geometry that gives the best Lift-to-Drag (L/D) ratio for a given cruise condition, without violating payload or structural constraints.

Built on top of [AeroSandbox][def].\
*Huge thanks to Peter Sharpe for this wonderful open-source utility for us to tinker around with :)*

## 🌟 Overview

Finding the perfect wing geometry means balancing area, span, stall speed, and drag. Doing this manually is tedious because of the sheer number of variables at work and their interdependencies. This project automates the process by letting an optimizer tweak the wing's half-span, root chord, taper ratio, rectangular fraction, and angle of attack simultaneously to hunt down the design with the lowest possible drag that still meets the target lift required for cruising

## 🚀 Key Technical Features

- **3D Nonlinear Lifting Line Theory**: Uses an LLT solver to accurately calculate spanwise lift distributions and induced drag, giving realistic 3D aerodynamic results without the massive computational cost of full CFD.
- **Neural Network Airfoil Polars**: Speeds up 2D aerodynamics by using **NeuralFoil** to instantly predict lift and drag coefficients across different attack angles and Reynolds numbers.
- **Gradient-Based Optimization**: Uses **IPOPT** (via CasADI, inbuilt in Aerosandbox) to quickly navigate the complex tradeoff space and converge on the best design.
- **Built-in Sanity Checks**: The script catches obvious user errors early (e.g. asking for mathematically impossible lift given the bounds) and automatically loosens constraints to prevent the solver from crashing outright.


## How to Run

1. **Install Dependencies**: Ensure you have Python installed, then install the required libraries:
   ```python
   pip install aerosandbox[full] pandas
   ```
   *(Note: `[full]` includes visualization and neural network dependencies needed for this project to run neuralfoil solvers.)*

2. **Configure Design Variables**: Open `wing_opti.py` and modify the parameters in the **USER CONFIGURATION** section near the beginning of the code to match your aircraft's requirements:
   
   ```python
   AIRFOIL_NAME = "e387"
   MTOW = 15
   VELOCITY = 35
   # ... [along with their corresponding bounds]
   ```
   *(Note: Feel Free to branch out and create new optimization variables and constraints)*

3. **Run the Optimizer**: Execute the main script from your terminal:
   ```bash
   python wing_opti.py
   ```

4. **Check Results**: 
   - **Console Output**: Real-time optimization progress and final wing parameters are printed directly to the terminal.
   - **CSV Export**: The complete iteration history is saved to `wing_opti_history.csv`.
   - **Text Summary**: A neat text file called `optimization_summary.txt` is created with a breakdown of your baseline vs optimized wing.
   - **Visual Plots**: The script automatically displays several graphs comparing the before and after, as well as the optimization history.

## 📊 What You Get (Demo Output)

When you run the script with its default test case values, it produces these helpful visualizations and files to make everything easy to digest:

1. **`wing_opti_history.csv`**: A spreadsheet of every step the optimizer took.
2. **`optimization_summary.txt`**: A clean, text-based report comparing the initial and final wing designs.
3. **Visual Plots**: A pop-up showing different graphs of the optimization journey:
   - **Planform Comparison**: See exactly how the physical shape of the wing changed.
   - **L/D History**: Watch the Lift-to-Drag ratio climb as the optimizer does its magic.
   - **Lift and Drag Forces**: Track how forces balance out to meet your requirements.
   - **CL and CD History**: See the evolution of aerodynamic coefficients.

   ![Planform Comparison](Baseline_vs_optimal%20planform.png)

   ![L/D History](L_D_optimization_history.png)

   ![Lift and Drag](Lift_and_Drag_Forces.png)

   ![CL and CD](CL_and_CD_history.png)

## 🛠️ How to Debug

If the optimizer fails to find a valid solution or returns unexpected results, consider the following troubleshooting steps:

- **Check Convergence**: Look at the IPOPT output in the console. If it says "Infeasible," your bounds (e.g., `BOUND_MIN_ALPHA`, `BOUND_MAX_HALF_SPAN`, etc.) might be too restrictive for the `LIFT_REQ`.  
  Cross Verify the values once and try changing airfoils or toggling optimization variables for a wider design space or to avoid too many variables. 
- **Inspect Intermediate Values**: The script uses `opti.debug.value()` in the callback to log values even if a solve fails. Check the `wing_opti_history.csv` to see where the optimizer was "stuck."
- **Visualization**: To see the wing geometry, you can add `base_airplane.draw()` or `opt_airplane.draw()` after they are defined (requires `plotly` or `matplotlib`).
- **Sanity Check Logs**: Pay attention to the Phase 1 in the code and the [WARNING] messages regarding auto-expanding bounds; these often indicate why the design space is problematic.

## 🗺️ Roadmap & Future Work

- **Structural Tradeoffs**: Add structural weight and root bending moments to the objective function to find the sweet spot between aerodynamic efficiency and lightweight construction.
- **Dihedral & Sweep**: Allow the optimizer to tweak geometric twist, sweep, and dihedral for lateral stability.
- **Airfoil Database Search**: Build a wrapper to automatically test out hundreds of airfoils from the UIUC database to find the best 2D cross-section for the 3D planform.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [AeroSandbox][def] - Core Aerodynamics and Optimization framework

## Contact

For questions or support, please open an issue or contact me at tahilramaniparas@gmail.com


[def]: https://github.com/peterdsharpe/AeroSandbox