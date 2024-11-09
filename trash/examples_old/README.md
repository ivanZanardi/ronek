# RONEK

**Reduced Order Modeling for Non-Equilibrium Kinetics**

---

### Examples

The available examples include:
- `RVC_O3`: Rovibrational collisional model for the O2 + O system at T = 10,000 K.
- `VC_O3_O4`: Vibrational collisional model for the O2 + O and O2 + O2 systems at T = 10,000 K.

Running the examples:

1. Navigate to the inputs folder of the corresponding example (e.g., `RVC_O3/inputs`).
2. Modify the input files (.json) in each subfolder as needed:
   - `gen_data`: Generates the data for testing.
   - `max_mom_2`: Constructs the Petrov-Galerkin model considering only the first 2 moments of the O$_2$ distribution function.
   - `max_mom_10`: Constructs the Petrov-Galerkin model considering the first 10 moments of the O$_2$ distribution function.
   
   Ensure that you update the paths in each input file and in the `allrun.sh` scripts located in each subfolder.
3. Once the appropriate Conda environment is activated, run the command `bash allrun.sh` in the main folder.
