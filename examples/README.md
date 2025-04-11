# RONEK

**Reduced Order Modeling for Non-Equilibrium Kinetics**

---

### Examples

The repository includes the following example cases:

- **`RVC_N3`**: Rovibrational collisional model for the N<sub>2</sub> + N system.
- **`RVC_O3`**: Rovibrational collisional model for the O<sub>2</sub> + O system.
- **`VC_O3_O4`**: Vibrational collisional model for the O<sub>2</sub> + O and O<sub>2</sub> + O<sub>2</sub> systems at T = 10\,000 K.

#### Running the Examples

1. **Navigate** to the `inputs` folder of the desired example (`<example>/inputs`).

2. **Edit Input Files**:
  Each subfolder contains a `.json` input file for a specific model setup:
    - `gen_data`: Generates data for testing.
    - `max_mom_2`: Builds a Petrov-Galerkin model using the first 2 moments of the molecule internal distribution function.
    - `max_mom_10`: Builds a Petrov-Galerkin model using the first 10 moments of the molecule internal distribution function.
  Make sure to update the file paths within each `.json` input and the `allrun.sh` scripts.

3. **Run the Model**:
  After activating the appropriate Conda environment, run:
    ```bash
    bash allrun.sh
    ```
    from the main directory of the selected example.

### Data Availability

The kinetic database (`<example>/database/kinetics.hdf5`) required to run these examples will be made available upon reasonable request.
