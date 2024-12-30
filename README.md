# partial_electrical_discharges-classification-quantum
Creation of electric discharge classifiers with quantum circuits



# Step A: Running the "binary_features_generation_git_hub.ipynb" notebook for feature generation

This notebook is responsible for generating the transformed images and obtaining the thirteen attributes (features) described in **Section 2: Image processing and feature extraction method** of the article. The process is based on feature extraction from electrical discharge images, which are stored in the `/IMAGES` directory and its respective subdirectories: `/IMAGES/ARC`, `/IMAGES/BREAK`, `/IMAGES/NO_PD`, and `/IMAGES/PD`.

**A1. Image visualization:**

The first cell of the notebook allows you to visualize the original images obtained with the sensor. To do this, simply modify the `image_path` variable with the path to the image you want to view. For example:

```python
image_path = '/IMAGES/BREAK/29.65 s.png'
A2. Feature extraction:
The next cell extracts the features of all images contained in a specific directory, defined by the variable image_directory. For example, to process the images in the /IMAGES/BREAK/ directory, you must set:
      image_directory = '/IMAGES/BREAK/'
    
For each image within the selected directory, the following thirteen attributes are extracted:
      X = (area_pixels, centroid_x, centroid_y, centroid_x_roi, centroid_y_roi, mean_coords_x, mean_coords_y, lado_px, image_width, image_height, mean_intensity, std_intensity, threshold, class)
    

Where the last field, class, is an integer that represents the class to which the image belongs (e.g., 0 for ARC, 1 for BREAK, 2 for NO_PD, 3 for PD).
After executing this cell, a CSV file named feature_BREAK.csv (for the /IMAGES/BREAK/ example) is generated in the /FEATURE_RESULTS/ directory.
This process must be repeated for each of the class directories: /IMAGES/NO_PD, /IMAGES/PD, and /IMAGES/ARC. This will generate the files feature_NO_PD.csv, feature_PD.csv, and feature_ARC.csv, respectively, in the /FEATURE_RESULTS/ directory.
A3. Concatenation of files for binary classification:
The last cell of the notebook concatenates the CSV files generated in the previous step, two by two, to create the necessary datasets for binary classification. These concatenated files are also saved in the /FEATURE_RESULTS/ directory and are the ones used in the training process with the two quantum methods: Quantum Variational (QVM) and Quantum Kernel Machine Learning (QKM).


The generated files are:

FEATURE_resultante_DP_NODP.csv     FEATURE_resultante_DP_BREAK.csv
FEATURE_resultante_DP_ARC.csv  FEATURE_resultante_BREAK_NOPD.csv

Note: The binary_features_generation_git_hub.ipynb notebook is self-explanatory and contains detailed comments in each cell to facilitate its understanding.




# Step B: Running the "FIT_DP_NODP_CIRCUIT_Github.ipynb" notebook to adjust the parameters of the Quantum Variational (QVM) model

This notebook is used to find the optimal parameters of the first **Quantum Variational (QVM)** model, the results of which are presented in **Table 1** of the article (**Optimal parameters for the QVM circuits with 10 and 11 parameters**). Specifically, this notebook focuses on optimizing the **10-parameter** circuit corresponding to the **PD_NOPD** column of that table.

B1 **Required Libraries:**

To run this notebook, you need to import the following libraries:

*   `qiskit_ibm_provider`
*   `qiskit`
*   `qiskit_algorithms`
*   `qiskit-ibm-runtime`
*   `qiskit-machine-learning[sparse]`
*   `qiskit-aer`

B2 **Main Functions:**

The notebook defines two main functions:

*   `ZZFeatureMap_10_parametros()`: Implements the 10-parameter variational quantum circuit shown in **Figure 6** of the article.
*   `ZZFeatureMap_11_parametros_MODI()`: Implements the 11-parameter variational quantum circuit shown in **Figure 7** of the article. Although this function is not used directly in this notebook, it is included for reference, as it is used to optimize the parameters of the other binary combinations (BREAK_NOPD, PD_ARC, and BREAK_ARC) presented in **Table 1**.

B3**Optimization Process:**

1. **Data Loading and Normalization:**
    *   The `normalizeData()` function loads the data from the `FEATURE_resultante_DP_NODP.csv` file (generated in **Step A**).
    *   The data is normalized to the range `[-0 * np.pi, 2 * np.pi]` using `MinMaxScaler` from `sklearn.preprocessing`.
    *   The original classes are renamed to 0 and 1 using `np.where()`. In this case, the original class 2 is renamed to 0.

2. **Quantum Circuit Definition:**
    *   The `ZZFeatureMap_10_parametros()` function is used to create an instance of the quantum circuit with 10 parameters.
    *   `x = ParameterVector('x', 2)`: represent the two input features, 'area_pixels' and 'mean_coords_x'.
    *   `teta = ParameterVector('teta', 10)`: represents the 10 variational parameters of the circuit.

3. **Cost Function Definition:**
    *   `classification_probability()`: Calculates the probability that the circuit correctly classifies a data point.
    *   `parity()`: Calculates the parity of a bitstring (0 or 1).
    *   `label_probability()`: Converts a dictionary of bitstrings and their counts to parity probabilities.
    *   `cross_entropy_loss()`: Calculates the cross-entropy loss between the predicted classification and the actual label.
    *   `cost_function()`: Defines the cost function to be minimized during optimization.

4. **Parameter Optimization:**
    *   The `COBYLA` optimizer from `qiskit_algorithms.optimizers` is used to minimize the cost function and find the optimal values of the 10 parameters.
    *   `initial_point`: Sets the initial values of the parameters for the optimizer. In this case an initial_point is provided, which is the end point of the previous optimization.
    *   The `log` variable (of type `OptimizerLog`) records the progress of the optimization, storing the cost function values and parameters at each iteration.
    *   `result`: contains the output of the optimizer, including the optimized parameters `result.x`.

5. **Model Evaluation:**
    *   Finally, the performance of the optimized model is evaluated using the `comparar_probabilidades()` function, which calculates the classification accuracy by comparing the predicted probabilities with the actual labels for both the training set (TRAIN) and the test set (TEST).

B4 **Execution and Results:**

When you run the notebook, the evolution of the cost function during optimization will be displayed, and the optimal values of the 10 parameters will be obtained, which correspond to the values presented in the **PD_NOPD** column of **Table 1** in the article. These parameters are stored in the `opt_var` variable. In addition, the accuracy of the model on the training and test sets is calculated and displayed.



**Note:** The notebook contains detailed comments in each cell to facilitate its understanding. The notebook FIT_BREAK_NOPD_CIRCUIT_GITHUB.ipynb is analogous to this binary combination(BREAK_NOPD).
The optimal parameters obtained are subsequently used in the **Step C** notebook to perform the classification on real quantum computers.


# Step C: Running the QVM model on real quantum computers or simulation with errors (Verification)

This notebook (`QVM_verification_two_qubits.ipynb`) is used to verify the performance of the **Quantum Variational (QVM)** model with the optimized parameters obtained in **Step B**, either on real IBM Quantum computers or through simulation with realistic noise models.

**Objective:**

The main objective of this step is to validate the results obtained during optimization in a real or simulated quantum hardware environment, taking into account the limitations and noise present in current devices.

**Workflow:**

1. **Environment Setup:**
    *   The necessary libraries are imported, including:
        *   `qiskit_ibm_runtime`: To interact with IBM Quantum services.
        *   `qiskit`: For the construction and manipulation of quantum circuits.
        *   `qiskit_algorithms`: For quantum algorithms.
        *   `qiskit_aer`: For local simulation.
        *   `pennylane`: For the calculation of certain metrics (optional, can be removed if not used directly).
        *   `matplotlib.pyplot`: For visualization of results.
        *   `time`: To measure execution time.
        *   `sklearn`: For data preprocessing and evaluation metrics.
        *   `scipy`: For scientific functions (optional).
        *   `numpy`: For numerical operations.
        *   `pandas`: For data handling.
    *   The IBM Quantum account credentials previously saved with `QiskitRuntimeService.save_account()` are loaded.
    *   The backend to be used is defined:
        *   For execution on real hardware:
           ```python
            service = QiskitRuntimeService()
            backend = service.least_busy(operational=True, simulator=False)
            ```
            This selects the least busy real IBM quantum computer at that moment.
        *   For simulation with a noise model:
            ```python
            from qiskit_ibm_runtime.fake_provider import FakeOsaka
            backend = FakeOsaka()
            ```
            This uses a simulator that emulates the behavior and errors of the real `ibmq_osaka` quantum computer.

        *   For simulation without errors
            ```python
            backend = Aer.get_backend('statevector_simulator')
            ```

2. **Data and Parameter Loading:**
    *   The `normalizeData()` function is called to load and normalize the data (same as in **Step B**).
    *   The optimized parameters `opt_var` obtained in **Step B** are loaded.

3. **Circuit Definition:**
    *   The `ZZFeatureMap_10_parametros()` function, which constructs the 10-parameter variational quantum circuit, is defined (same as in **Step B**).

4. **Circuit Transpilation:**
    *   A function `VARIA_circuitos_DP_NODP` is created, which:
        *   Takes a data point and the optimized parameters `opt_var` as input.
        *   Creates an instance of the `ZZFeatureMap_10_parametros` circuit with the provided data and parameters.
        *   Uses `generate_preset_pass_manager` to transpile the circuit for the selected backend (optimization for specific hardware).
        *   Adds the transpiled circuit to a global list `circuits_aa`.

5. **Circuit Execution:**
    *   An instance of `SamplerV2` (or `Sampler` for older versions of Qiskit Runtime) is created to execute the circuits on the selected backend.
    *   `VARIA_circuitos_DP_NODP` is called in a `for` loop for each data point in the test set, generating all the transpiled circuits and storing them in `circuits_aa`.
    *   `n_iterations = 136`: Number of circuits to execute. In this case, it corresponds to the number of elements in the TEST_DATA list, which are the test images.
    *   The job is executed on the backend using `sampler.run(circuits_aa)`.
    *   The results are obtained with `job.result()`.
    *   The job_id of the executed job is saved.
    *   The results are saved in a text file for further analysis.

6. **Results Analysis:**
    *   The counts from the results for each executed circuit are extracted.
    *   The parity (0 or 1) is calculated for each circuit based on the counts. In this case, tag (label) zero is taken if the parity is odd, and one if the parity is even.
    *   The classification accuracy is calculated by comparing the predicted parity with the actual labels of the test set.
    *   The results are printed, and a bar chart comparing the actual and predicted labels is generated.

**Improvements in the graph:**

The notebook includes code to generate a graph that compares the actual labels (`TRAIN_LABELS`) with the predicted labels (`paridad`). The following improvements have been made:

*   Figure size increased to `figsize=(12, 8)`.
*   Font size of the X and Y axis labels increased to 14 and bold.
*   Font size of the title increased to 16 and bold.
*   Legend font size increased to 14.
*   Marker size increased.
*   Y-axis ticks adjusted to 0 and 1.
*   Grid added for better visualization.
*   Font size of the numbers on the X and Y axes increased to 14.
*   Spacing adjusted with `plt.tight_layout()`.
*   Accuracy is displayed in the legend title.

**Conclusion:**

This notebook allows verifying the performance of the optimized QVM model in a real or simulated quantum hardware environment, providing a more realistic evaluation of its generalization ability. The results obtained in this step are presented in **Table 3** of the article, in the row corresponding to the `PD_NOPD` combination, for the `IBM_Kyoto`, `IBM_Brisbane`, and `IBM_Osaka` computers, along with the ideal simulation results (`Accuracy simulation`).

**Note:** This notebook is also self-explanatory and contains detailed comments in each cell to facilitate its understanding.





Step D: Running the Quantum Kernel Machine Learning (QKM).model on real quantum computers .




















