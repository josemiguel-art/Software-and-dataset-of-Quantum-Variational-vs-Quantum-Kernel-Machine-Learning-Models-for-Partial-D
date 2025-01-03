#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install qiskit-ibm-runtime')


# In[2]:


get_ipython().system('pip install qiskit')


# In[3]:


pip install qiskit_algorithms


# In[4]:


get_ipython().system('pip install qiskit-aer')


# In[5]:


get_ipython().system('pip install pylatexenc')


# In[ ]:


from qiskit_aer import AerSimulator,Aer# ojo son distintos AerSimulator y Aer  AerSimulator es para entrar en estadisticas de IBM
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import svm
import scipy


# In[6]:


get_ipython().system('pip install pennylane')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")
# constants
n = 2
RANDOM_STATE = 42
LR = 1e-3
class_labels = ['0', '1']

def normalizeData(DATA_PATH = "./FEATURE_RESULTS/FEATURE_resultante_DP_ruptura.csv"):
    """
    Normalizes the data
    """
    # Reads the data
    data = pd.read_csv(DATA_PATH)
    data = shuffle(data, random_state=RANDOM_STATE)
    X, Y = data[['area_pixels', ' mean_coords_x']].values, data[' class'].values

    # normalize the data
    scaler = MinMaxScaler(feature_range=(-0 * np.pi, 2 * np.pi))
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
    return X_train, X_test, Y_train, Y_test





import qiskit_algorithms
from qiskit_algorithms.optimizers import SPSA

from qiskit import QuantumCircuit

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import Statevector

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")

TRAIN_DATA, TEST_DATA, TRAIN_LABELS, TEST_LABELS = normalizeData()
# Replace all occurrences of 2 with 1
TRAIN_LABELS = np.where(TRAIN_LABELS == 2, 1, TRAIN_LABELS)
TEST_LABELS = np.where(TEST_LABELS == 2, 1, TEST_LABELS)
#print(TRAIN_DATA)


# In[9]:


# 

from qiskit_aer import AerSimulator,Aer # eye,  are different AerSimulator and Aer AerSimulator is to enter IBM statistics
#import pennylane as qml
#from pennylane import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import svm
import scipy


# In[41]:


print(TRAIN_LABELS)
print(f"Dimensiones de TRAIN_LABELS: {TRAIN_LABELS.shape}")


# In[11]:


# FIRST METHOD FOR CALCULATING THE KERNEL WITH STATE VECTOR
import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector


# In[12]:


def feature_map(circuit, x):
    # ZZMap modificado
    for i in range(2):
        circuit.h(i)
    circuit.rz(2 * x[0], 0)
    circuit.rz(2 * x[1], 1)
    circuit.cx(0, 1)
    circuit.rz(2 * (np.pi - x[0]) * (np.pi - x[1]), 1)
    circuit.cx(0, 1)

def inverse_feature_map(circuit, x):
    # Invertir ZZMap
    circuit.cx(0, 1)
    circuit.rz(-2 * (np.pi - x[0]) * (np.pi - x[1]), 1)
    circuit.cx(0, 1)
    circuit.rz(-2 * x[1], 1)
    circuit.rz(-2 * x[0], 0)
    for i in range(2):
        circuit.h(i)

def scalar_product1(circuit, x, y):
    feature_map(circuit, y)
    inverse_feature_map(circuit, x)

    backend = Aer.get_backend('statevector_simulator')
    t_qc = transpile(circuit, backend)
    qobj = assemble(t_qc)
    result = backend.run(qobj).result()
    statevector = result.get_statevector(circuit)
    probs = np.abs(statevector)**2
    return probs

def scalar_product(circuit, x, y):
    feature_map(circuit, y)
    inverse_feature_map(circuit, x)
    #backend = AerSimulator('statevector_simulator')
    backend = Aer.get_backend('statevector_simulator')
    t_qc = transpile(circuit, backend)
    qobj = assemble(t_qc)
    result = backend.run(qobj).result()
    statevector = result.get_statevector(circuit)
    probs = np.abs(statevector)**2
    return probs




def kernel_gram_matrix_full(X1, X2):
    print("Calculating Gram matrix")

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        print(int(i / len(X1) * 100), "%")
        for j, x2 in enumerate(X2):
            qc = QuantumCircuit(2, 2)
            x1 = x1.flatten()
            x2 = x2.flatten()
            prob = scalar_product(qc, x1, x2)[0]  #Use only the probability of state '00'
            gram_matrix[i, j] = prob

    return gram_matrix


# In[13]:


def scalar_product(circuit, x, y):
    feature_map(circuit, y)
    inverse_feature_map(circuit, x)
    #backend = AerSimulator('statevector_simulator')
    backend = Aer.get_backend('statevector_simulator')
    t_qc = transpile(circuit, backend)
    qobj = assemble(t_qc)
    result = backend.run(qobj).result()
    statevector = result.get_statevector(circuit)
    probs = np.abs(statevector)**2
    return probs


# In[ ]:





# In[ ]:


#If the matrix is ​​square, this is easier.
# the dimension of x1 and x2 are equal


def kernel_gram_matrix_full_symmetric(X1, X2):
    print("Calculando matriz de Gram")

    num_samples_X1 = X1.shape[0]


    gram_matrix = np.zeros((num_samples_X1, num_samples_X1))  # Make sure you have the correct size

    # Calculate only the upper half of the matrix
    for i in range(num_samples_X1):
        print(int(i / num_samples_X1 * 100), "%")
        for j in range(i, num_samples_X1):  # Only calculate the top half
            qc = QuantumCircuit(2, 2)
            x1 = X1[i].flatten()
            x2 = X2[j].flatten()

            prob = scalar_product(qc, x1, x2)[0]
            gram_matrix[i, j] = prob
            gram_matrix[j, i] = prob  # Symmetry: assigning the same value to the lower half

    return gram_matrix
x_train=TRAIN_DATA[:]
import time
inicio_tiempo = time.time()
matrix_simetrica = kernel_gram_matrix_full_symmetric(x_train, x_train)
fin=time.time()
tiempo=inicio_tiempo-fin
print(tiempo)


# In[ ]:


import numpy as np



# Save the matrix to a .npy file
np.save('./FEATURE_RESULTS/QKM_RESULTS_TWO_QUBITS/simulation_TRAIN_kernel_matrix_DP_ruptura.npy', matrix_simetrica)


# In[ ]:


"""
 
This step is repeated for the other three combinations giving the results
within the directory /FEATURE_RESULTS .

With this simulation method
backend = Aer.get_backend('statevector_simulator')

simulation_TRAIN_kernel_matrix_DP_NODP.npy
simulation_TRAIN_kernel_matrix_DP_RAYOMAS.npy
simulation_TRAIN_kernel_matrix_DP_ruptura.npy
simulation_TRAIN_kernel_matrix_ruptura_NODP.npy


"""



# In[1]:


import numpy as np
matrix_simetrica=[]
# Supongamos que 'matrix' es la matriz del kernel que has calculado
#matrix = kernel_gram_matrix_full(x_train, x_train)

# load la matriz en un archivo .npy
matrix_simetrica=np.load('./FEATURE_RESULTS/QKM_RESULTS_TWO_QUBITS/simulation_TRAIN_kernel_matrix_DP_ruptura.npy', matrix_simetrica)


# In[ ]:


import matplotlib.pyplot as plt

def visualize_kernel_matrix(kernel_matrix):
    plt.imshow(kernel_matrix, cmap='viridis', interpolation='none', origin='lower', extent=[0, len(kernel_matrix), 0, len(kernel_matrix)])
    cbar = plt.colorbar(label='kernel matrix value')
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    cbar.set_label('kernel matrix value', weight='bold', fontsize=14)
    plt.title('Kernel Matrix Visualization', fontweight='bold', fontsize=16)
    plt.xlabel('Data Index (X1)', fontweight='bold', fontsize=14)
    plt.ylabel('Data Index (X2)', fontweight='bold', fontsize=14)

    # Aumentar el tamaño y poner en negrita los números de los ejes
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    plt.show()

# Supongamos que 'matrix' es la matriz del kernel que has calculado
# (asegúrate de que tiene valores entre 0 y 1)
# matrix = kernel_gram_matrix_full(x_train, x_train)

# Visualizar la matriz del kernel
visualize_kernel_matrix(matrix_simetrica)


# In[ ]:


matrix_simetrica[1,1]


# In[ ]:


"""


Brief explanation of the method:

"This code uses SVM (Support Vector Machine) for classification, specifically the implementation provided by scikit-learn in Python.

Here is a line-by-line explanation:

clf = svm.SVC(kernel="precomputed"): This creates an SVM classifier (SVC, which stands for Support Vector Classifier) with a precomputed kernel.
The kernel parameter specifies the type of kernel to use in the SVM algorithm.
A precomputed kernel means that the classifier expects a precomputed kernel matrix as input rather than the training data directly. 
This can be useful in situations where kernel computation is expensive and has already been performed outside the model training.

y_train = TRAIN_LABELS: This appears to be assigning the training labels to y_train. 
TRAIN_LABELS contains the labels corresponding to the training data used to train the model.

clf.fit(matrix, y_train): This line is fitting the SVM model to the training data. 
The matrix contains the precomputed kernel matrix (or the training data if a precomputed kernel is not used), 
and y_train contains the corresponding labels for this data. 
The fit() method adjusts the model to the provided data, which means it finds the optimal hyperplane 
that separates the different classes based on the training data and the associated labels. 
Once this process is complete, the model is ready to make predictions on new, unseen data."

"""


# In[43]:


import numpy as np


# Other matrices obtained for the other combination  DP_ruptura by the same procedure.
# Load matrix from .npy file
matrix_simetrica_cargada = np.load('./FEATURE_RESULTS/QKM_RESULTS_TWO_QUBITS/simulation_TRAIN_kernel_matrix_DP_ruptura.npy')

# Print the matrix to verify
print(matrix_simetrica_cargada)


# In[47]:


import time
from sklearn import svm
from time import time
# Assuming XTest is your test dataset and YLabels are the corresponding labels
# It is assumed that you already have the TRAIN_LABELS and matrix variables defined above

# Create an SVM classifier with the precomputed kernel


clf = svm.SVC(kernel="precomputed")

# Fit the model to the training data
"""
finds the optimal hyperplane that separates the different classes based on the training data and the associated labels.
The objective function of the SVM is convex, meaning there are no local minima.
"""

start_time = time()
clf.fit(matrix_simetrica, TRAIN_LABELS)
training_time = time() - start_time
print("Tiempo de entrenamiento:", training_time, "segundos")


# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# Suppose 'kernel_matrix' is the kernel matrix you computed 
# with state vector simulator

# Select row 40 of the matrix
row_40 = matrix_simetrica[39]  # Index 39 why lists in Python are 0-indexed

# Crear el diagrama de barras de la fila 40
plt.figure(figsize=(10, 6))
plt.bar(range(0, 25), row_40[0:25], color='b')
plt.title('Bar Chart of Row 40 of Kernel Matrix', fontweight='bold', fontsize=16)
plt.xlabel('Column Index', fontweight='bold', fontsize=14)
plt.ylabel('Value', fontweight='bold', fontsize=14)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(True)
plt.show()


# In[ ]:


#Or, optionally use the save_account() method to save your credentials for easy access later on, before initializing the service.
"""
This part is about estimating the kernel with a real quantum computer 
those used are IBM Osaka;IBM Kyoto;IBM Brisbane.
Calibration data for computers used in the experiments from IBM. in  Table 2
del articulo: 
Quantum Variational vs Quantum Kernel Machine Learning
Models for Partial Discharge Classification in Dielectric Oils
José Miguel Monzón-Verona , Santiago García-Alonso , and Francisco Jorge Santana-Martín
"""
from qiskit_ibm_runtime import QiskitRuntimeService

# Save an IBM Quantum account and set it as your default account.

# Save an IBM Quantum account and set it as your default account.
QiskitRuntimeService.save_account(channel="ibm_quantum", token="........8e5338f3752cf34eaa427........", overwrite=True, set_as_default=True)
# Load saved credentials
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
backend.name


# In[ ]:


from qiskit_ibm_runtime.fake_provider import FakeManilaV2,FakeBrisbane,FakeKyoto,FakeOsaka
backend =FakeOsaka()


# In[ ]:


from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2


# In[ ]:


from qiskit.circuit.library import ZZFeatureMap
zz_feature_map_reference = ZZFeatureMap(feature_dimension=2, reps=1)

zz_feature_map_reference.decompose().draw()


# In[ ]:





# In[ ]:


"""
In this part, the functions used to estimate the kernel that is represented in the
Figure 11. Generic structure of the quantum circuit and measurement used to estimate the kernel of
Equation (26). It is particularized for three features.
"""

counts= []
circuits_a = []
circuits_aa = []

def scalar_product_circuitos(x, y):
   
    zz_feature_map_reference1 = zz_feature_map_reference.assign_parameters(y)
    
    zz_feature_map_reference_inv = zz_feature_map_reference.assign_parameters(x)

    zzinv1=zz_feature_map_reference_inv.inverse()
    # Combine the circuits
    circuito_final = zz_feature_map_reference1.compose(zzinv1, qubits=[0, 1])
    # Add measurements for each qubit
    
    circuito_final.measure_all()
    circuits_a.append(circuito_final)

   
    # backend = FakeOsaka()  for simulation
    passmanager = generate_preset_pass_manager(optimization_level=3, backend=backend)
    
    transpiled_circuit = passmanager.run(circuito_final)
    circuits_aa.append(transpiled_circuit)


    return 1



def kernel_gram_matrix_circuitos(X1, X2):
    print("Calculando matriz de Gram")

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        print(int(i / len(X1) * 100), "%")
        for j, x2 in enumerate(X2):
            #qc = QuantumCircuit(3, 3)
            x1 = x1.flatten()
            x2 = x2.flatten()
            prob = scalar_product_circuitos(x1, x2)  # Use only the probability of state '00'
            gram_matrix[i, j] = prob

    return gram_matrix





# In[ ]:


# Transpile circuit for noisy basis gates
passmanager = generate_preset_pass_manager(optimization_level=3, backend=sim_noise)
circ_tnoise = passmanager.run(circ)



# In[ ]:


from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# Assuming you have already configured your Qiskit service and backend

# Initialize the sampler and run the circuits
sampler = Sampler(backend)
job = sampler.run(circuits_aa)
result = job.result()

# Iterate over the results to print and save the counts
with open('./FEATURE_RESULTS/QKM_RESULTS_TWO_QUBITS/DP_ruptura_IBM_kyoto__fila_kernel39_25_results_counts.txt', 'w') as f:
    for idx, pub_result in enumerate(result):
        counts = pub_result.data.meas.get_counts()
        f.write(f"Counts for pub {idx}: {counts}\n")
        print(f" > Counts for pub {idx}: {counts}")


# In[ ]:


import matplotlib.pyplot as plt


#Assuming you already have the results in the `result` variable
counts_data = []
for idx, pub_result in enumerate(result):
    counts = pub_result.data.meas.get_counts()
    counts_data.append(counts)
    print(f" > Counts for pub {idx}: {counts}")


#Extract the counts for '00' from each post and calculate the total sum of counts per post
counts_00 = []
for counts in counts_data:
    total_counts = sum(counts.values())
    count_00 = counts.get('00', 0)
    count_00_pu = count_00 / total_counts if total_counts != 0 else 0  # Evitar división por cero
    counts_00.append(count_00_pu)


#Print '00' counts in p.u. to verify
for idx, count in enumerate(counts_00):
    print(f"Counts of '00' in p.u. for pub {idx}: {count:.4f}")


# In[ ]:


# Create the bar chart
fig, ax = plt.subplots(figsize=(12, 8))

# Define the indexes and values ​​for the chart
pub_indices = range(len(counts_00))

# Create the bars
ax.bar(pub_indices, counts_00, color='blue')

# Add tags and title
ax.set_xlabel('Publicación')
ax.set_ylabel('Recuentos de 00 (p.u.)')
ax.set_title('Recuentos de 00 por publicación en p.u.')
ax.set_xticks(pub_indices)
ax.set_xticklabels([f'Pub {i}' for i in pub_indices], rotation=90)

# Mostrar el gráfico
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import ast

# Path to the saved counts data file
file_path = './FEATURE_RESULTS/QKM_RESULTS_TWO_QUBITS/DP_ruptura_IBM_Kyoto__fila_kernel39_25_results_counts.txt'

# Initialize an empty list to hold the counts data
counts_data = []

# Read the file and extract counts for each publication
with open(file_path, 'r') as f:
    for line in f:
        if line.startswith('Counts for pub'):
            parts = line.split(': ', 1)  # Split only at the first occurrence
            try:
                counts = ast.literal_eval(parts[1].strip())
                counts_data.append(counts)
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing line: {line.strip()}")
                print(e)

# Extract the counts for '00' from each publication and normalize them
counts_00 = []
for counts in counts_data:
    total_counts = sum(counts.values())
    counts_00.append(counts.get('00', 0) / total_counts if total_counts > 0 else 0)

# Print the normalized counts for '00' to verify
for idx, count in enumerate(counts_00):
    print(f"Normalized counts of '00' for pub {idx}: {count:.4f}")

# Assume 'matrix_simetrica' is the kernel matrix you have loaded


# Select row 40 from the matrix
row_40 = matrix_simetrica[39]  # Index 39 because Python lists are 0-indexed

# Ensure both lists have the same length
length = min(len(row_40[0:25]), len(counts_00))
row_40 = row_40[0:length]
counts_00 = counts_00[:length]

# Create the combined bar chart
fig, ax = plt.subplots(figsize=(12, 8))

# Define the indices for the plot
indices = range(length)

# Create the bars for row 40 of the kernel matrix
ax.bar(indices, row_40, color='b', alpha=0.6, label='Simulation')

# Create the bars for the normalized counts of 00
ax.bar(indices, counts_00, color='r', alpha=0.6, label='IBM Kyoto')

# Add labels and title in bold and increase font size
ax.set_xlabel('Column Index / Publication', fontweight='bold', fontsize=16)
ax.set_ylabel('Value / Counts of 00 ', fontweight='bold', fontsize=16)
ax.set_title('Comparison of Kernel Matrix Row 40 and IBM Kyoto', fontweight='bold', fontsize=16)

# Modify the x-axis tick labels
ax.set_xticks(indices)
ax.set_xticklabels([f'Pub {i}' for i in indices], rotation=90, fontsize=14, fontweight='bold')

# Modify the y-axis tick labels
plt.yticks(fontsize=14, fontweight='bold')


ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.02), fontsize=14)
# Show the plot
plt.show()


# In[ ]:


backend.name


# In[ ]:


print(countss)


# 

# In[3]:


print("Checking SVC with train...")
x_train=TRAIN_DATA
from sklearn import svm
import numpy as np
from time import time
# Load matrix from .npy file
K = np.load('./FEATURE_RESULTS/QKM_RESULTS_TWO_QUBITS/simulation_TRAIN_kernel_matrix_DP_ruptura.npy')
#K = np.load('./FEATURE_RESULTS/simulation_TRAIN_kernel_matrix_DP_NODP.npy')

clf = svm.SVC(kernel="precomputed")
clf.fit(K, TRAIN_LABELS)

#Record the start time
inicio_tiempo = time()
sol = clf.predict(K)

y_train=TRAIN_LABELS
success = 0


# Calculate the time difference
tiempo_ejecucion = time()- inicio_tiempo

print(f"Tiempo de ejecución: {tiempo_ejecucion} segundos")


# In[46]:


inicio_tiempo = time()
y_train=TRAIN_LABELS
success = 0
for i in range(len(y_train)):
    if sol[i] == y_train[i]:
        success += 1

print("Precisión del train: ", success/len(sol)*100, "%")

# Registra el tiempo de finalización
fin_tiempo = time()

# Calcula la diferencia de tiempo
tiempo_ejecucion = fin_tiempo - inicio_tiempo

print(f"Tiempo de ejecución de compro train: {tiempo_ejecucion} segundos")


# In[ ]:


print("Comprobando con test...")
x_test=TEST_DATA
# Registra el tiempo de inicio
inicio_tiempo = time.time()
sol = clf.predict(kernel_gram_matrix_full(x_test, x_train))

y_test=TEST_LABELS
success = 0


# Calcula la diferencia de tiempo
tiempo_ejecucion = time.time()- inicio_tiempo

print(f"Tiempo de ejecución: {tiempo_ejecucion} segundos")


# In[ ]:


y_test=TEST_LABELS
success = 0
for i in range(len(y_test)):
    if sol[i] == y_test[i]:
        success += 1

print("Precisión del train: ", success/len(sol)*100, "%")

# Registra el tiempo de finalización
fin_tiempo = time.time()

# Calcula la diferencia de tiempo
tiempo_ejecucion = fin_tiempo - inicio_tiempo

print(f"Tiempo de ejecución de compro test: {tiempo_ejecucion} segundos")


# In[ ]:


""" Reproducibility  Support vector machine, SVM  with quantum kernel estimation for 3 qubits

Table 5. Accuracy and execution times for three qubits Equation (28), with symmetric matrix,
test_size=80%.




"""






# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")
# constants
n = 2
RANDOM_STATE = 42
LR = 1e-3
class_labels = ['0', '1']

def normalizeData(DATA_PATH = "./FEATURE_RESULTS/FEATURE_resultante_DP_RAYOMAS.csv"):
    """
    Normalizes the data
    """
    # Reads the data for three qubits
    data = pd.read_csv(DATA_PATH)
    data = shuffle(data, random_state=RANDOM_STATE)
    X, Y = data[['area_pixels', ' mean_coords_x',' mean_coords_y']].values, data[' class'].values

    # normalize the data
    scaler = MinMaxScaler(feature_range=(-0 * np.pi, 2 * np.pi))
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=RANDOM_STATE)
    return X_train, X_test, Y_train, Y_test


# In[5]:


import qiskit_algorithms
from qiskit_algorithms.optimizers import SPSA

from qiskit import QuantumCircuit

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import Statevector

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")

TRAIN_DATA, TEST_DATA, TRAIN_LABELS, TEST_LABELS = normalizeData()
# Replace all occurrences of 2 with 1
TRAIN_LABELS = np.where(TRAIN_LABELS == 2, 1, TRAIN_LABELS)
TEST_LABELS = np.where(TEST_LABELS == 2, 1, TEST_LABELS)


# In[27]:


print(TRAIN_LABELS)
print(f"Dimensiones de TRAIN_LABELS: {TRAIN_LABELS.shape}")


# In[45]:


# FIRST METHOD FOR CALCULATING THE KERNEL WITH STATE VECTOR
import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector


# In[44]:


from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.circuit.library import ZZFeatureMap
zz_feature_map_reference = ZZFeatureMap(feature_dimension=3, reps=1)# 3 qubits


# In[17]:


counts= []
circuits_a = []
circuits_aa = []
#for multiple circuits
def scalar_product_circuitos(x, y):
    
    zz_feature_map_reference1 = zz_feature_map_reference.assign_parameters(y)
    
    zz_feature_map_reference_inv = zz_feature_map_reference.assign_parameters(x)

    zzinv1=zz_feature_map_reference_inv.inverse()
    # Combine the circuits
    circuito_final = zz_feature_map_reference1.compose(zzinv1, qubits=[0, 1,2])
    
    circuito_final.measure_all()
    circuits_a.append(circuito_final)

    """
    backend = Aer.get_backend('statevector_simulator')
    t_qc = transpile(circuito_final, backend)
    qobj = assemble(t_qc)
    result = backend.run(qobj).result()
    statevector = result.get_statevector(circuito_final)
    probs = np.abs(statevector)**2


"""
    #backend = FakeOsaka()
    passmanager = generate_preset_pass_manager(optimization_level=3, backend=backend)
    #circ_tnoise = passmanager.run(circ)
    #shots=52000
    transpiled_circuit = passmanager.run(circuito_final)
    circuits_aa.append(transpiled_circuit)
    return 1


# In[ ]:





# In[47]:


zz_feature_map_reference = ZZFeatureMap(feature_dimension=3, reps=1)# 3 qubits or n_qubits



def scalar_product_N_q(circuit, x, y):
    

    zz_feature_map_reference1 = zz_feature_map_reference.assign_parameters(y)
    
    zz_feature_map_reference_inv = zz_feature_map_reference.assign_parameters(x)

    zzinv1=zz_feature_map_reference_inv.inverse()
    #  Combine the circuits
    circuito_final = zz_feature_map_reference1.compose(zzinv1, qubits=[0, 1,2])  # for 3 qubits


    
    backend = Aer.get_backend('statevector_simulator')
    t_qc = transpile(circuito_final, backend)
    qobj = assemble(t_qc)
    result = backend.run(qobj).result()
    statevector = result.get_statevector(t_qc)
    probs = np.abs(statevector)**2
    return probs







def kernel_gram_matrix_full(X1, X2):
    print("Calculando matriz de Gram")

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        print(int(i / len(X1) * 100), "%")
        for j, x2 in enumerate(X2):
            qc = QuantumCircuit(3, 3)
            x1 = x1.flatten()
            x2 = x2.flatten()
            prob = scalar_product_N_q( qc,x1, x2)[0]  # Usar solo la probabilidad del estado '00'
            gram_matrix[i, j] = prob

    return gram_matrix







import numpy as np
import time
from qiskit import QuantumCircuit





def kernel_gram_matrix_full_symmetric(X1, X2):
    print("Calculando matriz de Gram")

    num_samples_X1 = X1.shape[0]


    gram_matrix = np.zeros((num_samples_X1, num_samples_X1))  # Make sure you have the correct size

    # Calculate only the upper half of the matrix
    for i in range(num_samples_X1):
        print(int(i / num_samples_X1 * 100), "%")
        for j in range(i, num_samples_X1):  # Only calculate the top half
            qc = QuantumCircuit(3, 3)
            x1 = X1[i].flatten()
            x2 = X2[j].flatten()

            prob = scalar_product_N_q(qc, x1, x2)[0]
            gram_matrix[i, j] = prob
            gram_matrix[j, i] = prob  # Symmetry: assigning the same value to the lower half

    return gram_matrix
x_train=TRAIN_DATA[:]
import time
inicio_tiempo = time.time()
matrix_simetrica = kernel_gram_matrix_full_symmetric(x_train, x_train)
fin=time.time()
tiempo=inicio_tiempo-fin
print(tiempo)


# In[48]:


import numpy as np



# Guardar la matriz en un archivo .npy
np.save('./FEATURE_RESULTS/QKM_RESULTS_THREE_QUBITS/simulation_tres_q_TRAIN_kernel_matrix_DP_rayomas.npy', matrix_simetrica)


# In[24]:


# load la matriz en un archivo .npy
matrix_simetrica_load=np.load('./FEATURE_RESULTS/QKM_RESULTS_THREE_QUBITS/simulation_tres_q_TRAIN_kernel_matrix_DP_rayomas.npy')


# In[49]:


import matplotlib.pyplot as plt

def visualize_kernel_matrix(kernel_matrix):
    plt.imshow(kernel_matrix, cmap='viridis', interpolation='none', origin='lower', extent=[0, len(kernel_matrix), 0, len(kernel_matrix)])
    cbar = plt.colorbar(label='kernel matrix value')
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    cbar.set_label('kernel matrix value', weight='bold', fontsize=14)
    plt.title('Kernel Matrix Visualization', fontweight='bold', fontsize=16)
    plt.xlabel('Data Index (X1)', fontweight='bold', fontsize=14)
    plt.ylabel('Data Index (X2)', fontweight='bold', fontsize=14)

    # Aumentar el tamaño y poner en negrita los números de los ejes
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    plt.show()



#  Viewing the kernel matrix
visualize_kernel_matrix(matrix_simetrica_load)


# In[50]:


import time
from sklearn import svm


# Assuming XTest is your test dataset and YLabels are the corresponding labels
# It is assumed that you already have the TRAIN_LABELS and matrix variables defined above


#Create a SVM classifier with the precomputed kernel
clf = svm.SVC(kernel="precomputed")


#Fitting the model to the training data
"""
 

Finds the optimal hyperplane that separates different classes based on the training data and associated labels.
The SVM objective function is convex, meaning there are no local minima.

 
"""
start_time = time.time()
clf.fit(matrix_simetrica, TRAIN_LABELS)
training_time = time.time() - start_time
print("Training time:", training_time, "s")


# In[51]:


#Record the start time
inicio_tiempo = time.time()
sol = clf.predict(matrix_simetrica)

y_train=TRAIN_LABELS
success = 0


# Calculate the time difference
tiempo_ejecucion = time.time()- inicio_tiempo

print(f"Tiempo de ejecución: {tiempo_ejecucion} segundos")


# In[52]:


y_train=TRAIN_LABELS
success = 0
for i in range(len(y_train)):
    if sol[i] == y_train[i]:
        success += 1

print("Train Accuracy: ", success/len(sol)*100, "%")


# In[53]:


print("Checking with test...")
x_test=TEST_DATA
# Record the start time
inicio_tiempo = time.time()
sol = clf.predict(kernel_gram_matrix_full(x_test, x_train))

y_test=TEST_LABELS
success = 0


# Calculate the time difference
tiempo_ejecucion = time.time()- inicio_tiempo

print(f"Tiempo de ejecución: {tiempo_ejecucion} segundos")


# In[54]:


y_test=TEST_LABELS
success = 0
for i in range(len(y_test)):
    if sol[i] == y_test[i]:
        success += 1

print("Precisión del train: ", success/len(sol)*100, "%")

# Registra el tiempo de finalización
fin_tiempo = time.time()

# Calcula la diferencia de tiempo
tiempo_ejecucion = fin_tiempo - inicio_tiempo

print(f"Tiempo de ejecución de compro test: {tiempo_ejecucion} segundos")


# In[57]:


zz_feature_map_reference = ZZFeatureMap(feature_dimension=8, reps=1)# 8 qubits or n_qubits



def scalar_product_N_q(circuit, x, y):
    

    zz_feature_map_reference1 = zz_feature_map_reference.assign_parameters(y)
    
    zz_feature_map_reference_inv = zz_feature_map_reference.assign_parameters(x)

    zzinv1=zz_feature_map_reference_inv.inverse()
    #  Combine the circuits
    circuito_final = zz_feature_map_reference1.compose(zzinv1, qubits=[0, 1,2,3,4,5,6,7])  # for 8 qubits


    
    backend = Aer.get_backend('statevector_simulator')
    t_qc = transpile(circuito_final, backend)
    qobj = assemble(t_qc)
    result = backend.run(qobj).result()
    statevector = result.get_statevector(t_qc)
    probs = np.abs(statevector)**2
    return probs


# In[58]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")
# constants
n = 2
RANDOM_STATE = 42
LR = 1e-3
class_labels = ['0', '1']

def normalizeData(DATA_PATH = "./FEATURE_RESULTS/FEATURE_resultante_DP_RAYOMAS.csv"):
    """
    Normalizes the data
    """
    # Reads the data
    data = pd.read_csv(DATA_PATH)
    data = shuffle(data, random_state=RANDOM_STATE)
   # For eight qubits the attributes are:
    X, Y = data[['area_pixels', ' mean_coords_x',' mean_coords_y',' centroid_x',' centroid_y',' mean_intensity',' std_intensity',' threshold']].values, data[' class'].values
    # normalize the data
    scaler = MinMaxScaler(feature_range=(-0 * np.pi, 2 * np.pi))
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=RANDOM_STATE)
    return X_train, X_test, Y_train, Y_test





import qiskit_algorithms
from qiskit_algorithms.optimizers import SPSA

from qiskit import QuantumCircuit

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import Statevector

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")

TRAIN_DATA, TEST_DATA, TRAIN_LABELS, TEST_LABELS = normalizeData()
# Replace all occurrences of 2 with 1
TRAIN_LABELS = np.where(TRAIN_LABELS == 2, 1, TRAIN_LABELS)
TEST_LABELS = np.where(TEST_LABELS == 2, 1, TEST_LABELS)
#print(TRAIN_DATA)


# In[59]:


# para 8 qubits

x_train=TRAIN_DATA[:]
import time
inicio_tiempo = time.time()
matrix_simetrica = kernel_gram_matrix_full_symmetric(x_train, x_train)
fin=time.time()
tiempo=inicio_tiempo-fin
print(tiempo)


# In[60]:


import numpy as np



# Guardar la matriz en un archivo .npy
np.save('./FEATURE_RESULTS/QKM_RESULTS_EIGHT_QUBITS/simulation_eight_q_TRAIN_kernel_matrix_DP_rayomas.npy', matrix_simetrica)


# In[73]:


# Ajustar el modelo a los datos de entrenamiento
"""
 encuentra el hiperplano óptimo que separa las diferentes clases en función de los datos de entrenamiento y las etiquetas asociadas.
 La función objetivo del SVM es convexa, lo que significa que no hay mínimos locales.
"""
start_time = time.time()
clf.fit(matrix_simetrica, TRAIN_LABELS)
training_time = time.time() - start_time
print("Tiempo de entrenamiento:", training_time, "segundos")


# In[62]:


#Record the start time
inicio_tiempo = time.time()
sol = clf.predict(matrix_simetrica)

y_train=TRAIN_LABELS
success = 0


# Calculate the time difference
tiempo_ejecucion = time.time()- inicio_tiempo

print(f"Tiempo de ejecución: {tiempo_ejecucion} segundos")


# In[63]:


y_train=TRAIN_LABELS
success = 0
for i in range(len(y_train)):
    if sol[i] == y_train[i]:
        success += 1

print("Precisión del train: ", success/len(sol)*100, "%")


# In[77]:


zz_feature_map_reference = ZZFeatureMap(feature_dimension=8, reps=1)# 8 qubits or n_qubits
def scalar_product_N_q(circuit, x, y):
    

    zz_feature_map_reference1 = zz_feature_map_reference.assign_parameters(y)
    
    zz_feature_map_reference_inv = zz_feature_map_reference.assign_parameters(x)

    zzinv1=zz_feature_map_reference_inv.inverse()
    # Combinar los circuitos
    circuito_final = zz_feature_map_reference1.compose(zzinv1, qubits=[0, 1,2,3,4,5,6,7])  # for 3 qubits


    
    backend = Aer.get_backend('statevector_simulator')
    t_qc = transpile(circuito_final, backend)
    qobj = assemble(t_qc)
    result = backend.run(qobj).result()
    statevector = result.get_statevector(t_qc)
    probs = np.abs(statevector)**2
    return probs


def kernel_gram_matrix_full(X1, X2):
    print("Calculando matriz de Gram")

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        print(int(i / len(X1) * 100), "%")
        for j, x2 in enumerate(X2):
            qc = QuantumCircuit(8, 8)
            x1 = x1.flatten()
            x2 = x2.flatten()
            prob = scalar_product_N_q(qc,x1, x2)[0]  # Usar solo la probabilidad del estado '00'
            gram_matrix[i, j] = prob

    return gram_matrix


# In[78]:


print("Comprobando con test...")
x_test=TEST_DATA
# Registra el tiempo de inicio
inicio_tiempo = time.time()
sol = clf.predict(kernel_gram_matrix_full(x_test, x_train))

y_test=TEST_LABELS
success = 0


# Calcula la diferencia de tiempo
tiempo_ejecucion = time.time()- inicio_tiempo

print(f"Tiempo de ejecución: {tiempo_ejecucion} segundos")


# In[79]:


y_test=TEST_LABELS
success = 0
for i in range(len(y_test)):
    if sol[i] == y_test[i]:
        success += 1

print("Precisión del test: ", success/len(sol)*100, "%")


# In[ ]:




