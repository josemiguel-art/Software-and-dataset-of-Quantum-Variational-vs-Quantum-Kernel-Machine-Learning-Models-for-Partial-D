#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install qiskit_ibm_provider')


# In[ ]:


# Create circuit to test transpiler on
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator, Diagonal

# Use Statevector object to calculate the ideal output
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

# Qiskit Runtime


# In[ ]:


oracle = Diagonal([1] * 15 + [-1])
print(oracle)
qc = QuantumCircuit(4)
qc.h([0, 1,2,3])
qc = qc.compose(GroverOperator(oracle))

qc.draw()


# In[ ]:


ideal_distribution = Statevector.from_instruction(qc).probabilities_dict()

plot_histogram(ideal_distribution)


# In[6]:


get_ipython().system('pip install qiskit')


# In[7]:


get_ipython().system('pip install qiskit_algorithms')


# In[8]:


get_ipython().system('pip install qiskit-ibm-runtime')


# In[ ]:


get_ipython().system('pip install pylatexenc')


# In[ ]:


get_ipython().system("pip install 'qiskit-machine-learning[sparse]'")


# In[ ]:


get_ipython().system('pip install qiskit-aer')


# In[ ]:


import numpy as np
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
import numpy as np

from qiskit import QuantumCircuit

# 10-parameter variational quantum model
# define your parameters
x = ParameterVector('x', 2)
teta=ParameterVector('teta',10)
# define your parameters
def ZZFeatureMap_10_parametros(feature_dimension=2, reps=1, theta_param=np.pi/2, x=x, teta=teta):
    circuit = QuantumCircuit(feature_dimension)
#1.06324998,  0.49609634
    for i in range(reps):
        if i == 0:
            circuit.h(range(feature_dimension))

        for j in range(feature_dimension):
            circuit.p(theta_param * x[j] * 1, j)

        circuit.cx(0, 1)

        circuit.p(theta_param * (np.pi - x[0] *0.49609634) * (np.pi - x[1] *1.06324998), 1)

        circuit.cx(0, 1)

        # Ensure that the indices are within the range of feature_dimension
        if feature_dimension > 1:
            circuit.ry(teta[0], 0)
            circuit.ry(teta[1], 1)

        if feature_dimension > 1:
            circuit.cx(0, 1)
            circuit.ry(teta[2], 0)
            circuit.ry(teta[3], 1)
        if feature_dimension > 1:
            circuit.cx(0, 1)
            circuit.ry(teta[4], 0)
            circuit.ry(teta[5], 1)
            circuit.cx(0, 1)
        if feature_dimension > 1:
            circuit.cx(0, 1)
            circuit.ry(teta[6], 0)
            circuit.ry(teta[7], 1)
            circuit.cx(0, 1)
            circuit.ry(teta[8], 0)
            circuit.ry(teta[9], 1)
            circuit.cx(0, 1)
            circuit.measure_all()

    return circuit

feature_map = ZZFeatureMap_10_parametros(feature_dimension=2, reps=1, theta_param=2, x=x,teta=teta)
#print(feature_map)


feature_map.decompose().draw("mpl")


# In[ ]:


#FakeProviderForBackendV2()	Fake provider containing fake V2 backends.
#FakeProvider()	Fake provider containing fake V1 backends.
# https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake_provider

from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime.fake_provider import FakeManilaV2,FakeBrisbane,FakeKyoto,FakeOsaka

#FakeKyoto()	A fake 127 qubit backend.
  #FakeMontrealV2()	A fake 27 qubit backend.
# FakeMumbaiV2()	A fake 27 qubit backend.
#FakeNairobiV2()	A fake 7 qubit backend.
#FakeOsaka()	A fake 127 qubit backend.
# FakeOslo()	A fake 7 qubit backend.
import numpy as np
from qiskit.circuit import ParameterVector

import numpy as np
# 11-parameter variational quantum model
# define your parameters
x = ParameterVector('x', 2)
teta=ParameterVector('teta',11)
# define your parameters
def ZZFeatureMap_11_parametros_MODI(feature_dimension=2, reps=1, x=x, teta=teta):
    circuit = QuantumCircuit(feature_dimension)
#1.06324998,  0.49609634
#0.6810406 ,  0.93371965
#opt_var7=np.array([0.68958857, 0.93659037])#0.765
    for i in range(reps):
        if i == 0:
            circuit.h(range(feature_dimension))

        for j in range(feature_dimension):
            circuit.p( x[j] * 1, j)

        circuit.cx(0, 1)

        circuit.p( (x[0]*teta[10]) * (x[1]), 1)

        circuit.cx(0, 1)

        # Ensure that the indices are within the range of feature_dimension
        if feature_dimension > 1:
            circuit.ry(teta[0], 0)
            circuit.ry(teta[1], 1)

        if feature_dimension > 1:
            circuit.cx(0, 1)
            circuit.ry(teta[2], 0)
            circuit.ry(teta[3], 1)
        if feature_dimension > 1:
            circuit.cx(0, 1)
            circuit.ry(teta[4], 0)
            circuit.ry(teta[5], 1)
            circuit.cx(0, 1)
        if feature_dimension > 1:

            circuit.ry(teta[6], 0)
            circuit.ry(teta[7], 1)
            circuit.cx(0, 1)
            circuit.ry(teta[8], 0)
            circuit.ry(teta[9], 1)
            circuit.cx(0, 1)
            circuit.measure_all()

    return circuit

feature_map = ZZFeatureMap_11_parametros_MODI(feature_dimension=2, reps=1, x=x,teta=teta)
#print(feature_map)
feature_map.decompose().draw("mpl")


# In[10]:


import qiskit_algorithms
from qiskit_algorithms.optimizers import SPSA,COBYLA

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


# Normalizing data from 0 to 2pi

def normalizeData(DATA_PATH = "./FEATURE_RESULTS/FEATURE_resultante_DP_NODP.csv"):
    """
    Normalizes the data
    """
    # Reads the data mean_coords_x,  mean_coords_y  centroid_y_roi, mean_coords_x
    data = pd.read_csv(DATA_PATH)
    data = shuffle(data, random_state=RANDOM_STATE)
    X, Y = data[['area_pixels', ' mean_coords_x']].values, data[' class'].values
    #X, Y = data[[' mean_coords_x', ' centroid_y_roi']].values, data[' class'].values
    #X, Y = data[[' mean_coords_x', ' mean_coords_y']].values, data[' class'].values
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
TRAIN_LABELS = np.where(TRAIN_LABELS == 2, 0, TRAIN_LABELS)
TEST_LABELS = np.where(TEST_LABELS == 2, 0, TEST_LABELS)
print(TRAIN_LABELS)
# 
# Count the number of elements that are 0
num_zeros = np.count_nonzero(TRAIN_LABELS == 0)

# Count the number of elements that are  1
num_unos = np.count_nonzero(TRAIN_LABELS == 1)



print("Count the number of elements that are 0 :", num_zeros)
print(" Count the number of elements that are 1:", num_unos)


# In[ ]:


TEST_LABELS


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import numpy as np
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
import numpy as np
#from qiskit import BasicAer, execute

from qiskit_ibm_runtime import QiskitRuntimeService

# define your parameters
x = ParameterVector('x', 2)
teta=ParameterVector('teta',11)
teta10=1.52596629


#Functions used in the optimization process of circuit parameters

def ZZFeatureMap_11_parametros_MODI(feature_dimension=2, reps=1, x=x, teta=teta):
    circuit = QuantumCircuit(feature_dimension)
#1.06324998,  0.49609634
#0.6810406 ,  0.93371965
#opt_var7=np.array([0.68958857, 0.93659037])#0.765
    for i in range(reps):
        if i == 0:
            circuit.h(range(feature_dimension))

        for j in range(feature_dimension):
            circuit.p( x[j] * 1, j)

        circuit.cx(0, 1)

        circuit.p( (x[0]*teta[10]) * (x[1]), 1)

        circuit.cx(0, 1)

        # Ensure that the indices are within the range of feature_dimension
        if feature_dimension > 1:
            circuit.ry(teta[0], 0)
            circuit.ry(teta[1], 1)

        if feature_dimension > 1:
            circuit.cx(0, 1)
            circuit.ry(teta[2], 0)
            circuit.ry(teta[3], 1)
        if feature_dimension > 1:
            circuit.cx(0, 1)
            circuit.ry(teta[4], 0)
            circuit.ry(teta[5], 1)
            circuit.cx(0, 1)
        if feature_dimension > 1:

            circuit.ry(teta[6], 0)
            circuit.ry(teta[7], 1)
            circuit.cx(0, 1)
            circuit.ry(teta[8], 0)
            circuit.ry(teta[9], 1)
            circuit.cx(0, 1)
            circuit.measure_all()

    return circuit
feature_map = ZZFeatureMap_11_parametros_MODI(feature_dimension=2, reps=1, x=x,teta=teta)
print(feature_map)
feature_map.draw('mpl')



def classification_probability(data, variational):
    """Classify data points using given parameters.
    Args:
        data (list): Set of data points to classify
        variational (list): Parameters for `VAR_FORM`
    Returns:
    #results = execute(circuits, backend, shots=shots_per_execution).result()
    job = backend.run(circuits,shots=4000)
    #counts = job.result().get_counts()
    #######33
    results = job.result()
        list[dict]: Probability of circuit classifying
                    each data point as 0 or 1.
    shots_per_execution = 100"""
    shots_per_execution = 50
    circuits = [circuit_instance_MIO(d, variational) for d in data]
    #backend = FakeManilaV2()
    backend = FakeOsaka()
    #results = execute(circuits, backend, shots=shots_per_execution).result()
    #results = execute(circuits, backend, shots=shots_per_execution).result()
    job = backend.run(circuits,shots=100)
#counts = job.result().get_counts()
    classification = [
        label_probability(job.result().get_counts(c)) for c in circuits]
    return classification

def parity(bitstring):
    """Returns 1 if parity of `bitstring` is even, otherwise 0."""
    hamming_weight = sum(int(k) for k in list(bitstring))
    return (hamming_weight+1) % 2
def label_probability(results):
    """Converts a dict of bitstrings and their counts,
    to parities and their counts"""
    shots = sum(results.values())
    probabilities = {0: 0, 1: 0}
    for bitstring, counts in results.items():
        label = parity(bitstring)
        probabilities[label] += counts / shots
    return probabilities

def cross_entropy_loss(classification, expected):

    p = classification.get(expected)  # Prob. of correct classification
    return -np.log(p + 1e-10)



def cost_function(data, labels, variational):
    """Evaluates performance of our circuit with `variational`
    parameters on `data`.

    Args:
        data (list): List of data points to classify
        labels (list): List of correct labels for each data point
        variational (list): Parameters to use in circuit

    Returns:
        float: Cost (metric of performance)
    """
    classifications = classification_probability(data, variational)
    cost = 0
    for i, classification in enumerate(classifications):
        cost += cross_entropy_loss(classification, labels[i])
    cost /= len(data)
    return cost

import time








class OptimizerLog:
    """Log to store optimizer's intermediate results"""
    def __init__(self):
        self.evaluations = []
        self.parameters = []
        self.costs = []

    def update(self, parameter):
        """Save intermediate results."""
        evaluation = len(self.evaluations) + 1  # Assuming evaluations are sequential
        cost = objective_function(parameter)
        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        self.costs.append(cost)





def marcar_probabilidades_altas(probabilidades, umbral=0.91):
    INDICE_CLASE_0 = []
    INDICE_CLASE_1 = []

    for idx, prob_dict in enumerate(probabilidades):
        # Obtener la clase con la probabilidad más alta
        clase_elegida = max(prob_dict, key=prob_dict.get)

        # Marcar el elemento si la probabilidad es mayor que el umbral
        if prob_dict[clase_elegida] > umbral:
            if clase_elegida == 0:
                INDICE_CLASE_0.append(idx)
            elif clase_elegida == 1:
                INDICE_CLASE_1.append(idx)

    return INDICE_CLASE_0, INDICE_CLASE_1




# In[ ]:


# Set up the optimizationfrom qiskit_algorithms.optimizers import SPSA  optimizer = COBYLA(maxiter=40)
#from qiskit.algorithms.optimizers import
import qiskit_algorithms
from qiskit_algorithms.optimizers import SPSA,COBYLA,ADAM
log = OptimizerLog()
#optimizer = SPSA(maxiter=200,callback=callback_graph)

bounds = [(0, 3), (0, 2)]
optimizer = SPSA(maxiter=100)
#optimizer = SPSA(maxiter=50, learning_rate=0.05, perturbation=0.05)

#optimizer=COBYLA(maxiter=40 rhobeg=2.0,callback=log.update)
optimizer = COBYLA(maxiter=100,callback=log.update)
#optimizer =ADAM(maxiter=5)
#optimizer = SPSA(maxiter=50)
def circuit_instance_MIO(data, variational):
    """Assigns parameter values to `AD_HOC_CIRCUIT`.
    Args:
        data (list): Data values for the feature map
        variational (list): Parameter values for `VAR_FORM`
    Returns:
        QuantumCircuit: `AD_HOC_CIRCUIT` with parameters assigned
    """

    return feature_map.assign_parameters({x:data,teta:variational})


# It is uncommented according to the circuit used 10 or 11 parameters

#feature_map =ZZFeatureMap_11_parametros_MODI(feature_dimension=2, reps=1, x=x,teta=teta)

feature_map = ZZFeatureMap_10_parametros(feature_dimension=2, reps=1, theta_param=2, x=x,teta=teta)









# optimal value obtained WITH A MEAN P OF 0.9 for DP_NODP
opt_var=np.array([-1.36465941,  0.72901008,  0.46274449, -0.22550087, 0.71628267, -5.0369175 ,  0.25267942,  3.20192607,  2.22427876,  2.87675972])




def objective_function(variational):
    """Cost function of circuit parameters on training data.
    The optimizer will attempt to minimize this."""
    return cost_function(TRAIN_DATA# clear objective value history

objective_func_vals = []
# Run the optimization
bounds = [(0, 1.1), (0, 1.1)]


# Record the start time


inicio_tiempo = time.time()

# PARAA 10 PARA
opt_var=np.array([-0.28500921,  3.49039855,  1.85757059,  0.09167011, -3.19778968,
       -6.05658208,  2.21860378,  5.5063756 ,  4.54796396,  2.70080547,
        1.55475791])

initial_point_FIN=np.array([-1.36465941,  0.72901008,  0.46274449, -0.22550087, 0.71628267, -5.0369175 ,  0.25267942,  3.20192607,  2.22427876,  2.87675972])


# for 10 PARAM. intial value
opt_var=np.array([-0.28500921,  3.49039855,  1.85757059,  0.09167011, -3.19778968,
       -6.05658208,  2.21860378,  5.5063756 ,  4.54796396,  2.70080547,
        1.55475791])

initial_point=opt_var
result = optimizer.minimize(objective_function, initial_point)

# Record the completion time
fin_tiempo = time.time()

# Calculate the time difference
tiempo_ejecucion = fin_tiempo - inicio_tiempo

# Print the execution time
print(f"Time of the execution: {tiempo_ejecucion} s")



#final param for DP_NODP
feature_map = ZZFeatureMap_10_parametros(feature_dimension=2, reps=1, theta_param=2, x=x,teta=teta)
opt_var_10parametros=np.array([-1.36465941,  0.72901008,  0.46274449, -0.22550087, 0.71628267, -5.0369175 ,  0.25267942,  3.20192607,  2.22427876,  2.87675972])


print(result)
opt_var=result.x


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Suppose you have a 'log' object with attributes 'evaluations' and 'costs'
# log = ... 'evaluations' y 'costs'
# log = ...

# Crear la figura
fig = plt.figure()

# Graficar los datos
plt.plot(log.evaluations, log.costs)
plt.xlabel('Steps')
plt.ylabel('Cost')


# In[11]:


import numpy as np
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
import numpy as np

from qiskit import QuantumCircuit
# define your parameters
x = ParameterVector('x', 2)
teta=ParameterVector('teta',10)
# define your parameters
def ZZFeatureMap_10_parametros(feature_dimension=2, reps=1, theta_param=np.pi/2, x=x, teta=teta):
    circuit = QuantumCircuit(feature_dimension)
#1.06324998,  0.49609634
    for i in range(reps):
        if i == 0:
            circuit.h(range(feature_dimension))

        for j in range(feature_dimension):
            circuit.p(theta_param * x[j] * 1, j)

        circuit.cx(0, 1)

        circuit.p(theta_param * (np.pi - x[0] *0.49609634) * (np.pi - x[1] *1.06324998), 1)

        circuit.cx(0, 1)

        # Ensure that the indices are within the range of feature_dimension
        if feature_dimension > 1:
            circuit.ry(teta[0], 0)
            circuit.ry(teta[1], 1)

        if feature_dimension > 1:
            circuit.cx(0, 1)
            circuit.ry(teta[2], 0)
            circuit.ry(teta[3], 1)
        if feature_dimension > 1:
            circuit.cx(0, 1)
            circuit.ry(teta[4], 0)
            circuit.ry(teta[5], 1)
            circuit.cx(0, 1)
        if feature_dimension > 1:
            circuit.cx(0, 1)
            circuit.ry(teta[6], 0)
            circuit.ry(teta[7], 1)
            circuit.cx(0, 1)
            circuit.ry(teta[8], 0)
            circuit.ry(teta[9], 1)
            circuit.cx(0, 1)
            circuit.measure_all()

    return circuit

feature_map = ZZFeatureMap_10_parametros(feature_dimension=2, reps=1, theta_param=2, x=x,teta=teta)
#print(feature_map)

feature_map.decompose().draw()

from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime.fake_provider import FakeManilaV2,FakeBrisbane,FakeKyoto,FakeOsaka
backend = FakeOsaka()



def circuit_instance_MIO(data, variational):
    """Assigns parameter values to `AD_HOC_CIRCUIT`.
    Args:
        data (list): Data values for the feature map
        variational (list): Parameter values for `VAR_FORM`
    Returns:
        QuantumCircuit: `AD_HOC_CIRCUIT` with parameters assigned
    """

    return feature_map.assign_parameters({x:data,teta:variational})





def classification_probability(data, variational):
    """Classify data points using given parameters.
    Args:
        data (list): Set of data points to classify
        variational (list): Parameters for `VAR_FORM`
    Returns:
    #results = execute(circuits, backend, shots=shots_per_execution).result()
    job = backend.run(circuits,shots=4000)
    #counts = job.result().get_counts()
    #######33
    results = job.result()
        list[dict]: Probability of circuit classifying
                    each data point as 0 or 1.
    shots_per_execution = 100"""
    shots = 2000
    circuits = [circuit_instance_MIO(d, variational) for d in data]
    #backend = FakeManilaV2()
    backend = FakeOsaka()
    #results = execute(circuits, backend, shots=shots_per_execution).result()
    #results = execute(circuits, backend, shots=shots_per_execution).result()
    job = backend.run(circuits,shots=shots)
#counts = job.result().get_counts()
    classification = [
        label_probability(job.result().get_counts(c)) for c in circuits]
    return classification

def parity(bitstring):
    """Returns 1 if parity of `bitstring` is even, otherwise 0."""
    hamming_weight = sum(int(k) for k in list(bitstring))
    return (hamming_weight+1) % 2
def label_probability(results):
    """Converts a dict of bitstrings and their counts,
    to parities and their counts"""
    shots = sum(results.values())
    probabilities = {0: 0, 1: 0}
    for bitstring, counts in results.items():
        label = parity(bitstring)
        probabilities[label] += counts / shots
    return probabilities

def cross_entropy_loss(classification, expected):
    """Calculate accuracy of predictions using cross entropy loss.
    Args:
        classification (dict): Dict where keys are possible classes,
                               and values are the probability our
                               circuit chooses that class.
        expected (int): Correct classification of the data point.

    Returns:
        float: Cross entropy loss
    """
    p = classification.get(expected)  # Prob. of correct classification
    return -np.log(p + 1e-10)







def comparar_probabilidades(probabilidades, labels):
    if len(probabilidades) != len(labels):
        raise ValueError("Lists of probabilities and labels must be the same length.")

    coincidencias = 0

    for prob_dict, label in zip(probabilidades, labels):
        # Get the class with the highest probability
        clase_predicha = max(prob_dict, key=prob_dict.get)

        # Check if the predicted class matches the actual label
        if clase_predicha == label:
            coincidencias += 1

    probabilidad_media = coincidencias / len(labels)
    return probabilidad_media





feature_map = ZZFeatureMap_10_parametros(feature_dimension=2, reps=1, theta_param=2, x=x,teta=teta)

opt_var=np.array([-1.36465941,  0.72901008,  0.46274449, -0.22550087, 0.71628267, -5.0369175 ,  0.25267942,  3.20192607,  2.22427876,  2.87675972])

""" In this final part we can calculate the accuracy with this set of parameters after the optimization process.
"""
resultado = comparar_probabilidades(classification_probability(TRAIN_DATA,opt_var), TRAIN_LABELS)

print(f"The average probability of matching is for train : {resultado}")

resultado = comparar_probabilidades(classification_probability(TEST_DATA,opt_var), TEST_LABELS)
print(f"The average probability of matching is for test : {resultado}")








