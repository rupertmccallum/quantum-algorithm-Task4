import numpy as np
from random import random
from scipy import array
from scipy.optimize import minimize

from qiskit import *
from qiskit.extensions.standard import *
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import ExactEigensolver

def hamiltonian_operator:
    """
    Creates the Hamilton for which we want to 
    estimate the least eigenvalue.
    
    """
    pauli_dict = {
        'paulis': [{"coeff": {"imag": 0.0, "real": -0.5}, "label": "II"},
                   {"coeff": {"imag": 0.0, "real": -0.5}, "label": "ZZ"},
                   {"coeff": {"imag": 0.0, "real": 0.5}, "label": "XX"},
                   {"coeff": {"imag": 0.0, "real": 0.5}, "label": "YY"}
                   ]
    }
    return WeightedPauliOperator.from_dict(pauli_dict)
    
def quantum_state_preparation(circuit, parameters):
    q = circuit.qregs[0:3] # q is the quantum register where the info about qubits is stored
    circuit.h(q[0])
    circuit.rz(parameters[0], q[0]) 
    circuit.cx(q[0:2])
    circult.x(q[1])
    return circuit
    
def vqe_circuit(parameters, measure):
    """
    Creates a device ansatz circuit for optimization.
    :param parameters_array: list of parameters for constructing ansatz state that should be optimized.
    :param measure: measurement type. E.g. 'ZZ' stands for ZZ measurement.
    :return: quantum circuit.
    """
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    circuit = QuantumCircuit(q, c)

    # quantum state preparation
    circuit = quantum_state_preparation(circuit, parameters)

    # measurement
    if measure == 'ZZ':
        circuit.measure(q[0:2], c[0:2])
    elif measure == 'XX':
        circuit.u2(0, np.pi, q[0:2])
        circuit.measure(q[0:2], c[0:2])
    elif measure == 'YY':
        circuit.u2(0, np.pi/2, q[0:2])
        circuit.measure(q[0:2], c[0:2])
    elif measure == 'IX':
        circuit.u2(0, np.pi, q[1])
        circuit.measure(q[1], c[1])
    elif measure == 'XI':
        circuit.u2(0, np.pi, q[0])
        circuit.measure(q[0], c[0])
    elif measure == 'IY':
        circuit.u2(0, np.pi/2, q[1])
        circuit.measure(q[1], c[1])
    elif measure == 'YI':
        circuit.u2(0, np.pi/2, q[0])
        circuit.measure(q[0], c[0])
    elif measure == 'IZ':
        circuit.measure(q[1], c[1])
    elif measure == 'ZI':
        circuit.measure(q[0], c[0])
    elif measure == 'XY':
        circuit.u2(0, np.pi, q[0])
        circuit.u2(0, np.pi/2, q[1])
        circuit.measure(q[0:2], c[0:2])
    elif measure == 'XZ':
        circuit.u2(0, np.pi, q[0])
        circuit.measure(q[0:2], c[0:2])
    elif measure == 'YX':
        circuit.u2(0, np.pi/2, q[0])
        circuit.u2(0, np.pi, q[1])
        circuit.measure(q[0:2], c[0:2])
    elif measure == 'YZ':
        circuit.u2(p, np.pi/2, q[0])
        circuit.measure(q[0:2], c[0:2])
    elif measure == 'ZX':
        circuit.u2(0, np.pi, q[1])
        circuit.measure(q[0:2], c[0:2])
    elif measure == 'ZY':
        circuit.u2(0, np.pi/2, q[1])
        circuit.measure(q[0:2], c[0:2]
    else:
        raise ValueError('Not valid input for measurement')

    return circuit
    
def get_or_else_zero(d: dict, key: str):
    """
    Utility for working with dictionaries. If key is missing
    than return 0 otherwise the corresponding value.
    :param dict: the dictionary.
    :param key: key (string) in interest.
    :return: 0 or value of corresponding key.
    """
    value = 0
    if key in d:
        value = d[key]
    return value
    
def quantum_module(parameters, measure):
    # measure
    if measure == 'II':
        return 1
    elif measure == 'IX':
        circuit = vqe_circuit(parameters, 'IX')
    elif measure == 'IY':
        circuit = vqe_circuit(parameters, 'IY')
    elif measure == 'IZ':
        circuit = vqe_circuit(parameters, 'IZ')
    elif measure == 'XI':
        circuit = vqe_circuit(parameters, 'XI')
    elif measure == 'XX':
        circuit = vqe_circuit(parameters, 'XX')
    elif measure == 'XY':
        circuit = vqe_circuit(parameters, 'XY')
    elif measure == 'XZ':
        circuit = vqe_circuit(parameters, 'XZ')
    elif measure == 'YI':
        circuit = vqe_circuit(parameters, 'YI')
    elif measure == 'YX':
        circuit = vqe_circuit(parameters, 'YX')
    elif measure == 'YY':
        circuit = vqe_circuit(parameters, 'YY')
    elif measure == 'YZ':
        circuit = vqe_circuit(parameters, 'YZ')
    elif measure == 'ZI':
        circuit = vqe_circuit(parameters, 'ZI')
    elif measure == 'ZX':
        circuit = vqe_circuit(parameters, 'ZX')
    elif measure == 'ZY':
        circuit = vqe_circuit(parameters, 'ZY')
    elif measure == 'ZZ':
        circuit = vqe_circuit(parameters, 'ZZ')
    else:
        raise ValueError('Not valid input for measurement')
    
    shots = 1000
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    expectation_value = (get_or_else_zero(counts, '0') - get_or_else_zero(counts,'1')) / shots
    
    return expectation_value

def pauli_operator_to_dict(pauli_operator):
    """
    from WeightedPauliOperator return a dict.
    """
    d = pauli_operator.to_dict()
    paulis = d['paulis']
    paulis_dict = {}

    for x in paulis:
        label = x['label']
        coeff = x['coeff']['real']
        paulis_dict[label] = coeff

    return paulis_dict
pauli_dict = pauli_operator_to_dict(hamiltonian_operator)

def vqe(parameters):
        
    # quantum_modules
    quantum_module_II = get_or_else_zero(pauli_dict, 'II') * quantum_module(parameters, 'II')
    quantum_module_IX = get_or_else_zero(pauli_dict, 'IX') * quantum_module(parameters, 'IX')
    quantum_module_IY = get_or_else_zero(pauli_dict, 'IY') * quantum_module(parameters, 'IY')
    quantum_module_IZ = get_or_else_zero(pauli_dict, 'IZ') * quantum_module(parameters, 'IZ')
    quantum_module_XI = get_or_else_zero(pauli_dict, 'XI') * quantum_module(parameters, 'XI')
    quantum_module_XX = get_or_else_zero(pauli_dict, 'XX') * quantum_module(parameters, 'XX')
    quamtum_module_XY = get_or_else_zero(pauli_dict, 'XY') * quantum_module(parameters, 'XY')
    quantum_module_XZ = get_or_else_zero(pauli_dict, 'XZ') * quantum_module(parameters, 'XZ')
    quantum_module_YI = get_or_else_zero(pauli_dict, 'YI') * quantum_module(parameters, 'YI')
    quantum_module_YX = get_or_else_zero(pauli_dict, 'YX') * quantum_module(parameters, 'YX')
    quantum_module_YY = get_or_else_zero(pauli_dict, 'YY') * quantum_module(parameters, 'YY')
    quantum_module_YZ = get_or_else_zero(pauli_dict, 'YZ') * quantum_module(parameters, 'YZ')
    quantum_module_ZI = get_or_else_zero(pauli_dict, 'ZI') * quantum_module(parameters, 'ZI')
    quantum_module_ZX = get_or_else_zero(pauli_dict, 'ZX') * quantum_module(parameters, 'ZX')
    quantum_module_ZY = get_or_else_zero(pauli_dict, 'ZY') * quantum_module(parameters, 'ZY')
    quantum_module_ZZ = get_or_else_zero(pauli_dict, 'ZZ') * quantum_module(parameters, 'ZZ')
    
    # summing the measurement results
    classical_adder = quantum_module_II + quantum_module_IX + quantum_module_IY + quantum_module_IZ + 
    quantum_module_XI + quantum_module_XX + quantum_module_XY + quantum_module_XZ + quantum_module_YI +
    quantum_module_YX + quantum_module_YY + quantum_module_YZ + quantum_module_ZI + quantum_module_ZX +
    quantum_module_ZY + quantum_module_ZZ 
    
    return classical_adder

parameters_array = array([np.pi])
tol = 1e-3 # tolerance for optimization precision.

vqe_result = minimize(vqe, parameters_array, method="Powell", tol=tol)
print('The estimated ground state energy from VQE algorithm is: {}'.format(vqe_result.fun))
