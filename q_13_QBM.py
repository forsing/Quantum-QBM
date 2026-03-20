"""
QBM - Quantum Boltzmann Machine
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize as scipy_minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
NUM_LAYERS = 3
MAXITER = 300


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def build_qbm_circuit(theta, n_qubits, n_layers):
    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.h(i)

    idx = 0
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.rx(theta[idx], i)
            idx += 1
        for i in range(n_qubits):
            qc.rz(theta[idx], i)
            idx += 1
        for i in range(n_qubits - 1):
            qc.rzz(theta[idx], i, i + 1)
            idx += 1
        qc.rzz(theta[idx], n_qubits - 1, 0)
        idx += 1

    return qc


def num_params():
    return NUM_LAYERS * (NUM_QUBITS * 2 + NUM_QUBITS)


def exact_born_dist(theta):
    qc = build_qbm_circuit(theta, NUM_QUBITS, NUM_LAYERS)
    sv = Statevector.from_instruction(qc)
    return sv.probabilities()


def train_qbm(target):
    n_p = num_params()
    theta0 = np.random.uniform(0, 2 * np.pi, n_p)

    def cost(theta):
        born = exact_born_dist(theta)
        kl = 0.0
        for i, pt in enumerate(target):
            if pt > 0:
                pb = born[i]
                if pb <= 0:
                    pb = 1e-10
                kl += pt * np.log(pt / pb)
        return float(kl)

    res = scipy_minimize(cost, theta0, method='COBYLA',
                         options={'maxiter': MAXITER, 'rhobeg': 0.5})
    return res.x, res.fun


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, prob in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    n_states = 1 << NUM_QUBITS

    print(f"\n--- QBM ({NUM_QUBITS}q, {NUM_LAYERS} sloja, "
          f"Rx+Rz+Rzz, COBYLA {MAXITER} iter) ---")
    print(f"  Parametara po modelu: {num_params()}")

    dists = []
    for pos in range(7):
        print(f"  Poz {pos+1}...", end=" ", flush=True)
        target = build_empirical(draws, pos)
        theta, final_loss = train_qbm(target)
        born = exact_born_dist(theta)
        dists.append(born)

        top_idx = np.argsort(born)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{born[i]:.3f}" for i in top_idx)
        print(f"KL={final_loss:.4f}  top: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QBM, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()



"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- QBM (5q, 3 sloja, Rx+Rz+Rzz, COBYLA 300 iter) ---
  Parametara po modelu: 45
  Poz 1... KL=0.0058  top: 1:0.162 | 2:0.147 | 3:0.131
  Poz 2... KL=0.0229  top: 8:0.080 | 6:0.075 | 11:0.070
  Poz 3... KL=0.0574  top: 14:0.075 | 12:0.071 | 15:0.058
  Poz 4... KL=0.0378  top: 22:0.067 | 16:0.060 | 20:0.059
  Poz 5... KL=0.1229  top: 25:0.066 | 29:0.065 | 30:0.057
  Poz 6... KL=0.0399  top: 33:0.089 | 30:0.081 | 31:0.075
  Poz 7... KL=0.1528  top: 37:0.143 | 7:0.110 | 38:0.106

==================================================
Predikcija (QBM, deterministicki, seed=39):
[1, 8, x, y, z, 33, 37]
==================================================
"""



"""
QBM - Quantum Boltzmann Machine

Pocinje sa H na svim qubitima (uniformna superpozicija) - kao termalno stanje
Rx + Rz + Rzz gejt struktura: Rzz simulira Ising-ovu interakciju izmedju susednih qubita (Boltzmann energija)
Ciklicni Rzz (poslednji-prvi qubit) za periodicne granicne uslove
3 sloja, 45 parametara, COBYLA 300 iteracija - dublje treniranje
Uci Born distribuciju koja aproksimira Boltzmannovu distribuciju podataka
Egzaktno, deterministicki, Statevector
"""
