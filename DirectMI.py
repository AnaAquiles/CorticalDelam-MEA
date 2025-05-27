import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

"""
    This code will need some outputs from SignalClassification.py
"""

def calc_MI(X, Y):
    binsY = np.histogram_bin_edges(Y, bins='sturges', range=(0, 5))
    binsX = np.histogram_bin_edges(X, bins='sturges', range=(0, 5))
    
    c_XY = np.histogram2d(X, Y, bins=[binsX, binsY])[0]
    c_X = np.histogram(X, binsX)[0]
    c_Y = np.histogram(Y, binsY)[0]
    
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)
    
    MI = H_X + H_Y - H_XY
    return MI

def shan_entropy(c):
    c_normalized = np.nan_to_num(c / float(np.sum(c)))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))  
    return H

def permutation_test(X, Y, num_permutations=1000):
    observed_mi = calc_MI(X, Y)
    permuted_mis = np.zeros(num_permutations)
    
    for i in range(num_permutations):
        np.random.shuffle(Y)  # Shuffle Y to break any true dependency
        permuted_mis[i] = calc_MI(X, Y)
    
    p_value = np.sum(permuted_mis >= observed_mi) / num_permutations
    return observed_mi, p_value

def compute_directionality(set_a, set_b, lags=np.arange(-50, 51), num_permutations=1000, alpha=0.05):
    """
    Compute directionality between two signal sets (Set A -> Set B and Set B -> Set A) with statistical testing.
    
    """
    results = {
        "A_to_B": np.zeros((len(set_a), len(set_b), len(lags))),
        "B_to_A": np.zeros((len(set_b), len(set_a), len(lags))),
        "best_lags_A_to_B": np.zeros((len(set_a), len(set_b))),
        "best_lags_B_to_A": np.zeros((len(set_b), len(set_a))),
        "p_values_A_to_B": np.ones((len(set_a), len(set_b), len(lags))),
        "p_values_B_to_A": np.ones((len(set_b), len(set_a), len(lags))),
    }
    
    for i, a_signal in enumerate(tqdm(set_a, desc="Processing Set A")):
        for j, b_signal in enumerate(tqdm(set_b, desc=f"Processing Set B [{i+1}/{len(set_a)}]", leave=False)):
            mi_a_to_b = []
            mi_b_to_a = []
            p_a_to_b = []
            p_b_to_a = []
            
            for lag in tqdm(lags, desc=f"Lags [{i+1},{j+1}]", leave=False):
                if lag < 0:
                    lagged_b = np.roll(b_signal, lag)
                    lagged_a = a_signal
                else:
                    lagged_a = np.roll(a_signal, lag)
                    lagged_b = b_signal
                
                mi_ab, p_ab = permutation_test(lagged_a, lagged_b, num_permutations)
                mi_ba, p_ba = permutation_test(lagged_b, lagged_a, num_permutations)
                
                mi_a_to_b.append(mi_ab)
                mi_b_to_a.append(mi_ba)
                p_a_to_b.append(p_ab)
                p_b_to_a.append(p_ba)
            
            results["A_to_B"][i, j, :] = mi_a_to_b
            results["B_to_A"][j, i, :] = mi_b_to_a
            results["p_values_A_to_B"][i, j, :] = p_a_to_b
            results["p_values_B_to_A"][j, i, :] = p_b_to_a
            
            # Choose best lag based on MI, but only if significant
            significant_mis_A_to_B = np.array(mi_a_to_b) * (np.array(p_a_to_b) < alpha)
            significant_mis_B_to_A = np.array(mi_b_to_a) * (np.array(p_b_to_a) < alpha)
            
            if np.any(significant_mis_A_to_B):
                results["best_lags_A_to_B"][i, j] = lags[np.argmax(significant_mis_A_to_B)]
            else:
                results["best_lags_A_to_B"][i, j] = np.nan  # No significant causality
            
            if np.any(significant_mis_B_to_A):
                results["best_lags_B_to_A"][j, i] = lags[np.argmax(significant_mis_B_to_A)]
            else:
                results["best_lags_B_to_A"][j, i] = np.nan  # No significant causality
    
    return results



# # Compute directionality
lags9 = np.arange(-100, 100)
results9 = compute_directionality(Act1, Act2, lags9)  # changes with all the possible combinations


### save the result dictionary
with open('name.pkl', 'wb') as fp:
    pickle.dump(results9, fp)
              

# Aggregate and plot results
A_to_B = results9["A_to_B"]
B_to_A = results9["B_to_A"]

# Plot example pair (
plt.figure(figsize=(10, 6))
plt.plot(lags9, A_to_B[0, 0, :], label="MI(A1 -> B1)", color="blue")
plt.plot(lags9, B_to_A[0, 0, :], label="MI(B1 -> A1)", color="red")
plt.axvline(0, color='black', linestyle='--', label='Zero lag')
plt.xlabel("Lag (time steps)")
plt.ylabel("Mutual Information")
plt.title("Directionality Flow: A1 <-> B1")
plt.legend()
plt.show()



