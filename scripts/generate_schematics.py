"""
Generate presentation schematics:
1. Kernel Convolution Concept
   - Visualizing Surface vs Baseflow response to a single storm.
2. Simulation vs One-Step Ahead conceptual comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

def generate_kernel_concept():
    plt.figure(figsize=(10, 6))
    
    x = np.linspace(0, 50, 500)
    
    # 1. Surface Flow (Fast, Intense)
    # Shape ~ 2, Scale ~ 2 => Peak ~ 2 days
    k1 = gamma.pdf(x, a=2.5, scale=1.5)
    plt.plot(x, k1, lw=3, color='#E67E22', label='Surface Flow (Fast Response)')
    plt.fill_between(x, k1, color='#E67E22', alpha=0.2)
    
    # 2. Interflow (Medium)
    # Shape ~ 3, Scale ~ 4 => Peak ~ 8 days
    k2 = gamma.pdf(x, a=3.0, scale=4.0) * 0.6
    plt.plot(x, k2, lw=3, color='#D35400', label='Interflow (Medium Response)')
    plt.fill_between(x, k2, color='#D35400', alpha=0.2)
    
    # 3. Baseflow (Slow, Sustained)
    # Shape ~ 5, Scale ~ 8 => Peak ~ 32 days
    k3 = gamma.pdf(x, a=5.0, scale=8.0) * 0.3
    plt.plot(x, k3, lw=3, color='#BA4A00', label='Baseflow (Slow Response)')
    plt.fill_between(x, k3, color='#BA4A00', alpha=0.2)
    
    plt.title('Conceptual Hydrological Response Kernels', fontsize=16)
    plt.xlabel('Lag Time (Days)', fontsize=12)
    plt.ylabel('Influence Strength', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/images/kernel_concept.png', dpi=300)
    plt.close()

def generate_mode_concept():
    # Conceptual Time Series
    t = np.arange(50)
    y_true = np.sin(t/5) + np.random.normal(0, 0.2, 50) + 2
    y_pred_sim = np.sin(t/5) + 2 # Smooth
    y_pred_krig = y_true + np.random.normal(0, 0.05, 50) # Updated
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Simulation Mode
    ax1.plot(t, y_true, 'k--', label='Observed Streamflow', alpha=0.5)
    ax1.plot(t, y_pred_sim, color='#E67E22', lw=2, label='Simulation (Input Only)')
    ax1.set_title("Simulation Mode: Pure Process Understanding", fontsize=14, color='#E67E22')
    ax1.text(25, 1.5, r"$y_t = f(Rainfall_{t-L:t}) + \epsilon$", fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Forecasting Mode
    ax2.plot(t, y_true, 'k--', label='Observed Streamflow', alpha=0.5)
    ax2.plot(t, y_pred_krig, color='#BA4A00', lw=2, label='One-Step Forecast (Input + History)')
    ax2.set_title("Forecasting Mode: Operational Prediction", fontsize=14, color='#BA4A00')
    ax2.text(25, 1.5, r"$y_{t+1} = f(Rainfall) + \Sigma_{new} \Sigma_{old}^{-1} (y_t - \hat{y}_t)$", fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/modes_concept.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_kernel_concept()
    generate_mode_concept()
