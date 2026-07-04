import numpy as np
import pywt
from scipy.stats import ttest_ind

def haar_wavelet_transform(data):
    """
    Apply Haar wavelet transform to the input time series data.
    """
    coeffs = pywt.wavedec(data, 'haar', level=None)
    return coeffs

def detect_anomalies(data, significance_level=0.05):
    """
    Detect anomalies in time series data using Haar wavelet transform and t-test.
    
    Parameters:
        data (numpy.ndarray): Input time series data.
        significance_level (float): Significance level for the t-test.
        
    Returns:
        numpy.ndarray: Boolean array indicating anomalies (True for anomaly).
    """
    # Apply Haar wavelet transform
    coeffs = haar_wavelet_transform(data)
    
    # Reconstruct approximation coefficients (low-frequency components)
    approx_coeffs = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], 'haar')
    
    # Compute residuals (high-frequency components)
    residuals = data - approx_coeffs
    
    # Split residuals into two groups: first half and second half
    mid_point = len(residuals) // 2
    group1 = residuals[:mid_point]
    group2 = residuals[mid_point:]
    
    # Perform t-test between the two groups
    t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
    
    # Identify anomalies based on p-value
    anomalies = np.abs(residuals) > np.percentile(np.abs(residuals), 100 * (1 - significance_level))
    
    return anomalies

if __name__ == '__main__':
    # Generate dummy time series data
    np.random.seed(42)
    normal_data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    anomaly_data = normal_data.copy()
    anomaly_data[50:55] += 3  # Inject anomalies
    
    # Detect anomalies
    anomalies = detect_anomalies(anomaly_data, significance_level=0.05)
    
    # Print results
    print("Anomalies detected at indices:", np.where(anomalies)[0])