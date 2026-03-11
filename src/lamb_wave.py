"""
Lamb Wave Analysis for Structural Health Monitoring
====================================================
Signal processing and damage localization algorithms for
inkjet-printed piezoelectric ultrasonic sensor arrays.

Author: Maximilian Kopp
"""

import numpy as np
from scipy.signal import hilbert, correlate


def time_of_flight(signal_tx: np.ndarray, signal_rx: np.ndarray, fs: float) -> float:
    """Estimate time-of-flight between transmitter and receiver signals.

    Uses cross-correlation to find the time delay between
    transmitted and received Lamb wave packets.

    Parameters
    ----------
    signal_tx : np.ndarray
        Transmitted signal.
    signal_rx : np.ndarray
        Received signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    float
        Time-of-flight in seconds.
    """
    correlation = correlate(signal_rx, signal_tx, mode="full")
    lag_index = np.argmax(np.abs(correlation)) - (len(signal_tx) - 1)
    return lag_index / fs


def envelope_extraction(signal: np.ndarray) -> np.ndarray:
    """Extract signal envelope using Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        Input ultrasonic signal.

    Returns
    -------
    np.ndarray
        Signal envelope (amplitude modulation).
    """
    analytic = hilbert(signal)
    return np.abs(analytic)


def localize_damage(
    sensor_positions: np.ndarray,
    tof_matrix: np.ndarray,
    wave_velocity: float,
) -> np.ndarray:
    """Localize structural damage using time-of-flight triangulation.

    Parameters
    ----------
    sensor_positions : np.ndarray
        Array of sensor (x, y) coordinates, shape (N_sensors, 2).
    tof_matrix : np.ndarray
        Symmetric matrix of time-of-flight differences between
        baseline and damaged states, shape (N_sensors, N_sensors).
    wave_velocity : float
        Group velocity of Lamb wave mode in m/s.

    Returns
    -------
    np.ndarray
        Estimated damage location [x, y] in meters.
    """
    n_sensors = len(sensor_positions)
    grid_size = 200
    x_range = np.linspace(
        sensor_positions[:, 0].min() - 0.05,
        sensor_positions[:, 0].max() + 0.05,
        grid_size,
    )
    y_range = np.linspace(
        sensor_positions[:, 1].min() - 0.05,
        sensor_positions[:, 1].max() + 0.05,
        grid_size,
    )

    damage_image = np.zeros((grid_size, grid_size))

    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            delta_tof = tof_matrix[i, j]
            if abs(delta_tof) < 1e-12:
                continue
            for xi, x in enumerate(x_range):
                for yi, y in enumerate(y_range):
                    d_i = np.sqrt(
                        (x - sensor_positions[i, 0]) ** 2
                        + (y - sensor_positions[i, 1]) ** 2
                    )
                    d_j = np.sqrt(
                        (x - sensor_positions[j, 0]) ** 2
                        + (y - sensor_positions[j, 1]) ** 2
                    )
                    expected_tof = (d_i + d_j) / wave_velocity
                    residual = abs(expected_tof - abs(delta_tof))
                    damage_image[yi, xi] += 1.0 / (residual + 1e-6)

    max_idx = np.unravel_index(np.argmax(damage_image), damage_image.shape)
    return np.array([x_range[max_idx[1]], y_range[max_idx[0]]])


def damage_index(baseline: np.ndarray, current: np.ndarray) -> float:
    """Compute damage index from baseline and current signals.

    Parameters
    ----------
    baseline : np.ndarray
        Baseline (healthy) signal.
    current : np.ndarray
        Current (potentially damaged) signal.

    Returns
    -------
    float
        Damage index (0 = no change, higher = more damage).
    """
    correlation = np.corrcoef(baseline, current)[0, 1]
    return 1.0 - abs(correlation)
