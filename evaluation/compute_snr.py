def compute_snr(reference, estimate):
    noise = reference - estimate
    signal_power = np.sum(reference**2)
    noise_power = np.sum(noise**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else np.inf
    return snr

# Compute SNR for all 6 test images
snr_values = []
for i in range(6):
    snr = compute_snr(ground_truth_test[i], predicted_output[i])
    snr_values.append(snr)
    print(f"Image {i+1} → SNR: {snr:.2f} dB")
