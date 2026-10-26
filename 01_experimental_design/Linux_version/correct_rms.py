import numpy as np

# Final experimental RMS used during the experiment
rms_exp = 0.1

# Participant-specific threshold RMS values
threshold_rms = {
    "ID2":  0.00010537619208999369,
    "ID5":  0.00006129895910320457,
    "ID6":  0.0005113351279121514,
    "ID8":  0.000149597963964315,
    "ID10": 0.00013700900346767426,
    "ID13": 0.0007247619644548085
}

print("Corrected estimated level above threshold\n")

for participant, rms_ref in threshold_rms.items():

    level_db = 20 * np.log10(
        rms_exp / rms_ref
    )

    print(
        f"{participant}: "
        f"Threshold RMS = {rms_ref:.12f}, "
        f"Experimental RMS = {rms_exp:.2f}, "
        f"Estimated level above threshold = {level_db:.2f} dB"
    )
    