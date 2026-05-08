import numpy as np

# PHYSICAL CONSTANTS
FS = 2500
SPEED_OF_SOUND = 343 #m/s
DUCT_DIAMATER = 110 # mm
CUTOFF_FREQ = 1827.0

# DUCT GEOMETRY (Meters)
# Distances measured from the noise source
DIST_REF_MIC = 0.2            # Distance to Reference
DIST_SEC_SPK = 1.5            # Distance to Secondary
DIST_ERR_MIC = 1.8            # Distance to Error Microphone

# Calculated sample delay
PRIMARY_DELAY_SAMPLES = int((DIST_ERR_MIC / SPEED_OF_SOUND) * FS) # Source -> Error Mic
SECONDARY_DELAY_SAMPLES = int(((DIST_ERR_MIC - DIST_SEC_SPK) / SPEED_OF_SOUND) * FS) # Secondary -> Error Mic

# MODELLING Stage, the system learning the map of the secondary path
M_TAPS = 127                  # Number of taps for the Controller H(z) (FIR Filter)
S_TAPS = 50                  # Number of taps for the Secondary Path Model S_hat(z)
MU_CONTROLLER = 0.0005         # Step size for the FxLMS update (Engine B)
MU_MODELING = 0.1          # Step size for the S_hat update (Engine A)

# Invasive Modeling paramaters
V_NOISE_AMP = 0.1            # Volume level of the "Invasive" white noise injection


# Non-invasive Modeling parameters


# Signal Characteristics
TONES = [100, 200, 400]       # Target frequencies
