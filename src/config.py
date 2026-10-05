from pathlib import Path

# Demo vocabulary: 10 hospital-reception signs + "idle" (hands up but NOT signing).
# Learn the correct ISL version of each sign from a trusted ISL dictionary (ISLRTC).
SIGNS = ["hello", "thanks", "help", "water", "doctor",
         "pain", "yes", "no", "medicine", "where", "idle"]

SEQ_LEN = 30            # frames per sign
N_SEQUENCES = 40        # recordings per sign per person
FEATURES = 225          # 33 pose*3 + 21 left hand*3 + 21 right hand*3
CONF_THRESHOLD = 0.85

DATA_DIR = Path("../data/raw")
MODEL_PATH = Path("../models/sign_model.keras")
LABELS_PATH = Path("../models/labels.json")