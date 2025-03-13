"""
Sharp-PINNs for pitting corrosion with 2d-1pit
"""

class Config:
    EPOCHS = 100000
    N_SAMPLES = 20
    ADAPTIVE_SAMPLES = 8000
    ADAPTIVE_BASE_RATE = 5
    LR = 5e-4
    DECAY = 0.9
    DECAY_EVERY = 200
    STAGGER_PERIOD = 25
    EMB_SCALE = (1.5, 2.0) # emb sacle for (x, t)
    EMB_DIM = 64

    DOMAIN = ((-0.5, 0.5), (-0.5, 0.5), (0, 0.5), (0, 1.0))
    DATA_PATH = "./data/3d-1pit/"
    LOG_DIR = "/root/tf-logs"
    PREFIX = "3d-1pit"
    TS = [0.000, 1.968, 6.401, 9.357]

    NUM_LAYERS = 6
    HIDDEN_DIM = 128
    OUT_DIM = 2


    ACT_NAME = "tanh"
    ARCH_NAME = "modified_mlp"
    FOURIER_EMB = True
    CAUSAL_WEIGHT = True

    ALPHA_PHI = 1.03e-4
    OMEGA_PHI = 1.76e7
    MM = 7.94e-18
    DD = 8.5e-10
    AA = 5.35e7
    LP = 2.0
    CSE = 1.0
    CLE = 5100 / 1.43e5


    Lc = 1e-4
    Tc = 10.0
    AC_PRE_SCALE = 1e6
    CH_PRE_SCALE = 1e0


    CAUSAL_CONFIGS = {
        "ac_eps": 1e-5,
        "ch_eps": 1e-5,
        "step_size": 10,
        "max_last_weight": 0.99,
        "min_mean_weight": 0.5,
        "max_eps": 1e0,
        "chunks": 24,
    }


# if __name__ == "__main__":
#     for key, value in Config.__dict__.items():
#         # 将所以的item作为全局变量，key = value 的形式
#         if not key.startswith("__"):
#             globals()[key] = value