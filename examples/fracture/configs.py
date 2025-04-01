"""
Sharp-PINNs for pitting corrosion with 2d-1pit
"""


class Config:
    EPOCHS = 2000
    N_SAMPLES = 20
    ADAPTIVE_SAMPLES = 2000
    ADAPTIVE_BASE_RATE = 5
    LR = 5e-4
    DECAY = 0.9
    DECAY_EVERY = 200
    STAGGER_PERIOD = 25
    EMB_SCALE = (1.5, 0.5)  # emb sacle for (x, t)
    EMB_DIM = 64

    DOMAIN = [[-0.5, 0.5], [0, 0.5], [0, 2.0]]
    DATA_PATH = "./data/fracture/"
    LOG_DIR = "/root/tf-logs"
    PREFIX = "corrosion/fracture/irr"
    TS = [0.000, 3.582, 9.726, 19.966]

    NUM_LAYERS = 6
    HIDDEN_DIM = 128
    OUT_DIM = 3

    ACT_NAME = "gelu"
    ARCH_NAME = "modified_mlp"
    ASYMMETRIC = True
    FOURIER_EMB = True
    CAUSAL_WEIGHT = False
    IRR = True

    GC = 2.7
    L = 0.015
    LMBDA = 121.1538e3
    MU = 80.7692e3

    Lc = 1.0
    Tc = 1.0
    DISP_PRE_SCALE = 1e3
    AC_PRE_SCALE = 1e6
    CH_PRE_SCALE = 1e0

    CAUSAL_CONFIGS = {
        "eps": 1e-4,
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
