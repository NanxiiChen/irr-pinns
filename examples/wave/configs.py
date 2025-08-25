import math

class Config:
    EPOCHS = 50000
    N_SAMPLES = 30
    ADAPTIVE_SAMPLES = 500
    ADAPTIVE_BASE_RATE = 5
    LR = 5e-4
    DECAY = 0.9
    DECAY_EVERY = 2000
    SAVE_EVERY = 100
    EMB_SCALE = (4.0, 4.0)  # emb sacle for (x, t)
    EMB_DIM = 64

    DOMAIN = [[0, 2.0], [0, 1.0]]
    LOG_DIR = "/root/autodl-tmp/tf-logs"
    DATA_PATH = "./data/wave/"
    PREFIX = "wave/irr"
    RESUME = None
    TS = [0.0, 0.25, 0.50, 0.75, 1.0]

    NUM_LAYERS = 6
    HIDDEN_DIM = 64
    OUT_DIM = 1

    ACT_NAME = "tanh"
    ARCH_NAME = "modified_mlp"
    OPTIMIZER = "adam"
    FOURIER_EMB = True
    CAUSAL_WEIGHT = False
    IRR = True

    VELOCITY = 1.0
    ALPHA = 0.1
    AMPLITUDE = 1.0
    OMEGA = 4.0 * math.pi

    Lc = 1.0
    Tc = 10.0
    PDE_PRE_SCALE = 1.0

    CAUSAL_CONFIGS = {
        "eps": 1e-2,
        "step_size": 10,
        "max_last_weight": 0.99,
        "min_mean_weight": 0.5,
        "max_eps": 1e-3,
        "chunks": 24,
    }


# if __name__ == "__main__":
#     for key, value in Config.__dict__.items():
#         # 将所以的item作为全局变量，key = value 的形式
#         if not key.startswith("__"):
#             globals()[key] = value
