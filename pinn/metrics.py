import datetime
from tensorboardX import SummaryWriter


class MetricsTracker(SummaryWriter):

    def __init__(self, log_path: str):
        super().__init__(log_path)

    def register_scalars(self, step: int, names: list, values: list):
        for name, value in zip(names, values):
            self.add_scalar(name, value, step)

    def register_figure(self, step: int, fig, fig_name: str):
        self.add_figure(fig_name, fig, step)
