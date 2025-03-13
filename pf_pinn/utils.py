class StaggerSwitch:
    def __init__(self, pde_names=["ac", "ch", "ch", "ch"], stagger_period=10):
        self.pde_names = pde_names
        self.stagger_period = stagger_period
        self.epoch = 0

    def step_epoch(self):
        self.epoch += 1

    def decide_pde(self):
        epoch_round = len(self.pde_names) * self.stagger_period
        idx = (self.epoch % epoch_round) // self.stagger_period
        return self.pde_names[idx]
    

if __name__ == "__main__":
    stagger_switch = StaggerSwitch()
    for i in range(50):
        print(i, stagger_switch.decide_pde())
        stagger_switch.step_epoch()