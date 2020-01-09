

class NoamOptimizer:
    """
        This Hook implements the optimization strategy presented in the "Attention is all you need" paper
        Section 5.3.
    """

    timing = "pre"
    name = "NoamOptimizerHook"
    call_for_each_param = False

    def __init__(self, num_warmup_steps, factor, model_size):
        self.num_warmup_steps = num_warmup_steps
        self.factor = factor
        self.model_size = model_size
        self.iteration = 0

    def __call__(self, optimizer):
        self.iteration += 1

        warmup_vs_step_num = min(self.iteration ** (-0.5), self.iteration * self.num_warmup_steps ** (-1.5))
        optimizer.alpha = self.factor * self.model_size ** (-0.5) * warmup_vs_step_num
