class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count