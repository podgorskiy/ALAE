import csv
import matplotlib.pyplot as plt


class RunningMean:
    def __init__(self):
        self.mean = 0.0
        self.n = 0

    def __lshift__(self, value):
        self.mean = (float(value) + self.mean * self.n)/(self.n + 1)
        self.n += 1


class LossTracker:
    def __init__(self):
        self.tracks = {}
        self.epochs = []
        self.means_over_epochs = {}

    def add(self, name):
        assert name not in self.tracks, "Name is already used"
        track = RunningMean()
        self.tracks[name] = track
        self.means_over_epochs[name] = []
        return track

    def register_means(self, epoch):
        self.epochs.append(epoch)
        for key, value in self.tracks.items():
            self.means_over_epochs[key].append(value.mean)
            value.mean = 0.0
            value.n = 0

        with open('log.csv', mode='w') as csv_file:
            fieldnames = ['epoch'] + list(self.tracks.keys())
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fieldnames)
            for i in range(len(self.epochs)):
                writer.writerow([self.epochs[i]] + [self.means_over_epochs[x][i] for x in self.tracks.keys()])

    def __str__(self):
        result = ""
        for key, value in self.tracks.items():
            result += "%s: %.3f, " % (key, value.mean)
        return result[:-2]

    def plot(self):
        for key in self.tracks.keys():
            plt.plot(self.epochs, self.means_over_epochs[key], label=key)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig('plot.png')
        plt.close()
