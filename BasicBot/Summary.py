
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd


class Summary(object):

    attributes = ['accuracy', 'wall-time', 'learning-rate', 'train-loss', 'test-loss']

    def __init__(self, csv_files, attrs=None):
        csv_files = sorted(csv_files, reverse=True)
        if attrs:
            self.attributes = attrs

        self.data = dict()
        for attr in self.attributes:
            self.data[attr] = pd.DataFrame(index=range(10))

        for csv_file in csv_files:
            kind = os.path.split(csv_file)[-2]
            df = pd.read_csv(csv_file, index_col=0)

            for attr in self.attributes:
                if attr in df.columns:
                    self.data[attr] = pd.concat([self.data[attr], df[[attr]]], axis=1)
                    cols = [col for col in self.data[attr].columns]
                    cols[-1] = kind
                    self.data[attr].columns = cols

    def plot(self, key, grouped=True, out_file=None, legend=True, title=True, logy=False, alpha=0.2, **kwargs):
        df = self.data[key]
        df = df.dropna(axis=1, how='all')
        if grouped:
            df.columns = [col.rsplit('-', 1)[0] for col in df.columns]
        df = df.transpose().reset_index()
        df = df.groupby('index').agg(['max', 'median', 'min']).transpose()

        labels = list()
        lines = list()

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        if title:
            ax.set_title(key.title())

        if logy:
            ax.set_yscale('log')

        for column in df.columns:
            max_vals = df.loc(axis=0)[:, 'max'][[column]].values
            min_vals = df.loc(axis=0)[:, 'min'][[column]].values
            median_vals = df.loc(axis=0)[:, 'median'][[column]].values
            count = median_vals.shape[0]

            line, = plt.plot(median_vals.reshape(count))
            color = line.get_color()
            ax.fill_between(range(count), max_vals.reshape(count), min_vals.reshape(count),
                            facecolor=color, alpha=alpha)
            labels.append(column)
            lines.append(line)

        if legend:
            plt.legend(lines, labels)
            
        if kwargs.get('xlabel', None):
            plt.xlabel(kwargs['xlabel'])
            
        if kwargs.get('ylabel', None):
            plt.ylabel(kwargs['ylabel'])

        plt.tight_layout()

        if out_file:
            fig.savefig(out_file)

if __name__ == '__main__':

    files = glob.glob('output/*-*/*.csv')
    summary = Summary(files, ['test-score', 'train-score', 'loss'])

    attr = 'test-score'
    print(summary.data[attr])
    summary.plot(attr, out_file='assets/test-score.png', grouped=True, alpha=0.2, legend=False, title=False, xlabel='epoch', ylabel='score')
    plt.show()
    
    attr = 'train-score'
    print(summary.data[attr])
    summary.plot(attr, out_file='assets/train-score.png', grouped=True, alpha=0.2, legend=False, title=False, xlabel='epoch', ylabel='score')
    plt.show()

    attr = 'loss'
    print(summary.data[attr])
    summary.plot(attr, out_file='assets/loss.png', grouped=True, alpha=0.2, legend=False, title=False, logy=True, xlabel='epoch', ylabel='loss')
    plt.show()
