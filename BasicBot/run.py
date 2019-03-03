
from subprocess import Popen
from subprocess import PIPE
from shlex import split
from time import sleep
from glob import glob
import psutil
import matplotlib.pyplot as plt
import IPython
from Summary import Summary


commands = [
    ['python', 'BasicBot.py'],
    ['java', '-cp', 'FightingICE.jar:lib/gameLib.jar:lib/lwjgl.jar:lib/lwjgl_util.jar:lib/javatuples-1.2.jar'
                    ':lib/commons_csv.jar:lib/jinput.jar:lib/fileLib.jar:lib/py4j0.10.4.jar:data/ai/MctsAi.jar'
                    ':data/ai/RandomAI.jar', '-Djava.library.path=lib/native/linux', 'Main', '--py4j']]

training_steps = 30000
evaluation_sec = 120
max_epoch = 15
n_cpu = 2
n_experiments = 5
monitor = 'pygame'

cmd_format = 'python BasicBot.py train -ts {training_steps} -es {evaluation_sec} ' \
             '-o SandBag -n {n_cpu} --max-epoch {max_epoch} -m {monitor} ' \
             '-p output/SandBag-{n_expr} --render none'


if __name__ == '__main__':

    for n_expr in range(n_experiments):
        cmd = cmd_format.format_map(globals())
        print('>>> {}'.format(cmd))
        p = Popen(split(cmd), stdout=PIPE, stderr=PIPE)

        while True:
            buff = p.stderr.readline()
            buff = buff.decode('utf-8')
            if buff.strip() == 'Press ctrl + c many times to exit'.strip():
                p.terminate()
                ################################################
                ##### WARNING: KILL ALL MATCHED PROCESSES #####
                ################################################
                for proc in psutil.process_iter():
                    for cmdline in commands:
                        s1 = set(cmdline)
                        s2 = set(proc.cmdline())
                        if s1.issubset(s2):
                            print('Kill: ' + ' '.join(proc.cmdline()))
                            proc.kill()
                sleep(3)
                break
            else:
                print(buff, end='')

        print('** EXPR {} END **'.format(n_expr))
    print('** END ALL EXPRs **')

    files = glob('output/SandBag-*/*.csv')
    cols = ['test-score', 'train-score', 'loss']
    summary = Summary(files, cols)

    for col in cols:
        print('\n** PLOT: {} **'.format(col.title()))
        print(summary.data[col])
        logy = True if col == 'loss' else False
        summary.plot(col, out_file='assets/{}.png'.format(col), grouped=True, alpha=0.2,
                     legend=False, title=False, logy=logy)
        plt.show()

    IPython.embed()




