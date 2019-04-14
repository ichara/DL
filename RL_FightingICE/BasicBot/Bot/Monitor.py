# -*- coding: utf-8 -*-

import time
import logging
import numpy as np
from threading import Thread
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .Config import resolution


logger = logging.getLogger('BasicBot')


class PyGameMonitor(object):
    # show input data using pygame surface
    def __init__(self, height, width, debug_queue):
        import pygame
        import skimage.transform

        pygame.init()
        pygame.display.set_caption('FTG3')
        self.display = pygame.display
        self.pixelcopy = pygame.pixelcopy
        self.resize = skimage.transform.resize

        self.debug_port = debug_queue
        self.scale = 3
        self.n_frame = resolution[0]
        self.height = height
        self.width = width
        self.img_buff = np.zeros((self.height, self.n_frame * self.width, 3))
        self.screen = pygame.display.set_mode(
            (self.n_frame * self.scale * self.width, self.scale * self.height))
        self.surf = pygame.Surface(
            (self.n_frame * self.scale * self.width, self.scale * self.height))

    def set_image(self, screen):
        for i in range(self.n_frame):
            ob = screen[i].reshape((self.height, self.width))
            zeros = np.zeros((self.height, self.width))
            my_char = np.where(ob > 0, ob, zeros)
            opp_char = np.where(ob < 0, - ob, zeros)
            self.img_buff[:, i * self.width:(i + 1) * self.width, :] \
                = np.stack([my_char, 0.5 * (my_char + opp_char), opp_char], axis=2)

    def show(self):
        img = self.resize(self.img_buff,
                          (self.scale * self.height, self.n_frame * self.scale * self.width))
        img = (255 * img).astype(np.uint8).swapaxes(0, 1)
        self.pixelcopy.array_to_surface(self.surf, img)
        self.screen.blit(self.surf, (0, 0))
        self.display.flip()

    def run(self):
        while True:
            screen, info = self.debug_port.get()
            self.set_image(screen)
            self.show()
            time.sleep(0.016)


class MatplotlibMonitor(object):
    # show input image state and energy value and output action
    def __init__(self, height, width, debug_queue):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
        except ImportError as exc:
            logger.error('Install matplotlib')
            logger.error(exc)
            exit(-1)

        self.height, self.width = height, width
        self.debug_port = debug_queue

    def run(self):
        channel, height, width = resolution
        display_buffers = [np.zeros((resolution[1], resolution[2])) for _ in range(channel)]
        text_buffer = ['' for _ in range(channel)]
        tick_time = 16.67

        def frames():
            cnt = 0
            screens, info = np.random.random(resolution) * 0.5 - 0.25, dict()

            while True:
                if not self.debug_port.empty():
                    cnt = 0
                    screens, info = self.debug_port.get_nowait()
                else:
                    cnt += 1
                    if cnt > 10:
                        screens, info = np.random.random(resolution) * 0.5 - 0.25, dict()

                yield screens, info
                # time.sleep(0.016)

        fig, axes = plt.subplots(1, 4, figsize=(30, 4), sharey=True)
        fig.suptitle('Vision')
        imgs = list()
        txts = list()
        for i, ax in enumerate(axes):
            ax.set_title('t - {}'.format(channel - i - 1))
            img = ax.imshow(np.zeros((resolution[1], resolution[2]), dtype=np.float32),
                            vmin=-1, vmax=1, cmap='bwr', interpolation='none')
            imgs.append(img)

            text = ax.text(5, 15, '----', fontsize=18)
            txts.append(text)

        def set_data(data):
            screens, info = data
            for i, (screen, disp_buffer, img, text) in enumerate(zip(screens, display_buffers, imgs, txts)):
                disp_buffer[:] = screen[:]
                text_buffer[i] = 'A: {}\nE: {}\nR :{}'.format(
                    info.get('action', 'No data'),
                    info.get('energy', 'No data'),
                    info.get('reward', 'No data'))

                img.set_data(disp_buffer)
                text.set_text(text_buffer[i])

            return imgs + txts

        ani = animation.FuncAnimation(fig, set_data, frames, blit=True, interval=tick_time, repeat=False)
        plt.show()


def make_monitor(kind, env_id=0, player_no=1):

    def func_do_nothing(a, b, c):
        return a, b, c

    from multiprocessing import Queue
    from multiprocessing import Process

    try:
        import pygame
    except ImportError:
        logger.warning("Can't import pygame")
        return func_do_nothing

    try:
        import skimage.transform
    except ImportError:
        logger.warning("Can't import scikit-image")
        import skimage.transform
        return func_do_nothing

    debug_port = Queue()
    if kind == 'pygame':
        logger.debug('Make PyGame monitor')
        monitor = PyGameMonitor(resolution[1], resolution[2], debug_port)
        thread = Thread(target=monitor.run)
        thread.daemon = True
        thread.start()
    elif kind == 'matplotlib':
        logger.debug('Make Matplotlib monitor')
        monitor = MatplotlibMonitor(resolution[1], resolution[2], debug_port)
        proc = Process(target=monitor.run)
        proc.daemon = True
        proc.start()
    else:
        monitor = None
        return lambda a, b, c: func_do_nothing

    def connect_monitor(eid, pno, player_obj):
        if eid == env_id and pno == player_no and player_obj is not None:
            setattr(player_obj, 'debug_port', monitor.debug_port)
        else:
            if player_obj is None:
                logger.warning("Can't monitor java bot")
            return func_do_nothing

    return connect_monitor