# -*- coding: utf-8 -*-

import os
import logging
import platform
from subprocess import Popen
import shlex
import sys
import time
from threading import Thread
from .Config import FTG_PATH
sys.path.append(os.path.join(FTG_PATH, 'python'))
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters
from py4j.java_gateway import CallbackServerParameters
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters
from py4j.java_gateway import CallbackServerParameters


logger = logging.getLogger('BasicBot')


class FTGEnv(object):
    # launching Fighting Game AI platform as a separated process and manage it
    def __init__(self, _id, player_1, player_2, *args, **kwargs):
        self.id = _id
        self.train = kwargs.get('train', True)
        self.port = kwargs.get('port', 4242)
        self.black_bg = kwargs.get('black_bg', True)
        self.disable_window = kwargs.get('disable_window', True)
        self.verbose = kwargs.get('verbose', False)
        self.player_1 = player_1  # player 1 or 2 class
        self.player_2 = player_2  # not instance
        self.inverted_player_number = kwargs.get('inverted_player', 2)
        self.starts_with_energy = kwargs.get('starts_with_energy', False)

        self.log_off = kwargs.get('log_off', True)

        self.game_proc = None  # FTG game server process
        self.game_thread = None  # game client thread (belong to this process)
        self.manager = None
        self.gate_way = None

    def __del__(self):
        self.close()

    def close(self):
        if self.game_proc:
            logger.info('Terminate FTG {}'.format(self.id))
            self.game_proc.terminate()

    def _start_server(self, **kwargs):
        root_path = kwargs.get('root_path', '')

        #cps = ['FightingICE.jar', 'AIToolKit.jar', 'lib/lwjgl/*', 'lib/*', 'lib/natives/{}/*'.format(platform.system().lower())]
        cps = ['./bin', './lib/javax.json-1.0.4.jar', './lib/py4j0.10.4.jar', './lib/lwjgl/*', './lib/natives/{}/*'.format(platform.system().lower())]
        cps += [os.path.join('data/ai', p) for p in os.listdir('data/ai') if p.endswith('.jar')]
        cps = [os.path.join(root_path, cp) for cp in cps]
        #print('CLASSPATH', ':'.join(cps)) # FIXME
        #cmd = ['java', '-cp', ';'.join(cps), 'Main', '--py4j']
        cmd = ['java', '-cp', ':'.join(cps), 'Main', '--py4j']

        # cps = ['FightingICE.jar', 'lib/gameLib.jar', 'lib/lwjgl.jar',
        #        'lib/lwjgl_util.jar', 'lib/javatuples-1.2.jar', 'lib/commons_csv.jar',
        #        'lib/jinput.jar', 'lib/fileLib.jar', 'lib/py4j0.10.4.jar']
        # cps += [os.path.join('data/ai', p) for p in os.listdir('data/ai') if p.endswith('.jar')]
        # cps = [os.path.join(root_path, cp) for cp in cps]
        # cmd = [
        #     'java',
        #     '-cp', ':'.join(cps),
        #     '-Djava.library.path="{}"'.format(os.path.join(root_path, 'lib/native/linux')),
        #     'Main',
        #     '--py4j',
        # ]

        if self.black_bg:
            cmd.append('--black-bg')

        cmd.append('--inverted-player {}'.format(self.inverted_player_number))
        # cmd.append('--inverted-player 2')
        cmd.append('--port %d' % self.port)
        cmd.append('--limithp 400 400')

        if self.disable_window:
            cmd.append("--disable-window")

        # if self.log_off:
        #     cmd.append('--off')
        #     cmd.append('--del')

        if self.starts_with_energy:
            cmd.append('-t')

        if not self.verbose:
            server_log = open(os.devnull, 'w')
        else:
            import sys
            server_log = sys.stdout
            
        #print(cmd)

        # FTG process start
        self.game_proc = Popen(shlex.split(' '.join(cmd)), stdout=server_log, stderr=server_log)
        time.sleep(3)

    def _create_player(self, gateway, manager, player):
        if isinstance(player, str):
            # java bot: string input
            player_name, player_character = player.split(':')
            player_obj = None
        else:
            # python bot: python class input
            player._on_train = self.train
            player_obj = player(gateway)
            player_name = player_obj.__class__.__name__
            manager.registerAI(player_name, player_obj)
            player_character = player_obj.getCharacter()
        return player_character, player_name, player_obj

    def _make_a_game(self, p1, p2, n_game, **kwargs):
        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=self.port),
                                   callback_server_parameters=CallbackServerParameters(port=0), )
        python_port = self.gateway.get_callback_server().get_listening_port()
        # logger.info(str(self.gateway.java_gateway_server.getCallbackClient().getAddress(), python_port))
        self.gateway.java_gateway_server.resetCallbackClient(
            self.gateway.java_gateway_server.getCallbackClient().getAddress(), python_port)
        self.manager = self.gateway.entry_point

        connect_monitor = kwargs.get('ai_monitor', lambda a, b, c: None)

        def run_a_game():
            # n = 0
            # while n_game is None or n < n_game:
            #     p1_character, p1_name, p1_obj = self._create_player(self.gateway, self.manager, p1)
            #     connect_monitor(self.id, 1, p1_obj)
            #     p2_character, p2_name, p2_obj = self._create_player(self.gateway, self.manager, p2)
            #     connect_monitor(self.id, 2, p2_obj)
            #     game = self.manager.createGame(p1_character, p2_character, p1_name, p2_name, 1)
            #     self.manager.runGame(game)
            #     n += 1
            
            p1_character, p1_name, p1_obj = self._create_player(self.gateway, self.manager, p1)
            connect_monitor(self.id, 1, p1_obj)
            p2_character, p2_name, p2_obj = self._create_player(self.gateway, self.manager, p2)
            connect_monitor(self.id, 2, p2_obj)
            game = self.manager.createGame(p1_character, p2_character, p1_name, p2_name, n_game)
            print("GAME", p1_character, p2_character, p1_name, p2_name, n_game)
            self.manager.runGame(game)
            
            self.gateway.shutdown_callback_server(raise_exception=False)
            self.gateway.shutdown(raise_exception=False)
            sys.stdout.flush()

        return run_a_game

    def run(self, path=FTG_PATH, block=True, n_game=None, **kwargs):
        self.close()
        orig_path = os.path.abspath('.')
        os.chdir(path)
        # change directory to FTG platform
        self._start_server(**kwargs)
        game = self._make_a_game(self.player_1, self.player_2, n_game, **kwargs)
        os.chdir(orig_path)

        if block:
            game()
        else:
            self.game_thread = Thread(target=game)
            self.game_thread.daemon = True
            self.game_thread.start()
