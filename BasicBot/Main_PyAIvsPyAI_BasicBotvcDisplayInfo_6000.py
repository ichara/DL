import argparse
import sys
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters
from py4j.java_gateway import CallbackServerParameters
from py4j.java_gateway import get_field
from BasicBot import BasicBot
from DisplayInfo import DisplayInfo


def start_game(n_game):
    for i in range(n_game):
        basic_bot, display_info = BasicBot(gateway), DisplayInfo(gateway)
        manager.registerAI("BasicBot", basic_bot)
        manager.registerAI("DisplayInfo", display_info)
        print("Start game", i)
    
        game = manager.createGame(
                        basic_bot.getCharacter(), display_info.getCharacter(), 
                        "BasicBot", "DisplayInfo")
        manager.runGame(game)
    
        print("After game", i)
        sys.stdout.flush()


def close_gateway():
    gateway.close_callback_server()
    gateway.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Basic bot vs. Display info')
    parser.add_argument('--port', default=6000, type=int, help='game server port')
    parser.add_argument('-n', '--number', default=1, type=int, help='number of game')
    args = parser.parse_args()

    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=args.port),
                          callback_server_parameters=CallbackServerParameters(port=0))
    python_port = gateway.get_callback_server().get_listening_port()
    gateway.java_gateway_server.resetCallbackClient(
        gateway.java_gateway_server.getCallbackClient().getAddress(), python_port)
    manager = gateway.entry_point

    start_game(args.number)
    close_gateway()


