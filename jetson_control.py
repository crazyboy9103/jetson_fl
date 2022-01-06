import logging, socket, paramiko.ssh_exception
import fabric
from fabric import Connection, Config, SerialGroup, ThreadingGroup, exceptions, runners
from fabric.exceptions import GroupException
import argparse
import requests

error_ports = [20113]
import time
import threading

class Jetson:
    def __init__(self, min_port, max_port):
        self.address = "147.47.200.209"
        self.username, self.password = "jetson", "jetson"
        self.ports = [i for i in range(int(min_port), int(max_port)+1) if 1<=i%10<=6 and i not in error_ports]
        self.ssh_ports = []
        self.connections = []
        
    def check(self):
        for port in self.ports:
            con = Connection(f'{self.username}@{self.address}:{port}', connect_kwargs ={"password":self.password})
            command = 'ls'
            print(f'----------------{port}----------------')
            try:
                con.run(command)
                self.ssh_ports.append(port)
                self.connections.append(con)
            except:
                print('ERROR')

        print("Available ports", self.ssh_ports)
        return len(self.ssh_ports)
            
    
    
    def send_command(self, command):
        for port, con in zip(self.ssh_ports, self.connections): 
            print(f'----------------{port}----------------')
            try:
                con.run(command)

            except:
                print('ERROR')

                        
    def start_fed(self, experiment, delay, max_round, num_samples):        
        for i, (port, con) in enumerate(zip(self.ssh_ports, self.connections)):
            command = f'docker exec client python3 /ambient_fl/client.py --round {max_round} --delay {delay} --num {num_samples} --id {i} --exp {experiment}'
            print(f'----------------{port}----------------')
            try:
                t=threading.Thread(target=con.run,args=(command,))
                t.start()
                time.sleep(delay)
            except:
                print('ERROR')


# jetson.send_command("docker ps -a")

# jetson.send_command("docker stop $(docker ps -a -q)")

# jetson.send_command("docker start client")

#jetson.send_command("docker pull crazyboy9103/jetson_fl:latest")

#jetson.send_command("docker run -d -ti --name client --gpus all --privileged crazyboy9103/jetson_fl:latest")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", '-r', type=int, help="max round")
    parser.add_argument("--num", '-n', type=int, help="number of samples (overridden if exp == 3")
    parser.add_argument("--id", type=int, help="client id")
    parser.add_argument("--exp", type=int, help="experiment number")
    parser.add_argument("--delay", type=int, help="time delay")
    parser.add_argument("--ip", type=str, help="ip address of the server")

    args = parser.parse_args()

    while True:
        user_input = input("Type in the command to send :")
        MIN_PORT = 20101
        MAX_PORT = 20136
        jetson = Jetson(min_port = MIN_PORT,
                        max_port  = MAX_PORT)

        if user_input == "start":
            CLIENT_NUM = jetson.check()
            init = requests.get(f"http://{args.ip}/initialize/{CLIENT_NUM}/{args.exp}/{args.round}")
            
            #jetson.send_command("docker ps -a")
            jetson.send_command("docker stop $(docker ps -a -q)")
            jetson.send_command("docker start client")

            print(f"Federated learning started with {CLIENT_NUM} clients")
            jetson.start_fed(experiment= args.exp,
                            delay = args.delay, 
                            max_round = args.round, 
                            num_samples = args.num)
            break

        jetson.send_command(user_input)

    

