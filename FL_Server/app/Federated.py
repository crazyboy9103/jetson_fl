import copy
import numpy as np
import tensorflow as tf
import json
from app import numpy_encoder
import os
import random
import time
from time import gmtime, strftime
class FederatedServer:
    client_number = 5 # 전체 클라이언트 개수
    server_weight = None # 현재 서버에 저장되어있는 weight
    local_weights = {} # 각 클라이언트에서 받아온 parameter들의 dict

    experiment = 1 # Uniform by default

    done_clients = 0 # Task가 끝난 클라이언트의 개수
    server_round = 0 # 현재 라운드
    max_round = 5 #

    num_data = {}
    client_model_accuracy = {}
    server_model_accuracy = []
    

    
    @classmethod
    def build_model(cls, type="cnn"):
        if type == "cnn":
            model = tf.keras.models.Sequential([
                        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
                        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                        tf.keras.layers.Dropout(0.25),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dense(10, activation='softmax')
                ])
            model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model

    @classmethod
    def initialize(cls, client_num, experiment, max_round):
        cls.reset() # reset the variables when initialized
        cls.client_number = client_num
        cls.experiment = experiment
        cls.max_round = max_round
        cls.model = cls.build_model("cnn")
        print(f"Initialized server with {client_num} clients, experiment  {experiment}, max round {max_round}")
        return "Initialized server"

    @classmethod
    def update_num_data(cls, client_id, num_data):
        cls.num_data[client_id] = num_data
        print(f"Number of data for {client_id} updated")

    @classmethod
    def update(cls, client_id, local_weight):
        if not local_weight:
            print("Client id", str(client_id), "network error")
            if client_id in cls.client_model_accuracy:
                cls.client_model_accuracy[client_id].append(0)
            else:
                cls.client_model_accuracy[client_id] = [0]
            
        else:
            print("Client id", str(client_id), "updated")
            local_param = list(map(lambda weight: np.array(weight, dtype=np.float32), local_weight))
            cls.local_weights[client_id] = local_param
            cls.evaluateClientModel(client_id, local_param)

        cls.done_clients = cls.done_clients + 1 # increment current count

        if cls.done_clients == cls.client_number:
            cls.FedAvg() # fed avg
            cls.evaluateServerModel()
            cls.next_round()

        if cls.server_round + 1 == cls.max_round: # federated learning finished
            cls.save() # save all history into json file
            cls.reset()

    @classmethod
    def FedAvg(cls):
        print(f"FedAvg at round {cls.server_round}")
        print("Total client number:", cls.client_number)
        print("Experiment:", cls.experiment)
        print("Total done client number:", cls.done_clients)
        print("Server round:", cls.server_round)
        print("Max round:", cls.max_round)
        print("Number of data:", cls.num_data)

        weight = list(map(lambda block: np.zeros_like(block, dtype=np.float32), random.choice(list(cls.local_weights.values()))))
        total_num_data = 0
        for client_id in cls.local_weights:
            total_num_data += cls.num_data[client_id]

        for client_id, client_weight in cls.local_weights.items():
            client_num_data = cls.num_data[client_id]

            for i in range(len(weight)):
                weighted_weight = client_weight[i] * (client_num_data/total_num_data)
                weight[i] += weighted_weight

        cls.set_server_weight(weight)

    @classmethod
    def evaluateClientModel(cls, client_id, weight):
        print("evaluateclientmodel start")
        cls.model.set_weights(cls.local_weights[client_id]) # change to local weight

        mnist = tf.keras.datasets.mnist
        (_, _), (test_images, test_labels) = mnist.load_data()
        n = len(test_images)
        indices = np.random.choice([i for i in range(n)], n//10)
        test_images = test_images[indices]
        test_labels = test_labels[indices]
        test_images = test_images / 255
        test_images = test_images.reshape(-1,28, 28, 1)

        acc = cls.model.evaluate(test_images, test_labels)

        if client_id not in cls.client_model_accuracy:
            cls.client_model_accuracy[client_id] = []

        cls.client_model_accuracy[client_id].append(acc[1])

        if cls.server_weight != None:
            cls.model.set_weights(cls.server_weight) # revert to server weight

        print("evaluateclientmodel finish")

    @classmethod
    def evaluateServerModel(cls):
        mnist = tf.keras.datasets.mnist
        (_, _), (test_images, test_labels) = mnist.load_data()
        n = len(test_images)
        indices = np.random.choice([i for i in range(n)], n//10)

        test_images = test_images[indices]
        test_labels = test_labels[indices]
        test_images = test_images / 255
        test_images = test_images.reshape(-1,28, 28, 1)

        acc = cls.model.evaluate(test_images, test_labels)[1] # index 1 is acc
        cls.server_model_accuracy.append(acc)

    @classmethod
    def next_round(cls):
        cls.done_clients = 0 
        cls.server_round += 1 # proceed to next round
        cls.num_data = {}
        cls.local_weights = {}

    @classmethod
    def save(cls):
        result = {"client number": cls.client_number,
                  "experiment":cls.experiment,
                  "max round": cls.max_round,
                  "clients acc" : cls.client_model_accuracy,
                  "server acc" : cls.server_model_accuracy} 
        
        timestamp = strftime("%Y%m%d_%H%M%S", gmtime())
        with open("../Logs/"+timestamp+".json", 'w') as f:
            json.dump(result, f)
        
        log = f'#Json file saved as ../Logs/", {timestamp}+".json#'
        print("#"*len(log))
        print(log)
        print("#"*len(log))


    @classmethod
    def reset(cls):
        cls.client_model_accuracy = {}
        cls.server_model_accuracy = []
        cls.server_weight = None
        cls.local_weights = {}
        cls.done_clients = 0
        cls.server_round = 0
        cls.num_data = {}

    @classmethod
    def set_server_weight(cls, weight):
        cls.server_weight = weight

    @classmethod
    def get_server_weight(cls):
        return cls.server_weight

    @classmethod
    def get_done_clients(cls):
        return cls.done_clients

    @classmethod
    def get_server_round(cls):
        return cls.server_round
