import argparse
import json
import threading
import time
from random import random
import numpy as np
import requests
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Quiet tensorflow error messages

class Client:
    def __init__(self, max_round, time_delay = 5, num_samples=600, client_id = 0, experiment = 1, model_type="cnn", ip="http://147.47.200.178:9103/"):
        self.base_url = ip # Base Url
    
        self.client_id = client_id
        self.time_delay = time_delay

        self.current_round = 0 # 0 when initialized

        self.max_round = max_round # Set the maximum number of rounds
        
        # download mnist dataset and partition the data
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.prepare_images()
        self.split_train_images, self.split_train_labels = self.data_split(experiment, num_samples)
        self.local_data_num = len(self.split_train_labels)
        
        # Build model
        if model_type == "cnn":
            self.model = self.build_cnn_model()
    
        #elif model_type == "resnet50":

        #elif model_type == "mobilenetv2":

        # Session for requests
        self.session = requests.Session()

    def prepare_images(self):
        """
        IN: X
        OUT: entire mnist dataset
        """
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images, test_images = train_images / 255, test_images / 255
        
        # For CNN, add dummy channel to feed the images to CNN
        train_images=train_images.reshape(-1, 28, 28, 1)
        test_images=test_images.reshape(-1, 28, 28, 1)
        return train_images, train_labels, test_images, test_labels
    
    def build_cnn_model(self):
        """
        IN: X
        OUT: cnn model
        """
        #This model definition must be same in the server (Federated.py)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.SGD(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model
        
    def data_split(self, experiment, num_samples):
        """
        IN: experiment number, the number of data
        OUT:
            If experiment is 1: Uniform data split: We take equal amount of data from each class (iid)
            If experiment is 2: Random data split1: We take equal amount of data, but not uniformly distributed across classes
            If experiment is 3: Random data split2: We take different amount of data and not uniformly distributed across classes
        """
        
        train_index_list = [[] for _ in range(10)]
        test_index_list = [[] for _ in range(10)]

        for i, v in enumerate(self.train_labels):
            train_index_list[v].append(i)

        for i, v in enumerate(self.test_labels):
            test_index_list[v].append(i)

        split_train_images = []
        split_train_labels = []
        
     
        if experiment == 1: #uniform data split
            self.local_data_num = num_samples
            
            for i in range(len(train_index_list)):
                indices = train_index_list[i]
                random_indices = np.random.choice(indices, size=num_samples//10)
                
                split_train_images.extend(self.train_images[random_indices])
                split_train_labels.extend(self.train_labels[random_indices])
            

        elif experiment == 2: # Randomly selected, equally sized dataset
            self.local_data_num = num_samples
            random_indices = np.random.choice([i for i in range(len(self.train_labels))], size=num_samples)
            split_train_images = self.train_images[random_indices]
            split_train_labels = self.train_labels[random_indices]

        
            
        elif experiment == 3: # Randomly selected, differently sized dataset
            n = np.random.randint(low=16, high=200)
            self.local_data_num = n
            random_indices = np.random.choice([i for i in range(len(self.train_labels))], size=n)
            split_train_images = self.train_images[random_indices]
            split_train_labels = self.train_labels[random_indices]
            
      
                
        split_train_images = np.array(split_train_images)
        split_train_labels = np.array(split_train_labels)
        return split_train_images, split_train_labels

        
        
    def update_total_num_data(self, num_data):
        """
        IN: the number of training images that the current client has
        OUT: X
            update the number of training images for a client stored in the server
        """
        update_num_data_url =  f"{self.base_url}/update_num_data/{self.client_id}/{num_data}"
        self.session.get(update_num_data_url)
        

    
    def request_global_round(self):
        """
        IN: X
        OUT: Current global round at the server
        """
        round_url =  self.base_url + "get_server_round" 
        result = self.session.get(round_url)
        result = int(result.text)
        return result
    
    def request_global_weight(self):
        """
        IN: X
        OUT: server's most recent model parameters
        """
        get_weight_url =  f"{self.base_url}/get_server_weight" # Url that we send or fetch weight parameters
        result = self.session.get(get_weight_url)
        result_data = result.json()
        
        global_weight = None
        if result_data is not None:
            global_weight = []
            for i in range(len(result_data)):
                temp = np.array(result_data[i], dtype=np.float32)
                global_weight.append(temp)
            
        return global_weight

    def upload_local_weight(self, local_weight):
        """
        IN: local weight after training
        OUT: X
        """
        put_weight_url =  f"{self.base_url}/put_weight/{self.client_id}"
        for i in range(len(local_weight)):
            local_weight[i] = local_weight[i].tolist() # convert np.array weight to list 
        local_weight_to_json = json.dumps(local_weight) # serialize into json
        self.session.put(put_weight_url, data=local_weight_to_json) 
        
    def train_local_model(self):
        print(f"{self.client_id} started training")
        """
        IN: X
        OUT: local weight of the current client after training
        """
        global_weight = self.request_global_weight()
        if global_weight != None:
            self.model.set_weights(global_weight)
            
        self.model.fit(self.split_train_images, self.split_train_labels, epochs=10, batch_size=4, verbose=0)
        local_weight = self.model.get_weights()
        return local_weight
    
    def task(self):
        """
        IN: X
        OUT: Federated learning task
        """
        global_round = self.request_global_round()
        
        print("global round", global_round)
        print("current round", self.current_round)
        # If the current round is larger than the max round, finish
        if self.current_round >= self.max_round:
            print("Client", self.client_id, "finished")
            return 
        # If the global round = current client's round, the client needs to update the weight
        if global_round == self.current_round: #need update 
            print("Client "+ str(self.client_id) + "needs update")
            self.split_train_images, self.split_train_labels = self.data_split(num_samples=self.local_data_num)
            self.update_total_num_data(self.local_data_num)        
            self.current_round += 1
            local_weight = self.train_local_model()
            self.upload_local_weight(local_weight)
            time.sleep(self.time_delay)
            return self.task()
        # Otherwise, we need to wait until other clients finish training/uploading the weight
        else: 
            print("need wait")
            time.sleep(self.time_delay)
            return self.task()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", '-r', default=5, type=int, help="max round")
    parser.add_argument("--num", '-n', default=200, type=int, help="number of samples (overridden if exp == 3, 4")
    parser.add_argument("--id", default=0, type=int, help="client id")
    parser.add_argument("--exp", default=1, type=int, help="experiment number")
    parser.add_argument("--delay", default=5, type=int, help="time delay")
    parser.add_argument("--ip", default="127.0.0.1", type=str, help="ip address of the server")
    parser.add_argument("--model", default="cnn", type=str, help="model type (cnn, resnet50, mobilenetv2)")

    args = parser.parse_args()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass

    client = Client(args.round, args.delay, args.num, args.id, args.exp, args.model, args.ip)
    client.task()
    
