import json
from django.http import HttpResponse
# Create your views here.
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from app.Federated import FederatedServer
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    '''
        통신에 필요한 serialize 과정 진행 (ndarray -> list)
    '''
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


@api_view(['GET'])
def index(request):
    return HttpResponse("Server ok", status.HTTP_200_OK)

@api_view(['GET'])
def get_server_round(request):
    return HttpResponse(FederatedServer.get_server_round(), status.HTTP_200_OK)

@api_view(['PUT'])
def set_client_num(request, client_num):
    FederatedServer.client_number = client_num
    return HttpResponse(f"Total client number is set to {client_num}", status.HTTP_200_OK)

@api_view(['GET'])
def get_server_weight(request):
    server_weight = FederatedServer.get_server_weight()
    server_weight_to_json = json.dumps(server_weight, cls=NumpyEncoder)
    return HttpResponse(server_weight_to_json, status.HTTP_200_OK)

@api_view(['PUT'])
def put_weight(request, client_id):
    try:
        weight = JSONParser().parse(request)
        return HttpResponse(FederatedServer.update(client_id, weight), status.HTTP_200_OK)
    except:
        weight = None
        return HttpResponse(FederatedServer.update(client_id, weight), status.HTTP_200_OK)

@api_view(['GET'])
def update_num_data(request, client_id, num_data):
    return HttpResponse(FederatedServer.update_num_data(client_id, num_data), status.HTTP_200_OK)


@api_view(['GET'])
def reset(request):
    FederatedServer.reset()
    return HttpResponse("Server reset", status.HTTP_200_OK)

@api_view(['GET'])
def initialize(request, client_num, experiment, max_round):
    return HttpResponse(FederatedServer.initialize(client_num, experiment, max_round), status.HTTP_200_OK)

