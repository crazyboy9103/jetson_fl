from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path('index', views.index),
    path('admin/', admin.site.urls),
    path('reset/', views.reset),
    path('initialize/<int:client_num>/<int:experiment>/<int:max_round>', views.initialize),
    path('get_server_round', views.get_server_round),
    path('get_server_weight', views.get_server_weight),
    path('put_weight/<int:client_id>', views.put_weight),
    path("update_num_data/<int:client_id>/<int:num_data>", views.update_num_data),
    path('get_experiment', views.get_experiment),
    path('set_experiment/<int:experiment>', views.set_experiment)
 ]
