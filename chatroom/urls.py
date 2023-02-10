from django.urls import path
from . import views
from django.conf.urls import url
from django.urls import re_path as url
from . import HtmlTest

app_name = 'chatroom'
urlpatterns = [
   # ex) /chatroom/
   path('', views.index, name='index'),
   # ex) /chatroom/bert/
   path('<str:lm_name>/', views.room, name='room'),
   path('parameter/', views.get_post),
   path('room/', views.post_view),

   #path('<str:lm_name>/<str:num>/', views.helloworld, name='helloworld'),
   path('<str:lm_name>/<str:message>/', views.message, name='message'),

   # ex) /chatroom/newDataset
   path('<str:lm_name>/ajax/post_dataset/', views.post_dataset, name='post_dataset'),
]
