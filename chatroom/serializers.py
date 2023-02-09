from rest_framework import serializers
from .models import Room


class TutorialSerializer(serializers.ModelSerializer):

    class Meta:
        model = Room
        fields = ('keywords',
                  'text',
                  'title',
                  'url')