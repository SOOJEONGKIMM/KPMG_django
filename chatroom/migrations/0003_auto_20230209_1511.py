# Generated by Django 3.2.8 on 2023-02-09 06:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatroom', '0002_userinputdataset'),
    ]

    operations = [
        migrations.AddField(
            model_name='room',
            name='keywords',
            field=models.CharField(default='', max_length=70),
        ),
        migrations.AddField(
            model_name='room',
            name='text',
            field=models.CharField(default='', max_length=200),
        ),
        migrations.AddField(
            model_name='room',
            name='title',
            field=models.CharField(default='', max_length=200),
        ),
        migrations.AddField(
            model_name='room',
            name='url',
            field=models.CharField(default='', max_length=200),
        ),
    ]
