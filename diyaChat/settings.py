"""
Django settings for diyaChat project.

Generated by 'django-admin startproject' using Django 2.2.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""
from nltk.stem.porter import PorterStemmer
import os
import pymongo

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'm$j3usm7=l9vk*4-no3tu#ro_+99ayj(v__9hc(o21-6_y!-x1'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']

APPEND_SLASH = False
# Application definition

INSTALLED_APPS = [
    'channels',
    'tf_model.apps.TfModelConfig',
    'chatroom.apps.ChatroomConfig',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # CORS
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'diyaChat.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'diyaChat.wsgi.application'


# Channels
# This should be included to use websocket
ASGI_APPLICATION = 'diyaChat.routing.application'
'''
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379, '0.0.0.0')],
        },
    },
}
'''

# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases
'''
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'ENFORCE_SCHEMA': True,
        'LOGGING': {
            'version': 1,
            'loggers': {
                'djongo': {
                    'level': 'DEBUG',
                    'propogate': False,
                }
            },
         },
        'NAME': 'test',
        'CLIENT': {
            'host': '127.0.0.1',
            'port': 27017,
            'username': 'kpmg',
            'password': "1212",
            'authSource': 'admin',
            'authMechanism': 'SCRAM-SHA-1'
        }
    }
}
'''
DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'test',
        'HOST': '127.0.0.1',
        'PORT': 27017,
    }
}

# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'
#LANGUAGE_CODE = 'zh-hans'
#TIME_ZONE = 'UTC'
TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = '/static/'

#client = pymongo.MongoClient("whatever you like")
client = pymongo.MongoClient(host='localhost', port=27017)
db = client.test
collection = db.doc
#debug
#x = collection.find_one()
#print(x)

import mongoengine

# Other settings are here

# Connect with MongoEngine
'''
mongoengine.connect(
    host='mongodb+srv://<mongodb-user>:<password>'
    '@<mongodb-host>/<database-name>'
    '?retryWrites=true&w=majority',
    connect=False,
)

word_wij_invert_nk = []
doc_dj_keyword_weight_title_url_time_text = [None]
term_index_dict = {}

index = 0

print("获取 word_wij_invert_nk")
for elem in collection.find():
    #print("elem:",elem)
    word_wij_invert_nk.append(elem)
    term_index_dict[elem['word']] = index
    index = index + 1
#print("term_index_dict:",term_index_dict)
db2 = client.test
collection2 = db.doc

print("获取 doc_dj_keyword_weight_title_url_time_text")
for elem in collection2.find():
    doc_dj_keyword_weight_title_url_time_text.append(elem)
'''

porter_stemmer = PorterStemmer()