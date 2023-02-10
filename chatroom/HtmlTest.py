from django.http import HttpResponse
from django.template import loader

from .MongoDbManager import MongoDbManager
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def specific_user(request, doc_id):
    def get():
        db_user_data = MongoDbManager().get_users_from_collection({'doc_id': doc_id})

        user = db_user_data[0]
        del user['_id']

        template = loader.get_template('room.html')
        return HttpResponse(template.render({'userData': [user]}, request))


    if request.method == 'GET':
        return get()
    else:
        return HttpResponse(status=405)


def all_users(request):
    def get():
        dbUserData = MongoDbManager().get_users_from_collection({})
        print(dbUserData)

        users = []
        for user in dbUserData:
            del user['_id']
        users.append(user)

        template = loader.get_template('room.html')
        return HttpResponse(template.render({'userData': users}, request))


    if request.method == 'GET':
        return get()
    else:
        return HttpResponse(status=405)