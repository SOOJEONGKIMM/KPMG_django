from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.utils.safestring import mark_safe
from django.urls import reverse
from django.views.generic.edit import CreateView
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from rest_framework import status
from .serializers import RoomSerializer
from .models import Room, Answer, Question
from .models import UserInputDataset
from matplotlib import rc
import matplotlib.font_manager as fm
import random as rd
from .forms import DatasetPostForm
import pymongo
import json
from datetime import datetime

import requests
import random
import tensorflow as tf
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
#flask로 웹페이지 간단하게 만들고, mongodb, postgresql, 증상 웹페이지에 인풋 -> 의심되는 병 진단 결과 아웃풋,,
#서버에 bert api로 인풋을 get전송,,  flask 웹프레임워크,, 3090에서 다처리하고 처리한걸 flask로 보내기. (이걸 django로 바꾸기)


###### import for transformer model inferencing ######
from .transformer import data as tranformer_data
from .transformer import model as transformer
from .transformer.configs import DEFINES as DEFINES_transformer

###### import for seq2seq model inferencing #######
from .seq2seq import data as seq2seq_data
from .seq2seq import model as seq2seq
from .seq2seq.configs import DEFINES as DEFINES_seq2seq


###### import for attention_seq2seq model inferencing ######
from .attention_seq2seq import data as attention_seq2seq_data
from .attention_seq2seq import model as attention_seq2seq
from .attention_seq2seq.configs import DEFINES as DEFINES_attention_seq2seq

def index(request):
    chatroom_list = Room.objects.order_by('rank')[:5]
    context = {
        'chatroom_list': chatroom_list
    }
    return render(request, 'chatroom/index.html', context)

@api_view(['GET', 'PUT', 'DELETE','POST'])
def room(request, lm_name):
    chatroom_list = Room.objects.order_by('rank')[:5]
    client = pymongo.MongoClient(host='localhost',
                                 port=27017)
    db = client.test
    baidu = db.doc
    id = '63e3773f83a35219e5fca40a'
    #baidu = Room.objects.all()
    item = baidu.find_one()
    # print(x)
    #챗봇 현재시간 추력
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time)

    #READ
    context = {
        'keywords': item['keywords'],
        'url': item['url'],
        'title': item['title'],
        'current_time' : current_time
    }
    print("debugging from views:", context)
    if request.method == 'GET':
        #유저가 보낸 data를 UserInputDataset()모델로 db에 저장
        new_dataset = UserInputDataset()       # save to DB
        new_dataset.question = request.GET.get('question')
        print("new_dataset.question",new_dataset.question)
        new_dataset.save()
        print("success to insert new QA dataset from the user")

        create_graph(new_dataset.question)
        summary = rag_model(request, new_dataset.question)
        answer = "temp_ans_from_kordpr"
        new_dataset = { #for request html
            'question': new_dataset.question,
            'summary': summary,
            'answer': answer
        }

        return render(request, 'chatroom/room.html', new_dataset)
    '''
    context = {
        'lm_name': mark_safe(json.dumps(lm_name)),
        'chatroom_list': chatroom_list
    }
    '''
    #return redirect('room')
    return render(request, 'chatroom/room.html', {'context':context, 'current_time': current_time})

def create_graph(keyword):
    font_name = fm.FontProperties(fname="C:/Users\poohl/anaconda3\Lib\site-packages\matplotlib\mpl-data/fonts/ttf/NanumGothic.ttf").get_name()
    rc('font', family=font_name)
    plt.title('Knowledge Graph for Keyword')
    #print(keyword)#debug
    plt.figure(figsize=(16,16))
    plt.margins(x=0.1, y=0.02)

    example = {'연금시장': {'채권시장': 0.4153662323951721, '시장': 0.41456204652786255, '주식시장': 0.3486636281013489, '경제': 0.25629764795303345, '단계': 0.24219422042369843, '규모': 0.24082091450691223, '거래': 0.22897380590438843, '국내': 0.2052801549434662, '주선': 0.1902499794960022, '리스': 0.1724136471748352},
        '경제': {'주선': 0.2807018756866455, '채권시장': 0.2597755193710327, '하락': 0.2146831750869751, '산업': 0.21218357980251312, '경쟁': 0.2039693295955658, '한국': 0.20326876640319824, '단계': 0.20303569734096527, '리스': 0.1892050802707672, '수익': 0.17960302531719208, '최근': 0.17808492481708527},
        '채권': {'측정': 0.3019123673439026, '특성': 0.27175581455230713, '채권시장': 0.2508116364479065, '우수': 0.23209211230278015, '수준': 0.22973835468292236, '관련': 0.22968417406082153, '조원': 0.21655943989753723, '금액': 0.20694774389266968, '인력': 0.19489142298698425, '주식시장': 0.18427613377571106}}
    DG = nx.DiGraph(example)
    d = dict(DG.degree)
    #connect = nx.connected_components(DG.subgraph(d.keys()))

    G = nx.Graph()
    G_weight = [(a, b) for a, b in d.items()]
    G.add_nodes_from([a for a, b in d.items()])
   # G.add_edges_from(        (e[0], e[1]) for e in        DG.edges(data=True)    )
    #G = nx.DiGraph(example)
    # G = dict(DG.degree)
    # 아래 layout 중에서 마음에 드는 것을 쓰면 됩니다.
    pos = nx.spring_layout(G)
    # pos = graphviz_layout(G, prog='twopi')
    #pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(
        G, pos, node_color=[n[1] for n in G_weight], node_shape='h',
        node_size=3000, cmap=plt.cm.Blues#, alpha=0.9
    )

    nx.draw_networkx_edges(
        DG, pos, edge_color='black',
        width=1, edge_cmap=plt.cm.Greys, style='dashed'
    )

    nx.draw_networkx_labels(
        G, pos, font_family=font_name, font_color='black', font_size=10, font_weight='bold'
    )
    plt.axis('off')

    #pos = nx.shell_layout(G)
    #nx.draw_shell(G)
    #nx.draw(G, pos, font_family=font_name, font_size=15, with_labels=True, node_size=[v for v in d.values()]),
       # node_color=color_map)#_networkx_labels
    plt.savefig("D:/2022_2_GraphTransformer\graphtransformer\diya-chat\diyaChat\chatroom\static\chatroom\img\graph.png")


def rag_model(request, input_text):#TODO
    #TODO POST or GET?
    response = requests.get('http://ccf1-35-196-152-123.ngrok.io/predict', {'input_text': input_text})
    #summarized_text = response.text

    print("summarized_text:",{'summarized': response.text})
    return response.text
    #return render(request, 'chatroom/room.html', summarized_text)

def detail(request, lm_name):

    return HttpResponse("You're looking at chatroom using %s." % lm_name)

def post_view(request):
    return render(request, 'chatroom/room.html')

def get_post(request):
    if request.method == 'GET':
        id = request.GET['id']
        data = {
            'data': id,
        }
        return render(request, 'chatroom/parameter.html', data)

    elif request.method == 'POST':
        id = request.POST['id']
        name = request.POST['name']
        data = {
            'id': id,
            'name': name
        }
        return render(request, 'chatroom/parameter.html', data)
def message(request, message, lm_name):
    # if lm_name == 'tranformer':
    # if lm_name == 'seq2seq':
    # if lm_name == 'bert':
    #return HttpResponse("answer: %s" % (message))
    answer_obj = Answer.objects.all()
    question_obj = Question.objects.all()
    if "너" in message and "이름" in message:
        return HttpResponse("%s" % ("제 이름은 diya chat이에요 :)"))


    if lm_name == 'transformer':
        print(lm_name)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        arg_length = len(sys.argv)

        if (arg_length < 2):
            raise Exception("Don't call us. We'll call you")

        # Construct vocab file
        char2idx, idx2char, vocabulary_length = tranformer_data.load_vocabulary()

        ################# Encoder process #################
        input = message

        print(input)
        predic_input_enc, predic_input_enc_length = tranformer_data.enc_processing([input], char2idx)
        # Since it is not train process, there no input to decoder.
        # "" (empty string) is delivered just for fitting in the fixed structure
        predic_output_dec, predic_output_dec_length = tranformer_data.dec_output_processing([""], char2idx)
        # Since it is not train process, there no output from decoder.
        # "" (empty string) is delivered just for fitting in the fixed structure
        predic_target_dec = tranformer_data.dec_target_processing([""], char2idx)


        # Construct tf estimator
        transformer_classifier = tf.estimator.Estimator(
            model_fn=transformer.Model,
            model_dir=DEFINES_transformer.check_point_path_transformer,
            params={  # Parameter passing to model
                'embedding_size': DEFINES_transformer.embedding_size_transformer,
                'model_hidden_size': DEFINES_transformer.model_hidden_size_transformer,
                'ffn_hidden_size': DEFINES_transformer.ffn_hidden_size_transformer,
                'attention_head_size': DEFINES_transformer.attention_head_size_transformer,
                'learning_rate': DEFINES_transformer.learning_rate_transformer,
                'vocabulary_length': vocabulary_length,
                'embedding_size': DEFINES_transformer.embedding_size_transformer,
                'layer_size': DEFINES_transformer.layer_size_transformer,
                'max_sequence_length': DEFINES_transformer.max_sequence_length_transformer,
                'xavier_initializer': DEFINES_transformer.xavier_initializer_transformer
            })


        answer = ""

        for i in range(25):
            if i > 0:
                predic_output_dec, predic_output_decLength = tranformer_data.dec_output_processing([answer], char2idx)
                predic_target_dec = tranformer_data.dec_target_processing([answer], char2idx)

            ########### Get predicted answer ##########
            predictions = transformer_classifier.predict(
                input_fn=lambda: tranformer_data.eval_input_fn(predic_input_enc, predic_output_dec, predic_target_dec, 1))

            answer, finished = tranformer_data.pred_next_string(predictions, idx2char)

            if finished:
                break

        print("answer: ", answer)



    elif lm_name == 'seq2seq':
        print(lm_name)

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        arg_length = len(sys.argv)

        if(arg_length < 2):
            raise Exception("Don't call us. We'll call you")

        char2idx,  idx2char, vocabulary_length = seq2seq_data.load_vocabulary()
        input = message

        print(input)

        predic_input_enc, predic_input_enc_length = seq2seq_data.enc_processing([
                                                                        input], char2idx)
        predic_output_dec, predic_output_dec_length = seq2seq_data.dec_input_processing([
                                                                                ""], char2idx)
        predic_target_dec = seq2seq_data.dec_target_processing([""], char2idx)

        classifier = tf.estimator.Estimator(
            model_fn=seq2seq.model,
            model_dir=DEFINES_seq2seq.check_point_path_seq2seq,
            params={
                'hidden_size': DEFINES_seq2seq.hidden_size_seq2seq,
                'layer_size': DEFINES_seq2seq.layer_size_seq2seq,
                'learning_rate': DEFINES_seq2seq.learning_rate_seq2seq,
                'vocabulary_length': vocabulary_length,
                'embedding_size': DEFINES_seq2seq.embedding_size_seq2seq,
                'embedding': DEFINES_seq2seq.embedding_seq2seq,
                'multilayer': DEFINES_seq2seq.multilayer_seq2seq,
            })

        predictions = classifier.predict(
            input_fn=lambda: seq2seq_data.eval_input_fn(predic_input_enc, predic_output_dec, predic_target_dec, DEFINES_seq2seq.batch_size_seq2seq))

        answer = seq2seq_data.pred2string(predictions, idx2char)


    elif lm_name == 'attention_seq2seq':
        print(lm_name)

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        arg_length = len(sys.argv)

        if(arg_length < 2):
            raise Exception("Don't call us. We'll call you")

        char2idx,  idx2char, vocabulary_length = attention_seq2seq_data.load_vocabulary()

        input = message

        print(input)
        predic_input_enc, predic_input_enc_length = attention_seq2seq_data.enc_processing([
                                                                                          input], char2idx)

        predic_target_dec, _ = attention_seq2seq_data.dec_target_processing([
                                                                            ""], char2idx)

        if DEFINES_attention_seq2seq.serving_attention_seq2seq == True:

            predictor_fn = tf.contrib.predictor.from_saved_model(
                export_dir="/home/evo_mind/DeepLearning/NLP/Work/ChatBot2_Final/data_out/model/1541575161"
            )
        else:
            # Construct estimator
            classifier = tf.estimator.Estimator(
                model_fn=attention_seq2seq.Model,
                model_dir=DEFINES_attention_seq2seq.check_point_path_attention_seq2seq,
                params={
                    'hidden_size': DEFINES_attention_seq2seq.hidden_size_attention_seq2seq,
                    'layer_size': DEFINES_attention_seq2seq.layer_size_attention_seq2seq,
                    'learning_rate': DEFINES_attention_seq2seq.learning_rate_attention_seq2seq,
                    'teacher_forcing_rate': DEFINES_attention_seq2seq.teacher_forcing_rate_attention_seq2seq, # 학습시 디코더 인풋 정답 지원율 설정
                    'vocabulary_length': vocabulary_length,
                    'embedding_size': DEFINES_attention_seq2seq.embedding_size_attention_seq2seq,
                    'embedding': DEFINES_attention_seq2seq.embedding_attention_seq2seq,
                    'multilayer': DEFINES_attention_seq2seq.multilayer_attention_seq2seq,
                    'attention': DEFINES_attention_seq2seq.attention_attention_seq2seq,
                    'teacher_forcing': DEFINES_attention_seq2seq.teacher_forcing_attention_seq2seq,
                    'loss_mask': DEFINES_attention_seq2seq.loss_mask_attention_seq2seq, # PAD에 대한 마스크를 통해 loss를 제한
                    'serving': DEFINES_attention_seq2seq.serving_attention_seq2seq
                })

        if DEFINES_attention_seq2seq.serving_attention_seq2seq == True:
            predictions = predictor_fn({'input':predic_input_enc, 'output':predic_target_dec})

        else:
            predictions = classifier.predict(
                input_fn=lambda: attention_seq2seq_data.eval_input_fn(predic_input_enc, predic_target_dec, DEFINES_attention_seq2seq.batch_size_attention_seq2seq))

        # convert indexed sentence into string sentence
        answer = attention_seq2seq_data.pred2string(predictions, idx2char)

    else:
        answer = "죄송합니다. 아직 서비스 준비중 입니다. 빠른 시일 내에 완성하겠습니다."


    #return HttpResponse("%s" % answer)#TODO
    return render(requests, 'chatroom/room.html', {'answer_obj':answer_obj, 'question_obj':question_obj,'answer':answer})

# this is only for test
'''
def helloworld(request, num, lm_name):
    n1 = int(num)
    n2 = 1
    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.int32)

    add = tf.add(x, y)
    with tf.Session() as sess:
        z = sess.run(add, feed_dict={x:n1, y:n2})

    return HttpResponse("sum of %s and 1 is %s" %(num, z))
'''


@api_view(['GET', 'POST', 'DELETE'])
def tutorial_list(request):
    if request.method == 'GET':
        tutorials = Room.objects.all()

        title = request.GET.get('title', None)
        if title is not None:
            tutorials = tutorials.filter(title__icontains=title)

        tutorials_serializer = RoomSerializer(tutorials, many=True)
        return JsonResponse(tutorials_serializer.data, safe=False)
        # 'safe=False' for objects serialization

    elif request.method == 'POST':
        tutorial_data = JSONParser().parse(request)
        tutorial_serializer = RoomSerializer(data=tutorial_data)
        if tutorial_serializer.is_valid():
            tutorial_serializer.save()
            return JsonResponse(tutorial_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        count = Room.objects.all().delete()
        return JsonResponse({'message': '{} Tutorials were deleted successfully!'.format(count[0])},
                            status=status.HTTP_204_NO_CONTENT)


@api_view(['GET', 'PUT', 'DELETE'])
def tutorial_detail(request, pk):
    try:
        tutorial = Room.objects.get(pk=pk)
    except Room.DoesNotExist:
        return JsonResponse({'message': 'The tutorial does not exist'}, status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        tutorial_serializer = RoomSerializer(tutorial)
        return JsonResponse(tutorial_serializer.data)

    elif request.method == 'PUT':
        tutorial_data = JSONParser().parse(request)
        tutorial_serializer = RoomSerializer(tutorial, data=tutorial_data)
        if tutorial_serializer.is_valid():
            tutorial_serializer.save()
            return JsonResponse(tutorial_serializer.data)
        return JsonResponse(tutorial_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        tutorial.delete()
        return JsonResponse({'message': 'Tutorial was deleted successfully!'}, status=status.HTTP_204_NO_CONTENT)


@api_view(['GET'])
def tutorial_list_published(request):
    tutorials = Room.objects.filter(published=True)

    if request.method == 'GET':
        tutorials_serializer = RoomSerializer(tutorials, many=True)
        return JsonResponse(tutorials_serializer.data, safe=False)