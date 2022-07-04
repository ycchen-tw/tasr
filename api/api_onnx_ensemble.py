import sys
sys.modules['apex']=None
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json
import torch
from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, MT5ForConditionalGeneration

from argparse import ArgumentParser
import datetime
import hashlib
import time

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

import onnx
import onnxruntime
assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
import numpy as np
from transformers import AutoTokenizer
from onnx_helper import generate, generate_with_io_binding
from collections import Counter
from tqdm.auto import tqdm

NUM_GPU = 8
tokenizer_path = "./onnx_models/tokenizer"
encoder_model_paths = [
    "./onnx_models/f0_best/checkpoint-3000_encoder_decoder_init.onnx",
    "./onnx_models/f1_best/checkpoint-3750_encoder_decoder_init.onnx",
    "./onnx_models/f2_best/checkpoint-3500_encoder_decoder_init.onnx",
    "./onnx_models/f3_best/checkpoint-2750_encoder_decoder_init.onnx",
]
decoder_model_paths = [
    "./onnx_models/f0_best/checkpoint-3000_decoder.onnx",
    "./onnx_models/f1_best/checkpoint-3750_decoder.onnx",
    "./onnx_models/f2_best/checkpoint-3500_decoder.onnx",
    "./onnx_models/f3_best/checkpoint-2750_decoder.onnx",
]

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
gpu_models = []
for i_gpu in tqdm(range(NUM_GPU)):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    
    encoder_model_path = encoder_model_paths[i_gpu%4]
    decoder_model_path = decoder_model_paths[i_gpu%4]
    
    print(f'Load {encoder_model_path} to GPU {i_gpu}')
    
#     print(f'Loading {i_gpu}-th onnx encoder...')
    onnx_encoder = onnxruntime.InferenceSession(encoder_model_path, sess_options=options, providers=['CPUExecutionProvider'])
    onnx_encoder.set_providers(['CUDAExecutionProvider'], provider_options=[{'device_id': i_gpu}])
#     print(f'Loading {i_gpu}-th onnx decoder...')
    onnx_decoder = onnxruntime.InferenceSession(decoder_model_path, sess_options=options, providers=['CPUExecutionProvider'])
    onnx_decoder.set_providers(['CUDAExecutionProvider'], provider_options=[{'device_id': i_gpu}])
    gpu_models.append([onnx_encoder, onnx_decoder])

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'asdzxcasdzxcus@gmail.com'
SALT = 'my_salt_salt'                        
#########################################


def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid

i_call = 0
def predict(sentence_list, phoneme_sequence_list):
    """ Predict your model result.

    @param:
        sentence_list (list): an list of sentence sorted by probability.
        phoneme_sequence_list (list): an list of phoneme sequence sorted by probability.
    @returns:
        prediction (str): a sentence.
    """

    global i_call
    i_cuda = i_call % 8

    input_text = '</s>'.join(sentence_list).replace(' ', '')
    
    
    if len(sentence_list[0].replace(' ', '')) <= 12:
        i_call += 4
        
        print(f'Use GPU {[i%8 for i in range(i_cuda, i_cuda+4)]}')
        predictions = []
        for ii_cuda in [i%8 for i in range(i_cuda, i_cuda+4)]:
            onnx_encoder, onnx_decoder = gpu_models[ii_cuda]
            gen_outputs = generate_with_io_binding(input_text, tokenizer, onnx_encoder, onnx_decoder, buffer_device=f'cuda:{ii_cuda}')
            prediction = tokenizer.decode(gen_outputs, skip_special_tokens=True)
            predictions.append(prediction)
        if len(set(predictions)) > 1:
            print(predictions)
        prediction = Counter(predictions).most_common(1)[0][0]
        
    else:
        i_call += 1
        
        print(f'Use GPU {i_cuda}')
        onnx_encoder, onnx_decoder = gpu_models[i_cuda]
        gen_outputs = generate_with_io_binding(input_text, tokenizer, onnx_encoder, onnx_decoder, buffer_device=f'cuda:{i_cuda}')
        prediction = tokenizer.decode(gen_outputs, skip_special_tokens=True)


    ####################################################
    if _check_datatype_to_string(prediction):
        return prediction


def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.

    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')


@app.route('/inference', methods=['POST'])
def inference():
    start_time = time.time()
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)

    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = data['esun_timestamp']

    # 取 sentence list 中文
    sentence_list = data['sentence_list']
    # 取 phoneme sequence list (X-SAMPA)
    phoneme_sequence_list = data['phoneme_sequence_list']

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)

    try:
        ####### PUT YOUR MODEL INFERENCING CODE HERE #######
        answer = predict(sentence_list, phoneme_sequence_list)

    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e
    server_timestamp = time.time()

    response_dict = {'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': server_timestamp}
    response_record = [data, response_dict, answer]

    now = datetime.datetime.now() + datetime.timedelta(hours=8)
    dt_string = now.strftime("%Y%m%d_%H%M%S_%f")

    with open(f'day1_log/{dt_string}.json', 'w', encoding='utf-8') as file:
        json.dump(response_record, file, ensure_ascii=False, indent=4)

    use_time = round(time.time() - start_time, 5)
    print('**', use_time, answer)

    return jsonify(response_dict)


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=5002, help='port')
    arg_parser.add_argument('-d', '--debug', default=False, help='debug')
    options = arg_parser.parse_args()
    
    app.run(host='0.0.0.0', port=options.port, debug=options.debug)