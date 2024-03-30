# coding = utf-8
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import json
from flask import Flask, request, jsonify
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("Qwen-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen-7B", trust_remote_code=True).cuda()
model = model.to(device)
model.generation_config = GenerationConfig.from_pretrained("Qwen-7B")

def predict_model(data):
    text = data["message"][0]["content"]
    inputs = tokenizer(text, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_new_tokens=data["max_tokens"], top_k=data["top_k"], top_p=data["top_p"], temperature=data["temperature"], repetition_penalty=data["repetition_penalty"], num_beams=data["num_beams"])
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response

app = Flask(import_name=__name__)
@app.route("/generate", methods=["POST", "GET"])
def generate():
    data = json.loads(request.data)
    print(data)
    try:
        res = predict_model(data)
        label = "success"
    except Exception as e:
        res = ""
        label = "error"
        print(e)
    return jsonify({"output":[res], "status":label})
   
if __name__ == '__main__':
    app.run(port=3001, debug=False, host='0.0.0.0')  # 如果是0.0.0.0，则可以被外网访问
