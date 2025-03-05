import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from datasets import load_dataset

import numpy as np

#import openvino as ov
#from openvino.runtime import opset13 as opset

class LinearFP8(nn.Module):
    def __init__(self, module):
        super().__init__()
        assert isinstance(module, nn.Linear)
        self.module = module
        self.run_as_fp8 = False
        self.name = ""
        
        self.scale = module.weight
        
        self.w_scale = nn.Parameter(
            torch.ones((module.out_features, 1)).to(torch.float32)
        )
        self.a_scale_in = nn.Parameter(
            torch.ones((1, 1)).to(torch.float32)
        )
        self.a_scale_out = nn.Parameter(
            torch.ones((1, 1)).to(torch.float32)
        )
    
    def convert_weight(self):
        max_val = torch.finfo(torch.float8_e4m3fn).max
        #max_val = max_val.to(torch.float32)

        w_type = self.module.weight.dtype

        w_scale = self.module.weight.abs().max(dim=0)[0] / max_val
        self.w_scale.data = w_scale.unsqueeze(0)
        w = self.module.weight / w_scale
        w = w.to(torch.float8_e4m3fn)
        w = w.to(w_type) * self.w_scale.to(w_type)

        self.module.weight.data = w.to(self.module.weight.data.device)
        
    
    def forward(self, x):
        if self.run_as_fp8:
            return self.forward_fp8(x)
        return self.module(x)

    def forward_fp8(self, x):
        x_type = x.dtype
        max_val = torch.finfo(torch.float8_e4m3fn).max
        min_val = torch.finfo(torch.float8_e4m3fn).min

        x_fp8 = (x / self.a_scale_in)
        x_fp8 = torch.clamp(x_fp8, min_val, max_val)
        x_fp8 = x_fp8.to(torch.float8_e4m3fn)
        x_fp8 = x_fp8.to(x_type) * self.a_scale_in
        
        res = self.module(x_fp8)

        if not ('q_proj' in self.name or 'k_proj' in self.name):
            return res
        res_fp8 = (res / self.a_scale_out)
        res_fp8 = torch.clamp(res_fp8, min_val, max_val)
        res_fp8 = res_fp8.to(torch.float8_e4m3fn)
        res_fp8 = res_fp8.to(x_type) * self.a_scale_out

        return res_fp8
        


def convert_embeddings(layer):
    max_val = torch.finfo(torch.float8_e4m3fn).max
    w = layer.weight
    w_type = w.dtype
    scale = w.abs().max(dim=1)[0]
    scale = scale.unsqueeze(1)
    scale = scale / max_val
    
    w = (w / scale).to(torch.float8_e4m3fn)
    w = w.to(w_type) * scale
    layer.weight.data = w


def collect_stats(model, tokenizer):    
    input_activations = {}
    output_activations = {}

    def get_activations(layer_name):
        def hook(model, inputs, outputs):
            tensor_in = inputs[0] if isinstance(inputs, tuple) else inputs
            tensor_out = outputs[0] if isinstance(outputs, tuple) else outputs
            if not layer_name in input_activations:
                input_activations[layer_name] = tensor_in.abs().max(1)[0]
                output_activations[layer_name] = tensor_out.abs().max(1)[0]
            else:    
                input_activations[layer_name] = torch.maximum(tensor_in.abs().max(1)[0], input_activations[layer_name])
                output_activations[layer_name] = torch.maximum(tensor_out.abs().max(1)[0], output_activations[layer_name])
        return hook   

    all_hooks = []

    for name, module in model.named_modules():
        #print(name, type(module))
        if type(module) == LinearFP8:
            print('Add hook: ', name, type(module))
            hook = module.register_forward_hook(get_activations(name))
            all_hooks.append(hook)

    model.eval()
    
    input_text = "Who is the most famous composer?"
    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    output = model.generate(**input_ids, max_new_tokens=128, do_sample=False, temperature=0.0)
    print("FP model: ", tokenizer.decode(output[0], skip_special_tokens=True))
        
    dataset = load_dataset('wikitext', 'wikitext-2-v1', split='train')
    dataset = dataset.filter(lambda example: len(example["text"]) > 128)

    for i, data in enumerate(dataset):
        inputs = tokenizer(data['text'], return_tensors='pt').to(model.device)
        with torch.inference_mode():
            model(**inputs)
        if i >= 128:
            break

    for hook in all_hooks:
        hook.remove()
    
    #print(all_activations)

    max_val = torch.finfo(torch.float8_e4m3fn).max
    n_converts = 0
    for name, module in model.named_modules():
        if type(module) == LinearFP8:
            module.run_as_fp8 = True
            module.convert_weight()
            scale = input_activations[name].max()
            s_type = scale.dtype
            scale = scale / max_val
            scale = scale.to(s_type)
            module.a_scale_in.data = scale #all_activations[name].max() #/ max_val
            
            scale = output_activations[name].max()
            s_type = scale.dtype
            scale = scale / max_val
            scale = scale.to(s_type)
            module.a_scale_out.data = scale

            n_converts += 2
        if type(module) == nn.Embedding:
            convert_embeddings(module)
            n_converts += 1
    print("Number of FakeConverts: ", n_converts)
    
    output = model.generate(**input_ids, max_new_tokens=128, do_sample=False, temperature=0.0)
    print("FP8 model: ", tokenizer.decode(output[0], skip_special_tokens=True))



def set_value(obj, path, val):
    first, sep, rest = path.partition(".")
    # if first.isnumeric():
    #     first = int(first)
    if rest:
        new_obj = getattr(obj, first)
        set_value(new_obj, rest, val)
    else:
        setattr(obj, first, val)

def get_value(obj, path):
    first, sep, rest = path.partition(".")
    # if first.isnumeric():
    #     first = int(first)
    if rest:
        new_obj = getattr(obj, first)
        return get_value(new_obj, rest)
    else:
        return getattr(obj, first)

def replace_linear(model):
    replace_names = []
    for name, module in model.named_modules():
        #print(name, type(module))
        if type(module) == nn.Linear:
            print('replaced: ', name, type(module))
            # linear_fp8 = LinearFP8(module)
            # setattr(model, name, linear_fp8)
            #name = name[6:]
            replace_names.append(name)
    for name in replace_names:
        layer = get_value(model, name)
        linear_fp8 = LinearFP8(layer)
        linear_fp8.name = name
        set_value(model, name, linear_fp8)

def wrap_model(model):
    replace_linear(model)
    return model


def wrap_and_find_params(model, tokenizer):
    model_fp8 = wrap_model(model)
    #print(model_fp8)
    collect_stats(model_fp8, tokenizer)
    return model


def compress_decompress_ov_model(in_shape):
    data = opset.parameter(in_shape, name="input")
    scale = opset.constant(np.array([1.0], dtype=np.float32), name="scale")
    fake_convert = opset.fake_convert(data, scale)
    
    model = ov.Model([fake_convert], [data])

    compiled_model = ov.compile_model(model)
    return compiled_model

    #return lambda parameters: compiled_model(parameters)[0]


def compress_decompress_pt(data):
    pt_data = torch.tensor(data)
    data_type = pt_data.dtype
    
    pt_data = pt_data.to(torch.float8_e4m3fn).to(data_type)
    
    return pt_data.numpy()


def cmp_ov_pt(in_shape=1000):
    data = np.random.rand(in_shape) * 440
    
    ov_model = compress_decompress_ov_model([in_shape])
    
    data_ov = ov_model(data)[0]
    data_pt = compress_decompress_pt(data)
    
    print("OV - PT: ", np.mean(np.abs(data_ov - data_pt)))
    print("FP - OV: ", np.mean(np.abs(data_ov - data)))
    print("FP - PT: ", np.mean(np.abs(data - data_pt)))


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer
    
    device = 'cuda:1'
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_fp = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_fp8 = wrap_model(model_fp)
    print(model_fp8)
    collect_stats(model_fp8, tokenizer)

    # for i in range(10):
    #     cmp_ov_pt()
