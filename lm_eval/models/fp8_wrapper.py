import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from datasets import load_dataset
import os
import numpy as np

import openvino as ov
from openvino.runtime import opset13 as opset

from matplotlib import pyplot as plt

def ov_model_fp8(in_shape):
    data = opset.parameter(in_shape, name="input")
    scale = opset.constant(np.array([1.0], dtype=np.float32), name="scale")
    fake_convert = opset.fake_convert(data, scale)
    
    model = ov.Model([fake_convert], [data])

    compiled_model = ov.compile_model(model)
    return compiled_model

ov_model_w = ov_model_fp8([-1, -1])
ov_model_a = ov_model_fp8([-1, -1, -1])

use_ov = False


def quant_dequant_fp8e4m3(x, clamp_val):
    x_type = x.dtype
    max_val = torch.finfo(torch.float8_e4m3fn).max
    min_val = torch.finfo(torch.float8_e4m3fn).min

    scale = clamp_val / max_val
    x_fp8 = (x / scale)
    x_fp8 = torch.clamp(x_fp8, min_val, max_val)
    x_fp8 = x_fp8.to(torch.float8_e4m3fn).to(x_type)
    x_fp8 = x_fp8 * scale
    
    return x_fp8


def get_mse_max(data, name, op):
    if isinstance(data, list):
        data = torch.vstack(data)

    cur_max = data.max()
    
    
    
    data_fp8 = quant_dequant_fp8e4m3(data, cur_max)
    
    best_diff = torch.mean((data - data_fp8)**2)
    best_max = cur_max
    print("Init ", name, op, cur_max, best_diff)
    
    for i in range(50):
        cur_max = 0.98 * cur_max
        data_fp8 = quant_dequant_fp8e4m3(data, cur_max)
        diff = torch.mean((data - data_fp8)**2)
        if diff < best_diff:
            best_diff = diff
            best_max = cur_max
    print("Best ", name, op, best_max, best_diff)
    
    return best_max
        
    
    

def get_clamp_max_(data, name, op):
    cur_max = data.max()
    return torch.clamp(cur_max, 0, 50)

def get_clamp_max(data, name, op):
    quants = [0.0000e+00, 1.9531e-03, 3.9062e-03, 5.8594e-03, 7.8125e-03, 9.7656e-03,
                1.1719e-02, 1.3672e-02, 1.5625e-02, 1.7578e-02, 1.9531e-02, 2.1484e-02,
                2.3438e-02, 2.5391e-02, 2.7344e-02, 2.9297e-02, 3.1250e-02, 3.5156e-02,
                3.9062e-02, 4.2969e-02, 4.6875e-02, 5.0781e-02, 5.4688e-02, 5.8594e-02,
                6.2500e-02, 7.0312e-02, 7.8125e-02, 8.5938e-02, 9.3750e-02, 1.0156e-01,
                1.0938e-01, 1.1719e-01, 1.2500e-01, 1.4062e-01, 1.5625e-01, 1.7188e-01,
                1.8750e-01, 2.0312e-01, 2.1875e-01, 2.3438e-01, 2.5000e-01, 2.8125e-01,
                3.1250e-01, 3.4375e-01, 3.7500e-01, 4.0625e-01, 4.3750e-01, 4.6875e-01,
                5.0000e-01, 5.6250e-01, 6.2500e-01, 6.8750e-01, 7.5000e-01, 8.1250e-01,
                8.7500e-01, 9.3750e-01, 1.0000e+00, 1.1250e+00, 1.2500e+00, 1.3750e+00,
                1.5000e+00, 1.6250e+00, 1.7500e+00, 1.8750e+00, 2.0000e+00, 2.2500e+00,
                2.5000e+00, 2.7500e+00, 3.0000e+00, 3.2500e+00, 3.5000e+00, 3.7500e+00,
                4.0000e+00, 4.5000e+00, 5.0000e+00, 5.5000e+00, 6.0000e+00, 6.5000e+00,
                7.0000e+00, 7.5000e+00, 8.0000e+00, 9.0000e+00, 1.0000e+01, 1.1000e+01,
                1.2000e+01, 1.3000e+01, 1.4000e+01, 1.5000e+01, 1.6000e+01, 1.8000e+01,
                2.0000e+01, 2.2000e+01, 2.4000e+01, 2.6000e+01, 2.8000e+01, 3.0000e+01,
                3.2000e+01, 3.6000e+01, 4.0000e+01, 4.4000e+01, 4.8000e+01, 5.2000e+01,
                5.6000e+01, 6.0000e+01, 6.4000e+01, 7.2000e+01, 8.0000e+01, 8.8000e+01,
                9.6000e+01, 1.0400e+02, 1.1200e+02, 1.2000e+02, 1.2800e+02, 1.4400e+02,
                1.6000e+02, 1.7600e+02, 1.9200e+02, 2.0800e+02, 2.2400e+02, 2.4000e+02,
                2.5600e+02, 2.8800e+02, 3.2000e+02, 3.5200e+02, 3.8400e+02, 4.1600e+02,
                4.4800e+02]
    
    quant_per_channel = torch.zeros(data.shape)
    
    cur_max = data.max()
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    th_quant = 90
    
    cur_data = torch.clamp(data, 0, cur_max) / cur_max * max_fp8
    for i in range(len(quants)):
        quant_per_channel[cur_data >= quants[i]] += 1
    
    #mean_quant_per_channel = torch.mean(quant_per_channel)
    mean_quant_per_channel = torch.median(quant_per_channel)
    
    print(name, cur_max, mean_quant_per_channel)
    process = False
    while mean_quant_per_channel < th_quant:
        process = True
        cur_max = cur_max * 0.98
        quant_per_channel[:] = 0
        
        cur_data = torch.clamp(data, 0, cur_max) / cur_max * max_fp8
        for i in range(len(quants)):
            quant_per_channel[cur_data >= quants[i]] += 1
        
        #mean_quant_per_channel = torch.mean(quant_per_channel)
        mean_quant_per_channel = torch.median(quant_per_channel)
    
    if process:
        print(' ' * len(name), cur_max, mean_quant_per_channel)
    
    # plt.hist(data.detach().to(torch.float32).cpu().numpy())
    # plt.show()
    
    # if not os.path.exists("quants"):
    #     os.mkdir("quants")
    
    # torch.save(quant_per_channel, "quants/" + name + "_" + op + ".pt")
    
      
    return cur_max
    

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
        if use_ov:
            w = torch.tensor(ov_model_w(w.to(torch.float32).numpy())[0])
        else:
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
        
        if use_ov:
            x_fp8 = torch.tensor(ov_model_a(x_fp8.to(torch.float32).numpy())[0])
        else:
            x_fp8 = x_fp8.to(torch.float8_e4m3fn)
        x_fp8 = x_fp8.to(x_type) * self.a_scale_in
        
        res = self.module(x_fp8)

        if not ('q_proj' in self.name or 'k_proj' in self.name):# or 'layers.0.self_attn.k_proj' in self.name:
            return res
        res_fp8 = (res / self.a_scale_out)
        res_fp8 = torch.clamp(res_fp8, min_val, max_val)
        
        if use_ov:
            res_fp8 = torch.tensor(ov_model_a(res_fp8.to(torch.float32).numpy())[0])
        else:
            res_fp8 = res_fp8.to(torch.float8_e4m3fn)
        res_fp8 = res_fp8.to(x_type) * self.a_scale_out
        
        #print("Diff: ", torch.mean(torch.abs(res - res_fp8)))

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


def collect_stats(model, tokenizer, use_clamp_max=True):    
    input_activations = {}
    output_activations = {}

    def get_activations(layer_name):
        def hook(model, inputs, outputs):
            tensor_in = inputs[0] if isinstance(inputs, tuple) else inputs
            tensor_out = outputs[0] if isinstance(outputs, tuple) else outputs
            tensor_in = tensor_in.squeeze()
            tensor_out = tensor_out.squeeze()
            # if not layer_name in input_activations:
            #     input_activations[layer_name] = tensor_in.abs().max(1)[0]
            #     output_activations[layer_name] = tensor_out.abs().max(1)[0]
            # else:    
            #     input_activations[layer_name] = torch.maximum(tensor_in.abs().max(1)[0], input_activations[layer_name])
            #     output_activations[layer_name] = torch.maximum(tensor_out.abs().max(1)[0], output_activations[layer_name])
            if not layer_name in input_activations:
                input_activations[layer_name] = [tensor_in]
                output_activations[layer_name] = [tensor_out]
            else:    
                input_activations[layer_name].append(tensor_in)
                output_activations[layer_name].append(tensor_out)
        return hook   

    all_hooks = []

    for name, module in model.named_modules():
        #print(name, type(module))
        if type(module) == LinearFP8:
            print('Add hook: ', name, type(module))
            hook = module.register_forward_hook(get_activations(name))
            all_hooks.append(hook)

    model.eval()
    
    # input_text = "Who is the most famous composer?"
    # input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    # output = model.generate(**input_ids, max_new_tokens=128, do_sample=False, temperature=0.0)
    # print("FP model: ", tokenizer.decode(output[0], skip_special_tokens=True))
        
    dataset = load_dataset('wikitext', 'wikitext-2-v1', split='train')
    dataset = dataset.filter(lambda example: len(example["text"]) > 128)

    for i, data in enumerate(dataset):
        inputs = tokenizer(data['text'], return_tensors='pt').to(model.device)
        with torch.inference_mode():
            model(**inputs)
        if i >= 32:
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
            if use_clamp_max:
                #scale = get_clamp_max(input_activations[name].max(dim=0)[0], name, 'in')
                #scale = get_clamp_max(input_activations[name].max(dim=1)[0], name, 'in')
                scale = get_mse_max(input_activations[name], name, 'in')
            else:
                scale = input_activations[name].max()
            s_type = scale.dtype
            scale = scale / max_val
            scale = scale.to(s_type)
            module.a_scale_in.data = scale #all_activations[name].max() #/ max_val
            if not 'lm_head' in name:
                if use_clamp_max:
                    #scale = get_clamp_max(output_activations[name].max(dim=1)[0], name, 'out')
                    #scale = get_clamp_max(output_activations[name].max(dim=0)[0], name, 'out')
                    scale = get_mse_max(output_activations[name], name, 'out')
                else:
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
    
    # output = model.generate(**input_ids, max_new_tokens=128, do_sample=False, temperature=0.0)
    # print("FP8 model: ", tokenizer.decode(output[0], skip_special_tokens=True))



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

@torch.no_grad()
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
    data = np.random.rand(in_shape) * 10
    
    ov_model = compress_decompress_ov_model([in_shape])
    
    data_ov = ov_model(data)[0]
    data_pt = compress_decompress_pt(data)
    
    data_pt_ov = compress_decompress_pt(data_ov)
    data_ov_pt = ov_model(data_pt)[0]
    
    for i in range(in_shape):
        if data_ov[i] != data_pt[i]:
            print(i, data[i], data_ov[i], data_pt[i], "|", data_pt_ov[i], abs(data_ov[i] - data[i]), abs(data_pt[i] - data[i]))
    
    print("OV - PT: ", np.mean(np.abs(data_ov - data_pt)))
    print("OV - OV_PT: ", np.mean(np.abs(data_ov - data_pt_ov)))
    print("OV - PT_OV: ", np.mean(np.abs(data_pt - data_ov_pt)))
    print("FP - OV: ", np.mean(np.abs(data_ov - data)))
    print("FP - PT: ", np.mean(np.abs(data - data_pt)))


def print_fp8_values():
    max_val = torch.finfo(torch.float8_e4m3fn).max
    fp8_data = []
    for i in range(10):
        fp8_data.append(torch.randn((10000)) * 448/2**i)
    fp8_data = torch.vstack(fp8_data).flatten()
    fp8_data = fp8_data.abs()
    fp8_data = torch.clamp(fp8_data, -1, max_val)
    fp8_data = fp8_data.to(torch.float8_e4m3fn).to(torch.float32)

    fp8_data = torch.unique(fp8_data)
    fp8_data = torch.sort(fp8_data).values

    print(len(fp8_data))
    print(fp8_data)

if __name__ == "__main__":
    # from transformers import AutoModelForCausalLM
    # from transformers import AutoTokenizer
    
    # device = 'cuda:1'
    # model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # model_fp = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map=device)
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model_fp8 = wrap_model(model_fp)
    # print(model_fp8)
    # collect_stats(model_fp8, tokenizer)

    # for i in range(10):
    #     cmp_ov_pt()
    print_fp8_values()
