import torch
import numpy as np
import onnxruntime
assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
from onnxruntime.transformers.io_binding_helper import IOBindingHelper

def get_encode_output_shapes(input_len):
    output_shapes = {}
    output_shapes['logits'] = (1, 1, 28166)
    output_shapes['encoder_hidden_states'] = (1, input_len, 1024)
    for i in range(24):
        output_shapes[f"present_key_self_{i}"] = (1, 16, 1, 64)
        output_shapes[f"present_value_self_{i}"] = (1, 16, 1, 64)
    for i in range(24):
        output_shapes[f"present_key_cross_{i}"] = (1, 16, input_len, 64)
        output_shapes[f"present_value_cross_{i}"] = (1, 16, input_len, 64)
        
    return output_shapes

def get_decode_output_shapes(gen_seq_len):
    output_shapes = {}
    output_shapes['logits'] = (1, 1, 28166)
    for i in range(24):
        output_shapes[f"present_key_self_{i}"] = (1, 16, gen_seq_len, 64)
        output_shapes[f"present_value_self_{i}"] = (1, 16, gen_seq_len, 64)
        
    return output_shapes

def get_output_buffers(output_shapes, device, is_float16=False):
    data_type = torch.float16 if is_float16 else torch.float32

    output_buffers = {}
    for name, shape in output_shapes.items():
        output_buffers[name] = torch.empty(np.prod(shape), dtype=data_type, device=device)
    return output_buffers

def ort_type_to_numpy_type(ort_type: str):
        ort_type_to_numpy_type_map = {
            "tensor(int64)": np.longlong,
            "tensor(int32)": np.intc,
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(bool)": bool,
        }
        if ort_type not in ort_type_to_numpy_type_map:
            raise ValueError(f"{ort_type} not found in map")

        return ort_type_to_numpy_type_map[ort_type]
    
def get_io_numpy_type_map(ort_session):
    name_to_numpy_type = {}
    for inp in ort_session.get_inputs():
        name_to_numpy_type[inp.name] = ort_type_to_numpy_type(inp.type)

    for output in ort_session.get_outputs():
        name_to_numpy_type[output.name] = ort_type_to_numpy_type(output.type)
    return name_to_numpy_type

def get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes, return_numpy=True):
    """Copy results to cpu. Returns a list of numpy array."""
    ort_outputs = []
    for output in ort_session.get_outputs():
        output_name = output.name
        buffer = output_buffers[output_name]
        shape = output_shapes[output_name]
        copy_tensor = buffer[0 : np.prod(shape)].reshape(shape).clone().detach()
        if return_numpy:
            ort_outputs.append(copy_tensor.cpu().numpy())
        else:
            ort_outputs.append(copy_tensor)
    return ort_outputs

def prepare_io_binding(onnx_model, onnx_input, output_buffers, output_shapes):
    io_binding = onnx_model.io_binding()
    name_to_np_type = get_io_numpy_type_map(onnx_model)
    
    for name, value in onnx_input.items():
        assert value.is_contiguous()
        io_binding.bind_input(
            name,
            value.device.type,
            0 if value.device.type == 'cpu' else value.device.index,
            name_to_np_type[name],
            list(value.size()),
            value.data_ptr(),
        )
        
    for output in onnx_model.get_outputs():
        output_name = output.name
        output_buffer = output_buffers[output_name]
        io_binding.bind_output(
            output_name,
            output_buffer.device.type,
            0 if value.device.type == 'cpu' else value.device.index,
            name_to_np_type[output_name],
            output_shapes[output_name],
            output_buffer.data_ptr(),
        )
        
    return io_binding

def encode_with_io_binding(onnx_encoder, encode_input, input_len, buffer_device='cuda:1'):
    output_shapes = get_encode_output_shapes(input_len)
    output_buffers = get_output_buffers(output_shapes, device=buffer_device)
    
    io_binding = prepare_io_binding(onnx_encoder, encode_input, output_buffers, output_shapes)
    onnx_encoder.run_with_iobinding(io_binding)
    
    outputs = get_outputs_from_io_binding_buffer(onnx_encoder, output_buffers, output_shapes, return_numpy=False)
    return outputs

def decode_with_io_binding(onnx_decoder, decode_input, now_seq_len, buffer_device='cuda:1'):
    output_shapes = get_decode_output_shapes(now_seq_len)
    output_buffers = get_output_buffers(output_shapes, device=buffer_device)
    
    io_binding = prepare_io_binding(onnx_decoder, decode_input, output_buffers, output_shapes)
    onnx_decoder.run_with_iobinding(io_binding)
    
    outputs = get_outputs_from_io_binding_buffer(onnx_decoder, output_buffers, output_shapes, return_numpy=False)
    return outputs

def generate(input_text, tokenizer, onnx_encoder, onnx_decoder):
    result_ids = []
    
    max_encoder_length = 192
    tokens = tokenizer(input_text, max_length=max_encoder_length, truncation=True, return_tensors="np")

    encoder_input = {}
    encoder_input["encoder_input_ids"] = tokens['input_ids']
    encoder_input["encoder_attention_mask"] = tokens['attention_mask']
    encoder_input["decoder_input_ids"] = np.zeros([1, 1], dtype=np.int64)
    
    encoder_outputs = onnx_encoder.run(None, encoder_input)
    gen_tokenid = encoder_outputs[0].argmax()
    result_ids.append(gen_tokenid)
    
    decode_input = {}
    decode_input['input_ids'] = np.array([[gen_tokenid]])
    decode_input['encoder_attention_mask'] = tokens['attention_mask']
    decode_input['encoder_hidden_states'] = encoder_outputs[1]
    for i_layer in range(24):
        decode_input[f"past_key_self_{i_layer}"] = encoder_outputs[i_layer*2+2]
        decode_input[f"past_value_self_{i_layer}"] = encoder_outputs[i_layer*2+3]
        decode_input[f"past_key_cross_{i_layer}"] = encoder_outputs[i_layer*2+2+48]
        decode_input[f"past_value_cross_{i_layer}"] = encoder_outputs[i_layer*2+3+48]
    
    for i_token in range(32):
        decode_outputs = onnx_decoder.run(None, decode_input)
        gen_tokenid = decode_outputs[0].argmax()
        result_ids.append(gen_tokenid)
        if gen_tokenid == 1:
            break

        decode_input['input_ids'] = np.array([[gen_tokenid]])
        for i_layer in range(24):
            decode_input[f"past_key_self_{i_layer}"] = decode_outputs[i_layer*2+1]
            decode_input[f"past_value_self_{i_layer}"] = decode_outputs[i_layer*2+2]
        
    return result_ids

def generate_with_io_binding(input_text, tokenizer, onnx_encoder, onnx_decoder, buffer_device='cuda:0'):
    result_ids = []
    
    max_encoder_length = 192
    tokens = tokenizer(input_text, max_length=max_encoder_length, truncation=True, return_tensors="np")

    encode_input = {}
    encode_input["encoder_input_ids"] = tokens['input_ids']
    encode_input["encoder_attention_mask"] = tokens['attention_mask']
    encode_input["decoder_input_ids"] = np.zeros([1, 1], dtype=np.int64)
    encode_input = {name: torch.from_numpy(value) for name, value in encode_input.items()}
    
    encoder_outputs = encode_with_io_binding(onnx_encoder, encode_input, input_len=len(tokens['input_ids'][0]), buffer_device=buffer_device)
    gen_tokenid = int(encoder_outputs[0].argmax().cpu())
    result_ids.append(gen_tokenid)
    
    decode_input = {}
    decode_input['input_ids'] = torch.tensor([[gen_tokenid]])
    decode_input['encoder_attention_mask'] = encode_input["encoder_attention_mask"]
    decode_input['encoder_hidden_states'] = encoder_outputs[1]
    for i_layer in range(24):
        decode_input[f"past_key_self_{i_layer}"] = encoder_outputs[i_layer*2+2]
        decode_input[f"past_value_self_{i_layer}"] = encoder_outputs[i_layer*2+3]
        decode_input[f"past_key_cross_{i_layer}"] = encoder_outputs[i_layer*2+2+48]
        decode_input[f"past_value_cross_{i_layer}"] = encoder_outputs[i_layer*2+3+48]
    decode_input = {name: value for name, value in decode_input.items()}
    
    for i_token in range(32):
        decode_outputs = decode_with_io_binding(onnx_decoder, decode_input, now_seq_len=i_token+2, buffer_device=buffer_device)
        gen_tokenid = int(decode_outputs[0].argmax().cpu())
        result_ids.append(gen_tokenid)
        if gen_tokenid == 1:
            break

        decode_input['input_ids'] = torch.tensor([[gen_tokenid]])
        for i_layer in range(24):
            decode_input[f"past_key_self_{i_layer}"] = decode_outputs[i_layer*2+1].clone().detach()
            decode_input[f"past_value_self_{i_layer}"] = decode_outputs[i_layer*2+2].clone().detach()
        
    return result_ids