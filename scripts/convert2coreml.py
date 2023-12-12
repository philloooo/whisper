import whisper
import numpy as np
import torch
import coremltools as ct

from coremltools.converters.mil.mil import Builder as mb


def load_models():
    model = whisper.load_model("tiny")
    return model.encoder, model.decoder

def convert_encoder_to_tvm(model):
    model.eval()

    input_shape = (1, 80, 3000)
    input_data = torch.randn(input_shape)
    traced_model = torch.jit.trace(model, input_data)

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="logmel_data", shape=input_shape)]
    )

    return model

def convert_decoder_to_tvm(model):
    model.eval()

    tokens_shape = []
    max_token = 448
    # max_token = 448, max number of EnumeratedShapes supported by coreml = 128
    segment = max_token//128 + 1
    i = segment
    while(i<max_token+segment):
        tokens_shape.append([1, i])
        i += segment

    audio_shape = (1, 1500, 384)
    token_data = (1000*torch.rand(tokens_shape[0])).long()
    audio_data = torch.rand(audio_shape)
    traced_model = torch.jit.trace(model, (token_data, audio_data))

    token_flexible_shape = ct.EnumeratedShapes(shapes=tokens_shape, default=tokens_shape[0])


    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="token_data", shape=token_flexible_shape, dtype=int),
            ct.TensorType(name="audio_data", shape=audio_shape)
        ]
    )

    return model

def main():
    encoder, decoder = load_models()

    decoder = convert_decoder_to_tvm(decoder)
    decoder.save("decoder.mlpackage")

    encoder = convert_encoder_to_tvm(encoder)
    encoder.save("encoder.mlpackage")

if __name__ == "__main__":
    main()