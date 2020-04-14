import numpy
import cupy

from yukarin_autoreg.config import ModelConfig
from yukarin_autoreg.model import create_predictor
from yukarin_autoreg.network.fast_forward import get_fast_forward_params, fast_generate

import yukarin_autoreg_cpp

max_batch_size = 4
graph_length = 1000
length = 24000
config = ModelConfig(
    dual_softmax=False,
    bit_size=10,
    conditioning_size=80,
    embedding_size=16,
    hidden_size=896,
    linear_hidden_size=896,
    local_size=5,
    local_scale=1,
    local_layer_num=1,
    speaker_size=0,
    speaker_embedding_size=0,
    weight_initializer=None,
)
# config = ModelConfig(
#     dual_softmax=False,
#     bit_size=10,
#     gaussian=False,
#     input_categorical=True,
#     conditioning_size=None,
#     embedding_size=16,
#     hidden_size=896,
#     linear_hidden_size=896,
#     local_size=0,
#     local_scale=None,
#     local_layer_num=None,
#     speaker_size=0,
#     speaker_embedding_size=None,
#     weight_initializer=None,
# )
model = create_predictor(config)
model.to_device(0)

local_size = config.conditioning_size * 2 if config.conditioning_size is not None else 0

base_x = model.xp.array(numpy.random.randint(0, config.bit_size ** 2, size=(max_batch_size,)).astype(numpy.int32))
base_l_array = model.xp.array(numpy.random.rand(length, max_batch_size, local_size).astype(numpy.float32))
base_hidden = model.xp.array(numpy.random.rand(max_batch_size, config.hidden_size).astype(numpy.float32))

params = get_fast_forward_params(model)

def to_numpy(a):
    if isinstance(a, cupy.ndarray):
        a = cupy.asnumpy(a)
    return numpy.ascontiguousarray(a)

# C++
yukarin_autoreg_cpp.initialize(
    graph_length=graph_length,
    max_batch_size=max_batch_size,
    local_size=local_size,
    hidden_size=config.hidden_size,
    embedding_size=config.embedding_size,
    linear_hidden_size=config.linear_hidden_size,
    output_size=2 ** config.bit_size,
    x_embedder_W=to_numpy(params['x_embedder_W']),
    gru_xw=to_numpy(params['gru_xw']),
    gru_xb=to_numpy(params['gru_xb']),
    gru_hw=to_numpy(params['gru_hw']),
    gru_hb=to_numpy(params['gru_hb']),
    O1_W=to_numpy(params['O1_W']),
    O1_b=to_numpy(params['O1_b']),
    O2_W=to_numpy(params['O2_W']),
    O2_b=to_numpy(params['O2_b']),
)

before_output = None
for batch_size in [1, 2, 4]:
    x = model.xp.copy(base_x[:batch_size])
    l_array = model.xp.copy(base_l_array[:, :batch_size])
    hidden = model.xp.copy(base_hidden[:batch_size])

    # x = model.xp.zeros_like(x)
    # l_array = model.xp.zeros_like(l_array)
    # hidden = model.xp.zeros_like(hidden)

    output = numpy.ones((length, batch_size), dtype=numpy.int32) * -1
    r = yukarin_autoreg_cpp.inference(
        batch_size=batch_size,
        length=length,
        output=output,
        x=to_numpy(x),
        l_array=to_numpy(l_array),
        hidden=to_numpy(hidden),
    )
    print(output)

    if before_output is not None:
        min_batch_size = min(before_output.shape[1], output.shape[1])
        print('equal', numpy.all(before_output[:, :min_batch_size] == output[:, :min_batch_size]))
        assert numpy.all(before_output[:, :min_batch_size] == output[:, :min_batch_size])
    before_output = output


expected = cupy.array(fast_generate(
    length=length,
    x=model.xp.copy(base_x),
    l_array=model.xp.copy(model.xp.transpose(base_l_array, [1, 0, 2])),
    h=model.xp.copy(base_hidden),
    **params,
))

if output.shape == expected.shape:
    print('equal', numpy.all(output == cupy.asnumpy(expected)))
    assert numpy.all(output == cupy.asnumpy(expected))
else:
    print(expected)
