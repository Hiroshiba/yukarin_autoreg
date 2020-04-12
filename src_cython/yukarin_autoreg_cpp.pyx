cimport numpy

cpdef initialize(
   int graph_length,
   int max_batch_size,
   int local_size,
   int hidden_size,
   int embedding_size,
   int linear_hidden_size,
   int output_size,
   numpy.ndarray[numpy.float32_t, ndim=2] x_embedder_W,
   numpy.ndarray[numpy.float32_t, ndim=2] gru_xw,
   numpy.ndarray[numpy.float32_t, ndim=1] gru_xb,
   numpy.ndarray[numpy.float32_t, ndim=2] gru_hw,
   numpy.ndarray[numpy.float32_t, ndim=1] gru_hb,
   numpy.ndarray[numpy.float32_t, ndim=2] O1_W,
   numpy.ndarray[numpy.float32_t, ndim=1] O1_b,
   numpy.ndarray[numpy.float32_t, ndim=2] O2_W,
   numpy.ndarray[numpy.float32_t, ndim=1] O2_b,
):
   c_initialize(
      graph_length,
      max_batch_size,
      local_size,
      hidden_size,
      embedding_size,
      linear_hidden_size,
      output_size,
      <float*> x_embedder_W.data,
      <float*> gru_xw.data,
      <float*> gru_xb.data,
      <float*> gru_hw.data,
      <float*> gru_hb.data,
      <float*> O1_W.data,
      <float*> O1_b.data,
      <float*> O2_W.data,
      <float*> O2_b.data,
   )


cpdef inference(
   int batch_size,
   int length,
   numpy.ndarray[numpy.int32_t, ndim=2] output,
   numpy.ndarray[numpy.int32_t, ndim=1] x,
   numpy.ndarray[numpy.float32_t, ndim=3] l_array,
   numpy.ndarray[numpy.float32_t, ndim=2] hidden,
):
   c_inference(
      batch_size,
      length,
      <int*> output.data,
      <int*> x.data,
      <float*> l_array.data,
      <float*> hidden.data,
   )
