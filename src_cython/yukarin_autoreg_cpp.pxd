cdef extern from "yukarin_autoreg_cpp.h":
    void c_initialize "initialize" (
        int graph_length,
        int max_batch_size,
        int local_size,
        int hidden_size,
        int embedding_size,
        int linear_hidden_size,
        int output_size,
        float* h_x_embedder_W,
        float* h_gru_xw,
        float* h_gru_xb,
        float* h_gru_hw,
        float* h_gru_hb,
        float* h_O1_W,
        float* h_O1_b,
        float* h_O2_W,
        float* h_O2_b
    )

    void c_inference "inference" (
        int batch_size,
        int length,
        int* h_output,
        int* h_x,
        float* h_l_array,
        float* h_hidden
    )
