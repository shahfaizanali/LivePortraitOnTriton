FROM nvcr.io/nvidia/tritonserver:24.11-py3
COPY libgrid_sample_3d_plugin.so /opt/tritonserver/backends/tensorrt/
ENV LD_LIBRARY_PATH=/opt/tritonserver/backends/tensorrt:$LD_LIBRARY_PATH
ENV LD_PRELOAD=/opt/tritonserver/backends/tensorrt/libgrid_sample_3d_plugin.so

