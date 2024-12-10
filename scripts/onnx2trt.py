import os
import sys
import logging
import argparse
import platform

import tensorrt as trt
import ctypes
import numpy as np

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

def load_plugins(logger: trt.Logger):
    # Load plugin library
    if platform.system().lower() == 'linux':
        ctypes.CDLL("./libgrid_sample_3d_plugin.so", mode=ctypes.RTLD_GLOBAL)
    else:
        ctypes.CDLL("./checkpoints/liveportrait_onnx/grid_sample_3d_plugin.dll", mode=ctypes.RTLD_GLOBAL, winmode=0)
    # Initialize TensorRT plugin library
    trt.init_libnvinfer_plugins(logger, namespace="")

class EngineBuilder:
    def __init__(self, verbose=False):
        # Logger initialization for 10.7
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.VERBOSE

        # Plugin initialization
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        # Builder creation
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()

        # Workspace size (12 GB)
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 * (2 ** 30))

        # Create optimization profile
        self.profile = self.builder.create_optimization_profile()

        # # Set flags
        # self.config.set_flag(trt.BuilderFlag.STRICT_NANS)
        
        self.batch_size = None
        self.network = None
        self.parser = None

        # Load custom plugins
        load_plugins(self.trt_logger)

    def create_network(self, onnx_path):
    # Network creation flags
      network_flags = trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH

      self.network = self.builder.create_network(network_flags)
      self.parser = trt.OnnxParser(self.network, self.trt_logger)

      onnx_path = os.path.realpath(onnx_path)
      with open(onnx_path, "rb") as f:
          if not self.parser.parse(f.read()):
              log.error(f"Failed to load ONNX file: {onnx_path}")
              for error in range(self.parser.num_errors):
                  log.error(self.parser.get_error(error))
              sys.exit(1)

      inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
      outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

      log.info("Network Description")
      for input in inputs:
          self.batch_size = input.shape[0]
          log.info(f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}")
      for output in outputs:
          log.info(f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}")

      # Update the optimization profile with batch size
      
      for input in inputs:
          # Set min, opt, and max batch sizes
          self.profile.set_shape(
              input.name, 
              (1, *input.shape[1:]),  # min shape
              (1, *input.shape[1:]),  # opt shape 
              (1, *input.shape[1:])   # max shape
          )
      self.config.add_optimization_profile(self.profile)

    def create_engine(self, engine_path, precision):
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info(f"Building {precision} Engine in {engine_path}")

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)

        # Engine building and serialization
        with self.builder.build_serialized_network(self.network, self.config) as engine, open(engine_path, "wb") as f:
            log.info(f"Serializing engine to file: {engine_path}")
            f.write(engine)

def main(args):
    builder = EngineBuilder(args.verbose)
    builder.create_network(args.onnx)
    builder.create_engine(
        args.engine,
        args.precision
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", required=True, help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument(
        "-p",
        "--precision",
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    args = parser.parse_args()
    if args.engine is None:
        args.engine = args.onnx.replace(".onnx", ".trt")
    main(args)