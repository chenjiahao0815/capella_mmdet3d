import tensorrt as trt

def build_engine(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("spatial_features", (1,64,400,400), (1,64,400,400), (1,64,400,400))
    config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine)

build_engine("pointpillars_backbone_multilevel.onnx", "pointpillars_backbone_multilevel_fp16.engine")