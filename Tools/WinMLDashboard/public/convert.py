import argparse
from pathlib import Path
import onnxmltools

def parse_args():
    parser = argparse.ArgumentParser(description='Convert model to ONNX.')
    parser.add_argument('source', help='source  model')
    parser.add_argument('framework', help='source framework model comes from')
    parser.add_argument('ONNXVersion', help='which ONNX Version to convert to')
    parser.add_argument('inputNames', help='names of input nodes')
    parser.add_argument('outputNames', help='names of output nodes')
    parser.add_argument('destination', help='destination ONNX model (ONNX or prototxt extension)')
    parser.add_argument('--name', default='WimMLDashboardConvertedModel', help='(ONNX output only) model name')
    return parser.parse_args()


def get_extension(path):
    return Path(path).suffix[1:].lower()


def save_onnx(onnx_model, destination):
    destination_extension = get_extension(destination)
    if destination_extension == 'onnx':
        onnxmltools.utils.save_model(onnx_model, destination)
    else:
        raise RuntimeError(
            f'Conversion to extension {destination_extension} is not supported'
        )

def get_opset(ONNXVersion):
    if ONNXVersion == '1.2':
        return 7
    elif ONNXVersion == '1.3':
        return 8
    elif ONNXVersion == '1.4':
        return 9
    elif ONNXVersion == '1.5':
        return 10
    elif ONNXVersion == '1.6':
        return 11
    elif ONNXVersion == '1.7':
        return 12
    elif ONNXVersion == '1.8':
        return 13
    elif ONNXVersion == '1.9':
        return 14
    else:
        print(
            f'WARNING: ONNX Version {ONNXVersion} does not map to any known opset version, defaulting to opset V7'
        )
        return 7


def coreml_converter(args):
    # When imported, CoreML tools checks for the current version of Keras and TF and prints warnings if they are
    # outside its expected range. We don't want it to import these packages (since they are big and take seconds to
    # load) and we don't want to clutter the console with unrelated Keras warnings when converting from CoreML.


    import sys
    sys.modules['keras'] = None
    import coremltools
    from onnxmltools.convert import convert_coreml
    source_model = coremltools.utils.load_spec(args.source)
    return convert_coreml(
        source_model, name=args.name, target_opset=get_opset(args.ONNXVersion)
    )


def keras_converter(args):
    from onnxmltools.convert import convert_keras
    from keras.models import load_model
    source_model = load_model(args.source)
    return convert_keras(
        source_model, name=args.name, target_opset=get_opset(args.ONNXVersion)
    )

def scikit_learn_converter(args):
    from sklearn.externals import joblib
    source_model = joblib.load(args.source)
    from onnxmltools.convert.common.data_types import FloatTensorType
    from onnxmltools.convert import convert_sklearn

    return convert_sklearn(
        source_model,
        initial_types=[('input', FloatTensorType(source_model.coef_.shape))],
        target_opset=get_opset(args.ONNXVersion),
    )

def xgboost_converter(args):
    from sklearn.externals import joblib
    source_model = joblib.load(args.source)
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    return convert_xgboost(
        source_model,
        initial_types=[('input', FloatTensorType(shape=[1, 'None']))],
        target_opset=get_opset(args.ONNXVersion),
    )

def libSVM_converter(args):
    # not using target_opset for libsvm convert since the converter is only generating operators in ai.onnx.ml domain
    # but just passing in target_opset for consistency
    from libsvm.svmutil import svm_load_model
    source_model = svm_load_model(args.source)
    from onnxmltools.convert import convert_libsvm
    from onnxmltools.convert.common.data_types import FloatTensorType
    return convert_libsvm(
        source_model,
        initial_types=[('input', FloatTensorType([1, 'None']))],
        target_opset=get_opset(args.ONNXVersion),
    )

def convert_tensorflow_file(filename, opset, input_names, output_names):
    import tensorflow
    from tensorflow.core.framework import graph_pb2
    from tensorflow.python.tools import freeze_graph
    import onnx
    import tensorflow as tf

    graph_def = graph_pb2.GraphDef()
    with open(filename, 'rb') as file:
        graph_def.ParseFromString(file.read())
    converted_model = onnxmltools.convert_tensorflow(graph_def, target_opset=opset, input_names=input_names, output_names=output_names)
    onnx.checker.check_model(converted_model)
    return converted_model

def tensorFlow_converter(args):
    return convert_tensorflow_file(args.source, get_opset(args.ONNXVersion), args.inputNames.split(), args.outputNames.split())

def onnx_converter(args):
    return onnxmltools.load_model(args.source)

framework_converters = {
    '': onnx_converter,
    'coreml': coreml_converter,
    'keras': keras_converter,
    'scikit-learn': scikit_learn_converter,
    'xgboost': xgboost_converter,
    'libsvm': libSVM_converter,
    'tensorflow': tensorFlow_converter
}

suffix_converters = {
    'h5': keras_converter,
    'keras': keras_converter,
    'mlmodel': coreml_converter,
    'onnx': onnx_converter,
}


def main(args):
    # TODO: framework converter check.
    source_extension = get_extension(args.source)
    framework = args.framework.lower()
    frame_converter = framework_converters.get(framework)
    suffix_converter = suffix_converters.get(source_extension) 

    if not (frame_converter or suffix_converter):
        raise RuntimeError(
            f'Conversion from extension {source_extension} is not supported'
        )

    if frame_converter and suffix_converter and (frame_converter != suffix_converter):
        raise RuntimeError(
            f'model with extension {source_extension} do not come from {framework}'
        )

    onnx_model = None
    if frame_converter:
        onnx_model = frame_converter(args)
    else:
        onnx_model = suffix_converter(args)


    if framework == 'tensorflow':
        with open(args.destination, 'wb') as file:
            file.write(onnx_model.SerializeToString())
    else:
        save_onnx(onnx_model, args.destination)


if __name__ == '__main__':
    main(parse_args())
