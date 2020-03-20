# Run these code using TensorFlow 1, not TensorFlow 2.0
# conda activate keras-tf-gpu-updated
# cd /d D:\Projects\med-3d-segmentation\src\tools\model\exporters
# python protobuf.py

from pathlib import Path

import keras
import tensorflow as tf
from keras import backend as K


MODEL_PATH = Path(r'D:\Temp\Test\2020.03.06-Class1-Unet-densenet201-binary_crossentropy_plus_dice_loss-352x352-Batch9-T2_tse_SkipEmptySlices.h5')
EXPORTED_PATH = Path(r'D:\Temp\Test\2020.03.06-Class1-Unet-densenet201-binary_crossentropy_plus_dice_loss-352x352-Batch9-T2_tse_SkipEmptySlices.pb')


def export(src_path: Path, dst_path: Path):
    # https://www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/

    # This line must be executed before loading Keras model.
    K.set_learning_phase(0)

    model = keras.models.load_model(str(src_path), compile=False)
    print(f'model.outputs: {model.outputs}')
    print(f'model.inputs: {model.inputs}')

    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])

    tf.train.write_graph(frozen_graph, "model", str(dst_path), as_text=False)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def main():
    export(MODEL_PATH, EXPORTED_PATH)


if __name__ == '__main__':
    main()
