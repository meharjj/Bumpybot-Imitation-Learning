import torch

def Export(model,x,verify=True):
    torch.onnx.export(model,                # model being run
                x,                          # model input (or a tuple for multiple inputs)
                "model.onnx",               # where to save the model (can be a file or file-like object)
                export_params=True,         # store the trained parameter weights inside the model file
                do_constant_folding=True,   # whether to execute constant folding for optimization
                input_names = ["input"],    # the model's input names
                output_names = ["output"])  # the model's output names
    if verify:
        import onnx
        onnx_model = onnx.load("model.onnx")
        onnx.checker.check_model(onnx_model)
        print("ONNX Conversion Successful.")