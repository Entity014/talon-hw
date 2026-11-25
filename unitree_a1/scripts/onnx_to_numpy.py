"""Convert PROMO ONNX model to pure NumPy .npz weights directly.
No PyTorch checkpoint required!
Extracts Linear weights, biases, and ELU activations directly from onnx.load(model).
"""

import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

def convert_onnx_to_numpy_npz(onnx_path: Path, out_path: Path, meta_path: Path = None):
    model = onnx.load(str(onnx_path))
    meta = {}
    if meta_path and meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

    # Map initializers by name
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}

    weights = {}
    weights["meta"] = np.array(json.dumps({
        "format": "promo_numpy_v1",
        "onnx_source": str(onnx_path),
        "obs_dim": meta.get("obs_dim", 45),
        "hist_len": meta.get("hist_len", 10),
        "vel_hist": 5,
        "latent": 45,
        "film_actor": meta.get("film_actor", False),
        "action_scale": meta.get("action_scale", 0.25),
        "kp": meta.get("kp", 55.0),
        "kd": meta.get("kd", 0.8),
        "default_q": [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]
    }))

    # We extract gemm nodes or matmul/add nodes in graph order
    # Let's inspect nodes in order
    linears = []
    for node in model.graph.node:
        if node.op_type == "Gemm":
            # inputs: [A, B, C] where B is weight, C is bias
            w_name = node.input[1]
            b_name = node.input[2]
            w = inits[w_name]
            b = inits[b_name]
            # Gemm attributes transB usually 1
            transB = 0
            for attr in node.attribute:
                if attr.name == "transB":
                    transB = attr.i
            if not transB:
                w = w.T
            linears.append((node.name, w, b))
        elif node.op_type == "MatMul":
            w_name = node.input[1]
            if w_name in inits:
                w = inits[w_name].T
                linears.append((node.name, w, None))
        elif node.op_type == "Add":
            b_name = node.input[1] if node.input[1] in inits else (node.input[0] if node.input[0] in inits else None)
            if b_name and len(linears) > 0 and linears[-1][2] is None:
                name, w, _ = linears[-1]
                linears[-1] = (name, w, inits[b_name])

    print(f"Found {len(linears)} linear layers in ONNX graph.")
    for idx, (name, w, b) in enumerate(linears):
        print(f"  [{idx}] {name}: weight shape {w.shape}, bias shape {b.shape if b is not None else None}")

    # base_v2 structure (film_actor = False, 10 layers)
    # oursA structure (film_actor = True, 12 layers)
    if len(linears) == 10:
        # hist_enc (0, 1, 2)
        for i in range(3):
            weights[f"hist_enc.{i}.weight"] = linears[i][1].astype(np.float32)
            weights[f"hist_enc.{i}.bias"] = linears[i][2].astype(np.float32)
        # vel_est (3, 4, 5)
        for i in range(3):
            weights[f"vel_est.{i}.weight"] = linears[3 + i][1].astype(np.float32)
            weights[f"vel_est.{i}.bias"] = linears[3 + i][2].astype(np.float32)
        # actor (6, 7, 8, 9)
        for i in range(4):
            weights[f"actor.{i}.weight"] = linears[6 + i][1].astype(np.float32)
            weights[f"actor.{i}.bias"] = linears[6 + i][2].astype(np.float32)
        weights["film_actor"] = np.array(False)
    elif len(linears) == 12:
        # hist_enc (0, 1, 2)
        for i in range(3):
            weights[f"hist_enc.{i}.weight"] = linears[i][1].astype(np.float32)
            weights[f"hist_enc.{i}.bias"] = linears[i][2].astype(np.float32)
        # vel_est (3, 4, 5)
        for i in range(3):
            weights[f"vel_est.{i}.weight"] = linears[3 + i][1].astype(np.float32)
            weights[f"vel_est.{i}.bias"] = linears[3 + i][2].astype(np.float32)
        # film (6, 7)
        for i in range(2):
            weights[f"film.{i}.weight"] = linears[6 + i][1].astype(np.float32)
            weights[f"film.{i}.bias"] = linears[6 + i][2].astype(np.float32)
        # actor layers (8, 9, 10)
        for i in range(3):
            weights[f"actor_layer.{i}.weight"] = linears[8 + i][1].astype(np.float32)
            weights[f"actor_layer.{i}.bias"] = linears[8 + i][2].astype(np.float32)
        # actor out (11)
        weights["actor_out.weight"] = linears[11][1].astype(np.float32)
        weights["actor_out.bias"] = linears[11][2].astype(np.float32)
        weights["film_actor"] = np.array(True)
    else:
        raise ValueError(f"Unexpected layer count {len(linears)} for model")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **weights)
    print(f"Successfully converted ONNX -> {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--meta", default=None)
    args = p.parse_args()
    convert_onnx_to_numpy_npz(Path(args.onnx), Path(args.out), Path(args.meta) if args.meta else None)
