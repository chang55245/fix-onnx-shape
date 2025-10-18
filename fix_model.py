#!/usr/bin/env python3
"""
replace_simplified_ln_combined.py
Replaces all SimplifiedLayerNormalization ops (any domain) with
pure ONNX RMSNorm-style subgraphs, preserving dtype and external data.
Then runs safe shape inference to produce a clean, portable ONNX model.
"""

import sys
import os
import uuid
import onnx
from onnx import helper, numpy_helper, TensorProto, shape_inference


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_valueinfo_shape(model, tensor_name, dtype=TensorProto.INT64, shape=[1]):
    """Ensure that tensor_name has value_info with the given dtype/shape."""
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            # Update dims if missing or empty
            vi.type.tensor_type.elem_type = dtype
            dims = vi.type.tensor_type.shape.dim
            if len(dims) == 0 or all(not d.HasField("dim_value") for d in dims):
                del dims[:]
                for d in shape:
                    vi.type.tensor_type.shape.dim.add().dim_value = d
            print(f"🔧 Updated existing value_info for {tensor_name} -> shape {shape}")
            return model

    # Not found → create new one
    vi = helper.make_tensor_value_info(tensor_name, dtype, shape)
    model.graph.value_info.append(vi)
    print(f"🔧 Added new value_info for {tensor_name} -> shape {shape}")
    return model

def unique(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def dtype_of_value(model, name):
    for coll in (model.graph.value_info, model.graph.input, model.graph.output):
        for vi in coll:
            if vi.name == name and vi.type and vi.type.tensor_type.elem_type:
                return vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        if init.name == name:
            return init.data_type
    return None


def to_fp32_if_needed(name, orig_dtype, new_name):
    if orig_dtype in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
        cast = helper.make_node("Cast", [name], [new_name], to=TensorProto.FLOAT)
        return [cast], new_name
    return [], name


def cast_back_if_needed(name, target_dtype, new_name):
    if target_dtype in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
        cast = helper.make_node("Cast", [name], [new_name], to=target_dtype)
        return [cast], new_name
    if target_dtype and target_dtype != TensorProto.FLOAT:
        cast = helper.make_node("Cast", [name], [new_name], to=target_dtype)
        return [cast], new_name
    return [], name


def all_nodes(graph):
    for n in graph.node:
        yield n
        for attr in n.attribute:
            if attr.g:
                yield from all_nodes(attr.g)


# ---------------------------------------------------------------------
# Core replacement logic
# ---------------------------------------------------------------------
def replace_simplified_layernorm(model):
    g = model.graph
    new_nodes = []
    replaced = 0

    for node in list(g.node):
        is_sln = (
            node.op_type in ("SimplifiedLayerNormalization", "SimplifiedLayerNorm")
            and node.domain in ("", "com.microsoft", "ai.onnx.contrib")
        )
        if not is_sln:
            new_nodes.append(node)
            continue

        replaced += 1
        X = node.input[0]
        scale = node.input[1] if len(node.input) > 1 else ""
        out_Y = node.output[0]
        epsilon = 1e-5
        for a in node.attribute:
            if a.name.lower() in ("epsilon", "eps"):
                epsilon = float(a.f)

        x_dtype = dtype_of_value(model, X)
        scale_dtype = dtype_of_value(model, scale)

        local_nodes = []

        # Cast to fp32 if needed
        casts, X_fp32 = to_fp32_if_needed(X, x_dtype, unique(X + "_fp32"))
        local_nodes.extend(casts)
        casts, scale_fp32 = to_fp32_if_needed(scale, scale_dtype, unique(scale + "_fp32")) if scale else ([], "")
        local_nodes.extend(casts)

        # X^2 → mean(X^2) → +eps → sqrt → reciprocal → X * inv → *scale
        sq = unique("sq")
        local_nodes.append(helper.make_node("Mul", [X_fp32, X_fp32], [sq]))

        rm = unique("rm")
        local_nodes.append(helper.make_node("ReduceMean", [sq], [rm], axes=[-1], keepdims=1))

        eps_name = unique("eps")
        eps_tensor = numpy_helper.from_array(
            __import__("numpy").array(epsilon, dtype="float32"), name=eps_name
        )
        model.graph.initializer.append(eps_tensor)

        add_eps = unique("add_eps")
        local_nodes.append(helper.make_node("Add", [rm, eps_name], [add_eps]))

        sqrt = unique("sqrt")
        inv = unique("inv")
        local_nodes.append(helper.make_node("Sqrt", [add_eps], [sqrt]))
        local_nodes.append(helper.make_node("Reciprocal", [sqrt], [inv]))

        norm = unique("norm")
        local_nodes.append(helper.make_node("Mul", [X_fp32, inv], [norm]))

        y_fp32 = unique("y_fp32")
        if scale_fp32:
            local_nodes.append(helper.make_node("Mul", [norm, scale_fp32], [y_fp32]))
        else:
            y_fp32 = norm

        back_nodes, y_final = cast_back_if_needed(y_fp32, x_dtype, out_Y)
        local_nodes.extend(back_nodes)
        if y_final != out_Y:
            local_nodes.append(helper.make_node("Identity", [y_fp32], [out_Y]))

        new_nodes.extend(local_nodes)

    if replaced == 0:
        print("⚠️  No SimplifiedLayerNormalization nodes found.")
    else:
        print(f"✅ Replaced {replaced} SimplifiedLayerNormalization node(s).")

    del g.node[:]
    g.node.extend(new_nodes)
    return model

def safe_save_model(model, path):
    """Safely save ONNX model, removing any stale external data."""
    bin_path = os.path.join(os.path.dirname(path), "model_data.bin")
    if os.path.exists(path):
        os.remove(path)
    if os.path.exists(bin_path):
        os.remove(bin_path)
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model_data.bin",
    )


def safe_infer_and_save(model, dst):
    """Run shape inference using a temporary ONNX file and preserve external data."""
    try:
        print("Running safe shape inference...")

        tmp = "tmp_replaced.onnx"
        safe_save_model(model, tmp)

        shape_inference.infer_shapes_path(tmp, dst)

        # Load with external data intact
        inferred = onnx.load(dst, load_external_data=True)
        print(f"✅ Inferred model written to {dst} (nodes: {len(inferred.graph.node)})")
        return inferred
    except Exception as e:
        print("⚠️ Shape inference failed:", e)
        return model

def fix_concat_constants(model):
    fixed = 0

    # Find all Concat input names
    concat_inputs = set()
    for node in model.graph.node:
        if node.op_type == "Concat":
            concat_inputs.update(node.input)

    # Fix Constant nodes feeding those inputs
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] in concat_inputs:
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    t = attr.t
                    if len(t.dims) == 0:
                        # promote scalar -> 1D tensor
                        t.dims.append(1)
                        # ensure data field still valid
                        if not (t.int64_data or t.float_data or t.raw_data):
                            # fallback if data was empty, copy single value
                            if t.data_type == TensorProto.INT64:
                                t.int64_data.append(-1)
                            elif t.data_type == TensorProto.FLOAT:
                                t.float_data.append(0.0)
                        fixed += 1
                        print(f"🔧 Promoted scalar Constant '{node.name}' to shape [1].")

    if fixed:
        print(f"✅ Fixed {fixed} scalar Constant(s) feeding Concat.")
    else:
        print("ℹ️  No scalar Constants feeding Concat were found.")
    return model

def fix_scalar_shape_constant(model):
    fixed = 0
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == "/model/constants/TensorProto.INT64/0D/1":
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    t = attr.t
                    if len(t.dims) == 0:
                        t.dims.append(1)
                        fixed += 1
                        print(f"🔧 Promoted scalar Constant '{node.name}' to 1-D tensor [1].")
    if not fixed:
        print("ℹ️  Constant '/model/constants/TensorProto.INT64/0D/1' not found or already fixed.")
    return model
# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def ensure_concat_output_is_vector(model, tensor_name="/model/pos_ids_reformat/Concat/output_0"):
    """Force the given tensor to have a 1-D INT64 value_info."""
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            vi.type.tensor_type.elem_type = TensorProto.INT64
            dims = vi.type.tensor_type.shape.dim
            del dims[:]  # clear old dims
            vi.type.tensor_type.shape.dim.add().dim_value = 1
            print(f"🔧 Updated existing value_info for {tensor_name} -> shape [1]")
            return model

    vi = helper.make_tensor_value_info(tensor_name, TensorProto.INT64, [1])
    model.graph.value_info.append(vi)
    print(f"🔧 Added new value_info for {tensor_name} -> shape [1]")
    return model
def ensure_rank1_valueinfos_for_shape_constants(model):
    """Add or update value_info for known shape constants used by Reshape."""
    shape_consts = [
        "/model/constants/TensorProto.INT64/1D/0, -1, 128",
        "/model/constants/TensorProto.INT64/1D/0, -1, 2048",
        "/model/constants/TensorProto.INT64/1D/0, -1, 1024",
    ]
    existing_names = {vi.name for vi in model.graph.value_info}
    for name in shape_consts:
        if name in existing_names:
            for vi in model.graph.value_info:
                if vi.name == name:
                    vi.type.tensor_type.elem_type = TensorProto.INT64
                    dims = vi.type.tensor_type.shape.dim
                    del dims[:]
                    vi.type.tensor_type.shape.dim.add().dim_value = 3
                    print(f"🔧 Updated value_info for {name} → shape [3]")
        else:
            vi = helper.make_tensor_value_info(name, TensorProto.INT64, [3])
            model.graph.value_info.append(vi)
            print(f"🔧 Added value_info for {name} → shape [3]")
    return model

def main():
    if len(sys.argv) != 3:
        print("Usage: python replace_simplified_ln_combined.py input.onnx output.onnx")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]

    print(f"Loading model: {src}")
    model = onnx.load(src, load_external_data=True)
    print(f"Nodes before: {len(model.graph.node)}")

    model = replace_simplified_layernorm(model)
    print(f"Nodes after replacement: {len(model.graph.node)}")

    model = safe_infer_and_save(model, dst)
    model = fix_concat_constants(model)
    model = ensure_valueinfo_shape(model, "/model/constants/TensorProto.INT64/1D/-1", TensorProto.INT64, [1])
    model = fix_scalar_shape_constant(model)
    model = ensure_concat_output_is_vector(model)
    model = ensure_rank1_valueinfos_for_shape_constants(model)
    safe_save_model(model, dst)

    print("✅ Done.")
    print(f"Output model: {dst}")
    print(f"File size: {os.path.getsize(dst)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
