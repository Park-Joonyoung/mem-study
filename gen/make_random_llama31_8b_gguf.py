#!/usr/bin/env python3
"""Generate a random-weight Llama-3.1-8B-architecture gguf (f16).

For memory-trace experiments only: kernels read every weight regardless of
value, so random weights produce the exact same KV/weight access pattern as
real ones. Tokenizer metadata is copied verbatim from the test vocab gguf
(models/ggml-vocab-llama-bpe.gguf) so llama.cpp can load the model.

Shapes follow convert_hf_to_gguf.py output for Meta-Llama-3.1-8B:
  n_vocab 128256, n_embd 4096, n_layer 32, n_head 32, n_head_kv 8,
  head_dim 128, n_ff 14336, rope_theta 500000. (~16 GB f16 on disk.)
Rope-scaling keys (and rope_freqs tensor) are omitted -> vanilla rope; fine
for ctx <= 8192 experiments.
"""

import sys
from pathlib import Path

import numpy as np

LLAMA_CPP = Path("/home/mem-study/llama.cpp")
sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402

VOCAB_GGUF = LLAMA_CPP / "models" / "ggml-vocab-llama-bpe.gguf"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/mem-study/models/llama31_8b_random_f16.gguf")

N_VOCAB, N_EMBD, N_LAYER = 128256, 4096, 32
N_HEAD, N_HEAD_KV, HEAD_DIM, N_FF = 32, 8, 128, 14336

rng = np.random.default_rng(0)


def rand_f16(*shape):
    return (rng.standard_normal(np.prod(shape), dtype=np.float32) * 0.02) \
        .astype(np.float16).reshape(shape)


def ones_f32(n):
    return np.ones(n, dtype=np.float32)


def copy_tokenizer_fields(reader: GGUFReader, writer: GGUFWriter) -> int:
    """Copy every tokenizer.* KV field from the vocab gguf verbatim."""
    n = 0
    for field in reader.fields.values():
        if not field.name.startswith("tokenizer."):
            continue
        val = field.contents()
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            writer.add_array(field.name, val)
        elif vtype == gguf.GGUFValueType.STRING:
            writer.add_string(field.name, val)
        elif vtype in (gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32):
            writer.add_uint32(field.name, int(val))
        elif vtype == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(field.name, float(val))
        elif vtype == gguf.GGUFValueType.BOOL:
            writer.add_bool(field.name, bool(val))
        else:
            raise RuntimeError(f"unhandled vocab field type {vtype} for {field.name}")
        n += 1
    return n


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    reader = GGUFReader(VOCAB_GGUF)
    writer = GGUFWriter(str(OUT), arch="llama")

    writer.add_block_count(N_LAYER)
    writer.add_context_length(8192)
    writer.add_embedding_length(N_EMBD)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_HEAD_KV)
    writer.add_rope_dimension_count(HEAD_DIM)
    writer.add_rope_freq_base(500000.0)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(N_VOCAB)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    n_tok = copy_tokenizer_fields(reader, writer)
    print(f"copied {n_tok} tokenizer fields from {VOCAB_GGUF.name}")

    writer.add_tensor("token_embd.weight", rand_f16(N_VOCAB, N_EMBD))
    for i in range(N_LAYER):
        p = f"blk.{i}"
        writer.add_tensor(f"{p}.attn_norm.weight", ones_f32(N_EMBD))
        writer.add_tensor(f"{p}.attn_q.weight", rand_f16(N_EMBD, N_EMBD))
        writer.add_tensor(f"{p}.attn_k.weight", rand_f16(N_HEAD_KV * HEAD_DIM, N_EMBD))
        writer.add_tensor(f"{p}.attn_v.weight", rand_f16(N_HEAD_KV * HEAD_DIM, N_EMBD))
        writer.add_tensor(f"{p}.attn_output.weight", rand_f16(N_EMBD, N_EMBD))
        writer.add_tensor(f"{p}.ffn_norm.weight", ones_f32(N_EMBD))
        writer.add_tensor(f"{p}.ffn_gate.weight", rand_f16(N_FF, N_EMBD))
        writer.add_tensor(f"{p}.ffn_up.weight", rand_f16(N_FF, N_EMBD))
        writer.add_tensor(f"{p}.ffn_down.weight", rand_f16(N_EMBD, N_FF))
        print(f"  blk.{i} tensors added", flush=True)
    writer.add_tensor("output_norm.weight", ones_f32(N_EMBD))
    writer.add_tensor("output.weight", rand_f16(N_VOCAB, N_EMBD))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
