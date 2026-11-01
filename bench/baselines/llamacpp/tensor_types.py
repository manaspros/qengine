"""Per-type byte summary of a GGUF file (what gguf-dump lists per tensor, aggregated).
Bytes read per decode token = all tensors except the token embedding (gathered, one row) and the
MTP/nextn tensors (only used when MTP drafting is on).
Usage: python3 tensor_types.py <file.gguf>   (needs llama.cpp's gguf-py on PYTHONPATH)"""
import collections, sys
import gguf

r = gguf.GGUFReader(sys.argv[1])
# Embedded MTP layers are the last nextn_predict_layers blocks (qwen35: blk.64); decode skips them.
f = {k: v for k, v in r.fields.items()}
arch = bytes(f["general.architecture"].parts[-1]).decode()
nblk = int(f[f"{arch}.block_count"].parts[-1][0]) if f"{arch}.block_count" in f else 0
nmtp = int(f[f"{arch}.nextn_predict_layers"].parts[-1][0]) if f"{arch}.nextn_predict_layers" in f else 0
mtp_blocks = {f"blk.{i}." for i in range(nblk - nmtp, nblk)} if nmtp else set()
mtp_bytes = 0
by_type, decode_bytes, total = collections.Counter(), 0, 0
kinds = collections.defaultdict(collections.Counter)  # tensor role (name suffix) -> type -> count
for t in r.tensors:
    ty = t.tensor_type.name
    n = int(t.n_bytes)
    by_type[ty] += n; total += n
    role = t.name.split(".", 2)[-1] if t.name.startswith("blk.") else t.name
    kinds[role][ty] += 1
    is_mtp = "nextn" in t.name or "mtp" in t.name or any(t.name.startswith(b) for b in mtp_blocks)
    mtp_bytes += n if is_mtp else 0
    if t.name != "token_embd.weight" and not is_mtp:
        decode_bytes += n
print(f"{len(r.tensors)} tensors, {total/1e9:.2f} GB; streamed per decode token {decode_bytes/1e9:.2f} GB "
      f"(excludes token_embd and the embedded MTP block(s) {sorted(mtp_blocks)}: {mtp_bytes/1e9:.3f} GB)")
for ty, n in by_type.most_common():
    print(f"  {ty:>8} {n/1e9:6.2f} GB")
print("tensor role -> types (count):")
for role in sorted(kinds):
    print(f"  {role:<40} " + ", ".join(f"{k}x{v}" for k, v in kinds[role].most_common()))
