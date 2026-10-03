#!/usr/bin/env python3
"""Account for the current two-D700 target graph, without running GPU work.

Counts are dense-equivalent multiply/add FLOPs (one FMA = two FLOPs), not
instruction counts. Weight bytes are payloads, not measured DRAM traffic.
Optionally validate every tensor against the GGUF header, reading no weights.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess


D, F, VOCAB, WINDOW = 3840, 15360, 262144, 1024
LAYERS, SLIDING, GLOBAL = 48, 40, 8
ROOT = Path(__file__).resolve().parents[1]


def shapes():
    """(name, count, global K replicated?, K, full output rows)."""
    result = []
    for kind, count, dim, kv in [('sliding', SLIDING, 256, 8), ('global', GLOBAL, 512, 1)]:
        for name, k, n in [('attn_q', D, 16 * dim), ('attn_k', D, kv * dim),
                           ('attn_output', 16 * dim, D), ('ffn_gate', D, F),
                           ('ffn_up', D, F), ('ffn_down', F, D)]:
            result.append((f'{kind}.{name}', count, kind == 'global' and name == 'attn_k', k, n))
        if kind == 'sliding':
            result.append((f'{kind}.attn_v', count, False, D, kv * dim))
    return result


def expected_tensors():
    result = {'token_embd.weight': ([D, VOCAB], 14),
              'output_norm.weight': ([D], 0), 'rope_freqs.weight': ([256], 0)}
    for layer in range(LAYERS):
        kind = 'global' if layer % 6 == 5 else 'sliding'
        dim = 512 if kind == 'global' else 256
        for name, _, _, k, n in shapes():
            if name.startswith(kind + '.'):
                result[f'blk.{layer}.{name.split(".")[1]}.weight'] = ([k, n], 2)
        for name in ['attn_norm', 'post_attention_norm', 'ffn_norm', 'post_ffw_norm']:
            result[f'blk.{layer}.{name}.weight'] = ([D], 0)
        for name in ['attn_q_norm', 'attn_k_norm']:
            result[f'blk.{layer}.{name}.weight'] = ([dim], 0)
        result[f'blk.{layer}.layer_output_scale.weight'] = ([1], 0)
    return result


def validate_gguf(path):
    """Minimal GGUF v3 directory reader; seek over tokenizer data and weights."""
    formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
               7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    with path.open('rb') as f:
        def unpack(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]

        def string(keep=True):
            length = unpack('Q')
            if keep:
                return f.read(length).decode('utf-8')
            f.seek(length, 1)

        def value(ty, keep=False):
            if ty in formats:
                v = unpack(formats[ty])
                return v if keep else None
            if ty == 8:
                return string(keep)
            if ty == 9:
                element, count = unpack('I'), unpack('Q')
                if not keep and element in formats:
                    f.seek(count * struct.calcsize('<' + formats[element]), 1)
                    return None
                values = []
                for _ in range(count):
                    v = value(element, keep)
                    if keep:
                        values.append(v)
                return values if keep else None
            raise ValueError(f'unknown GGUF metadata type {ty}')

        if f.read(4) != b'GGUF' or unpack('I') != 3:
            raise ValueError('expected GGUF v3')
        tensor_count, metadata_count = unpack('Q'), unpack('Q')
        meta = {}
        for _ in range(metadata_count):
            name, ty = string(), unpack('I')
            keep = name.startswith('gemma4.') or name == 'general.architecture'
            v = value(ty, keep)
            if keep:
                meta[name] = v
        found = {}
        for _ in range(tensor_count):
            name, nd = string(), unpack('I')
            dims = [unpack('Q') for _ in range(nd)]
            ty = unpack('I')
            unpack('Q')  # Offset, relative to aligned tensor payload.
            found[name] = (dims, ty)
        expected = expected_tensors()
        errors = [name for name in sorted(set(expected) | set(found))
                  if expected.get(name) != found.get(name)]
        if errors:
            raise ValueError(f'GGUF tensors differ from this cost model: {errors}')
        pattern = meta['gemma4.attention.sliding_window_pattern']
        if [bool(v) for v in pattern] != [i % 6 != 5 for i in range(LAYERS)]:
            raise ValueError('unexpected sliding/global layer pattern')
        if meta['general.architecture'] != 'gemma4':
            raise ValueError('unexpected architecture')
        end = f.tell()
        f.seek(0)
        digest = hashlib.sha256(f.read(end)).hexdigest()
    return {'path': str(path), 'file_bytes': path.stat().st_size,
            'validated_tensor_count': tensor_count, 'header_bytes': end,
            'header_sha256': digest, 'weights_read': False}


def profile_summary(path):
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    result = []
    for card in range(2):
        work = sum(float(r[f'gpu{card}_ms']) for r in rows)
        gaps = sum(float(r[f'gap{card}_ms']) for r in rows)
        result.append({'gpu': card, 'segment_work_ms': work,
                       'gap_ms': gaps, 'timestamp_span_ms': work + gaps,
                       'last_segment_ms': float(rows[-1][f'gpu{card}_ms'])})
    return {'path': str(path), 'scope': 'one final token, not an average', 'cards': result}


def source_provenance(ref):
    """Keep committed engine sources separate from the running audit script."""
    names = ['engine/src/model.rs', 'engine/src/quant.rs']
    if ref:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--verify', '--end-of-options', ref + '^{commit}'],
            cwd=ROOT, text=True).strip()
        sources = {name: subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
                   for name in names}
        provenance = {'kind': 'git_commit', 'requested_ref': ref, 'resolved_commit': commit}
    else:
        sources = {name: (ROOT / name).read_bytes() for name in names if (ROOT / name).exists()}
        provenance = {'kind': 'working_tree_snapshot' if sources else 'engine_sources_unavailable'}
    return provenance, {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, help='optional CPU-only header validation')
    parser.add_argument('--batch', type=int, default=512, help='ordinary prefill chunk length')
    parser.add_argument('--prefix', type=int, default=0, help='tokens preceding prefill chunk')
    parser.add_argument('--core-mhz', type=float, default=850)
    parser.add_argument('--memory-mhz', type=float, default=1370)
    parser.add_argument('--hop-us', type=float, default=42, help='illustrative serialized hop assumption')
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--source-ref', help='hash engine sources from this Git commit, not the working tree')
    args = parser.parse_args()
    if not 1 <= args.batch <= 512 or args.prefix < 0:
        parser.error('batch must be 1..512 and prefix nonnegative')
    if min(args.core_mhz, args.memory_mhz) <= 0 or args.hop_us < 0:
        parser.error('clocks must be positive and hop latency nonnegative')
    try:
        provenance, source_hashes = source_provenance(args.source_ref)
    except (OSError, subprocess.CalledProcessError) as error:
        parser.error(f'cannot resolve source provenance: {error}')

    matrices = []
    for name, count, replicated, k, n in shapes():
        params = k * n if replicated else k * n // 2
        matrices.append({'matrix': name, 'layer_count': count, 'full_shape_K_N': [k, n],
                         'replicated': replicated, 'parameters_per_gpu_per_layer': params,
                         'original_q4_bytes_per_gpu_all_layers': count * params * 18 // 32,
                         'repacked_bytes_per_gpu_all_layers': count * params * 20 // 32,
                         'dense_flops_per_gpu_per_token_all_layers': 2 * count * params})
    q4_params = sum(m['parameters_per_gpu_per_layer'] * m['layer_count'] for m in matrices)
    q4_bytes = q4_params * 20 // 32
    head_params, head_bytes = D * VOCAB // 2, D * VOCAB // 2 * 5 // 4
    weight_bytes = q4_bytes + head_bytes
    bandwidth = args.memory_mhz * 1e6 * 4 * (384 // 8)
    compute = args.core_mhz * 1e6 * 2048 * 2
    hops_s = 96 * args.hop_us / 1e6
    prefill_layer_flops = 2 * q4_params * args.batch
    head_flops = 2 * head_params
    sliding_pairs = sum(min(args.prefix + t + 1, WINDOW) for t in range(args.batch))
    global_pairs = sum(args.prefix + t + 1 for t in range(args.batch))
    attention_flops = 4 * (SLIDING * 8 * 256 * sliding_pairs + GLOBAL * 8 * 512 * global_pairs)
    prefill_flops = prefill_layer_flops + head_flops + attention_flops
    contexts = []
    for context in [512, 1024, 8192, 32768]:
        sliding_payload = SLIDING * min(context, WINDOW) * 4 * 256 * 2 * 2
        global_payload = GLOBAL * context * 1 * 512 * 2 * 2
        attn_flops = 4 * (SLIDING * 8 * 256 * min(context, WINDOW) + GLOBAL * 8 * 512 * context)
        bytes_once = weight_bytes + sliding_payload + global_payload
        contexts.append({'visible_tokens': context,
                         'unique_KV_f16_bytes_per_gpu': sliding_payload + global_payload,
                         'logical_KV_f16_bytes_requested_by_query_heads_per_gpu':
                             2 * sliding_payload + 8 * global_payload,
                         'attention_dot_product_flops_per_gpu': attn_flops,
                         'ideal_unique_payload_bandwidth_tok_s': bandwidth / bytes_once,
                         'illustrative_payload_plus_hops_tok_s': 1 / (bytes_once / bandwidth + hops_s)})
    tensors = expected_tensors()
    type_bytes = {0: (1, 4), 2: (32, 18), 14: (256, 210)}
    gguf_tensor_bytes = sum(math.prod(dims) // type_bytes[ty][0] * type_bytes[ty][1]
                            for dims, ty in tensors.values())
    original_matrix_bytes = (2 * q4_params - GLOBAL * D * 512) * 18 // 32 + D * VOCAB * 210 // 256
    result = {
        'units': {'GB': '10^9 bytes', 'GiB': '2^30 bytes', 'FMA': 'two dense-equivalent FLOPs'},
        'assumptions': ['one token/sequence, two GPUs, global K replicated',
                        'Q4 repack: 20 bytes/32 weights; Q6 head: 5 bytes/4 weights',
                        'ordinary prefill computes one LM-head column per chunk; verification computes every column',
                        'attention counts QK and PV only, excluding online softmax rescaling, exp and reductions',
                        'hardware rooflines exclude dequantization instructions, norms, activation traffic, cache misses, driver work and synchronization',
                        'hop scenario is illustrative; transfer and semaphore latency can overlap GPU work',
                        'KV byte counts are logical f16 payloads; reuse/transactions require profiling'],
        'hardware_per_gpu': {'core_mhz': args.core_mhz, 'memory_mhz': args.memory_mhz,
                             'bandwidth_B_s': bandwidth, 'fp32_FLOP_s': compute},
        'model': {'tensor_count': len(tensors), 'parameter_count': sum(math.prod(x[0]) for x in tensors.values()),
                  'gguf_tensor_payload_bytes': gguf_tensor_bytes, 'original_matrix_bytes': original_matrix_bytes,
                  'q4_parameters_per_gpu': q4_params, 'q4_repacked_bytes_per_gpu': q4_bytes,
                  'head_parameters_per_gpu': head_params, 'head_repacked_bytes_per_gpu': head_bytes,
                  'decode_matrix_bytes_per_gpu': weight_bytes,
                  'decode_matrix_flops_per_gpu': 2 * (q4_params + head_params),
                  'four_column_verification_matrix_flops_per_gpu': 8 * (q4_params + head_params),
                  'matrices': matrices},
        'decode_rooflines': {'weight_only_ms': weight_bytes / bandwidth * 1000,
                             'weight_only_tok_s': bandwidth / weight_bytes,
                             'hop_us_assumption': args.hop_us, '96_hops_ms': hops_s * 1000,
                             'weight_plus_hops_scenario_tok_s': 1 / (weight_bytes / bandwidth + hops_s),
                             'contexts': contexts},
        'prefill_per_gpu': {'batch': args.batch, 'prefix_tokens': args.prefix,
                            'layer_matrix_flops': prefill_layer_flops, 'one_head_flops': head_flops,
                            'attention_dot_product_flops': attention_flops, 'counted_flops': prefill_flops,
                            'counted_flops_per_token': prefill_flops / args.batch,
                            'ideal_counted_compute_tok_s': args.batch * compute / prefill_flops,
                            'at_2p2_TFLOP_s_counted_work_tok_s': args.batch * 2.2e12 / prefill_flops,
                            'P2P_bytes_sent_per_gpu': 96 * D * 4 * args.batch},
        'source_provenance': provenance,
        'source_sha256': source_hashes,
        'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.gguf:
        result['gguf_header_validation'] = validate_gguf(args.gguf)
    if args.profile:
        result['recorded_profile'] = profile_summary(args.profile)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
