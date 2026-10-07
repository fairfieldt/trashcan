"""Summarize llama-perplexity KL-divergence logs (one per candidate) into a TSV with the recipe and size.
usage: kld_table.py OUT.tsv NAME... (reads kld/NAME.log and quant-NAME.log)"""
import re
import sys

Q = '/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash'
RECIPES = {
    'E-q4_0': 'routed experts Q4_0',
    'E-mxfp4': 'routed experts MXFP4',
    'E-iq4xs': 'routed experts IQ4_XS gate/up, IQ4_NL down',
    'E-iq2xxs': 'routed experts IQ2_XXS gate/up, IQ4_NL down',
    'E-iq3xxs': 'routed experts IQ3_XXS gate/up, IQ4_NL down',
    'E-iq4mx': 'routed experts IQ4_XS gate/up, MXFP4 down',
    'E-iq2mx': 'routed experts IQ2_XXS gate/up, MXFP4 down',
    'N-q4_0': 'all dense matrices Q4_0 (mixers, DeltaNet, attention, shared expert, n-gram projections)',
    'T-q4_0': 'n-gram table Q4_0',
    'A-hc': 'hyper-connection mixers Q4_0',
    'A-gdn': 'DeltaNet projections Q4_0',
    'A-attn': 'QSA projections Q4_0',
    'A-shexp': 'shared experts Q4_0',
    'G-iq4': 'routed and shared experts IQ4_XS gate/up, IQ4_NL down; head Q6_K (six-Mac recipe)',
    'S-q2dn-r': 'G-iq4 with ISTA GSQ Q2_0 routed down projections on layers 1 5 8 12 16 20 24 30 33 36 40 45',
    'S-q2dn-gsq': 'S-q2dn-r with the shared down projections of those layers at ggml Q2_0 (absmax) too',
    'S-q2dn-absmax': 'S-q2dn-gsq with ggml Q2_0 (absmax) routed down projections instead of ISTA GSQ',
}
rows = []
for name in sys.argv[2:]:
    log = open(f'{Q}/kld/{name}.log', errors='replace').read()

    def get(label):
        m = re.search(re.escape(label) + r'\s*:\s*([-0-9.]+)(?:\s*±\s*([0-9.]+))?', log)
        return (m.group(1), m.group(2) or '') if m else ('', '')

    try:
        q = open(f'{Q}/quant-{name.replace("-gsq", "").replace("-absmax", "")}.log', errors='replace').read()
        size = re.search(r'quant size\s*=\s*([0-9.]+) MiB \(([0-9.]+) BPW\)', q)
    except FileNotFoundError:
        size = None
    ppl, _ = get('Mean PPL(Q)')
    ratio, ratio_e = get('Mean PPL(Q)/PPL(base)')
    kld, kld_e = get('Mean    KLD')
    k99, _ = get('99.0%   KLD')
    same, same_e = get('Same top p')
    rms, _ = get('RMS Δp')
    rows.append([name, RECIPES.get(name, ''), size.group(1) if size else '', size.group(2) if size else '', ppl,
                 ratio, ratio_e, kld, kld_e, k99, same, same_e, rms])
hdr = ['name', 'changed_from_q8_0', 'gguf_mib_incl_q8_ngram_table', 'gguf_bpw', 'ppl', 'ppl_ratio', 'ppl_ratio_err',
       'kld_mean', 'kld_err', 'kld_p99', 'same_top_pct', 'same_top_err', 'rms_dp_pct']
with open(sys.argv[1], 'w') as f:
    f.write('\t'.join(hdr) + '\n')
    for r in rows:
        f.write('\t'.join(r) + '\n')
print(open(sys.argv[1]).read())
