#!/usr/bin/env bash
# Fetch Cerebras's "How we deploy trillion-parameter frontier models on Cerebras" and swap in the six-Mac
# trashcan numbers, for a private parody copy. The numbers and their sources are in notes.html.
#
#   docs/article/cerebras-parody.sh [OUT.html]     # writes OUT.html (default: ./mac-pro-parody.html)
#   docs/article/cerebras-parody.sh --check        # only reports how often each substitution matched
#   SRC=page.html docs/article/cerebras-parody.sh  # use a saved copy instead of fetching
#
# Text substitutions apply to text between tags only, so URLs and markup stay intact. Scripts are stripped so the
# page's own JavaScript can't re-render the original text. Each substitution that matches nothing is reported:
# if Cerebras edits the post, adjust that pair.
set -euo pipefail

URL=https://www.cerebras.ai/blog/how-we-deploy-trillion-parameter-frontier-models-on-cerebras
CHECK=0
OUT=./mac-pro-parody.html
case "${1:-}" in
  --check) CHECK=1 ;;
  "") ;;
  *) OUT=$1 ;;
esac

src=$(mktemp "${OUT%/*}/.parody-src.XXXXXX" 2>/dev/null || mktemp ./.parody-src.XXXXXX)
trap 'rm -f "$src"' EXIT
if [[ -n "${SRC:-}" ]]; then
  cp "$SRC" "$src"
else
  curl -fsSL -A "Mozilla/5.0" -o "$src" "$URL"
fi

CHECK=$CHECK perl -CSD -Mutf8 - "$src" > "$src.out" <<'PERL'
use strict; use warnings;

local $/; my $html = <>;

# Ordered pairs: specific phrases first, then the generic wafer -> Mac Pro words.
my @text = (
  # title, byline
  ['How we deploy Trillion-parameter frontier models on Cerebras', 'How we deploy 180B-parameter frontier models on 2013 Mac Pros', 'optional'],
  ['How we deploy trillion-parameter frontier models on Cerebras', 'How we deploy 180B-parameter frontier models on 2013 Mac Pros', 'optional'],
  ['Sarah Chieng', 'The trashcan project'],

  # intro
  ['how we deploy large models on Cerebras wafer-scale chips', 'how we deploy large models on six 13-year-old Mac Pros connected over Thunderbolt'],
  ['A single Cerebras WSE-3T wafer has 44 GB of on-chip memory, and trillion-parameter frontier models need terabytes for their weights alone.',
   'A single 2013 Mac Pro has 12 GB of GPU memory (6 GB of GDDR5 on each of its two AMD FirePro D700s), and 180B-parameter frontier models need over a hundred gigabytes for their weights alone.'],
  ['split a large frontier model across multiple WSE-3T wafers, to run multi-trillion parameter models at 1000-2000+ tokens per second',
   'split a large frontier model across multiple 2013 Mac Pros, to run 180B-parameter models at 20+ tokens per second'],
  ['the open source model Kimi K3, with ~3 trillion parameters', 'the open-weights model Qwen3.8-Flash-Next, with ~177 billion parameters'],
  ['How we fit the full model into on-chip memory', 'How we fit the full model into GPU memory'],
  ['How we map the model across multiple chips', 'How we map the model across multiple Mac Pros'],
  ['How we handle a high # of concurrent requests', 'How we handle a # of concurrent requests'],

  # Breaking it down
  ['is an open source, 3T parameter model with 69 linear-attention layers and 24 dense layers',
   'is an open-weights, 177B parameter model with 36 linear-attention (Gated DeltaNet) layers and 12 sparse-attention layers'],
  ["It scores an impressive 88.3% on Terminal-Bench 2.1 (vs. GPT-5.6 Sol's 88.8%) and 91.2% on BrowseComp (vs. GPT-5.6 Sol's 90.4%).",
   "It scores an impressive 95.1% on picking the same next token as its own 8-bit release (mean KL divergence 0.039 over 39K tokens)."],
  ['These results demonstrate frontier-level coding and agentic capabilities, even outperforming leading proprietary models on some benchmarks.',
   'These results demonstrate frontier-level 4-bit quantization, even outperforming the 3-bit and 2-bit versions on every benchmark we ran.'],
  ['A 256k assumed average request size', 'A 32k assumed average request size'],
  ['4-bit weight quantization (like the Hugging Face implementation)', '~4-bit expert quantization (8-bit for dense weights)'],
  ['An 8-bit KV cache', 'A 16-bit KV cache'],

  # #1
  ['total number of Cerebras WSE-3T wafers', 'total number of 2013 Mac Pros'],
  ['has 3 trillion parameters. If all 3T parameters are stored at 4 bits each, the raw parameter payload is:',
   "has 177 billion parameters. Stored at our mix of 4 and 8 bits they take 116 GiB, of which 51 GiB is an n-gram table that stays on the first Mac's SSD, so the raw parameter payload for the GPUs is:"],
  ['Dividing that payload by 93 layers then gives:', 'Dividing that payload by 48 layers then gives:'],

  # #2
  ['Number of Wafers', 'Number of Mac Pros'],
  ['each layer needs about 16 GB, we can figure out how many layers fit on each 44 GB wafer',
   'each layer needs about 1.4 GB, we can figure out how many layers fit on each 12 GB Mac Pro'],
  ['Dense layers need additional room for a KV cache, because each dense (full-attention) layer',
   'Sparse-attention layers need additional room for a KV cache, because each sparse-attention layer'],
  ['So, we place one dense layer per wafer and leave the remaining SRAM for its KV cache.',
   'So, we place one sparse-attention layer per GPU, two per Mac Pro, and leave the remaining VRAM for its KV cache.'],
  ['That leaves room for two linear layers per wafer (2 × 16 GB = 32 GB).',
   'That leaves room for three linear layers per GPU, six per Mac Pro (6 × 1.4 GB ≈ 8 GB).'],
  ["Kimi K3's 69 linear layers and 24 dense layers", "Qwen3.8-Flash-Next's 36 linear layers and 12 sparse-attention layers"],
  ['This gives us a total wafer count of 35 + 24 = 59 wafers.', 'This gives us a total Mac Pro count of max(6, 6) = 6 Mac Pros.'],

  # #3
  ['Cerebras executes models in a pipeline. After the Cerebras compiler maps the model layers across regions of wafers',
   'trashcan executes models in a pipeline. After the trashcan engine maps the model layers across the GPUs of the Mac Pros'],
  ['and this can be done via RDMA over Ethernet. This is roughly 2 microseconds per hop.',
   'and this can be done via USB4STREAM over Thunderbolt 2. This is roughly 25 microseconds per round trip.'],
  ['because each Cerebras wafer has 44 GB of on-chip SRAM and each layer is only 16 GB',
   'because each 2013 Mac Pro has 12 GB of GDDR5 and each layer is only 1.4 GB'],
  ['directly on the chip', 'directly on the GPU'],

  # #4
  ['Wafer 1 runs model layers 1–2 and sends its activations to wafer 2. Wafer 2 runs layers 3–4 and passes its results to Wafer 3',
   'incite runs model layers 1–8 and sends its activations to inflight. inflight runs layers 9–16 and passes its results to alight'],

  # #5
  ['No! If we only sent', 'Yes. If we only sent'],
  ['Instead, we keep the entire pipeline full with different requests. Because the model is split into pipeline stages already, each stage can work on a different request at the same time.',
   'That is what we do. Because the model is split into pipeline stages already, each stage could work on a different request at the same time.'],
  ['While stage 3 processes a token for request A, stage 2 processes one for request B, and stage 1 one for request C.',
   'While stage 3 processes a token for request A, stages 1 and 2 wait for request A.'],
  ['In our example, each stage is a layer. In production, Cerebras uses much finer-grained stages, and many more of them.',
   'In our example, each stage is a Mac Pro. In production, trashcan uses twelve stages, one per GPU.'],
  ['Each stage can also process a micro-batch', 'Each stage could also process a micro-batch'],
  ['So the number of requests in flight is roughly the number of stages times the micro-batch size.',
   'So the number of requests in flight would be roughly the number of stages times the micro-batch size. We have not built micro-batches.'],
  ['Loading player...', ''],

  # #6
  ['The amount of on-chip memory allocated to KV cache', 'The amount of GPU memory allocated to KV cache'],
  ['on each dense-layer wafer, so the size of on-chip memory sets', 'on each sparse-attention GPU, so the size of GPU memory sets'],
  ['KV cache requirements for dense layers', 'KV cache requirements for sparse-attention layers'],
  ['For each 44 GB wafer that already has one dense layer, we can calculate the remaining space for the KV cache with:',
   'For each 6 GB GPU that already has one sparse-attention layer, three DeltaNet layers and a twelfth of the output head, we can calculate the remaining space for the KV cache with:'],
  ['Total wafer storage - size of one dense layer. In this case:',
   "Driver memory budget - everything already allocated, including one request's cache. In this case:"],
  ['a single 256k request', 'a single 32k request'],
  ["Unlike a GPU batch, those requests don't wait for one another to finish, they stream through the pipeline continuously.",
   "Unlike a GPU batch, that request doesn't wait for any other to finish."],
  ['when measured at the same output generation speed.',
   'when measured at the same output generation speed. Ours is 15,150 tokens per second per megawatt (20.61 tokens per second at 1,360 W).'],

  # scaling and closing
  ['beyond 3T parameters', 'beyond 180B parameters'],
  ['Each additional wafer contributes compute and on-chip memory, allowing us to accommodate models that exceed the capacity of any single chip.',
   'Each additional Mac Pro contributes two GPUs and 12 GB of GDDR5, allowing us to accommodate models that exceed the capacity of any single Mac Pro.'],
  ['Interested in deploying a model on Cerebras or building with our inference cloud?',
   'Interested in deploying a model on Mac Pros or building with our inference trashcans?'],
  ['Get started with our free API: cloud.cerebras.ai', 'Get started with: trashcan serve'],
  ['Reach out to our sales team: sales@cerebras.ai', 'Reach out to: Tom'],
  ['cloud.cerebras.ai', 'trashcan serve', 'optional'],
  ['sales@cerebras.ai', 'Tom', 'optional'],

  # generic words, last
  ['Kimi K3', 'Qwen3.8-Flash-Next', 'optional'],
  ['WSE-3T wafers', '2013 Mac Pros', 'optional'],
  ['WSE-3T wafer', '2013 Mac Pro', 'optional'],
  ['wafers', 'Mac Pros', 'optional'],
  ['Wafers', 'Mac Pros', 'optional'],
  ['wafer', 'Mac Pro', 'optional'],
  ['Wafer', 'Mac Pro', 'optional'],
  ['on-chip SRAM', 'GDDR5', 'optional'],
  ['on-chip memory', 'GPU memory', 'optional'],
);

# Figures: replace each original image (by its CDN id) with our own HTML.
sub calc { my $l = join '', map { "<div>$_</div>" } @_;
  qq{<div style="background:#111;color:#f0efec;border-radius:16px;padding:28px 32px;margin:24px 0;font:18px/1.7 ui-monospace,Menlo,monospace;overflow-x:auto;white-space:pre">$l</div>} }
sub n { qq{<span style="color:#f2793a">$_[0]</span>} }
my $chips = sub { join '', map {
  my ($t, $c) = @$_; qq{<span style="display:inline-block;width:88px;margin:3px;padding:8px 0;text-align:center;border-radius:8px;color:#fff;font:600 12px ui-monospace,Menlo,monospace;background:$c">$t</span>} } @_ };
my @grp = (['DELTANET', '#0e6b6b'], ['DELTANET', '#0e6b6b'], ['DELTANET', '#0e6b6b'], ['SPARSE ATTN', '#6b8cff']);
my $box = 'background:#f2f2f0;border-radius:16px;padding:28px 32px;margin:24px 0;font:16px/1.5 system-ui,sans-serif;color:#111';
my $macs = join ' → ', map { "<b>$_->[0]</b> (layers $_->[1])" }
  (['incite', '1–8'], ['inflight', '9–16'], ['alight', '17–24'], ['indite', '25–32'], ['invite', '33–40'], ['enlight', '41–48']);

my %fig = (
  'c124705d630730f1fc2f68f350bbbc026f574c07' => qq{<div style="$box"><div style="font-size:40px;font-weight:700">48 Layers</div>
    <div style="font-size:22px;margin-bottom:12px"><span style="color:#0e6b6b">36 Linear</span> + <span style="color:#6b8cff">12 Sparse attention</span></div>
    <div>Group 1: } . $chips->(@grp) . qq{</div><div>Group 2: } . $chips->(@grp) . qq{</div><div style="color:#777">… 10 more groups</div></div>},
  '1de256fc77c92de1fe3e2297026f583dde63176c' => qq{<div style="$box"><div style="font-weight:700;margin-bottom:8px">Same next token as the 8-bit release (39K tokens)</div>
    <table style="border-collapse:collapse;width:100%"><tr><td>Six-Mac recipe (IQ4 experts)</td><td style="text-align:right"><b>95.1%</b></td></tr>
    <tr><td>Q4_0 experts</td><td style="text-align:right">94.9%</td></tr><tr><td>MXFP4 experts</td><td style="text-align:right">93.7%</td></tr>
    <tr><td>IQ3_XXS experts</td><td style="text-align:right">92.6%</td></tr><tr><td>IQ2_XXS experts</td><td style="text-align:right">88.6%</td></tr></table></div>},
  '61b6d6a611991bbb7069b23faa5a0c73b8a9fe86' => calc(n(116) . ' GiB − ' . n(51) . ' GiB n-gram table = ' . n(65) . ' GiB'),
  '5622b906b303fbbede7600901e2c4d00dd37f691' => calc(n(65) . ' GiB ÷ ' . n(48) . ' layers ≈ ' . n('1.4') . ' GiB per layer'),
  'e0d95625e101f6936906687df4d1d0817e5d3af7' => calc(n(36) . ' linear layers ÷ ' . n(6) . ' per Mac Pro = ' . n(6),
                                                     n(12) . ' sparse-attention layers ÷ ' . n(2) . ' per Mac Pro = ' . n(6)),
  '263c843c6a5e1e7c44c69dfc344af39233aacd4f' => qq{<div style="$box"><div style="font-size:32px;font-weight:700;margin-bottom:12px">Large models scale to multiple Mac Pros</div>$macs → back to incite, over Thunderbolt 2</div>},
  '81b3ca230cf8f9e2383929aed2bf49267dcdc022' => qq{<div style="$box">token → $macs → next token</div>},
  '4ca12e7b7b3f0cfb782bdd5d05a60bc50b6085e5' => calc(n('5,869') . ' MiB budget − ' . n('5,830') . ' MiB allocated = ' . n(39) . ' MiB'),
  'cddda82a24db97053fa39e071d29e836a59502ad' => calc(n('32,768') . ' positions × ' . n('2,048') . ' bytes = ' . n(64) . ' MiB'),
  'f9afc17d3c841b866108579fd474420cb6e79261' => calc(n(39) . ' MiB ÷ ' . n(64) . ' MiB = ' . n('0.6') . ' → ' . n(1) . ' concurrent request'),
);

sub pat {   # literal text, tolerant of curly/straight apostrophes, entities, non-breaking spaces and,
            # with $_[1], inline tags (links, bold) inside the phrase, which the replacement drops
  my $gap = $_[1] ? '(?:<[^<>]+>)*' : '';
  my $p = join $gap, map {
    $_ eq "'" || $_ eq '’' ? q{(?:'|’|&#x27;|&#39;|&rsquo;)} :
    $_ eq ' ' ? q{(?:\s|&nbsp;|\x{a0})+} :
    $_ eq '&' ? q{(?:&|&amp;)} : quotemeta } split //, $_[0];
  qr/$p/;
}

my %count;
$html =~ s{<script\b.*?</script>}{}gis;                        # no hydration back to the original text
$html =~ s{<img\b[^>]*image\.mux\.com[^>]*>}{}gi;               # the two pipeline animations
for my $id (keys %fig) {
  $count{"figure $id"} = ($html =~ s{<img\b[^>]*\Q$id\E[^>]*>}{$fig{$id}}g) || 0;
}
my %optional;
for my $pair (@text) {
  my ($from, $to, $opt) = @$pair;
  my $n = 0;
  if (length $from > 25) {             # whole sentences: may span inline tags, never occur inside a tag
    my $re = pat($from, 1);
    $n = ($html =~ s/$re/$to/g) || 0;
  } else {                             # short words: text between tags only, so URLs stay intact
    my $re = pat($from, 0);
    my @parts = split /(<[^>]*>)/, $html;
    for (@parts) { next if /^</; $n += s/$re/$to/g; }
    $html = join '', @parts;
  }
  $count{$from} = $n;
  $optional{$from} = 1 if $opt;
}
$html =~ s{<head>}{<head><base href="https://www.cerebras.ai/">}i;
$html =~ s{(<body\b[^>]*>)}{$1<div style="position:fixed;left:0;right:0;bottom:0;z-index:2147483647;font:13px system-ui;padding:6px 12px;background:#ffd;color:#333">Parody, not affiliated with Cerebras. Original: <a href="https://www.cerebras.ai/blog/how-we-deploy-trillion-parameter-frontier-models-on-cerebras">How we deploy trillion-parameter frontier models on Cerebras</a></div>}i;

my @missing = grep { !$count{$_} && !$optional{$_} } sort keys %count;
printf STDERR "%d of %d required substitutions matched\n", (grep { !$optional{$_} } keys %count) - @missing, scalar grep { !$optional{$_} } keys %count;
print STDERR "no match: $_\n" for @missing;
print $html unless $ENV{CHECK};
PERL

if (( CHECK )); then
  rm -f "$src.out"
else
  mv "$src.out" "$OUT"
  echo "wrote $OUT" >&2
fi
