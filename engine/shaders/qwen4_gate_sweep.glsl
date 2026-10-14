// Body of qwen4_gate_sweep (volatile polls) and qwen4_gate_sweep_nt (nontemporal polls), which define GATE_ACCESS.
// A bounded gate that evicts its line from the GPU's L2 between polls: one wave polls word 0 of `g` (host memory)
// until the host sets it, and before every poll after the first it reads p.m lines of `s` (device-local, p.stride
// dwords apart), so that the next poll misses the L2 and reads host memory again. At most p.n polls and
// p.k << 16 shader-clock ticks. p.m must be a multiple of 512 (0: no sweep).
// Results in the gate's second line: word 16 how (1 opened, 2 out of polls, 3 out of time), 17 polls,
// 18-19 ticks spent, 20-21 ticks in poll loads, 22 fewest ticks of one poll load, 23 most, 24-25 ticks in sweeps.
layout(local_size_x=64) in;
layout(set=2,binding=0,std430) coherent volatile GATE_ACCESS buffer Gate { uint v[]; } g;
layout(set=2,binding=1,std430) volatile readonly buffer Sweep { uint w[]; } s;
layout(push_constant, std430) uniform Push {
    uint64_t a,b,c,d,e,f,params,extra;
    uint n,k,m,stride,heads,kvheads,dim,cap,window,kind,offset,flags,r0,r1,r2,r3;
} p;

void main() {
    uint lane = gl_LocalInvocationIndex, step = p.stride;
    uint64_t t0 = clockARB(), limit = uint64_t(p.k) << 16, load_ticks = 0, sweep_ticks = 0;
    uint fewest = 0xffffffffu, most = 0u, i = 0u, how = 1u, acc = 0u;
    for (;;) {
        uint64_t a = clockARB();
        uint x = g.v[0];
        memoryBarrierBuffer();
        uint64_t b = clockARB();
        uint dt = uint(b - a);
        load_ticks += dt;
        fewest = min(fewest, dt);
        most = max(most, dt);
        if (x != 0u) {
            break;
        }
        if (++i >= p.n) {
            how = 2u;
            break;
        }
        if (b - t0 >= limit) {
            how = 3u;
            break;
        }
        for (uint j = lane; j < p.m; j += 512u) {
            uint o = j * step;
            acc += s.w[o] + s.w[o + 64u * step] + s.w[o + 128u * step] + s.w[o + 192u * step]
                + s.w[o + 256u * step] + s.w[o + 320u * step] + s.w[o + 384u * step] + s.w[o + 448u * step];
        }
        memoryBarrierBuffer();
        sweep_ticks += clockARB() - b;
    }
    uint64_t spent = clockARB() - t0;
    // Never true for the zeroed sweep buffer; keeps the sweep's loads.
    if (acc == 0x9e3779b9u) {
        g.v[31] = acc;
    }
    if (lane == 0u) {
        g.v[16] = how;
        g.v[17] = i;
        g.v[18] = uint(spent);
        g.v[19] = uint(spent >> 32);
        g.v[20] = uint(load_ticks);
        g.v[21] = uint(load_ticks >> 32);
        g.v[22] = fewest;
        g.v[23] = most;
        g.v[24] = uint(sweep_ticks);
        g.v[25] = uint(sweep_ticks >> 32);
    }
}
