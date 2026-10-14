#!/usr/bin/env python3
"""Temporary fine-grained timers in the restructured draft.rs (profiling on incite; reverted after)."""
import sys
p = sys.argv[1] + '/engine/src/qwen4/draft.rs'
s = open(p).read()


def rep(a, b, count=1):
    global s
    assert s.count(a) == count, (a, s.count(a))
    s = s.replace(a, b)


rep("/// A job for the pool", """pub mod tprof {
    use std::sync::Mutex;
    use std::time::Instant;
    pub static P: Mutex<Option<(Instant, Vec<(&'static str, f64, usize)>)>> = Mutex::new(None);
    pub fn start() {
        let mut g = P.lock().unwrap();
        match g.as_mut() {
            Some((t, _)) => *t = Instant::now(),
            None => *g = Some((Instant::now(), Vec::new())),
        }
    }
    pub fn tick(l: &'static str) {
        let mut g = P.lock().unwrap();
        if let Some((t, v)) = g.as_mut() {
            let now = Instant::now();
            let dt = (now - *t).as_secs_f64() * 1e3;
            *t = now;
            match v.iter_mut().find(|e| e.0 == l) {
                Some(e) => { e.1 += dt; e.2 += 1; }
                None => v.push((l, dt, 1)),
            }
        }
    }
    pub fn take() -> Vec<(&'static str, f64, usize)> {
        P.lock().unwrap().take().map(|x| x.1).unwrap_or_default()
    }
}

/// A job for the pool""")
# mixer
rep("""            rms_norm(row, Some(&m.norm[(i % hc) * d..(i % hc + 1) * d]), c.eps);
        }
        let xa = Act::new(&xn, w);""", """            rms_norm(row, Some(&m.norm[(i % hc) * d..(i % hc + 1) * d]), c.eps);
        }
        tprof::tick("mix.norm");
        let xa = Act::new(&xn, w);
        tprof::tick("mix.act");""")
rep("""        let outs = matvecs(&self.pool, &first, &xa, 16);
        let lo: Vec<f32> = outs[0].iter().map(|v| silu(v / hc as f32)).collect();
        let la = Act::new(&lo, c.hc_rank);""", """        let outs = matvecs(&self.pool, &first, &xa, 16);
        tprof::tick("mix.down");
        let lo: Vec<f32> = outs[0].iter().map(|v| silu(v / hc as f32)).collect();
        let la = Act::new(&lo, c.hc_rank);
        tprof::tick("mix.lo");""")
rep("""        let write = outs.get(1).map(|inj| inj.iter().map(|v| 2. * sigmoid(v / hc as f32)).collect());
        (u, write)""", """        tprof::tick("mix.up");
        let write = outs.get(1).map(|inj| inj.iter().map(|v| 2. * sigmoid(v / hc as f32)).collect());
        (u, write)""")
# attention
rep("""        let ua = Act::new(u, d);
        // The indexer queries""", """        tprof::tick("at.pre");
        let ua = Act::new(u, d);
        tprof::tick("at.act");
        // The indexer queries""")
rep("""        let outs = matvecs(&self.pool, &mats, &ua, 64);
        self.prof[6] += t0.elapsed().as_secs_f64() * 1e3;""", """        let outs = matvecs(&self.pool, &mats, &ua, 64);
        self.prof[6] += t0.elapsed().as_secs_f64() * 1e3;
        tprof::tick("at.proj");""")
rep("""        let visible: Vec<Vec<usize>> =""", """        tprof::tick("at.kv");
        let visible: Vec<Vec<usize>> =""")
rep("""        let scale = 1. / (hd as f32).sqrt();
        let mut o = vec![0f32; n * nh * hd];""", """        tprof::tick("at.sel");
        let scale = 1. / (hd as f32).sqrt();
        let mut o = vec![0f32; n * nh * hd];""")
rep("""        self.prof[7] += t0.elapsed().as_secs_f64() * 1e3;
        matvec(&self.pool, &self.o, &Act::new(&o, nh * hd), 32)""", """        self.prof[7] += t0.elapsed().as_secs_f64() * 1e3;
        tprof::tick("at.core");
        let oa = Act::new(&o, nh * hd);
        tprof::tick("at.oact");
        let r = matvec(&self.pool, &self.o, &oa, 32);
        tprof::tick("at.o");
        r""")
# moe
rep("""        let ua = Act::new(u, d);
        let logits = matvec(&self.pool, &e.router, &ua, 32);""", """        tprof::tick("moe.pre");
        let ua = Act::new(u, d);
        tprof::tick("moe.act");
        let logits = matvec(&self.pool, &e.router, &ua, 32);
        tprof::tick("moe.router");""")
rep("""        let slots = k + 1;
        // Gate and up rows""", """        tprof::tick("moe.top");
        let slots = k + 1;
        // Gate and up rows""")
rep("""        let mut down = vec![0f32; n * slots * d];""", """        tprof::tick("moe.gu");
        let mut down = vec![0f32; n * slots * d];""")
rep("""        let mut y = vec![0f32; n * d];
        for t in 0..n {
            for s in 0..slots {""", """        tprof::tick("moe.down");
        let mut y = vec![0f32; n * d];
        for t in 0..n {
            for s in 0..slots {""")
rep("""                    *o += wgt * v;
                }
            }
        }
        y""", """                    *o += wgt * v;
                }
            }
        }
        tprof::tick("moe.sum");
        y""")
# forward
rep("""        let et = self.emb.tensor("token_embd.weight").context("missing token_embd")?;
        // x_s""", """        tprof::start();
        let et = self.emb.tensor("token_embd.weight").context("missing token_embd")?;
        // x_s""")
rep("""        let (ea, ha) = (Act::new(&en, d), Act::new(&hn, d));
        let outs = matvecs_of(&self.pool, &[(&self.eh_e, &ea), (&self.eh_h, &ha)], 32);""", """        tprof::tick("in.embed");
        let (ea, ha) = (Act::new(&en, d), Act::new(&hn, d));
        tprof::tick("in.act");
        let outs = matvecs_of(&self.pool, &[(&self.eh_e, &ea), (&self.eh_h, &ha)], 32);
        tprof::tick("in.mv");""")
rep("""        lap(&mut self.prof, 0);
        let (u, write) = self.mix(&self.attn_mix, &x, n);""", """        tprof::tick("in.add");
        lap(&mut self.prof, 0);
        let (u, write) = self.mix(&self.attn_mix, &x, n);""")
rep("""        let y = self.attention(&u, pos, n);
        CpuStage::combine(&c, &mut x, &y, write.as_deref().unwrap(), n);""", """        let y = self.attention(&u, pos, n);
        tprof::tick("at.ret");
        CpuStage::combine(&c, &mut x, &y, write.as_deref().unwrap(), n);
        tprof::tick("combine");""")
rep("""        let y = self.moe(&u, n);
        CpuStage::combine(&c, &mut x, &y, write.as_deref().unwrap(), n);""", """        let y = self.moe(&u, n);
        tprof::tick("moe.ret");
        CpuStage::combine(&c, &mut x, &y, write.as_deref().unwrap(), n);
        tprof::tick("combine");""")
rep("""        let best = if head {
            let (hn, _) = self.mix(&self.head_mix, &x[(n - 1) * w..], 1);
            self.head_best(&hn)""", """        let best = if head {
            tprof::tick("pre.head");
            let (hn, _) = self.mix(&self.head_mix, &x[(n - 1) * w..], 1);
            tprof::tick("head.mix");
            let b = self.head_best(&hn);
            tprof::tick("head.best");
            b""")
open(p, 'w').write(s)

p2 = sys.argv[1] + '/engine/src/qwen4/mtp.rs'
s = open(p2).read()
a = """                    if s == 1 {
                        fast.prof = [0.; 8];
                    }"""
assert s.count(a) == 1
s = s.replace(a, """                    if s == 1 {
                        fast.prof = [0.; 8];
                        super::draft::tprof::take();
                    }""")
a = """            let (step_bytes, all_bytes) = fast.bytes();"""
assert s.count(a) == 1
s = s.replace(a, """            let table: Vec<serde_json::Value> = super::draft::tprof::take()
                .into_iter()
                .map(|(l, ms, c)| serde_json::json!([l, ms, c]))
                .collect();
            eprintln!("{}", serde_json::json!({"tprof": table}));
            let (step_bytes, all_bytes) = fast.bytes();""")
open(p2, 'w').write(s)
print("instrumented")
