use tang_compress::img::*;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let c = Ccx::parse(&std::fs::read(&a[1]).unwrap()).unwrap();
    let mut m = Model::from_ccx(&c, 1, 0, 0);
    let f = vec![12u32; 8];
    m.arm.layers.iter().for_each(|l| eprintln!("arm layer {}x{} s={} k[..6]={:?}", l.out, l.inp, l.qw.s, &l.qw.k[..6]));
    eprintln!("ifce_of {:?} nifce {} ctx {:?} slope {}", m.ifce_of, m.nifce, m.ctx, m.slope);
    let n = m.sizes.len();
    let grids: Vec<Vec<i32>> = (0..n).map(|g| c.t[&format!("latent.{g}")].1.iter().map(|&v| v as i32).collect()).collect();
    for g in 0..n { let v=&grids[g]; eprintln!("grid {g} {:?} hyper {} mean|v| {:.3} max {}", m.sizes[g], m.hyper[g], v.iter().map(|x| x.abs() as f64).sum::<f64>()/v.len() as f64, v.iter().map(|x| x.abs()).max().unwrap()); }
    // set fw
    let _ = f;
    let (bytes, _) = encode(&c, 1, 0, 0);
    let p = parse(&bytes).unwrap();
    let m = p.m;
    let g = a.get(2).map_or(0, |s| s.parse().unwrap());
    let (h, w) = m.sizes[g];
    let lay = Layout::new(h, w, &m);
    let mut gd = Guard::default();
    let ifc = ifce_ctx(&m, g, &grids, &mut gd);
    let mut inb = vec![0; m.arm.nin()];
    let mut o = [0; 2];
    let mut bits = 0.0;
    for y in 0..h { for x in 0..w {
        arm_input(&m, &lay, &grids[g], &ifc, y, x, 0, &mut inb);
        m.arm.run(&inb, &mut o, &mut gd);
        let mu = o[0] as f64 / 256.0; let b = (o[1] as f64 / 256.0).clamp(-5.0, 5.0).exp();
        let v = grids[g][y*w+x] as f64;
        let cdf = |t: f64| { let d = t - mu; if d < 0.0 {0.5*(d/b).exp()} else {1.0-0.5*(-d/b).exp()} };
        bits += -(cdf(v+0.5)-cdf(v-0.5)).max(1.0/65536.0).log2();
        if y == h/2 && x >= w/2 && x < w/2+6 { eprintln!("({y},{x}) v={v} mu={mu:.3} b={b:.3} in={:?}", &inb); }
    }}
    eprintln!("grid {g} float-rate {:.0} B, fw arm {:?}", bits/8.0, m.arm.layers.iter().map(|l| l.fw).collect::<Vec<_>>());
}
