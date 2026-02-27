# tang

The bedrock. Core math types generic over `Scalar` — write your physics once, get exact derivatives by swapping the type parameter.

## Types

```
Scalar ─── f32, f64, Dual<S>
  │
  ├── Vec2<S>   Vec3<S>   Vec4<S>
  ├── Point2<S>           Point3<S>
  ├── Dir3<S>             (normalized, derefs to Vec3)
  │
  ├── Mat3<S>   Mat4<S>
  ├── Quat<S>             (exp/log maps, slerp, Rodrigues)
  ├── Transform<S>        (rigid isometry = rotation + translation)
  │
  ├── SpatialVec<S>       (6D twist / wrench)
  ├── SpatialMat<S>       (6×6 as four 3×3 blocks)
  ├── SpatialTransform<S> (Plücker coordinates)
  └── SpatialInertia<S>   (mass + COM + rotational inertia)
```

## The `Scalar` trick

Every type is parameterized over `S: Scalar`. Since `Dual<f64>` implements `Scalar`, any computation you write automatically propagates derivatives:

```rust
use tang::{Vec3, Dual, Scalar};

fn spring_energy<S: Scalar>(x: Vec3<S>, rest: S) -> S {
    let stretch = x.norm() - rest;
    stretch * stretch * S::from_f64(0.5)
}

// plain evaluation
let e = spring_energy(Vec3::new(1.0, 0.0, 0.0), 0.5);

// exact gradient via dual numbers — same function, different type
let dx = Dual::new(1.0, 1.0);
let dy = Dual::new(0.0, 0.0);
let dz = Dual::new(0.0, 0.0);
let de = spring_energy(Vec3::new(dx, dy, dz), Dual::new(0.5, 0.0));
// de.dual is ∂E/∂x, exact to machine precision
```

## Spatial algebra

6D spatial vectors eliminate the pain of 6×6 indexing. `SpatialMat` stores four 3×3 blocks directly, and `SpatialTransform` applies motion/force transforms without ever constructing the full matrix:

```rust
use tang::{SpatialTransform, SpatialVec, Vec3, Quat};

let xform = SpatialTransform::new(Quat::identity(), Vec3::new(0.0, 0.0, 1.0));
let twist = SpatialVec::new(Vec3::zero(), Vec3::new(1.0, 0.0, 0.0));
let moved = xform.apply_motion(twist); // 18 ops, not 36
```

## Design

- **`#![no_std]`** with `alloc` — works on embedded, in WASM, wherever
- **`#[repr(C)]`** everywhere — `bytemuck::Pod` for GPU upload, zero-copy
- **Exact geometric predicates** — optional `exact` feature uses Shewchuk's algorithm for orient2d/3d, incircle/insphere
- **nalgebra compat** — `zeros()`, `norm_squared()`, `new_normalize()`, `from_diagonal(&v)` aliases all present

## Features

| Feature | Default | Purpose |
|---------|---------|---------|
| `std` | yes | Use inherent float methods |
| `bytemuck` | no | Pod/Zeroable for GPU interop |
| `serde` | no | Serialize everything |
| `exact` | no | Shewchuk geometric predicates |
| `libm` | no | no_std math fallback |

## License

[MIT](../../LICENSE)
