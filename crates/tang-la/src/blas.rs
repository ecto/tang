//! Safe wrappers around Apple Accelerate CBLAS routines.

const CBLAS_COL_MAJOR: i32 = 102;
const CBLAS_NO_TRANS: i32 = 111;

extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );

    fn cblas_dgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );

    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );

    fn cblas_dgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32,
    );
}

/// C = A * B  (f32, column-major)
///
/// A is m×k, B is k×n, C is m×n. All column-major with leading dims = nrows.
///
/// # Safety
/// Pointers must be valid for the given dimensions.
#[inline]
pub unsafe fn sgemm(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) {
    cblas_sgemm(
        CBLAS_COL_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        m, n, k,
        1.0,
        a, m,
        b, k,
        0.0,
        c, m,
    );
}

/// C = A * B  (f64, column-major)
#[inline]
pub unsafe fn dgemm(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: i32,
    n: i32,
    k: i32,
) {
    cblas_dgemm(
        CBLAS_COL_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        m, n, k,
        1.0,
        a, m,
        b, k,
        0.0,
        c, m,
    );
}

/// y = A * x  (f32, column-major)
///
/// A is m×n, x has length n, y has length m.
#[inline]
pub unsafe fn sgemv(
    a: *const f32,
    x: *const f32,
    y: *mut f32,
    m: i32,
    n: i32,
) {
    cblas_sgemv(
        CBLAS_COL_MAJOR,
        CBLAS_NO_TRANS,
        m, n,
        1.0,
        a, m,
        x, 1,
        0.0,
        y, 1,
    );
}

/// y = A * x  (f64, column-major)
#[inline]
pub unsafe fn dgemv(
    a: *const f64,
    x: *const f64,
    y: *mut f64,
    m: i32,
    n: i32,
) {
    cblas_dgemv(
        CBLAS_COL_MAJOR,
        CBLAS_NO_TRANS,
        m, n,
        1.0,
        a, m,
        x, 1,
        0.0,
        y, 1,
    );
}
