//! `HEALPix` sky pixelization.
//!
//! Implements the Hierarchical Equal Area isoLatitude Pixelization scheme
//! (Gorski et al. 2005, `ApJ` 622, 759), supporting both the RING and NESTED
//! pixel orderings. Tessellates the sphere into `12 * nside^2` pixels of
//! equal solid angle.
//!
//! `nside` must be a power of two. All angles are in radians.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

const JRLL: [i64; 12] = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
const JPLL: [i64; 12] = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7];

/// Position within one of the 12 `HEALPix` base faces.
#[derive(Debug, Clone, Copy)]
struct FaceCoords {
    face: u32,
    ix: u32,
    iy: u32,
}

/// Total number of pixels at the given resolution: `12 * nside^2`.
#[must_use]
pub fn n_pixels(nside: u32) -> u64 {
    12 * u64::from(nside) * u64::from(nside)
}

/// RING-scheme pixel index containing the given sky position.
#[must_use]
pub fn ang_to_ring(nside: u32, ra: f64, dec: f64) -> u64 {
    face_to_ring(nside, ang_to_face(nside, ra, dec))
}

/// NESTED-scheme pixel index containing the given sky position.
#[must_use]
pub fn ang_to_nested(nside: u32, ra: f64, dec: f64) -> u64 {
    face_to_nested(nside, ang_to_face(nside, ra, dec))
}

/// Center `(ra, dec)` of a RING-scheme pixel.
#[must_use]
pub fn ring_to_ang(nside: u32, pixel: u64) -> (f64, f64) {
    face_to_ang(nside, ring_to_face(nside, pixel))
}

/// Center `(ra, dec)` of a NESTED-scheme pixel.
#[must_use]
pub fn nested_to_ang(nside: u32, pixel: u64) -> (f64, f64) {
    face_to_ang(nside, nested_to_face(nside, pixel))
}

/// Convert a RING-scheme pixel index to NESTED.
#[must_use]
pub fn ring_to_nested(nside: u32, pixel: u64) -> u64 {
    face_to_nested(nside, ring_to_face(nside, pixel))
}

/// Convert a NESTED-scheme pixel index to RING.
#[must_use]
pub fn nested_to_ring(nside: u32, pixel: u64) -> u64 {
    face_to_ring(nside, nested_to_face(nside, pixel))
}

/// Strict upper bound on the angular distance from any pixel center to any
/// of its vertices, across all pixels at this resolution.
///
/// Useful for converting a strict disc query (centers within radius) into
/// an inclusive one (any pixel that may overlap) by adding this value to
/// the search radius.
#[must_use]
pub fn max_pixel_radius(nside: u32) -> f64 {
    1.5 / f64::from(nside)
}

/// Vertex positions of a RING-scheme pixel, in the order N, E, S, W.
#[must_use]
pub fn ring_vertices(nside: u32, pixel: u64) -> [(f64, f64); 4] {
    face_vertices(nside, ring_to_face(nside, pixel))
}

/// Vertex positions of a NESTED-scheme pixel, in the order N, E, S, W.
#[must_use]
pub fn nested_vertices(nside: u32, pixel: u64) -> [(f64, f64); 4] {
    face_vertices(nside, nested_to_face(nside, pixel))
}

/// All RING-scheme pixels whose centers lie within `radius` of `(ra, dec)`.
///
/// To find every pixel that may overlap the disc, expand `radius` by
/// [`max_pixel_radius`].
#[must_use]
pub fn query_disc_ring(nside: u32, ra: f64, dec: f64, radius: f64) -> Vec<u64> {
    query_disc(nside, ra, dec, radius)
}

/// All NESTED-scheme pixels whose centers lie within `radius` of `(ra, dec)`.
///
/// To find every pixel that may overlap the disc, expand `radius` by
/// [`max_pixel_radius`].
#[must_use]
pub fn query_disc_nested(nside: u32, ra: f64, dec: f64, radius: f64) -> Vec<u64> {
    query_disc(nside, ra, dec, radius)
        .into_iter()
        .map(|p| ring_to_nested(nside, p))
        .collect()
}

fn face_vertices(nside: u32, fc: FaceCoords) -> [(f64, f64); 4] {
    let x = f64::from(fc.ix);
    let y = f64::from(fc.iy);
    [
        face_xy_to_ang(nside, fc.face, x + 1.0, y + 1.0), // N
        face_xy_to_ang(nside, fc.face, x + 1.0, y),       // E
        face_xy_to_ang(nside, fc.face, x, y),             // S
        face_xy_to_ang(nside, fc.face, x, y + 1.0),       // W
    ]
}

/// Map a continuous position `(x, y)` within `face` to `(ra, dec)`.
/// Pixel centers correspond to half-integer `(x, y)`; pixel corners to
/// integer values in `[0, nside]`.
fn face_xy_to_ang(nside: u32, face: u32, x: f64, y: f64) -> (f64, f64) {
    let nside = f64::from(nside);
    let jrll = JRLL[face as usize] as f64;
    let jpll = JPLL[face as usize] as f64;
    let jr = jrll * nside - x - y;
    let inv_3nside_sq = 1.0 / (3.0 * nside * nside);

    let (nr, z) = if jr < nside {
        (jr, 1.0 - jr * jr * inv_3nside_sq)
    } else if jr > 3.0 * nside {
        let nr = 4.0 * nside - jr;
        (nr, nr * nr * inv_3nside_sq - 1.0)
    } else {
        (nside, (2.0 * nside - jr) * 2.0 / (3.0 * nside))
    };

    let phi = if nr == 0.0 {
        0.0
    } else {
        (jpll * nr + x - y) * (FRAC_PI_2 / (2.0 * nr))
    };
    (phi.rem_euclid(TAU), z.asin())
}

/// Disc query in the RING scheme. Walks only the rings that overlap in
/// latitude and computes the phi half-width per ring from spherical geometry.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "ring/pixel indices are bounded by nside"
)]
fn query_disc(nside: u32, ra: f64, dec: f64, radius: f64) -> Vec<u64> {
    let mut out = Vec::new();
    if !radius.is_finite() || radius <= 0.0 {
        return out;
    }
    let nside_i = i64::from(nside);
    let nside_f = f64::from(nside);
    let inv_3nside_sq = 1.0 / (3.0 * nside_f * nside_f);
    let nl4 = 4 * nside_i;

    let z_q = dec.sin();
    let cos_r = radius.cos();
    let one_minus_zq_sq = (1.0 - z_q * z_q).max(0.0);
    let phi_q = ra.rem_euclid(TAU);

    // Convert a z value to a real-valued ring index counted from the north.
    let ring_for_z = |z: f64| -> f64 {
        if z >= 2.0 / 3.0 {
            ((1.0 - z) / inv_3nside_sq).sqrt()
        } else if z >= -2.0 / 3.0 {
            2.0 * nside_f * (1.0 - 0.75 * z)
        } else {
            4.0 * nside_f - ((z + 1.0) / inv_3nside_sq).sqrt()
        }
    };

    let z_north = (dec + radius).min(FRAC_PI_2).sin();
    let z_south = (dec - radius).max(-FRAC_PI_2).sin();
    let iring_lo = (ring_for_z(z_north).floor() as i64).max(1);
    let iring_hi = (ring_for_z(z_south).ceil() as i64).min(nl4 - 1);

    for iring in iring_lo..=iring_hi {
        let (z, npix_in_ring, n_before, phi_offset) = ring_geometry(iring, nside_i);

        let denom = (one_minus_zq_sq * (1.0 - z * z)).sqrt();
        let dphi_max = if denom == 0.0 {
            // Query or ring is at a pole: ring is one point in (z, phi).
            PI
        } else {
            let cos_dphi = (cos_r - z_q * z) / denom;
            if cos_dphi > 1.0 {
                continue;
            } else if cos_dphi < -1.0 {
                PI
            } else {
                cos_dphi.acos()
            }
        };

        let dpix = TAU / npix_in_ring as f64;
        if dphi_max >= PI {
            out.extend(n_before..n_before + npix_in_ring);
            continue;
        }
        let ip_lo = ((phi_q - dphi_max - phi_offset) / dpix).ceil() as i64;
        let ip_hi = ((phi_q + dphi_max - phi_offset) / dpix).floor() as i64;
        for ip in ip_lo..=ip_hi {
            let ip_mod = ip.rem_euclid(npix_in_ring as i64) as u64;
            out.push(n_before + ip_mod);
        }
    }
    out
}

// Values are mathematically bounded: face 0..11, ix/iy 0..nside-1.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "values are bounded by construction"
)]
fn ang_to_face(nside: u32, ra: f64, dec: f64) -> FaceCoords {
    let nside = i64::from(nside);
    let z = dec.sin();
    let za = z.abs();
    let tt = ra.rem_euclid(TAU) / FRAC_PI_2; // in [0, 4)

    if za <= 2.0 / 3.0 {
        // Equatorial belt.
        let temp1 = nside as f64 * (0.5 + tt);
        let temp2 = nside as f64 * (z * 0.75);
        let jp = (temp1 - temp2).floor() as i64;
        let jm = (temp1 + temp2).floor() as i64;
        let ifp = jp / nside;
        let ifm = jm / nside;
        let face = match ifp.cmp(&ifm) {
            std::cmp::Ordering::Equal => (ifp | 4) as u32,
            std::cmp::Ordering::Less => ifp as u32,
            std::cmp::Ordering::Greater => (ifm + 8) as u32,
        };
        FaceCoords {
            face,
            ix: (jm & (nside - 1)) as u32,
            iy: (nside - (jp & (nside - 1)) - 1) as u32,
        }
    } else {
        // Polar caps. Use cos(dec) directly to retain accuracy near the poles.
        let ntt = (tt.floor() as i64).min(3);
        let tp = tt - ntt as f64;
        let s = dec.cos(); // sin(colatitude)
        let tmp = nside as f64 * s / ((1.0 + za) / 3.0).sqrt();
        let jp = ((tp * tmp).floor() as i64).clamp(0, nside - 1);
        let jm = (((1.0 - tp) * tmp).floor() as i64).clamp(0, nside - 1);
        if z >= 0.0 {
            FaceCoords {
                face: ntt as u32,
                ix: (nside - jm - 1) as u32,
                iy: (nside - jp - 1) as u32,
            }
        } else {
            FaceCoords {
                face: (ntt + 8) as u32,
                ix: jp as u32,
                iy: jm as u32,
            }
        }
    }
}

fn face_to_ang(nside: u32, fc: FaceCoords) -> (f64, f64) {
    let nside_i = i64::from(nside);
    let nl4 = 4 * nside_i;
    let fact2 = 4.0 / n_pixels(nside) as f64;
    let ix = i64::from(fc.ix);
    let iy = i64::from(fc.iy);
    let jr = JRLL[fc.face as usize] * nside_i - ix - iy - 1;

    let (nr, z, kshift) = if jr < nside_i {
        let nr = jr;
        (nr, 1.0 - (nr * nr) as f64 * fact2, 0_i64)
    } else if jr > 3 * nside_i {
        let nr = nl4 - jr;
        (nr, (nr * nr) as f64 * fact2 - 1.0, 0_i64)
    } else {
        let fact1 = (2 * nside_i) as f64 * fact2;
        (
            nside_i,
            (2 * nside_i - jr) as f64 * fact1,
            (jr - nside_i) & 1,
        )
    };

    let jp_raw = (JPLL[fc.face as usize] * nr + ix - iy + 1 + kshift) / 2;
    let jp = if jp_raw > nl4 {
        jp_raw - nl4
    } else if jp_raw < 1 {
        jp_raw + nl4
    } else {
        jp_raw
    };

    let phi = (jp as f64 - (kshift + 1) as f64 * 0.5) * (FRAC_PI_2 / nr as f64);
    (phi.rem_euclid(TAU), z.asin())
}

fn face_to_ring(nside: u32, fc: FaceCoords) -> u64 {
    let nside = i64::from(nside);
    let nl4 = 4 * nside;
    let ix = i64::from(fc.ix);
    let iy = i64::from(fc.iy);
    let jr = JRLL[fc.face as usize] * nside - ix - iy - 1;

    let (nr, n_before, kshift) = if jr < nside {
        let nr = jr;
        (nr, 2 * nr * (nr - 1), 0_i64)
    } else if jr > 3 * nside {
        let nr = nl4 - jr;
        (nr, 12 * nside * nside - 2 * (nr + 1) * nr, 0_i64)
    } else {
        (
            nside,
            2 * nside * (nside - 1) + (jr - nside) * nl4,
            (jr - nside) & 1,
        )
    };

    let jp_raw = (JPLL[fc.face as usize] * nr + ix - iy + 1 + kshift) / 2;
    let jp = if jp_raw > nl4 {
        jp_raw - nl4
    } else if jp_raw < 1 {
        jp_raw + nl4
    } else {
        jp_raw
    };
    (n_before + jp - 1).cast_unsigned()
}

// Values are mathematically bounded: face 0..11, ix/iy 0..nside-1.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "values are bounded by construction"
)]
fn ring_to_face(nside: u32, pix: u64) -> FaceCoords {
    let nside = i64::from(nside);
    let pix = pix.cast_signed();
    let ncap = 2 * nside * (nside - 1);
    let npix = 12 * nside * nside;
    let nl2 = 2 * nside;

    let (iring, iphi, kshift, nr, face) = if pix < ncap {
        // North polar cap.
        let iring = (1 + (1 + 2 * pix).isqrt()) >> 1;
        let iphi = pix + 1 - 2 * iring * (iring - 1);
        (iring, iphi, 0_i64, iring, ((iphi - 1) / iring) as u32)
    } else if pix < npix - ncap {
        // Equatorial belt.
        let ip = pix - ncap;
        let iring = ip / (4 * nside) + nside;
        let iphi = ip % (4 * nside) + 1;
        let kshift = (iring + nside) & 1;
        let ire = iring - nside + 1;
        let irm = nl2 + 2 - ire;
        let ifm = (iphi - ire / 2 + nside - 1) / nside;
        let ifp = (iphi - irm / 2 + nside - 1) / nside;
        let face = match ifp.cmp(&ifm) {
            std::cmp::Ordering::Equal => (ifp | 4) as u32,
            std::cmp::Ordering::Less => ifp as u32,
            std::cmp::Ordering::Greater => (ifm + 8) as u32,
        };
        (iring, iphi, kshift, nside, face)
    } else {
        // South polar cap.
        let ip = npix - pix;
        let i = (1 + (2 * ip - 1).isqrt()) >> 1;
        let iphi = 4 * i + 1 - (ip - 2 * i * (i - 1));
        (2 * nl2 - i, iphi, 0_i64, i, (8 + (iphi - 1) / i) as u32)
    };

    let irt = iring - JRLL[face as usize] * nside + 1;
    let ipt_raw = 2 * iphi - JPLL[face as usize] * nr - kshift - 1;
    let ipt = if ipt_raw >= nl2 {
        ipt_raw - 8 * nside
    } else {
        ipt_raw
    };
    FaceCoords {
        face,
        ix: ((ipt - irt) >> 1) as u32,
        iy: ((-(ipt + irt)) >> 1) as u32,
    }
}

fn face_to_nested(nside: u32, fc: FaceCoords) -> u64 {
    let npface = u64::from(nside) * u64::from(nside);
    u64::from(fc.face) * npface + spread_bits(fc.ix) + (spread_bits(fc.iy) << 1)
}

fn nested_to_face(nside: u32, pix: u64) -> FaceCoords {
    let npface = u64::from(nside) * u64::from(nside);
    let local = pix & (npface - 1);
    FaceCoords {
        face: (pix / npface) as u32,
        ix: compress_bits(local),
        iy: compress_bits(local >> 1),
    }
}

/// Deposit each of the 32 input bits into the even bit positions of a `u64`.
fn spread_bits(v: u32) -> u64 {
    let x = u64::from(v);
    let x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
    let x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
    let x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    let x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    (x | (x << 1)) & 0x5555_5555_5555_5555
}

/// Inverse of [`spread_bits`]: gather the even bits of `v` into a `u32`.
fn compress_bits(v: u64) -> u32 {
    let x = v & 0x5555_5555_5555_5555;
    let x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    let x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
    let x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF;
    let x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF;
    let x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF;
    x as u32
}

/// Geometry of RING-scheme ring `iring` (counted from the north pole, 1-based).
///
/// Returns `(z, npix_in_ring, n_before, phi_offset)` where `z` is the cosine
/// of colatitude, `n_before` is the number of pixels in earlier rings, and
/// pixel `ip` (0-indexed within the ring) has center phi
/// `phi_offset + ip * (2*pi / npix_in_ring)`.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "ring counts are bounded by 4*nside"
)]
fn ring_geometry(iring: i64, nside: i64) -> (f64, u64, u64, f64) {
    let nside_f = nside as f64;
    let inv_3nside_sq = 1.0 / (3.0 * nside_f * nside_f);
    if iring < nside {
        let z = 1.0 - (iring * iring) as f64 * inv_3nside_sq;
        let npix = 4 * iring;
        let n_before = 2 * iring * (iring - 1);
        let phi_offset = FRAC_PI_2 / iring as f64 * 0.5;
        (z, npix as u64, n_before as u64, phi_offset)
    } else if iring <= 3 * nside {
        let z = (2 * nside - iring) as f64 * (2.0 / (3.0 * nside_f));
        let npix = 4 * nside;
        let ncap = 2 * nside * (nside - 1);
        let n_before = ncap + (iring - nside) * 4 * nside;
        let kshift = (iring - nside) & 1;
        let phi_offset = if kshift == 0 {
            FRAC_PI_2 / nside_f * 0.5
        } else {
            0.0
        };
        (z, npix as u64, n_before as u64, phi_offset)
    } else {
        let from_south = 4 * nside - iring;
        let z = (from_south * from_south) as f64 * inv_3nside_sq - 1.0;
        let npix = 4 * from_south;
        let n_before = 12 * nside * nside - 2 * (from_south + 1) * from_south;
        let phi_offset = FRAC_PI_2 / from_south as f64 * 0.5;
        (z, npix as u64, n_before as u64, phi_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn n_pixels_matches_formula() {
        for &nside in &[1_u32, 2, 4, 8, 64, 1024] {
            assert_eq!(n_pixels(nside), 12 * u64::from(nside) * u64::from(nside));
        }
    }

    #[test]
    fn ring_indices_in_range() {
        let nside = 64;
        let n = n_pixels(nside);
        for i in 0..200 {
            let ra = f64::from(i) * 0.031;
            for j in 0..50 {
                let dec = -1.5 + f64::from(j) * 0.06;
                let p = ang_to_ring(nside, ra, dec);
                assert!(p < n, "ring pixel out of range: {p} >= {n}");
            }
        }
    }

    #[test]
    fn nested_indices_in_range() {
        let nside = 64;
        let n = n_pixels(nside);
        for i in 0..200 {
            let ra = f64::from(i) * 0.031;
            for j in 0..50 {
                let dec = -1.5 + f64::from(j) * 0.06;
                let p = ang_to_nested(nside, ra, dec);
                assert!(p < n, "nested pixel out of range: {p} >= {n}");
            }
        }
    }

    #[test]
    fn ring_covers_all_pixels() {
        let nside = 4;
        let n = n_pixels(nside) as usize;
        let mut hit = vec![false; n];
        for i in 0..1000 {
            let ra = f64::from(i) * 0.013;
            for j in 0..500 {
                let dec = -PI / 2.0 + f64::from(j) * (PI / 500.0);
                hit[ang_to_ring(nside, ra, dec) as usize] = true;
            }
        }
        assert!(hit.iter().all(|&h| h));
    }

    #[test]
    fn ring_roundtrip() {
        let nside = 64;
        for p in (0..n_pixels(nside)).step_by(37) {
            let (ra, dec) = ring_to_ang(nside, p);
            assert_eq!(ang_to_ring(nside, ra, dec), p);
        }
    }

    #[test]
    fn nested_roundtrip() {
        let nside = 64;
        for p in (0..n_pixels(nside)).step_by(37) {
            let (ra, dec) = nested_to_ang(nside, p);
            assert_eq!(ang_to_nested(nside, ra, dec), p);
        }
    }

    #[test]
    fn scheme_conversion_roundtrip() {
        let nside = 32;
        for p in 0..n_pixels(nside) {
            assert_eq!(nested_to_ring(nside, ring_to_nested(nside, p)), p);
            assert_eq!(ring_to_nested(nside, nested_to_ring(nside, p)), p);
        }
    }

    #[test]
    fn schemes_agree_via_face() {
        let nside = 16;
        for i in 0..50 {
            let ra = f64::from(i) * 0.12;
            for j in 0..30 {
                let dec = -1.4 + f64::from(j) * 0.1;
                let r = ang_to_ring(nside, ra, dec);
                let n = ang_to_nested(nside, ra, dec);
                assert_eq!(ring_to_nested(nside, r), n);
                assert_eq!(nested_to_ring(nside, n), r);
            }
        }
    }

    #[test]
    fn ang_returns_valid_ranges() {
        let nside = 8;
        for p in 0..n_pixels(nside) {
            let (ra, dec) = ring_to_ang(nside, p);
            assert!((0.0..TAU).contains(&ra), "ra out of range: {ra}");
            assert!(
                (-FRAC_PI_2..=FRAC_PI_2).contains(&dec),
                "dec out of range: {dec}"
            );
        }
    }

    #[test]
    fn poles_map_to_endpoints() {
        let nside = 64;
        let p_north = ang_to_ring(nside, 0.0, FRAC_PI_2 - 1e-9);
        assert!(p_north < 4);
        let p_south = ang_to_ring(nside, 0.0, -FRAC_PI_2 + 1e-9);
        assert!(p_south >= n_pixels(nside) - 4);
    }

    #[test]
    fn spread_compress_roundtrip() {
        for v in [0_u32, 1, 0xFF, 0x1234, 0xFFFF, 0xDEAD_BEEF, u32::MAX] {
            assert_eq!(compress_bits(spread_bits(v)), v);
        }
    }

    fn ang_sep(a: (f64, f64), b: (f64, f64)) -> f64 {
        // Spherical angular distance, robust at small separations.
        let (ra1, dec1) = a;
        let (ra2, dec2) = b;
        let dx = dec1.cos() * ra1.cos() - dec2.cos() * ra2.cos();
        let dy = dec1.cos() * ra1.sin() - dec2.cos() * ra2.sin();
        let dz = dec1.sin() - dec2.sin();
        let chord = (dx * dx + dy * dy + dz * dz).sqrt();
        2.0 * (chord / 2.0).asin()
    }

    #[test]
    fn vertices_surround_center() {
        let nside = 16;
        let bound = max_pixel_radius(nside);
        for p in 0..n_pixels(nside) {
            let center = ring_to_ang(nside, p);
            for v in ring_vertices(nside, p) {
                let sep = ang_sep(center, v);
                assert!(
                    sep <= bound,
                    "vertex sep {sep} exceeds bound {bound} at pixel {p}"
                );
            }
        }
    }

    #[test]
    fn vertices_match_across_schemes() {
        let nside = 8;
        for p_ring in 0..n_pixels(nside) {
            let p_nest = ring_to_nested(nside, p_ring);
            assert_eq!(ring_vertices(nside, p_ring), nested_vertices(nside, p_nest));
        }
    }

    #[test]
    fn vertices_shared_with_neighbors() {
        // Every vertex of every pixel coincides with a vertex of another pixel.
        let nside = 4;
        let all: Vec<_> = (0..n_pixels(nside))
            .map(|p| ring_vertices(nside, p))
            .collect();
        for (i, verts) in all.iter().enumerate() {
            for v in verts {
                let shared = all
                    .iter()
                    .enumerate()
                    .any(|(j, vs)| j != i && vs.iter().any(|vo| ang_sep(*v, *vo) < 1e-10));
                assert!(shared, "pixel {i} vertex {v:?} not shared");
            }
        }
    }

    #[test]
    fn query_disc_zero_radius_empty() {
        assert!(query_disc_ring(8, 1.0, 0.5, 0.0).is_empty());
    }

    #[test]
    fn query_disc_matches_brute_force() {
        let nside = 16;
        let cases = [
            (0.5_f64, 0.0_f64, 0.2_f64), // equator
            (1.0, 1.4, 0.15),            // near north pole
            (4.5, -1.3, 0.25),           // southern hemisphere
            (PI - 0.001, 0.0, 0.05),     // small disc near phi=pi
        ];
        for (ra, dec, radius) in cases {
            let mut got = query_disc_ring(nside, ra, dec, radius);
            got.sort_unstable();
            let expected: Vec<u64> = (0..n_pixels(nside))
                .filter(|&p| {
                    let c = ring_to_ang(nside, p);
                    ang_sep((ra, dec), c) <= radius
                })
                .collect();
            assert_eq!(got, expected, "ra={ra} dec={dec} r={radius}");
        }
    }

    #[test]
    fn query_disc_full_sky() {
        let nside = 8;
        let res = query_disc_ring(nside, 0.0, 0.0, PI);
        assert_eq!(res.len() as u64, n_pixels(nside));
    }

    #[test]
    fn query_disc_nested_matches_ring_via_conversion() {
        let nside = 16;
        let (ra, dec, r) = (2.0, 0.3, 0.1);
        let mut from_ring: Vec<u64> = query_disc_ring(nside, ra, dec, r)
            .into_iter()
            .map(|p| ring_to_nested(nside, p))
            .collect();
        let mut from_nest = query_disc_nested(nside, ra, dec, r);
        from_ring.sort_unstable();
        from_nest.sort_unstable();
        assert_eq!(from_ring, from_nest);
    }

    #[test]
    fn max_pixel_radius_is_actual_upper_bound() {
        for nside in [1, 2, 4, 8, 16, 32] {
            let bound = max_pixel_radius(nside);
            for p in 0..n_pixels(nside) {
                let center = ring_to_ang(nside, p);
                for v in ring_vertices(nside, p) {
                    let sep = ang_sep(center, v);
                    assert!(
                        sep <= bound,
                        "nside={nside} pixel={p} sep={sep} bound={bound}"
                    );
                }
            }
        }
    }
}
