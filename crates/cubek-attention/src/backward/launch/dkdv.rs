//! Kernel 3: compute `dK` and `dV`. KV-outer loop, accumulates in registers,
//! no atomics.
//!
//! Each program instance owns one KV-block and sweeps all Q-blocks. Both
//! `dK` and `dV` are accumulated independently across the sweep and written
//! once at the end.
//!
//! Implemented as two separate kernels (one for dK, one for dV) because the
//! combined kernel hits the wgpu backend's 8-storage-buffer-per-stage cap.
//! dV doesn't need `V` or `D` so its binding set is naturally smaller; dK
//! drops `dv` instead.

use cubecl::{CubeDim, calculate_cube_count_elemwise, client::Client, prelude::*};

use super::io_dtype;
use crate::backward::definition::BackwardConfig;
use crate::forward::definition::{AttentionGlobalTypes, AttentionSetupError};

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dv_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<f32>,
    dv: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] val_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: ElemType,
) {
    let row_idx = ABSOLUTE_POS;
    let seq_kv = k.shape(k.rank() - 2);
    let seq_q = q.shape(q.rank() - 2);

    let total_kv_rows = k.len() / head_dim;
    if row_idx >= total_kv_rows {
        terminate!();
    }

    let j = row_idx % seq_kv;
    let bh = row_idx / seq_kv;

    let k_base = row_idx * head_dim;
    let dv_base = row_idx * val_dim;
    let q_row_base = bh * seq_q * head_dim;
    let do_row_base = bh * seq_q * val_dim;
    let row_idx_base = bh * seq_q;


    let mut dv_acc = Array::<f32>::new(val_dim);
    for dd in 0..val_dim {
        dv_acc[dd] = 0.0_f32;
    }

    for i in 0..seq_q {
        let masked = causal && j + seq_q > i + seq_kv;
        if !masked {
            let q_base = q_row_base + i * head_dim;
            let do_base = do_row_base + i * val_dim;
            let lse_i = lse[row_idx_base + i];

            let mut dot = 0.0_f32;
            for dd in 0..head_dim {
                dot += f32::cast_from(q[q_base + dd]) * f32::cast_from(k[k_base + dd]);
            }
            let p_ij = (dot * scale - lse_i).exp();

            for dd in 0..val_dim {
                dv_acc[dd] += p_ij * f32::cast_from(do_[do_base + dd]);
            }
        }
    }

    for dd in 0..val_dim {
        dv[dv_base + dd] = E::cast_from(dv_acc[dd]);
    }
}

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dk_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<f32>,
    d: &Tensor<f32>,
    dk: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] val_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: ElemType,
) {
    let row_idx = ABSOLUTE_POS;
    let seq_kv = k.shape(k.rank() - 2);
    let seq_q = q.shape(q.rank() - 2);

    let total_kv_rows = k.len() / head_dim;
    if row_idx >= total_kv_rows {
        terminate!();
    }

    let j = row_idx % seq_kv;
    let bh = row_idx / seq_kv;

    let k_base = row_idx * head_dim;
    let v_base = row_idx * val_dim;
    let dk_base = row_idx * head_dim;
    let q_row_base = bh * seq_q * head_dim;
    let do_row_base = bh * seq_q * val_dim;
    let row_idx_base = bh * seq_q;


    let mut dk_acc = Array::<f32>::new(head_dim);
    for dd in 0..head_dim {
        dk_acc[dd] = 0.0_f32;
    }

    for i in 0..seq_q {
        let masked = causal && j + seq_q > i + seq_kv;
        if !masked {
            let q_base = q_row_base + i * head_dim;
            let do_base = do_row_base + i * val_dim;
            let lse_i = lse[row_idx_base + i];
            let d_i = d[row_idx_base + i];

            let mut dot = 0.0_f32;
            for dd in 0..head_dim {
                dot += f32::cast_from(q[q_base + dd]) * f32::cast_from(k[k_base + dd]);
            }
            let p_ij = (dot * scale - lse_i).exp();

            let mut dp = 0.0_f32;
            for dd in 0..val_dim {
                dp += f32::cast_from(do_[do_base + dd]) * f32::cast_from(v[v_base + dd]);
            }

            let s_ds = scale * p_ij * (dp - d_i);
            for dd in 0..head_dim {
                dk_acc[dd] += s_ds * f32::cast_from(q[q_base + dd]);
            }
        }
    }

    for dd in 0..head_dim {
        dk[dk_base + dd] = E::cast_from(dk_acc[dd]);
    }
}

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_dkdv_tiled_kernel<E: Float>(
    q: &Tensor<E>,
    k: &Tensor<E>,
    v: &Tensor<E>,
    do_: &Tensor<E>,
    lse: &Tensor<f32>,
    d: &Tensor<f32>,
    dk: &mut Tensor<E>,
    dv: &mut Tensor<E>,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] val_dim: usize,
    #[comptime] causal: bool,
    #[define(E)] _dtype: ElemType,
) {
    let c_d = comptime!(head_dim.div_ceil(32));
    let c_v = comptime!(val_dim.div_ceil(32));
    let q_sh_len = comptime!(32 * head_dim);
    let do_sh_len = comptime!(32 * val_dim);

    let seq_kv = k.shape(k.rank() - 2);
    let seq_q = q.shape(q.rank() - 2);
    let total_kv_rows = k.len() / head_dim;
    let total_bh = total_kv_rows / seq_kv;
    let kv_tiles_per_bh = seq_kv.div_ceil(4);

    let cube_id = CUBE_POS;
    let bh = cube_id / kv_tiles_per_bh;
    if bh >= total_bh {
        terminate!();
    }

    let kv_tile_idx = cube_id % kv_tiles_per_bh;
    let j_base = kv_tile_idx * 4;
    let w = UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let tid = w * 32 + lane;

    let j = j_base + w;
    let valid_j = j < seq_kv;
    let kv_row_idx = bh * seq_kv + j;

    let k_base = kv_row_idx * head_dim;
    let v_base = kv_row_idx * val_dim;
    let dk_base = kv_row_idx * head_dim;
    let dv_base = kv_row_idx * val_dim;
    let q_bh_base = bh * seq_q * head_dim;
    let do_bh_base = bh * seq_q * val_dim;
    let row_bh_base = bh * seq_q;

    let mut k_reg = Array::<f32>::new(c_d);
    let mut v_reg = Array::<f32>::new(c_v);
    let mut dk_acc = Array::<f32>::new(c_d);
    let mut dv_acc = Array::<f32>::new(c_v);

    #[unroll]
    for c in 0..c_d {
        let dd = c * 32 + lane;
        if valid_j && dd < head_dim {
            k_reg[c] = f32::cast_from(k[k_base + dd]);
        } else {
            k_reg[c] = 0.0_f32;
        }
        dk_acc[c] = 0.0_f32;
    }

    #[unroll]
    for c in 0..c_v {
        let dd = c * 32 + lane;
        if valid_j && dd < val_dim {
            v_reg[c] = f32::cast_from(v[v_base + dd]);
        } else {
            v_reg[c] = 0.0_f32;
        }
        dv_acc[c] = 0.0_f32;
    }

    let min_q = if causal && j_base + seq_q > seq_kv {
        j_base + seq_q - seq_kv
    } else {
        0usize
    };

    let mut q_sh = Shared::<[f32]>::new_slice(q_sh_len);
    let mut do_sh = Shared::<[f32]>::new_slice(do_sh_len);
    let mut lse_sh = Shared::<[f32]>::new_slice(32usize);
    let mut d_sh = Shared::<[f32]>::new_slice(32usize);

    let mut i_tile = min_q;
    while i_tile < seq_q {
        let rem = seq_q - i_tile;
        let tile_rows = if rem < 32usize { rem } else { 32usize };

        let q_tile_base = q_bh_base + i_tile * head_dim;
        let q_elems = tile_rows * head_dim;
        let mut idx = tid;
        while idx < q_elems {
            q_sh[idx] = f32::cast_from(q[q_tile_base + idx]);
            idx += 128;
        }

        let do_tile_base = do_bh_base + i_tile * val_dim;
        let do_elems = tile_rows * val_dim;
        let mut idx_do = tid;
        while idx_do < do_elems {
            do_sh[idx_do] = f32::cast_from(do_[do_tile_base + idx_do]);
            idx_do += 128;
        }

        if tid < tile_rows {
            lse_sh[tid] = lse[row_bh_base + i_tile + tid];
            d_sh[tid] = d[row_bh_base + i_tile + tid];
        }

        sync_cube();

        if valid_j {
            for t in 0..tile_rows {
                let i = i_tile + t;
                let masked = causal && j + seq_q > i + seq_kv;
                if !masked {
                    let q_off = t * head_dim;
                    let do_off = t * val_dim;
                    let lse_i = lse_sh[t];
                    let d_i = d_sh[t];

                    let mut q_val = Array::<f32>::new(c_d);
                    let mut dot_part = 0.0_f32;
                    #[unroll]
                    for c in 0..c_d {
                        let dd = c * 32 + lane;
                        if dd < head_dim {
                            let qv = q_sh[q_off + dd];
                            q_val[c] = qv;
                            dot_part += qv * k_reg[c];
                        } else {
                            q_val[c] = 0.0_f32;
                        }
                    }
                    let p_ij = (plane_sum(dot_part) * scale - lse_i).exp();

                    let mut dp_part = 0.0_f32;
                    #[unroll]
                    for c in 0..c_v {
                        let dd = c * 32 + lane;
                        if dd < val_dim {
                            let dov = do_sh[do_off + dd];
                            dv_acc[c] += p_ij * dov;
                            dp_part += dov * v_reg[c];
                        }
                    }
                    let dp = plane_sum(dp_part);
                    let s_ds = scale * p_ij * (dp - d_i);

                    #[unroll]
                    for c in 0..c_d {
                        dk_acc[c] += s_ds * q_val[c];
                    }
                }
            }
        }

        sync_cube();
        i_tile += 32;
    }

    if valid_j {
        #[unroll]
        for c in 0..c_d {
            let dd = c * 32 + lane;
            if dd < head_dim {
                dk[dk_base + dd] = E::cast_from(dk_acc[c]);
            }
        }
        #[unroll]
        for c in 0..c_v {
            let dd = c * 32 + lane;
            if dd < val_dim {
                dv[dv_base + dd] = E::cast_from(dv_acc[c]);
            }
        }
    }
}

/// Compute `dK` and `dV` into pre-allocated tensors (overwrites, not
/// accumulates).
///
/// Inputs (q/k/v/dO/dK/dV dtype taken from `global_dtypes`, accumulation in fp32):
/// - `q, k, v`: `[B, H, N, d]`.
/// - `do_`:     `[B, H, N, d]` upstream gradient.
/// - `lse`:     `[B, H, N]` fp32: saved from forward.
/// - `d`:       `[B, H, N]` fp32: from the prepass kernel.
///
/// Outputs:
/// - `dk`:      `[B, H, N, d]`: written cleanly.
/// - `dv`:      `[B, H, N, d]`: written cleanly.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward_dkdv(
    client: &Client,
    q: TensorBinding,
    k: TensorBinding,
    v: TensorBinding,
    do_: TensorBinding,
    lse: TensorBinding,
    d: TensorBinding,
    dk: TensorBinding,
    dv: TensorBinding,
    global_dtypes: &AttentionGlobalTypes,
    config: BackwardConfig,
) -> Result<(), AttentionSetupError> {
    let dtype = io_dtype(global_dtypes)?;
    let f32_size = f32::elem_type_native().size();

    let head_dim = q.shape[q.shape.len() - 1];
    let val_dim = v.shape[v.shape.len() - 1];

    let address_type = q
        .required_address_type(dtype.size())
        .max(k.required_address_type(dtype.size()))
        .max(v.required_address_type(dtype.size()))
        .max(do_.required_address_type(dtype.size()))
        .max(lse.required_address_type(f32_size))
        .max(d.required_address_type(f32_size))
        .max(dk.required_address_type(dtype.size()))
        .max(dv.required_address_type(dtype.size()));

    if client.properties().hardware.plane_size_max == 32 && head_dim <= 128 && val_dim <= 128 {
        let seq_kv = k.shape[k.shape.len() - 2];
        let total_kv_rows = k.shape.iter().product::<usize>() / head_dim;
        let total_bh = total_kv_rows / seq_kv;
        let kv_tiles_per_bh = seq_kv.div_ceil(4);
        let num_cubes = total_bh * kv_tiles_per_bh;
        let cube_dim = CubeDim::new_2d(32, 4);
        let cube_count = calculate_cube_count_elemwise(
            client,
            num_cubes * (cube_dim.num_elems() as usize),
            cube_dim,
        );
        flash_attention_backward_dkdv_tiled_kernel::launch(
            client,
            cube_count,
            cube_dim,
            address_type,
            q.into_tensor_arg(),
            k.into_tensor_arg(),
            v.into_tensor_arg(),
            do_.into_tensor_arg(),
            lse.into_tensor_arg(),
            d.into_tensor_arg(),
            dk.into_tensor_arg(),
            dv.into_tensor_arg(),
            config.scale,
            head_dim,
            val_dim,
            config.causal,
            dtype,
        );
    } else {
        let working_units = k.shape.iter().product::<usize>() / head_dim;
        let cube_dim = CubeDim::new(client, working_units);
        let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

        flash_attention_backward_dv_kernel::launch(
            client,
            cube_count.clone(),
            cube_dim,
            address_type,
            q.clone().into_tensor_arg(),
            k.clone().into_tensor_arg(),
            do_.clone().into_tensor_arg(),
            lse.clone().into_tensor_arg(),
            dv.into_tensor_arg(),
            config.scale,
            head_dim,
            val_dim,
            config.causal,
            dtype,
        );

        flash_attention_backward_dk_kernel::launch(
            client,
            cube_count,
            cube_dim,
            address_type,
            q.into_tensor_arg(),
            k.into_tensor_arg(),
            v.into_tensor_arg(),
            do_.into_tensor_arg(),
            lse.into_tensor_arg(),
            d.into_tensor_arg(),
            dk.into_tensor_arg(),
            config.scale,
            head_dim,
            val_dim,
            config.causal,
            dtype,
        );
    }

    Ok(())
}
