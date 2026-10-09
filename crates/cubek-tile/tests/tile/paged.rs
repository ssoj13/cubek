//! A paged KV cache, which is expert routing with the axis split.
//!
//! The physical pool is `[PAGE, OFFSET]` over one buffer, spelled with
//! [`PhysicalAxisMap::disjoint`] exactly as a quantization block splits `K`. A page table names
//! which physical page holds a sequence's logical one, and the walk routes `PAGE` to it.
//!
//! So the pool needs no ordering of its own, and the page size is the split's radix rather than a
//! number carried beside a lookup.
//!
//! The second half reads the same kind of pool with no walk over pages at all: the operand is
//! bound [`paged`](cubek_tile::launch::Arg::paged), the kernel walks logical positions, and the
//! table is resolved where each line's offset is taken.
#![allow(non_snake_case)]

use super::{Form, implied};
use cubecl::{prelude::*, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::launch::{Delivery, Refusal};
use cubek_tile::layout::{PageTableShape, PhysicalAxisMap};
use cubek_tile::*;

/// Which sequence.
const B: Axis = Axis(0);
/// Which of the sequence's own pages: what the table is addressed by.
const LP: Axis = Axis(1);
/// Which physical page of the pool: what the table names, and what the walk routes.
const PAGE: Axis = Axis(2);
/// Where inside a page.
const OFFSET: Axis = Axis(3);
/// The feature a slot carries.
const D: Axis = Axis(4);

const SEQS: usize = 2;
const PAGES_PER_SEQ: usize = 2;
const POOL_PAGES: usize = 4;
const PAGE_SIZE: usize = 2;
const FEATURES: usize = 2;
const SLOTS: usize = POOL_PAGES * PAGE_SIZE;

/// Page 2 is shared: sequence 0 holds it first, sequence 1 second. Nothing about the pool's order
/// says so, which is the point of a table.
const TABLE: [u32; SEQS * PAGES_PER_SEQ] = [2, 0, 3, 2];

/// `out[b] = Σ kv[slot]` over the slots of the pages sequence `b` holds.
#[cube(launch)]
fn paged_sum_kernel<E: Numeric>(
    kv: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    table: &Tensor<u32>,
    space: Partitioning,
    #[comptime] seq: Level,
    #[comptime] logical: Level,
    #[comptime] physical: Level,
    #[define(E)] _dtype: ElemType,
) {
    let kv = kv.tile(&space);
    let out = out.tile(&space);

    for sequence in space.over(&seq) {
        let b = sequence.coord(B);

        let mut total = out.at(&sequence);
        total.init(Monoid::identity::<E>(comptime!(Monoid::Sum)));

        for lp in sequence.over(&logical) {
            let phys = table[b * PAGES_PER_SEQ + lp.coord(LP)] as usize;

            // The same call MoE makes: the axis keeps the pool's four pages, the walk visits the
            // one the table named.
            for page in lp.over(&physical).routed(PAGE, phys) {
                let mut cell = out.at(&page);
                cell.reduce_axis_accumulate(&kv.at(&page), comptime!(Monoid::Sum));
            }
        }
    }
}

/// Every slot distinct, so reading a neighbouring page cannot land on the right answer.
fn kv_values() -> Vec<f32> {
    (0..SLOTS * FEATURES)
        .map(|i| {
            let (slot, d) = (i / FEATURES, i % FEATURES);
            (slot * 10 + d) as f32
        })
        .collect()
}

fn run(table: &[u32]) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (B, SEQS),
                (LP, PAGES_PER_SEQ),
                (PAGE, POOL_PAGES),
                (OFFSET, PAGE_SIZE),
                (D, FEATURES),
            ]),
            Levels::leaf(&[(B, 1), (LP, 1), (PAGE, 1)])
                .walk_every(&[PAGE])
                .walk_every(&[LP])
                .walk_every(&[B])
                .build(),
        ),
        Form::Static,
    );

    let (kv_handle, _) = TestInput::builder(client.clone(), Shape::new([SLOTS, FEATURES]))
        .dtype(f32_ty)
        .custom(kv_values())
        .generate_with_f32_host_data();
    let table_handle = TestInput::builder(client.clone(), Shape::new([TABLE.len()]))
        .dtype(u32_ty)
        .custom(table.iter().map(|&p| p as f32).collect())
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([SEQS, FEATURES]))
        .dtype(f32_ty)
        .custom(vec![-1.0; SEQS * FEATURES])
        .generate_without_host_data();

    paged_sum_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            kv_handle.binding().into_tensor_arg(),
            // The pool's one dimension read as two axes: `slot = page * PAGE_SIZE + offset`.
            TileSpec::new(Projection::new(
                &[PAGE, OFFSET, D],
                &[
                    PhysicalAxisMap::disjoint(&[(PAGE, PAGE_SIZE), (OFFSET, 1)]),
                    PhysicalAxisMap::of(D),
                ],
            )),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[B, D]),
        ),
        table_handle.binding().into_tensor_arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        launcher.partitioning().level(2),
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

/// The reference: each sequence over the slots of the pages its table names.
fn expected(table: &[u32]) -> Vec<Vec<f32>> {
    let values = kv_values();
    (0..SEQS)
        .map(|b| {
            (0..FEATURES)
                .map(|d| {
                    (0..PAGES_PER_SEQ)
                        .flat_map(|lp| {
                            let page = table[b * PAGES_PER_SEQ + lp] as usize;
                            (0..PAGE_SIZE).map(move |o| (page * PAGE_SIZE + o) * FEATURES + d)
                        })
                        .map(|i| values[i])
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// Two page tables over the same pool: the answer follows the table, which it could not if the
/// route did nothing, since the walk would fold the whole pool under both.
#[test]
fn a_routed_page_axis_folds_each_sequence_over_the_pages_it_holds() {
    for table in [TABLE, [0, 1, 1, 0]] {
        let got = run(&table);
        let want = expected(&table);
        for (b, row) in want.iter().enumerate() {
            for (d, cell) in row.iter().enumerate() {
                assert_eq!(
                    got.get_f32(&[b, d]),
                    *cell,
                    "sequence {b} feature {d}: pages {:?}",
                    &table[b * PAGES_PER_SEQ..(b + 1) * PAGES_PER_SEQ]
                );
            }
        }
    }
}

/// The paged reads below: a position, as the sequences see it, and nothing of the pool.
const S: Axis = Axis(1);
/// The feature a position carries.
const F: Axis = Axis(2);

const SEQUENCES: usize = 3;
const PAGES_EACH: usize = 3;
const PAGE_LEN: usize = 2;
const POOL: usize = 7;
const FEATURES_READ: usize = 3;
/// Positions a step reads: more than a page, so every step but the last crosses one.
const READ_BLOCK: usize = 4;

/// Each sequence's pages, scattered over the pool; page 4 is sequence 2's twice, and page 0 is
/// held by two sequences.
const PAGES: [[u32; PAGES_EACH]; SEQUENCES] = [[5, 0, 3], [1, 6, 2], [4, 4, 0]];

/// `out[b] = Σ pool[b's page of s, s % PAGE_LEN]` over the first `lens[b]` positions of sequence `b`,
/// read through [`TileArg::tile_paged`]: the kernel walks logical positions, a block of them a
/// step, and [`Tile::within`] cuts each sequence at its own length.
#[cube(launch)]
fn paged_read_sum_kernel<E: Numeric>(
    pool: &TileArg<'_, E, Const<1>>,
    pages: ComptimeOption<&Tensor<u32>>,
    lens: &Tensor<u32>,
    out: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] seq: Level,
    #[comptime] block: Level,
    #[define(E)] _dtype: ElemType,
) {
    let pool = pool.tile_paged(comptime!(space.clone()), pages);
    let out = out.tile(comptime!(space.clone()));

    for sequence in space.over(&seq) {
        let len = lens[sequence.coord(B)] as usize;

        let mut total = out.at(&sequence);
        total.init(Monoid::identity::<E>(comptime!(Monoid::Sum)));

        let positions = pool.within(S, 0, len);
        for blk in sequence.over(&block).range(0, len.div_ceil(READ_BLOCK)) {
            let mut cell = out.at(&blk);
            cell.reduce_axis_accumulate(&positions.at(&blk), comptime!(Monoid::Sum));
        }
    }
}

/// Every pool cell distinct: `page·100 + row·10 + feature`.
fn pool_values() -> Vec<f32> {
    (0..POOL * PAGE_LEN * FEATURES_READ)
        .map(|i| {
            let (page, row, f) = (
                i / (PAGE_LEN * FEATURES_READ),
                (i / FEATURES_READ) % PAGE_LEN,
                i % FEATURES_READ,
            );
            (page * 100 + row * 10 + f) as f32
        })
        .collect()
}

fn run_paged_read(lens: &[u32; SEQUENCES]) -> HostData {
    run_paged_read_over(lens, &PAGES)
}

/// [`run_paged_read`] through `pages`.
fn run_paged_read_over(
    lens: &[u32; SEQUENCES],
    pages: &[[u32; PAGES_EACH]; SEQUENCES],
) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (B, SEQUENCES),
                (S, PAGES_EACH * PAGE_LEN),
                (F, FEATURES_READ),
            ]),
            Levels::leaf(&[(B, 1), (S, READ_BLOCK)])
                .walk_every(&[S])
                .walk_every(&[B])
                .build(),
        ),
        Form::Static,
    );

    let (pool_handle, _) =
        TestInput::builder(client.clone(), Shape::new([POOL, PAGE_LEN, FEATURES_READ]))
            .dtype(f32_ty)
            .custom(pool_values())
            .generate_with_f32_host_data();
    let table_handle = TestInput::builder(client.clone(), Shape::new([SEQUENCES, PAGES_EACH]))
        .dtype(u32_ty)
        .custom(pages.iter().flatten().map(|&p| p as f32).collect())
        .generate_without_host_data();
    let lens_handle = TestInput::builder(client.clone(), Shape::new([SEQUENCES]))
        .dtype(u32_ty)
        .custom(lens.iter().map(|&l| l as f32).collect())
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([SEQUENCES, FEATURES_READ]))
        .dtype(f32_ty)
        .custom(vec![-1.0; SEQUENCES * FEATURES_READ])
        .generate_without_host_data();

    let pool = launcher
        .arg(pool_handle.binding())
        .axes(&[B, S, F])
        .paged(
            B,
            S,
            PageTableShape {
                sequences: SEQUENCES,
                pages_per_sequence: PAGES_EACH,
            },
        )
        .build()
        .unwrap();

    paged_read_sum_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        pool.arg(),
        ComptimeOptionArgs::Some(table_handle.binding().into_tensor_arg()),
        lens_handle.binding().into_tensor_arg(),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[B, F]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

/// The reference: each sequence's positions looked up in its row of the table on the host.
fn expected_paged_read(lens: &[u32; SEQUENCES]) -> Vec<Vec<f32>> {
    expected_paged_read_over(lens, &PAGES)
}

/// [`expected_paged_read`] through `pages`, an entry past the pool reading its last page.
fn expected_paged_read_over(
    lens: &[u32; SEQUENCES],
    pages: &[[u32; PAGES_EACH]; SEQUENCES],
) -> Vec<Vec<f32>> {
    let values = pool_values();
    (0..SEQUENCES)
        .map(|b| {
            (0..FEATURES_READ)
                .map(|f| {
                    (0..lens[b] as usize)
                        .map(|s| {
                            let page = (pages[b][s / PAGE_LEN] as usize).min(POOL - 1);
                            values[(page * PAGE_LEN + s % PAGE_LEN) * FEATURES_READ + f]
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// A sequence read through its page table folds the positions its pages hold, in its table's
/// order: steps that cross pages, a length that ends inside a page, a full sequence and one of a
/// single position. Reading the pool where it lies would fold pages 0, 1 and 2 for sequence 0.
#[test]
fn a_paged_operand_reads_each_sequence_through_its_table() {
    for lens in [[5, 6, 1], [6, 3, 4]] {
        let got = run_paged_read(&lens);
        let want = expected_paged_read(&lens);
        for (b, row) in want.iter().enumerate() {
            for (f, cell) in row.iter().enumerate() {
                assert_eq!(
                    got.get_f32(&[b, f]),
                    *cell,
                    "sequence {b} feature {f}: {} positions of pages {:?}",
                    lens[b],
                    PAGES[b]
                );
            }
        }
    }
}

/// A table entry past the pool reads its last page rather than past the buffer: a table is device
/// data a launch may never have written.
#[test]
fn a_paged_read_stays_in_the_pool_and_the_table() {
    let pages = [[5, 99, 3], [1, 6, 2], [4, u32::MAX, 0]];
    let lens = [6, 3, 5];
    let got = run_paged_read_over(&lens, &pages);
    let want = expected_paged_read_over(&lens, &pages);
    for (b, row) in want.iter().enumerate() {
        for (f, cell) in row.iter().enumerate() {
            assert_eq!(got.get_f32(&[b, f]), *cell, "sequence {b} feature {f}");
        }
    }
}

/// What a kernel could not read through a table is refused as the operand is bound: one dim for
/// both axes, pages that are not the outermost dim, a line that would cross a page, and a
/// delivery that moves runs.
#[test]
fn a_paged_operand_the_table_cannot_resolve_is_refused_at_build() {
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(B, SEQUENCES), (S, PAGES_EACH * 3), (F, 4)]),
            Levels::leaf(&[(B, 1), (S, READ_BLOCK)])
                .walk_every(&[S])
                .walk_every(&[B])
                .build(),
        ),
        Form::Static,
    );
    let table = PageTableShape {
        sequences: SEQUENCES,
        pages_per_sequence: PAGES_EACH,
    };
    let pool = |shape: [usize; 3]| {
        TestInput::builder(client.clone(), Shape::new(shape))
            .dtype(f32::elem_type_native())
            .zeros()
            .generate_without_host_data()
    };
    let [a, b, c, d] = [
        pool([POOL, 3, 4]),
        pool([3, POOL, 4]),
        pool([POOL, 4, 3]),
        pool([POOL, 3, 4]),
    ];
    let one_dim = launcher
        .arg(a.binding())
        .axes(&[B, S, F])
        .paged(S, S, table)
        .build()
        .err();
    assert_eq!(one_dim, Some(Refusal::PagingOfOneDim(S)));
    let inner_pages = launcher
        .arg(b.binding())
        .axes(&[S, B, F])
        .paged(B, S, table)
        .build()
        .err();
    assert_eq!(inner_pages, Some(Refusal::PagesNotOutermost(B)));
    let split_line = launcher
        .arg(c.binding())
        .axes(&[B, F, S])
        .vectorize(2)
        .paged(B, S, table)
        .build()
        .err();
    assert_eq!(
        split_line,
        Some(Refusal::PageSplitsALine {
            page_size: 3,
            width: 2
        })
    );
    let by_runs = launcher
        .arg(d.binding())
        .axes(&[B, S, F])
        .delivery(Delivery::Tma)
        .paged(B, S, table)
        .build()
        .err();
    assert_eq!(by_runs, Some(Refusal::PagingMovedByRuns));
}
