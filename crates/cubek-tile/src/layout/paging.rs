//! A pool of fixed-size pages read as sequences: [`Paging`] states it, [`PageTable`] resolves it.
//!
//! The buffer is a pool whose `sequence` dim holds pages and whose `position` dim a page's
//! positions. An operand bound over it is read as the sequences a table names: sequence `b`'s
//! position `s` lies in page `table[b, s / page_size]`, at row `s % page_size`. The projection stays
//! direct over the logical `[…, sequence, …, position, …]` axes, so every window, walk and bound
//! is the logical one; the table is resolved only where a coordinate becomes an offset
//! ([`BufferLayout`](super::projection::BufferLayout)), once per line.

use cubecl::prelude::*;

use crate::*;

/// Which axes of an operand a page table resolves: `sequence` names the page a table row lists,
/// `position` a run of `page_size` positions within one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Paging {
    pub sequence: Axis,
    pub position: Axis,
    /// Positions one page holds: the pool's extent along `position`.
    pub page_size: usize,
}

impl Paging {
    /// The buffer dims `projection` carries the two axes on, `(sequence, position)`.
    ///
    /// # Panics
    ///
    /// If either axis is not one dim's identity: a page is resolved digit for digit, so neither
    /// axis may share a dim or be gathered.
    pub(crate) fn dims(&self, projection: &Projection) -> (usize, usize) {
        let dim = |axis: Axis| {
            (0..projection.physical_rank())
                .find(|&pa| projection.physical_axis(pa).is_identity(axis))
                .unwrap_or_else(|| {
                    panic!(
                        "Paging: {axis:?} must be one buffer dim's identity to be resolved \
                         through a page table"
                    )
                })
        };
        (dim(self.sequence), dim(self.position))
    }
}

/// The extent of a page table, `[sequences, pages_per_sequence]`: what an operand bound over a
/// pool of pages reads as its sequences ([`Arg::paged`](crate::launch::Arg::paged)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PageTableShape {
    pub sequences: usize,
    pub pages_per_sequence: usize,
}

/// A page table as a kernel reads it: `u32`, `[sequences, pages_per_sequence]`, read by stride.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct PageTable {
    entries: Box<[u32]>,
    sequence_stride: u32,
    page_stride: u32,
    pages_per_sequence: u32,
    #[cube(comptime)]
    page_size: usize,
    /// The buffer dim the pages lie along.
    #[cube(comptime)]
    sequence_dim: usize,
    /// The buffer dim a page's positions lie along.
    #[cube(comptime)]
    position_dim: usize,
    /// Positions one step of the position dim's coordinate covers: a line's width where the
    /// positions run along the innermost dim, which is addressed in lines; one elsewhere.
    #[cube(comptime)]
    unit: usize,
}

#[cube]
impl PageTable {
    /// `table` resolving `paging`'s axes, which `projection` carries, an operand read `width`
    /// values a line along its innermost dim.
    pub(crate) fn new(
        table: &Tensor<u32>,
        #[comptime] paging: Paging,
        #[comptime] projection: Projection,
        #[comptime] width: usize,
    ) -> PageTable {
        let (sequence_dim, position_dim) = comptime!(paging.dims(&projection));
        let unit = comptime!(match position_dim == projection.physical_rank() - 1 {
            true => width,
            false => 1,
        });
        comptime!(assert!(
            paging.page_size.is_multiple_of(unit),
            "PageTable: a line of {unit} positions would cross a page of {}",
            paging.page_size
        ));
        PageTable {
            entries: unsafe { table.as_slice().as_boxed_unchecked() },
            sequence_stride: table.stride(0) as u32,
            page_stride: table.stride(1) as u32,
            pages_per_sequence: table.shape(1) as u32,
            page_size: comptime!(paging.page_size),
            sequence_dim,
            position_dim,
            unit,
        }
    }

    /// `digits`, a logical coordinate along each buffer dim, with the sequence's digit replaced
    /// by the page its position lies in and the position's by its row there.
    ///
    /// A position past the sequence's last page reads that page: a masked read may ask for its
    /// offset before the mask drops it, and the table holds nothing past its rows to index.
    pub(crate) fn resolve(&self, digits: &Coords<u32>) -> Coords<u32> {
        let page_size = comptime!(self.page_size as u32);
        let unit = comptime!(self.unit as u32);
        let position = digits.at(comptime!(self.position_dim)).times(unit);
        let logical = position
            .divided_by(page_size)
            .min_with(self.pages_per_sequence.minus(1u32));
        let at = digits
            .at(comptime!(self.sequence_dim))
            .times(self.sequence_stride)
            .plus(logical.times(self.page_stride));
        let page = self.entries[at.cast::<usize>()];
        let mut resolved = Coords::<u32>::new();
        #[unroll]
        for pa in 0..digits.len() {
            if comptime!(pa == self.sequence_dim) {
                resolved.push(page);
            } else if comptime!(pa == self.position_dim) {
                resolved.push(position.remainder(page_size).divided_by(unit));
            } else {
                resolved.push(digits.at(pa));
            }
        }
        resolved
    }
}
