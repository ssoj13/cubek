//! One strided operand as a single launch argument: the tensor and its comptime [`TileSpec`].

use cubecl::prelude::*;

use crate::*;

/// One strided operand as a single launch argument: the tensor and its comptime [`TileSpec`].
#[derive(CubeType, CubeLaunch)]
pub struct TileArg<'a, E: Numeric, V: Size> {
    pub tensor: &'a Tensor<Vector<E, V>>,
    #[cube(comptime)]
    pub spec: TileSpec,
}

#[cube]
impl<'a, E: Numeric, V: Size> TileArg<'a, E, V> {
    /// Serve the operand as a [`Tile`] over the kernel's one `partitioning`.
    pub fn tile(&self, partitioning: &Partitioning) -> Tile<E> {
        let space = comptime!(partitioning.clone());
        GlobalOperand::<E>::tensor(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
        )
        .tile(comptime!(space.levels().to_vec()))
    }

    /// [`tile`](TileArg::tile), read through `pages` where the operand is a pool of pages bound
    /// [`paged`](crate::launch::Arg::paged), and where it lies where `pages` is `None`: one
    /// kernel serves either. The launch decides which it compiles by the `pages` it passes, which
    /// must match the binding: a table for a pool bound `paged`, `None` for any other operand;
    /// a mismatch panics as the kernel expands.
    pub fn tile_paged(
        &self,
        partitioning: &Partitioning,
        pages: ComptimeOption<&Tensor<u32>>,
    ) -> Tile<E> {
        self.tile_as_paged::<E>(partitioning, pages)
    }

    /// [`tile_as`](TileArg::tile_as), read through `pages` as [`tile_paged`](TileArg::tile_paged)
    /// is. A pool of pages is never packed, so `O` is the stored element where `pages` is `Some`.
    pub fn tile_as_paged<O: Numeric>(
        &self,
        partitioning: &Partitioning,
        pages: ComptimeOption<&Tensor<u32>>,
    ) -> Tile<O> {
        #[comptime]
        match pages {
            ComptimeOption::Some(table) => {
                let space = comptime!(partitioning.clone());
                GlobalOperand::<O>::paged(
                    self.tensor,
                    table,
                    comptime!(space.space().clone()),
                    comptime!(self.spec.clone()),
                )
                .tile(comptime!(space.levels().to_vec()))
            }
            ComptimeOption::None => self.tile_as::<O>(partitioning),
        }
    }

    /// [`tile`](TileArg::tile) reading `O` out of the stored `E`, unpacked where the binding is
    /// [`packed`](TileSpec::packed).
    pub fn tile_as<O: Numeric>(&self, partitioning: &Partitioning) -> Tile<O> {
        let space = comptime!(partitioning.clone());
        GlobalOperand::<O>::stored(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
        )
        .tile(comptime!(space.levels().to_vec()))
    }

    /// [`tile`](TileArg::tile) for a partly runtime gather map, its dynamic values in
    /// [`GlobalOperand::gathered`]'s order.
    pub fn tile_gathered(
        &self,
        partitioning: &Partitioning,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> Tile<E> {
        let space = comptime!(partitioning.clone());
        GlobalOperand::<E>::gathered(
            self.tensor,
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
            coefficients,
            offsets,
        )
        .tile(comptime!(space.levels().to_vec()))
    }
}
