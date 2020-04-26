// Copyright 2020 Graydon Hoare <graydon@pobox.com>
// Licensed under the MIT and Apache-2.0 licenses.

use super::*;
use bit_set::BitSet;
use num_traits::bounds::Bounded;
use std::cmp::{Eq, Ord, Ordering, PartialOrd};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::ops::Add;

/// Write code that _uses_ lattices over this type, and it will delegate
/// to the functions of the parameter `LatticeDef`.

#[derive(Debug)]
pub struct LatticeElt<D: LatticeDef> {
    pub value: D::T,
}

impl<D: LatticeDef> Clone for LatticeElt<D>
where
    D::T: Clone,
{
    fn clone(&self) -> Self {
        LatticeElt::new_from(self.value.clone())
    }
}

impl<D: LatticeDef> LatticeDef for LatticeElt<D>
{
    type T = LatticeElt<D>;
    fn unit() -> Self::T {
        Self::default()
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T
    {
        lhs + rhs
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering>
    {
        D::partial_order(&lhs.value, &rhs.value)
    }
}

impl<D: LatticeDef> Copy for LatticeElt<D> where D::T: Copy {}

impl<D: LatticeDef> Hash for LatticeElt<D>
where
    D::T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state)
    }
}

impl<D: LatticeDef> Add<LatticeElt<D>> for LatticeElt<D> {
    type Output = LatticeElt<D>;
    fn add(self, other: Self) -> Self {
        LatticeElt::new_from(D::join(&self.value, &other.value))
    }
}

impl<D: LatticeDef> Add<&LatticeElt<D>> for LatticeElt<D> {
    type Output = LatticeElt<D>;
    fn add(self, other: &LatticeElt<D>) -> Self {
        LatticeElt::new_from(D::join(&self.value, &other.value))
    }
}

impl<'lhs, 'rhs, D: LatticeDef> Add<&'rhs LatticeElt<D>> for &'lhs LatticeElt<D> {
    type Output = LatticeElt<D>;
    fn add(self, other: &'rhs LatticeElt<D>) -> LatticeElt<D> {
        LatticeElt::new_from(D::join(&self.value, &other.value))
    }
}

impl<D: LatticeDef> Add<LatticeElt<D>> for &LatticeElt<D> {
    type Output = LatticeElt<D>;
    fn add(self, other: LatticeElt<D>) -> LatticeElt<D> {
        LatticeElt::new_from(D::join(&self.value, &other.value))
    }
}

impl<D: LatticeDef> Default for LatticeElt<D> {
    fn default() -> Self {
        LatticeElt { value: D::unit() }
    }
}

impl<D: LatticeDef> PartialEq for LatticeElt<D> {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

impl<D: LatticeDef> Eq for LatticeElt<D> where D::T: Eq {}

impl<D: LatticeDef> PartialOrd for LatticeElt<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        D::partial_order(&self.value, &other.value)
    }
}

impl<D: LatticeDef> Ord for LatticeElt<D>
where
    D::T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl<D: LatticeDef> LatticeElt<D> {
    pub fn new_from(t: D::T) -> Self {
        LatticeElt { value: t }
    }
    pub fn join(&self, other: &Self) -> Self {
        Self::new_from(D::join(&self.value, &other.value))
    }
}

// We cannot provide a blanket
//
//    `impl<D:LatticeDef> From<D::T> for LatticeElt<D>`
//
// because there's a blanket `From<T> for T` in libcore, and it is possible that
// `D::T` could be equal to `LatticeElt<D>`. Or at least rustc thinks so. I'm
// not smart enough to argue. But we can provide From<T> for each of the inner
// types of the specific `LatticeDef`s we define in this crate, which is close
// enough.
impl<M: Ord + Clone + MaxUnitDefault> From<M> for LatticeElt<MaxDef<M>> {
    fn from(t: M) -> Self {
        Self::new_from(t)
    }
}

impl<M: Ord + Clone + MaxUnitMinValue> From<M> for LatticeElt<MaxNum<M>> {
    fn from(t: M) -> Self {
        Self::new_from(t)
    }
}

impl<M: Ord + Clone> From<M> for LatticeElt<MinOpt<M>> {
    fn from(t: M) -> Self {
        Self::new_from(Some(t))
    }
}

impl<M: Ord + Clone + Bounded> From<M> for LatticeElt<MinNum<M>> {
    fn from(t: M) -> Self {
        Self::new_from(t)
    }
}

impl From<BitSet> for LatticeElt<BitSetWithUnion> {
    fn from(t: BitSet) -> Self {
        Self::new_from(t)
    }
}

impl From<BitSet> for LatticeElt<BitSetWithIntersection> {
    fn from(t: BitSet) -> Self {
        Self::new_from(Some(t))
    }
}

impl<K: Ord + Clone, VD: LatticeDef> From<BTreeMap<K, LatticeElt<VD>>>
    for LatticeElt<BTreeMapWithUnion<K, VD>>
where
    VD::T: Clone,
{
    fn from(t: BTreeMap<K, LatticeElt<VD>>) -> Self {
        Self::new_from(t)
    }
}

impl<K: Ord + Clone, VD: LatticeDef> From<BTreeMap<K, LatticeElt<VD>>>
    for LatticeElt<BTreeMapWithIntersection<K, VD>>
where
    VD::T: Clone,
{
    fn from(t: BTreeMap<K, LatticeElt<VD>>) -> Self {
        Self::new_from(Some(t))
    }
}

impl<U: Ord + Clone> From<BTreeSet<U>> for LatticeElt<BTreeSetWithUnion<U>> {
    fn from(t: BTreeSet<U>) -> Self {
        Self::new_from(t)
    }
}

impl<U: Ord + Clone> From<BTreeSet<U>> for LatticeElt<BTreeSetWithIntersection<U>> {
    fn from(t: BTreeSet<U>) -> Self {
        Self::new_from(Some(t))
    }
}

impl<A: LatticeDef, B: LatticeDef> From<(A::T, B::T)> for LatticeElt<Tuple2<A, B>> {
    fn from(t: (A::T, B::T)) -> Self {
        Self::new_from(t)
    }
}

impl<A: LatticeDef, B: LatticeDef, C: LatticeDef> From<(A::T, B::T, C::T)>
    for LatticeElt<Tuple3<A, B, C>>
{
    fn from(t: (A::T, B::T, C::T)) -> Self {
        Self::new_from(t)
    }
}

impl<A: LatticeDef, B: LatticeDef, C: LatticeDef, D: LatticeDef> From<(A::T, B::T, C::T, D::T)>
    for LatticeElt<Tuple4<A, B, C, D>>
{
    fn from(t: (A::T, B::T, C::T, D::T)) -> Self {
        Self::new_from(t)
    }
}

impl<A: LatticeDef, B: LatticeDef, C: LatticeDef, D: LatticeDef, E: LatticeDef>
    From<(A::T, B::T, C::T, D::T, E::T)> for LatticeElt<Tuple5<A, B, C, D, E>>
{
    fn from(t: (A::T, B::T, C::T, D::T, E::T)) -> Self {
        Self::new_from(t)
    }
}

