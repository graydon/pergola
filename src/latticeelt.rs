// Copyright 2020 Graydon Hoare <graydon@pobox.com>
// Licensed under the MIT and Apache-2.0 licenses.

use super::*;
use crate::latticedef::DefTraits;
use num_traits::bounds::Bounded;
use std::cmp::{Eq, Ord, Ordering, PartialOrd};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::ops::Add;

#[cfg(feature = "bits")]
use crate::latticedef::BitSetWrapper;

#[cfg(feature = "bits")]
use bit_set::BitSet;

#[cfg(feature = "im")]
use im::OrdMap as ArcOrdMap;

#[cfg(feature = "im")]
use im::OrdSet as ArcOrdSet;

#[cfg(feature = "im-rc")]
use im_rc::OrdMap as RcOrdMap;

#[cfg(feature = "im-rc")]
use im_rc::OrdSet as RcOrdSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Write code that _uses_ lattices over this type, and it will delegate
/// to the functions of the parameter `LatticeDef`.

#[cfg(feature = "serde")]
#[derive(Debug, Serialize, Deserialize)]
pub struct LatticeElt<D: LatticeDef> {
    pub value: D::T,
}

#[cfg(not(feature = "serde"))]
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

/// Trait to extract the Def back out of a given LatticeElt.
pub trait EltDef {
    type Def: LatticeDef;
}

impl<D: LatticeDef> EltDef for LatticeElt<D> {
    type Def = D;
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
impl<M: DefTraits + MaxUnitDefault> From<M> for LatticeElt<MaxDef<M>> {
    fn from(t: M) -> Self {
        Self::new_from(t)
    }
}

impl<M: DefTraits + MaxUnitMinValue> From<M> for LatticeElt<MaxNum<M>> {
    fn from(t: M) -> Self {
        Self::new_from(t)
    }
}

impl<M: DefTraits> From<M> for LatticeElt<MinOpt<M>> {
    fn from(t: M) -> Self {
        Self::new_from(Some(t))
    }
}

impl<M: DefTraits + Bounded> From<M> for LatticeElt<MinNum<M>> {
    fn from(t: M) -> Self {
        Self::new_from(t)
    }
}

#[cfg(feature = "bits")]
impl From<BitSet> for LatticeElt<BitSetWithUnion> {
    fn from(t: BitSet) -> Self {
        Self::new_from(BitSetWrapper(t))
    }
}

#[cfg(feature = "bits")]
impl From<BitSet> for LatticeElt<BitSetWithIntersection> {
    fn from(t: BitSet) -> Self {
        Self::new_from(Some(BitSetWrapper(t)))
    }
}

impl<K: DefTraits, VD: LatticeDef> From<BTreeMap<K, LatticeElt<VD>>>
    for LatticeElt<BTreeMapWithUnion<K, VD>>
where
    VD::T: Clone,
{
    fn from(t: BTreeMap<K, LatticeElt<VD>>) -> Self {
        Self::new_from(t)
    }
}

#[cfg(feature = "im")]
impl<K: DefTraits, VD: LatticeDef> From<ArcOrdMap<K, LatticeElt<VD>>>
    for LatticeElt<ArcOrdMapWithUnion<K, VD>>
where
    VD::T: Clone,
{
    fn from(t: ArcOrdMap<K, LatticeElt<VD>>) -> Self {
        Self::new_from(t)
    }
}

#[cfg(feature = "im")]
impl<K: DefTraits, VD: LatticeDef> From<ArcOrdMap<K, LatticeElt<VD>>>
    for LatticeElt<ArcOrdMapWithIntersection<K, VD>>
where
    VD::T: Clone,
{
    fn from(t: ArcOrdMap<K, LatticeElt<VD>>) -> Self {
        Self::new_from(Some(t))
    }
}

#[cfg(feature = "im-rc")]
impl<K: DefTraits, VD: LatticeDef> From<RcOrdMap<K, LatticeElt<VD>>>
    for LatticeElt<RcOrdMapWithUnion<K, VD>>
where
    VD::T: Clone,
{
    fn from(t: RcOrdMap<K, LatticeElt<VD>>) -> Self {
        Self::new_from(t)
    }
}

#[cfg(feature = "im-rc")]
impl<K: DefTraits, VD: LatticeDef> From<RcOrdMap<K, LatticeElt<VD>>>
    for LatticeElt<RcOrdMapWithIntersection<K, VD>>
where
    VD::T: Clone,
{
    fn from(t: RcOrdMap<K, LatticeElt<VD>>) -> Self {
        Self::new_from(Some(t))
    }
}

impl<K: DefTraits, VD: LatticeDef> From<BTreeMap<K, LatticeElt<VD>>>
    for LatticeElt<BTreeMapWithIntersection<K, VD>>
where
    VD::T: Clone,
{
    fn from(t: BTreeMap<K, LatticeElt<VD>>) -> Self {
        Self::new_from(Some(t))
    }
}

impl<U: DefTraits> From<BTreeSet<U>> for LatticeElt<BTreeSetWithUnion<U>> {
    fn from(t: BTreeSet<U>) -> Self {
        Self::new_from(t)
    }
}

impl<U: DefTraits> From<BTreeSet<U>> for LatticeElt<BTreeSetWithIntersection<U>> {
    fn from(t: BTreeSet<U>) -> Self {
        Self::new_from(Some(t))
    }
}

#[cfg(feature = "im")]
impl<U: DefTraits> From<ArcOrdSet<U>> for LatticeElt<ArcOrdSetWithUnion<U>> {
    fn from(t: ArcOrdSet<U>) -> Self {
        Self::new_from(t)
    }
}

#[cfg(feature = "im")]
impl<U: DefTraits> From<ArcOrdSet<U>> for LatticeElt<ArcOrdSetWithIntersection<U>> {
    fn from(t: ArcOrdSet<U>) -> Self {
        Self::new_from(Some(t))
    }
}

#[cfg(feature = "im-rc")]
impl<U: DefTraits> From<RcOrdSet<U>> for LatticeElt<RcOrdSetWithUnion<U>> {
    fn from(t: RcOrdSet<U>) -> Self {
        Self::new_from(t)
    }
}

#[cfg(feature = "im-rc")]
impl<U: DefTraits> From<RcOrdSet<U>> for LatticeElt<RcOrdSetWithIntersection<U>> {
    fn from(t: RcOrdSet<U>) -> Self {
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
