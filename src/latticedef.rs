// Copyright 2020 Graydon Hoare <graydon@pobox.com>
// Licensed under the MIT and Apache-2.0 licenses.

use super::LatticeElt;
use num_traits::bounds::Bounded;
use std::cmp::{Ord, Ordering, PartialOrd};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "bits")]
use bit_vec::BitVec;

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
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// `DefTraits` is used to constrain `LatticeDef`s and also type parameters of
/// structs that implement `LatticeDef`. This requires some explaining.
///
/// A `LatticeDef` is typically just a unit-struct with no content aside from
/// `PhantomData` -- it's essentially a module to be used as a parameter to a
/// `LatticeElt` -- so it doesn't obviously make sense to constrain them at
/// all.
///
/// But: since `LatticeDef`s wind up as type parameters for a variety of structs
/// in client libraries (that themselves contain `LatticeElt`s that use those
/// `LatticeDef`s), any attempt to derive standard traits on such _structs_ will
/// bump into a bug in derive -- https://github.com/rust-lang/rust/issues/26925
/// -- which prevents derived impls from working right if a struct's type
/// parameters don't themselves implement the derived traits.
///
/// So to keep derive working downstream, we insist all `LatticeDef`s provide
/// most standard derivable traits. Impls for them can be derived _on_ the
/// `LatticeDef`s trivially anyways, so this isn't much of a burden.

// The first (trait) part of this definition is a "necessary" condition to be DefTraits.
// Any time you want to be DefTraits, you must at least meet the given sum of traits.
#[cfg(feature = "serde")]
pub trait DefTraits: Debug + Ord + Clone + Hash + Default + Serialize + DeserializeOwned {}

#[cfg(not(feature = "serde"))]
pub trait DefTraits: Debug + Ord + Clone + Hash + Default {}

// The second (impl) part of this definition is a "sufficient" condition to be DefTraits.
// Any time you meet the given sum of traits, that's sufficient to be DefTraits.
#[cfg(feature = "serde")]
impl<T: Debug + Ord + Clone + Hash + Default + Serialize + DeserializeOwned> DefTraits for T {}

#[cfg(not(feature = "serde"))]
impl<T: Debug + Ord + Clone + Hash + Default> DefTraits for T {}

/// `ValTraits` is used to constrain the `LatticeDef::T` types to include basic
/// assumptions we need all datatypes to support. But notably not `Ord`! While
/// several `LatticeDef` type parameters do require `Ord` (which is because of
/// the the deriving-bug workaround described in the docs of `DefTraits`) the
/// _partial_ orders of the lattice are separate and defined by the
/// `LatticeDef`s themselves, and several important `LatticeDef::T` types are
/// not totally ordered at all (namely all the set-like and map-like ones).

#[cfg(feature = "serde")]
pub trait ValTraits: Debug + Eq + Clone + Hash + Default + Serialize + DeserializeOwned {}

#[cfg(not(feature = "serde"))]
pub trait ValTraits: Debug + Eq + Clone + Hash + Default {}

#[cfg(feature = "serde")]
impl<T: Debug + Eq + Clone + Hash + Default + Serialize + DeserializeOwned> ValTraits for T {}

#[cfg(not(feature = "serde"))]
impl<T: Debug + Eq + Clone + Hash + Default> ValTraits for T {}

/// Implement this trait on a (typically vacuous) type to define a specific
/// lattice as a type-with-some-choice-of-operators.
pub trait LatticeDef: DefTraits {
    type T: ValTraits;
    fn unit() -> Self::T;
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T;
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering>;
}

/// A marker trait here to pick out types where `Default::default` is safe to
/// use as a unit for a max-lattice. In particular it's _not_ safe in types like
/// signed integers, where there are many values less than `Default::default`.
pub trait MaxUnitDefault: Default {}
impl MaxUnitDefault for String {}
impl MaxUnitDefault for bool {}
impl MaxUnitDefault for char {}
impl MaxUnitDefault for () {}
impl MaxUnitDefault for u8 {}
impl MaxUnitDefault for u16 {}
impl MaxUnitDefault for u32 {}
impl MaxUnitDefault for u64 {}
impl MaxUnitDefault for u128 {}
impl MaxUnitDefault for &str {}
impl<T> MaxUnitDefault for &[T] {}
impl<T: MaxUnitDefault> MaxUnitDefault for Option<T> {}
impl<T: MaxUnitDefault> MaxUnitDefault for Box<[T]> {}
impl<T: MaxUnitDefault> MaxUnitDefault for Box<T> {}
impl<T: MaxUnitDefault> MaxUnitDefault for std::cell::Cell<T> {}
impl<T: MaxUnitDefault> MaxUnitDefault for std::cell::RefCell<T> {}
impl<T: MaxUnitDefault> MaxUnitDefault for std::rc::Rc<T> {}
impl<T: MaxUnitDefault> MaxUnitDefault for Vec<T> {}

/// A marker type for other types that use the `Bounded::min_value` as the unit
/// for a max-lattice.
pub trait MaxUnitMinValue: Bounded {}
impl MaxUnitMinValue for i8 {}
impl MaxUnitMinValue for i16 {}
impl MaxUnitMinValue for i32 {}
impl MaxUnitMinValue for i64 {}
impl MaxUnitMinValue for i128 {}

/// This lattice definition recycles the `Ord::max` and `Ord::cmp` of its
/// element type, as well as either `Default::default` as its unit. In other
/// words this is the "most normal" lattice over unsigned scalar, vector or
/// string types, probably the one you want most of the time.
#[cfg(feature = "serde")]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize)]
pub struct MaxDef<M: DefTraits> {
    phantom: PhantomData<M>,
}
#[cfg(not(feature = "serde"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct MaxDef<M: DefTraits> {
    phantom: PhantomData<M>,
}
impl<M: DefTraits + MaxUnitDefault> LatticeDef for MaxDef<M> {
    type T = M;
    fn unit() -> Self::T {
        M::default()
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        lhs.clone().max(rhs.clone())
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
        Some(lhs.cmp(rhs))
    }
}

/// This lattice definition recycles the `Ord::max` and `Ord::cmp` of its
/// element type, as well as `Bounded::min_value` as its unit. This is
/// similar to `MaxDef` except it works with signed types.
#[cfg(feature = "serde")]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize)]
pub struct MaxNum<M: DefTraits> {
    phantom: PhantomData<M>,
}
#[cfg(not(feature = "serde"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct MaxNum<M: DefTraits> {
    phantom: PhantomData<M>,
}
impl<M: DefTraits + MaxUnitMinValue> LatticeDef for MaxNum<M> {
    type T = M;
    fn unit() -> Self::T {
        M::min_value()
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        lhs.clone().max(rhs.clone())
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
        Some(lhs.cmp(rhs))
    }
}

/// This lattice is _similar_ to MaxDef but inverts the order, with the minimal
/// value according to `Ord::cmp` as its join, and the unit being a putative
/// "maximal" value of the element type. Since several Ord types do not _have_ a
/// maximal value (think strings, maps, etc.) `MinOpt` represents its element
/// using an Option<M> where None is the "maximal" value (that forms the lattice
/// unit) and Some(M) is for the rest.
///
/// Note this may not be quite what you want if your type _does_ have a maximal
/// element. For example this will make the unit of u32 still be None, not
/// u32::MAX. For those, use MinNum. Both are _safe_, but MinOpt is weird in
/// those cases.
#[cfg(feature = "serde")]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize)]
pub struct MinOpt<M: DefTraits> {
    phantom: PhantomData<M>,
}
#[cfg(not(feature = "serde"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct MinOpt<M: DefTraits> {
    phantom: PhantomData<M>,
}
impl<M: DefTraits> LatticeDef for MinOpt<M> {
    type T = Option<M>;
    fn unit() -> Self::T {
        None
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        match (lhs, rhs) {
            (None, None) => None,
            (Some(_), None) => lhs.clone(),
            (None, Some(_)) => rhs.clone(),
            (Some(a), Some(b)) => Some(a.clone().min(b.clone())),
        }
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
        match (lhs, rhs) {
            (None, None) => Some(Ordering::Equal),
            // NB: None is a putative "maximal element" for the underlying
            // "natural" order of the representation type, but this lattice
            // inverts the natural order (taking join as min) so None becomes
            // _minimal_ in the lattice's join-induced order.
            (None, Some(_)) => Some(Ordering::Less),
            (Some(_), None) => Some(Ordering::Greater),
            // Again: we invert the natural order in this lattice, so a<=b
            // iff b<=a in the underlying Ord-presented order.
            (Some(a), Some(b)) => Some(b.cmp(a)),
        }
    }
}

/// This is like `MinOpt` but for numeric (or specifically `Bounded`) types
/// that have a numeric upper bound: it uses that as the unit rather than
/// the additional "maximal value" tacked on in `MinOpt`. Best option for
/// numeric lattices with join as minimum.
#[cfg(feature = "serde")]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize)]
pub struct MinNum<M: DefTraits> {
    phantom: PhantomData<M>,
}
#[cfg(not(feature = "serde"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct MinNum<M: DefTraits> {
    phantom: PhantomData<M>,
}
impl<M: DefTraits + Bounded> LatticeDef for MinNum<M> {
    type T = M;
    fn unit() -> Self::T {
        M::max_value()
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        lhs.clone().min(rhs.clone())
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
        Some(rhs.cmp(lhs))
    }
}

/// Wrap a BitSet in a newtype so we can implement serde traits on it
/// (weirdly by delegating _to_ its inner BitVec).
#[cfg(feature = "bits")]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct BitSetWrapper(pub BitSet);

#[cfg(all(feature = "bits", feature = "serde"))]
impl Serialize for BitSetWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        self.0.get_ref().serialize(serializer)
    }
}

#[cfg(all(feature = "bits", feature = "serde"))]
impl<'a> Deserialize<'a> for BitSetWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::de::Deserializer<'a>,
    {
        let v = BitVec::deserialize(deserializer);
        match v {
            Ok(bv) => Ok(BitSetWrapper(BitSet::from_bit_vec(bv))),
            Err(e) => Err(e),
        }
    }
}

/// This lattice is a standard bitset-with-union.
///
/// Note: you _could_ use a `BitSet` in the `MaxStd` or `MinOpt` lattices
/// (`BitSet` satisfies the bounds) but the "set semantics" you usually want in
/// a set-of-sets lattice aren't achieved that way: the `Ord`-provided order on
/// `BitSet` is a _lexicographical total order_ on the _sequence_ of bits,
/// rather than set-theoretic sub/superset relation (which is only a partial
/// order), and of course joining by max (or min) of that order will not produce
/// a union (or intersection) as one would want.
#[cfg(all(feature = "bits", feature = "serde"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize)]
pub struct BitSetWithUnion;

#[cfg(all(feature = "bits", not(feature = "serde")))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct BitSetWithUnion;

#[cfg(feature = "bits")]
impl LatticeDef for BitSetWithUnion {
    type T = BitSetWrapper;
    fn unit() -> Self::T {
        BitSetWrapper(BitSet::default())
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        BitSetWrapper(lhs.0.union(&rhs.0).collect())
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
        if lhs.0 == rhs.0 {
            Some(Ordering::Equal)
        } else if lhs.0.is_subset(&rhs.0) {
            Some(Ordering::Less)
        } else if lhs.0.is_superset(&rhs.0) {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}

/// This lattice is a standard bitset-with-intersection.
///
/// As with `BitSetWithUnion`, this is a lattice over `BitSet` with
/// set-semantics rather than the lexicographical-total-order provided by the
/// `Ord` implementation on `BitSet`. And as with `MinOpt`, this provides a
/// putative "maximal value" for the underlying type (a superset of any actual
/// `Bitset`) as well as a join that inverts the typical order of a set-valued
/// lattice, taking set-intersections from the "maximal" unit upwards towards
/// the empty set (at the top of the lattice).
#[cfg(all(feature = "bits", feature = "serde"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize)]
pub struct BitSetWithIntersection;

#[cfg(all(feature = "bits", not(feature = "serde")))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct BitSetWithIntersection;

#[cfg(feature = "bits")]
impl LatticeDef for BitSetWithIntersection {
    type T = Option<BitSetWrapper>;
    fn unit() -> Self::T {
        None
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        match (lhs, rhs) {
            (None, None) => None,
            (None, Some(_)) => rhs.clone(),
            (Some(_), None) => lhs.clone(),
            (Some(a), Some(b)) => Some(BitSetWrapper(a.0.intersection(&b.0).collect())),
        }
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
        match (lhs, rhs) {
            (None, None) => Some(Ordering::Equal),
            (None, Some(_)) => Some(Ordering::Less),
            (Some(_), None) => Some(Ordering::Greater),
            (Some(a), Some(b)) => {
                if a.0 == b.0 {
                    Some(Ordering::Equal)
                } else if a.0.is_subset(&b.0) {
                    Some(Ordering::Greater)
                } else if b.0.is_subset(&a.0) {
                    Some(Ordering::Less)
                } else {
                    None
                }
            }
        }
    }
}

macro_rules! impl_map_with_union {
    ($LDef:ident, $Map:ident) => {
        /// This is a lattice for maps that contain other lattices as
        /// values. The join operator takes the union of (key, value) pairs for
        /// keys present in only one map -- equivalent to an elementwise
        /// join-with-unit -- and the elementwise join of values for keys that
        /// exist in both maps.
        ///
        /// As with `BitSet`, this avoids the typical _lexicographic_ order on
        /// maps in favour of the join-induced partial order: a subset relation
        /// extended with the lattice orders of the values when the same key is
        /// present in both maps.
        #[cfg(feature = "serde")]
        #[derive(
            Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize,
        )]
        pub struct $LDef<K: DefTraits, VD: LatticeDef> {
            phantom1: PhantomData<K>,
            phantom2: PhantomData<VD>,
        }
        #[cfg(not(feature = "serde"))]
        #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
        pub struct $LDef<K: DefTraits, VD: LatticeDef> {
            phantom1: PhantomData<K>,
            phantom2: PhantomData<VD>,
        }
        impl<K: DefTraits, VD: LatticeDef> LatticeDef for $LDef<K, VD>
        where
            VD::T: Clone,
        {
            type T = $Map<K, LatticeElt<VD>>;
            fn unit() -> Self::T {
                $Map::default()
            }
            fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
                let mut tmp: Self::T = (*lhs).clone();
                for (k, v) in rhs.iter() {
                    tmp.entry(k.clone())
                        .and_modify(|e| *e = e.join(v))
                        .or_insert(v.clone());
                }
                tmp
            }
            fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
                // This is a complicated partial order: lhs <= rhs if lhs has a subset
                // of the keys in rhs _and_ every lhs value of every common key is <=
                // the rhs value. If common-key values are ordered with any mix of
                // greater or lesser, or if any values on common keys are unordered, the
                // maps are unordered.
                let mut lhs_lt_rhs_at_some_key = false;
                let mut rhs_lt_lhs_at_some_key = false;
                for (k, lv) in lhs.iter() {
                    match rhs.get(k) {
                        None => rhs_lt_lhs_at_some_key = true,
                        Some(rv) => match lv.partial_cmp(rv) {
                            Some(Ordering::Equal) => (),
                            Some(Ordering::Less) => lhs_lt_rhs_at_some_key = true,
                            Some(Ordering::Greater) => rhs_lt_lhs_at_some_key = true,
                            None => return None,
                        },
                    }
                }
                for (k, rv) in rhs.iter() {
                    match lhs.get(k) {
                        None => lhs_lt_rhs_at_some_key = true,
                        Some(lv) => match lv.partial_cmp(rv) {
                            Some(Ordering::Equal) => (),
                            Some(Ordering::Less) => lhs_lt_rhs_at_some_key = true,
                            Some(Ordering::Greater) => rhs_lt_lhs_at_some_key = true,
                            None => return None,
                        },
                    }
                }
                match (lhs_lt_rhs_at_some_key, rhs_lt_lhs_at_some_key) {
                    (false, false) => Some(Ordering::Equal),
                    (true, false) => Some(Ordering::Less),
                    (false, true) => Some(Ordering::Greater),
                    (true, true) => None,
                }
            }
        }
    };
}

impl_map_with_union!(BTreeMapWithUnion, BTreeMap);

#[cfg(feature = "im")]
impl_map_with_union!(ArcOrdMapWithUnion, ArcOrdMap);

#[cfg(feature = "im-rc")]
impl_map_with_union!(RcOrdMapWithUnion, RcOrdMap);

macro_rules! impl_map_with_intersection {
    ($LDef:ident, $Map:ident) => {
        /// Similar to other intersection-based lattices in this crate, this
        /// lattice is a map that stores inner lattices and joins using
        /// intersection. Maps are represented as `Option<BTreeMap>` and the
        /// unit is again a putative "maximum" map-with-all-possible-keys
        /// (represented by `None`).
        #[cfg(feature = "serde")]
        #[derive(
            Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize,
        )]
        pub struct $LDef<K: DefTraits, VD: LatticeDef> {
            phantom1: PhantomData<K>,
            phantom2: PhantomData<VD>,
        }
        #[cfg(not(feature = "serde"))]
        #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
        pub struct $LDef<K: DefTraits, VD: LatticeDef> {
            phantom1: PhantomData<K>,
            phantom2: PhantomData<VD>,
        }
        impl<K: DefTraits, VD: LatticeDef> LatticeDef for $LDef<K, VD>
        where
            VD::T: Clone,
        {
            type T = Option<$Map<K, LatticeElt<VD>>>;
            fn unit() -> Self::T {
                None
            }
            fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
                match (lhs, rhs) {
                    (None, None) => None,
                    (Some(_), None) => lhs.clone(),
                    (None, Some(_)) => rhs.clone(),
                    (Some(lmap), Some(rmap)) => {
                        let mut tmp = $Map::<K, LatticeElt<VD>>::default();
                        for (k, lv) in lmap.iter() {
                            match rmap.get(k) {
                                None => (),
                                Some(rv) => {
                                    tmp.insert(k.clone(), lv.join(rv));
                                }
                            }
                        }
                        Some(tmp)
                    }
                }
            }
            fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
                // This is a complicated partial order: lhs <= rhs if lhs has a
                // superset of the keys in rhs _and_ every lhs value of every
                // common key is <= the rhs value. If common-key values are
                // ordered with any mix of greater or lesser, or if any values
                // on common keys are unordered, the maps are unordered.
                match (lhs, rhs) {
                    (None, None) => Some(Ordering::Equal),

                    // The None element is the unit, the
                    // map-with-all-possible-values, which is less-than all
                    // other maps in the intersection-based partial order.
                    (None, Some(_)) => Some(Ordering::Less),
                    (Some(_), None) => Some(Ordering::Greater),

                    // When we have two maps with definite subsets-of-all-keys,
                    // we look at them element-wise.
                    (Some(lmap), Some(rmap)) => {
                        let mut lhs_lt_rhs_at_some_key = false;
                        let mut rhs_lt_lhs_at_some_key = false;
                        for (k, lv) in lmap.iter() {
                            match rmap.get(k) {
                                // If lmap has a value and rmap hasn't, lmap is
                                // "less than" (has more values than) rmap in
                                // the intersection partial order. This is the
                                // opposite interpretation of present-vs-absent
                                // keys as we have above in the union map code.
                                None => lhs_lt_rhs_at_some_key = true,
                                Some(rv) => {
                                    // When we have keys in both maps, we defer
                                    // to the partial order of the values. Note
                                    // that we do _not_ invert the partial order
                                    // among the values here, so this branch
                                    // contains the same code as above in the
                                    // union map code.
                                    match lv.partial_cmp(rv) {
                                        Some(Ordering::Equal) => (),
                                        Some(Ordering::Less) => lhs_lt_rhs_at_some_key = true,
                                        Some(Ordering::Greater) => rhs_lt_lhs_at_some_key = true,
                                        None => return None,
                                    }
                                }
                            }
                        }
                        for (k, rv) in rmap.iter() {
                            match lmap.get(k) {
                                None => rhs_lt_lhs_at_some_key = true,
                                Some(lv) => match lv.partial_cmp(rv) {
                                    Some(Ordering::Equal) => (),
                                    Some(Ordering::Less) => lhs_lt_rhs_at_some_key = true,
                                    Some(Ordering::Greater) => rhs_lt_lhs_at_some_key = true,
                                    None => return None,
                                },
                            }
                        }
                        match (lhs_lt_rhs_at_some_key, rhs_lt_lhs_at_some_key) {
                            (false, false) => Some(Ordering::Equal),
                            (true, false) => Some(Ordering::Less),
                            (false, true) => Some(Ordering::Greater),
                            (true, true) => None,
                        }
                    }
                }
            }
        }
    };
}

impl_map_with_intersection!(BTreeMapWithIntersection, BTreeMap);

#[cfg(feature = "im")]
impl_map_with_intersection!(ArcOrdMapWithIntersection, ArcOrdMap);

#[cfg(feature = "im-rc")]
impl_map_with_intersection!(RcOrdMapWithIntersection, RcOrdMap);

#[cfg(any(feature = "im", feature = "im-rc"))]
macro_rules! impl_im_set_with_union {
    ($LDef:ident, $Set:ident) => {
        /// This is the same semantics as the `BitSetWithUnion` lattice, but
        /// covering sets of arbitrary ordered values.
        #[cfg(feature = "serde")]
        #[derive(
            Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize,
        )]
        pub struct $LDef<U: DefTraits> {
            phantom: PhantomData<U>,
        }
        #[cfg(not(feature = "serde"))]
        #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
        pub struct $LDef<U: DefTraits> {
            phantom: PhantomData<U>,
        }
        impl<U: DefTraits> LatticeDef for $LDef<U> {
            type T = $Set<U>;
            fn unit() -> Self::T {
                $Set::default()
            }
            fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
                lhs.clone().union(rhs.clone())
            }
            fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
                if lhs == rhs {
                    Some(Ordering::Equal)
                } else if lhs.is_subset(rhs) {
                    Some(Ordering::Less)
                } else if rhs.is_subset(lhs) {
                    Some(Ordering::Greater)
                } else {
                    None
                }
            }
        }
    };
}

#[cfg(feature = "im")]
impl_im_set_with_union!(ArcOrdSetWithUnion, ArcOrdSet);

#[cfg(feature = "im-rc")]
impl_im_set_with_union!(RcOrdSetWithUnion, RcOrdSet);

/// This is the same semantics as the `BitSetWithUnion` lattice, but covering
/// sets of arbitrary ordered values.
#[cfg(feature = "serde")]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize)]
pub struct BTreeSetWithUnion<U: DefTraits> {
    phantom: PhantomData<U>,
}
#[cfg(not(feature = "serde"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct BTreeSetWithUnion<U: DefTraits> {
    phantom: PhantomData<U>,
}
impl<U: DefTraits> LatticeDef for BTreeSetWithUnion<U> {
    type T = BTreeSet<U>;
    fn unit() -> Self::T {
        BTreeSet::default()
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        lhs.union(rhs).cloned().collect()
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
        if lhs == rhs {
            Some(Ordering::Equal)
        } else if lhs.is_subset(rhs) {
            Some(Ordering::Less)
        } else if lhs.is_superset(rhs) {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}

#[cfg(any(feature = "im", feature = "im-rc"))]
macro_rules! impl_im_set_with_intersection {
    ($LDef:ident, $Set:ident) => {
        /// This is the same semantics as the `BitSetWithIntersection` lattice, but
        /// covering sets of arbitrary ordered values.
        #[cfg(feature = "serde")]
        #[derive(
            Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize,
        )]
        pub struct $LDef<U: DefTraits> {
            phantom: PhantomData<U>,
        }
        #[cfg(not(feature = "serde"))]
        #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
        pub struct $LDef<U: DefTraits> {
            phantom: PhantomData<U>,
        }
        impl<U: DefTraits> LatticeDef for $LDef<U> {
            type T = Option<$Set<U>>;
            fn unit() -> Self::T {
                None
            }
            fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
                match (lhs, rhs) {
                    (None, None) => None,
                    (None, Some(_)) => rhs.clone(),
                    (Some(_), None) => lhs.clone(),
                    (Some(a), Some(b)) => Some(a.clone().intersection(b.clone())),
                }
            }
            fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
                match (lhs, rhs) {
                    (None, None) => Some(Ordering::Equal),
                    (None, Some(_)) => Some(Ordering::Less),
                    (Some(_), None) => Some(Ordering::Greater),
                    (Some(a), Some(b)) => {
                        if a == b {
                            Some(Ordering::Equal)
                        } else if a.is_subset(b) {
                            Some(Ordering::Greater)
                        } else if b.is_subset(a) {
                            Some(Ordering::Less)
                        } else {
                            None
                        }
                    }
                }
            }
        }
    };
}

#[cfg(feature = "im")]
impl_im_set_with_intersection!(ArcOrdSetWithIntersection, ArcOrdSet);

#[cfg(feature = "im-rc")]
impl_im_set_with_intersection!(RcOrdSetWithIntersection, RcOrdSet);

/// This is the same semantics as the `BitSetWithIntersection` lattice, but
/// covering sets of arbitrary ordered values.
#[cfg(feature = "serde")]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize)]
pub struct BTreeSetWithIntersection<U: DefTraits> {
    phantom: PhantomData<U>,
}
#[cfg(not(feature = "serde"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct BTreeSetWithIntersection<U: DefTraits> {
    phantom: PhantomData<U>,
}
impl<U: DefTraits> LatticeDef for BTreeSetWithIntersection<U> {
    type T = Option<BTreeSet<U>>;
    fn unit() -> Self::T {
        None
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        match (lhs, rhs) {
            (None, None) => None,
            (None, Some(_)) => rhs.clone(),
            (Some(_), None) => lhs.clone(),
            (Some(a), Some(b)) => Some(a.intersection(b).cloned().collect()),
        }
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
        match (lhs, rhs) {
            (None, None) => Some(Ordering::Equal),
            (None, Some(_)) => Some(Ordering::Less),
            (Some(_), None) => Some(Ordering::Greater),
            (Some(a), Some(b)) => {
                if a == b {
                    Some(Ordering::Equal)
                } else if a.is_subset(b) {
                    Some(Ordering::Greater)
                } else if a.is_superset(b) {
                    Some(Ordering::Less)
                } else {
                    None
                }
            }
        }
    }
}

/// Cartesian product lattices or 2, 3, 4, 5 inner lattices. Join joins elements
/// pairwise, order is the product order (_not_ lexicographical order) i.e. where
/// (a, b) <= (c, d) iff a <= c _and_ b <= d.
///
/// If you need more than 5-element tuples, maybe just nest these (or submit a
/// pull request).
#[cfg(feature = "serde")]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize)]
pub struct Tuple2<A: LatticeDef, B: LatticeDef> {
    phantom1: PhantomData<A>,
    phantom2: PhantomData<B>,
}
#[cfg(not(feature = "serde"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct Tuple2<A: LatticeDef, B: LatticeDef> {
    phantom1: PhantomData<A>,
    phantom2: PhantomData<B>,
}
impl<A: LatticeDef, B: LatticeDef> LatticeDef for Tuple2<A, B> {
    type T = (A::T, B::T);
    fn unit() -> Self::T {
        (A::unit(), B::unit())
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        let (la, lb) = lhs;
        let (ra, rb) = rhs;
        (A::join(la, ra), B::join(lb, rb))
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
        let (la, lb) = lhs;
        let (ra, rb) = rhs;
        match (A::partial_order(la, ra), B::partial_order(lb, rb)) {
            (Some(a), Some(b)) if a == b => Some(a),
            _ => None,
        }
    }
}

#[cfg(feature = "serde")]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize)]
pub struct Tuple3<A: LatticeDef, B: LatticeDef, C: LatticeDef> {
    phantom1: PhantomData<A>,
    phantom2: PhantomData<B>,
    phantom3: PhantomData<C>,
}
#[cfg(not(feature = "serde"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct Tuple3<A: LatticeDef, B: LatticeDef, C: LatticeDef> {
    phantom1: PhantomData<A>,
    phantom2: PhantomData<B>,
    phantom3: PhantomData<C>,
}
impl<A: LatticeDef, B: LatticeDef, C: LatticeDef> LatticeDef for Tuple3<A, B, C> {
    type T = (A::T, B::T, C::T);
    fn unit() -> Self::T {
        (A::unit(), B::unit(), C::unit())
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        let (la, lb, lc) = lhs;
        let (ra, rb, rc) = rhs;
        (A::join(la, ra), B::join(lb, rb), C::join(lc, rc))
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
        let (la, lb, lc) = lhs;
        let (ra, rb, rc) = rhs;
        match (
            A::partial_order(la, ra),
            B::partial_order(lb, rb),
            C::partial_order(lc, rc),
        ) {
            (Some(a), Some(b), Some(c)) if a == b && b == c => Some(a),
            _ => None,
        }
    }
}

#[cfg(feature = "serde")]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize)]
pub struct Tuple4<A: LatticeDef, B: LatticeDef, C: LatticeDef, D: LatticeDef> {
    phantom1: PhantomData<A>,
    phantom2: PhantomData<B>,
    phantom3: PhantomData<C>,
    phantom4: PhantomData<D>,
}
#[cfg(not(feature = "serde"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct Tuple4<A: LatticeDef, B: LatticeDef, C: LatticeDef, D: LatticeDef> {
    phantom1: PhantomData<A>,
    phantom2: PhantomData<B>,
    phantom3: PhantomData<C>,
    phantom4: PhantomData<D>,
}
impl<A: LatticeDef, B: LatticeDef, C: LatticeDef, D: LatticeDef> LatticeDef for Tuple4<A, B, C, D> {
    type T = (A::T, B::T, C::T, D::T);
    fn unit() -> Self::T {
        (A::unit(), B::unit(), C::unit(), D::unit())
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        let (la, lb, lc, ld) = lhs;
        let (ra, rb, rc, rd) = rhs;
        (
            A::join(la, ra),
            B::join(lb, rb),
            C::join(lc, rc),
            D::join(ld, rd),
        )
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
        let (la, lb, lc, ld) = lhs;
        let (ra, rb, rc, rd) = rhs;
        match (
            A::partial_order(la, ra),
            B::partial_order(lb, rb),
            C::partial_order(lc, rc),
            D::partial_order(ld, rd),
        ) {
            (Some(a), Some(b), Some(c), Some(d)) if a == b && b == c && c == d => Some(a),
            _ => None,
        }
    }
}

#[cfg(feature = "serde")]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default, Serialize, Deserialize)]
pub struct Tuple5<A: LatticeDef, B: LatticeDef, C: LatticeDef, D: LatticeDef, E: LatticeDef> {
    phantom1: PhantomData<A>,
    phantom2: PhantomData<B>,
    phantom3: PhantomData<C>,
    phantom4: PhantomData<D>,
    phantom5: PhantomData<E>,
}
#[cfg(not(feature = "serde"))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Default)]
pub struct Tuple5<A: LatticeDef, B: LatticeDef, C: LatticeDef, D: LatticeDef, E: LatticeDef> {
    phantom1: PhantomData<A>,
    phantom2: PhantomData<B>,
    phantom3: PhantomData<C>,
    phantom4: PhantomData<D>,
    phantom5: PhantomData<E>,
}
impl<A: LatticeDef, B: LatticeDef, C: LatticeDef, D: LatticeDef, E: LatticeDef> LatticeDef
    for Tuple5<A, B, C, D, E>
{
    type T = (A::T, B::T, C::T, D::T, E::T);
    fn unit() -> Self::T {
        (A::unit(), B::unit(), C::unit(), D::unit(), E::unit())
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        let (la, lb, lc, ld, le) = lhs;
        let (ra, rb, rc, rd, re) = rhs;
        (
            A::join(la, ra),
            B::join(lb, rb),
            C::join(lc, rc),
            D::join(ld, rd),
            E::join(le, re),
        )
    }
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering> {
        let (la, lb, lc, ld, le) = lhs;
        let (ra, rb, rc, rd, re) = rhs;
        match (
            A::partial_order(la, ra),
            B::partial_order(lb, rb),
            C::partial_order(lc, rc),
            D::partial_order(ld, rd),
            E::partial_order(le, re),
        ) {
            (Some(a), Some(b), Some(c), Some(d), Some(e))
                if a == b && b == c && c == d && d == e =>
            {
                Some(a)
            }
            _ => None,
        }
    }
}
