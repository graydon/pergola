/*!

This is a small crate that provides a generic unital-join-semilattice type
(hereafter: "lattice") along with a few common instances.

Lattices are defined in two separate pieces: a _definition_ trait `LatticeDef`
that provides the type-and-functions for a given lattice and a _user interface_
struct `LatticeElt` that's parameterized by a `LatticeDef` and provides
convenient methods to work with (including impls of standard Rust operator
traits).

This unusual split exists because many types have multiple equally viable
lattices you can build on them (eg. u32-with-min or u32-with-max) and we want to
avoid both coupling any given lattice definition to the type _or_ accidentally
inheriting an impl for any of the type's "standard semantics" as the lattice
semantics, eg. we don't want to inherit u32's standard partial order as any
lattice's partial order, unless explicitly building such a lattice.

*/

// TODO: Maybe add hash maps and sets
// TODO: Maybe split struct into struct + trait
// TODO: Maybe split such trait into lattices with/without unit, upper bound

use bit_set::BitSet;
use num_traits::bounds::Bounded;
use std::cmp::{Eq, Ord, Ordering, PartialOrd};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Add;

/// Implement this trait on a (typically vacuous) type to define a specific
/// lattice as a type-with-some-choice-of-operators.
pub trait LatticeDef {
    type T;
    fn unit() -> Self::T;
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T;
    fn partial_order(lhs: &Self::T, rhs: &Self::T) -> Option<Ordering>;
}

/// Write code that _uses_ lattices over this type, and it will delegate
/// to the functions of the parameter `LatticeDef`.
#[derive(Debug)]
pub struct LatticeElt<D: LatticeDef> {
    value: D::T,
}

impl<D: LatticeDef> Clone for LatticeElt<D>
where
    D::T: Clone,
{
    fn clone(&self) -> Self {
        LatticeElt::new_from(self.value.clone())
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
    fn new_from(t: D::T) -> Self {
        LatticeElt { value: t }
    }
    fn join(&self, other: &Self) -> Self {
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
#[derive(Debug)]
pub struct MaxDef<M> {
    phantom: PhantomData<M>,
}
impl<M: Ord + Clone + MaxUnitDefault> LatticeDef for MaxDef<M> {
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
#[derive(Debug)]
pub struct MaxNum<M> {
    phantom: PhantomData<M>,
}
impl<M: Ord + Clone + MaxUnitMinValue> LatticeDef for MaxNum<M> {
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
#[derive(Debug)]
pub struct MinOpt<M> {
    phantom: PhantomData<M>,
}
impl<M: Ord + Clone> LatticeDef for MinOpt<M> {
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
#[derive(Debug)]
pub struct MinNum<M> {
    phantom: PhantomData<M>,
}
impl<M: Ord + Clone + Bounded> LatticeDef for MinNum<M> {
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

/// This lattice is a standard bitset-with-union.
///
/// Note: you _could_ use a `BitSet` in the `MaxStd` or `MinOpt` lattices
/// (`BitSet` satisfies the bounds) but the "set semantics" you usually want in
/// a set-of-sets lattice aren't achieved that way: the `Ord`-provided order on
/// `BitSet` is a _lexicographical total order_ on the _sequence_ of bits,
/// rather than set-theoretic sub/superset relation (which is only a partial
/// order), and of course joining by max (or min) of that order will not produce
/// a union (or intersection) as one would want.
#[derive(Debug)]
pub struct BitSetWithUnion;
impl LatticeDef for BitSetWithUnion {
    type T = BitSet;
    fn unit() -> Self::T {
        BitSet::default()
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        lhs.union(rhs).collect()
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

/// This lattice is a standard bitset-with-intersection.
///
/// As with `BitSetWithUnion`, this is a lattice over `BitSet` with
/// set-semantics rather than the lexicographical-total-order provided by the
/// `Ord` implementation on `BitSet`. And as with `MinOpt`, this provides a
/// putative "maximal value" for the underlying type (a superset of any actual
/// `Bitset`) as well as a join that inverts the typical order of a set-valued
/// lattice, taking set-intersections from the "maximal" unit upwards towards
/// the empty set (at the top of the lattice).
#[derive(Debug)]
pub struct BitSetWithIntersection;
impl LatticeDef for BitSetWithIntersection {
    type T = Option<BitSet>;
    fn unit() -> Self::T {
        None
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        match (lhs, rhs) {
            (None, None) => None,
            (None, Some(_)) => rhs.clone(),
            (Some(_), None) => lhs.clone(),
            (Some(a), Some(b)) => Some(a.intersection(b).collect()),
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

/// This is a lattice for maps that contain other lattices as values. The join
/// operator takes the union of (key, value) pairs for keys present in only one
/// map -- equivalent to an elementwise join-with-unit -- and the elementwise
/// join of values for keys that exist in both maps.
///
/// As with `BitSet`, this avoids the typical _lexicographic_ order on maps in
/// favour of the join-induced partial order: a subset relation extended with
/// the lattice orders of the values when the same key is present in both maps.
#[derive(Debug)]
pub struct BTreeMapWithUnion<K: Ord + Clone, VD: LatticeDef> {
    phantom1: PhantomData<K>,
    phantom2: PhantomData<VD>,
}
impl<K: Ord + Clone, VD: LatticeDef> LatticeDef for BTreeMapWithUnion<K, VD>
where
    VD::T: Clone,
{
    type T = BTreeMap<K, LatticeElt<VD>>;
    fn unit() -> Self::T {
        BTreeMap::default()
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

/// Similar to other intersection-based lattices in this crate, this lattice is
/// a map that stores inner lattices and joins using intersection. Maps are
/// represented as `Option<BTreeMap>` and the unit is again a putative "maximum"
/// map-with-all-possible-keys (represented by `None`).
#[derive(Debug)]
pub struct BTreeMapWithIntersection<K: Ord + Clone, VD: LatticeDef> {
    phantom1: PhantomData<K>,
    phantom2: PhantomData<VD>,
}
impl<K: Ord + Clone, VD: LatticeDef> LatticeDef for BTreeMapWithIntersection<K, VD>
where
    VD::T: Clone,
{
    type T = Option<BTreeMap<K, LatticeElt<VD>>>;
    fn unit() -> Self::T {
        None
    }
    fn join(lhs: &Self::T, rhs: &Self::T) -> Self::T {
        match (lhs, rhs) {
            (None, None) => None,
            (Some(_), None) => lhs.clone(),
            (None, Some(_)) => rhs.clone(),
            (Some(lmap), Some(rmap)) => {
                let mut tmp = BTreeMap::<K, LatticeElt<VD>>::default();
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
        // This is a complicated partial order: lhs <= rhs if lhs has a superset
        // of the keys in rhs _and_ every lhs value of every common key is <=
        // the rhs value. If common-key values are ordered with any mix of
        // greater or lesser, or if any values on common keys are unordered, the
        // maps are unordered.
        match (lhs, rhs) {
            (None, None) => Some(Ordering::Equal),

            // The None element is the unit, the map-with-all-possible-values,
            // which is less-than all other maps in the intersection-based
            // partial order.
            (None, Some(_)) => Some(Ordering::Less),
            (Some(_), None) => Some(Ordering::Greater),

            // When we have two maps with definite subsets-of-all-keys, we look at
            // them element-wise.
            (Some(lmap), Some(rmap)) => {
                let mut lhs_lt_rhs_at_some_key = false;
                let mut rhs_lt_lhs_at_some_key = false;
                for (k, lv) in lmap.iter() {
                    match rmap.get(k) {
                        // If lmap has a value and rmap hasn't, lmap is "less
                        // than" (has more values than) rmap in the intersection
                        // partial order. This is the opposite interpretation of
                        // present-vs-absent keys as we have above in the union
                        // map code.
                        None => lhs_lt_rhs_at_some_key = true,
                        Some(rv) => {
                            // When we have keys in both maps, we defer to the
                            // partial order of the values. Note that we do
                            // _not_ invert the partial order among the values
                            // here, so this branch contains the same code as
                            // above in the union map code.
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

/// This is the same semantics as the `BitSetWithUnion` lattice, but covering
/// sets of arbitrary ordered values.
#[derive(Debug)]
pub struct BTreeSetWithUnion<U: Clone + Ord> {
    phantom: PhantomData<U>,
}
impl<U: Clone + Ord> LatticeDef for BTreeSetWithUnion<U> {
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

/// This is the same semantics as the `BitSetWithIntersection` lattice, but
/// covering sets of arbitrary ordered values.
#[derive(Debug)]
pub struct BTreeSetWithIntersection<U: Clone + Ord> {
    phantom: PhantomData<U>,
}
impl<U: Clone + Ord> LatticeDef for BTreeSetWithIntersection<U> {
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
#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen};
    use std::fmt::Debug;

    #[derive(Clone, Debug)]
    struct BitSetWrapper(BitSet);

    trait TestWrapper<T> {
        fn unwrap(self) -> T;
    }

    impl TestWrapper<BitSet> for BitSetWrapper {
        fn unwrap(self) -> BitSet {
            let BitSetWrapper(w) = self;
            w
        }
    }

    impl<D: LatticeDef> Arbitrary for LatticeElt<D>
    where
        D: 'static,
        D::T: Arbitrary,
    {
        fn arbitrary<G: Gen>(g: &mut G) -> LatticeElt<D> {
            LatticeElt::new_from(D::T::arbitrary(g))
        }
    }

    impl Arbitrary for BitSetWrapper {
        fn arbitrary<G: Gen>(g: &mut G) -> BitSetWrapper {
            let v = <Vec<u8> as Arbitrary>::arbitrary(g);
            BitSetWrapper(BitSet::from_bytes(&v[..]))
        }
    }

    fn join_assoc<D: LatticeDef>(a: LatticeElt<D>, b: LatticeElt<D>, c: LatticeElt<D>) -> bool
    where
        D: Debug,
        D::T: Debug,
    {
        (&a + (&b + &c)) == ((&a + &b) + &c)
    }

    fn join_comm<D: LatticeDef>(a: LatticeElt<D>, b: LatticeElt<D>) -> bool
    where
        D: Debug,
        D::T: Debug,
    {
        (&a + &b) == (&b + &a)
    }

    fn join_idem<D: LatticeDef>(a: LatticeElt<D>) -> bool
    where
        D: Debug,
        D::T: Debug,
    {
        (&a + &a) == a
    }

    fn join_unit<D: LatticeDef>(a: LatticeElt<D>) -> bool
    where
        D: Debug,
        D::T: Debug,
    {
        (&a + &LatticeElt::<D>::default()) == a
    }

    fn join_order<D: LatticeDef>(a: LatticeElt<D>, b: LatticeElt<D>) -> bool
    where
        D: Debug,
        D::T: Debug,
    {
        match a.partial_cmp(&b) {
            None | Some(Ordering::Equal) => true,
            Some(Ordering::Less) => (&a + &b) == b,
            Some(Ordering::Greater) => (&a + &b) == a,
        }
    }

    fn quickcheck_props<D: LatticeDef>()
    where
        LatticeElt<D>: Arbitrary,
        D: Debug,
        D::T: Debug,
    {
        quickcheck(join_assoc as fn(LatticeElt<D>, LatticeElt<D>, LatticeElt<D>) -> bool);
        quickcheck(join_idem as fn(LatticeElt<D>) -> bool);
        quickcheck(join_unit as fn(LatticeElt<D>) -> bool);
        quickcheck(join_comm as fn(LatticeElt<D>, LatticeElt<D>) -> bool);
        quickcheck(join_order as fn(LatticeElt<D>, LatticeElt<D>) -> bool);
    }

    #[test]
    fn quickcheck_others() {
        quickcheck_props::<MaxDef<u32>>();
        quickcheck_props::<MaxDef<String>>();
        quickcheck_props::<MaxNum<i8>>();
        quickcheck_props::<MinNum<i8>>();
        quickcheck_props::<MinNum<u64>>();
        quickcheck_props::<Tuple2<MaxDef<u32>, MaxDef<u32>>>();
        quickcheck_props::<Tuple3<MaxDef<u64>, MaxDef<String>, MinNum<i32>>>();
        quickcheck_props::<BTreeMapWithUnion<u32, MaxDef<u32>>>();
        quickcheck_props::<BTreeMapWithUnion<u32, Tuple2<MinNum<i8>, MaxDef<String>>>>();
        quickcheck_props::<BTreeMapWithIntersection<u32, MaxDef<Vec<u8>>>>();
        quickcheck_props::<Tuple2<BTreeSetWithUnion<String>, BTreeSetWithIntersection<char>>>();
    }

    // BitSet doesn't implement Arbitrary and it's a mess to try to adapt it
    // without interfering with our blanker LatticeElt Arbitrary impl above.
    #[test]
    fn quickcheck_bitsets() {
        fn bitset_union_join_assoc(a: BitSetWrapper, b: BitSetWrapper, c: BitSetWrapper) -> bool {
            join_assoc::<BitSetWithUnion>(a.unwrap().into(), b.unwrap().into(), c.unwrap().into())
        }
        fn bitset_union_join_idem(a: BitSetWrapper) -> bool {
            join_idem::<BitSetWithUnion>(a.unwrap().into())
        }
        fn bitset_union_join_unit(a: BitSetWrapper) -> bool {
            join_unit::<BitSetWithUnion>(a.unwrap().into())
        }
        fn bitset_union_join_comm(a: BitSetWrapper, b: BitSetWrapper) -> bool {
            join_comm::<BitSetWithUnion>(a.unwrap().into(), b.unwrap().into())
        }
        fn bitset_union_join_order(a: BitSetWrapper, b: BitSetWrapper) -> bool {
            join_order::<BitSetWithUnion>(a.unwrap().into(), b.unwrap().into())
        }
        quickcheck(
            bitset_union_join_assoc as fn(BitSetWrapper, BitSetWrapper, BitSetWrapper) -> bool,
        );
        quickcheck(bitset_union_join_idem as fn(BitSetWrapper) -> bool);
        quickcheck(bitset_union_join_unit as fn(BitSetWrapper) -> bool);
        quickcheck(bitset_union_join_comm as fn(BitSetWrapper, BitSetWrapper) -> bool);
        quickcheck(bitset_union_join_order as fn(BitSetWrapper, BitSetWrapper) -> bool);

        fn bitset_intersection_join_assoc(
            a: BitSetWrapper,
            b: BitSetWrapper,
            c: BitSetWrapper,
        ) -> bool {
            join_assoc::<BitSetWithIntersection>(
                a.unwrap().into(),
                b.unwrap().into(),
                c.unwrap().into(),
            )
        }
        fn bitset_intersection_join_idem(a: BitSetWrapper) -> bool {
            join_idem::<BitSetWithIntersection>(a.unwrap().into())
        }
        fn bitset_intersection_join_unit(a: BitSetWrapper) -> bool {
            join_unit::<BitSetWithIntersection>(a.unwrap().into())
        }
        fn bitset_intersection_join_comm(a: BitSetWrapper, b: BitSetWrapper) -> bool {
            join_comm::<BitSetWithIntersection>(a.unwrap().into(), b.unwrap().into())
        }
        fn bitset_intersection_join_order(a: BitSetWrapper, b: BitSetWrapper) -> bool {
            join_order::<BitSetWithIntersection>(a.unwrap().into(), b.unwrap().into())
        }
        quickcheck(
            bitset_intersection_join_assoc
                as fn(BitSetWrapper, BitSetWrapper, BitSetWrapper) -> bool,
        );
        quickcheck(bitset_intersection_join_idem as fn(BitSetWrapper) -> bool);
        quickcheck(bitset_intersection_join_unit as fn(BitSetWrapper) -> bool);
        quickcheck(bitset_intersection_join_comm as fn(BitSetWrapper, BitSetWrapper) -> bool);
        quickcheck(bitset_intersection_join_order as fn(BitSetWrapper, BitSetWrapper) -> bool);
    }
}
