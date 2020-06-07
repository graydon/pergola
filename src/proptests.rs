// Copyright 2020 Graydon Hoare <graydon@pobox.com>
// Licensed under the MIT and Apache-2.0 licenses.

use super::*;
#[cfg(feature = "bits")]
use bit_set::BitSet;

#[cfg(feature = "bits")]
use crate::latticedef::BitSetWrapper;

#[cfg(feature = "bits")]
use proptest::bits::bitset;

use proptest::prelude::*;
use proptest::{collection, option, strategy::Just};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt::Debug;

// Proptest stuff

fn prop_assoc<D: LatticeDef>(
    case: &str,
    a: &LatticeElt<D>,
    b: &LatticeElt<D>,
    c: &LatticeElt<D>,
) -> Result<(), TestCaseError>
where
    D: Debug,
    D::T: Debug,
{
    println!("join-associativity for {}", case);
    println!(
        "\tlhs: {:?} + ({:?} + {:?}) == {:?}",
        a.value,
        b.value,
        c.value,
        (a + (b + c)).value
    );
    println!(
        "\trhs: ({:?} + {:?}) + {:?} == {:?}",
        a.value,
        b.value,
        c.value,
        ((a + b) + c).value
    );
    prop_assert_eq!(a + (b + c), (a + b) + c);
    Ok(())
}

fn prop_comm<D: LatticeDef>(
    case: &str,
    a: &LatticeElt<D>,
    b: &LatticeElt<D>,
) -> Result<(), TestCaseError>
where
    D: Debug,
    D::T: Debug,
{
    println!("join-commutativity for {}", case);
    println!(
        "\tlhs: {:?} + {:?} == {:?}",
        a.value,
        b.value,
        (a + b).value
    );
    println!(
        "\trhs: {:?} + {:?} == {:?}",
        b.value,
        a.value,
        (b + a).value
    );
    prop_assert_eq!(a + b, b + a);
    Ok(())
}

fn prop_idem<D: LatticeDef>(case: &str, a: &LatticeElt<D>) -> Result<(), TestCaseError>
where
    D: Debug,
    D::T: Debug,
    LatticeElt<D>: Clone,
{
    println!("join-idempotence for {}", case);
    println!(
        "\tvalues: {:?} + {:?} == {:?}",
        a.value,
        a.value,
        (a + a).value
    );
    prop_assert_eq!(a + a, a.clone());
    Ok(())
}

fn prop_unit<D: LatticeDef>(case: &str, a: &LatticeElt<D>) -> Result<(), TestCaseError>
where
    D: Debug,
    D::T: Debug,
    LatticeElt<D>: Clone,
{
    let u = &LatticeElt::<D>::default();
    println!("join-unit for {}", case);
    println!(
        "\tvalues: {:?} + {:?} == {:?}",
        a.value,
        u.value,
        (a + u).value
    );
    prop_assert_eq!(a + u, a.clone());
    Ok(())
}

fn prop_induced_order<D: LatticeDef>(
    case: &str,
    a: &LatticeElt<D>,
    b: &LatticeElt<D>,
) -> Result<(), TestCaseError>
where
    D: Debug,
    D::T: Debug,
    LatticeElt<D>: Clone,
{
    println!("join-induced order matches explicit order for {}", case);
    match a.partial_cmp(b) {
        None => {
            println!("\tvalues: {:?} unordered wrt. {:?}", a.value, b.value);
            ()
        }
        Some(Ordering::Equal) => {
            println!("\tordering: {:?} == {:?} ", a.value, b.value);
            println!(
                "\tequality: {:?} + {:?} == {:?} == {:?}",
                a.value, b.value, a.value, b.value
            );
            prop_assert_eq!(a + b, b.clone());
            prop_assert_eq!(a + b, a.clone());
        }
        Some(Ordering::Less) => {
            println!("\tordering: {:?} < {:?} ", a.value, b.value);
            println!("\tequality: {:?} + {:?} == {:?}", a.value, b.value, b.value);
            prop_assert_eq!(a + b, b.clone());
        }
        Some(Ordering::Greater) => {
            println!("\tordering: {:?} > {:?} ", a.value, b.value);
            println!("\tequality: {:?} + {:?} == {:?}", a.value, b.value, a.value);
            prop_assert_eq!(a + b, a.clone());
        }
    }
    Ok(())
}

fn prop_order_refl<D: LatticeDef>(case: &str, a: &LatticeElt<D>) -> Result<(), TestCaseError>
where
    D: Debug,
    D::T: Debug,
    LatticeElt<D>: Clone,
{
    println!("explicit order is reflexive for {}", case);
    println!(
        "\tvalue: {:?} cmp {:?}: {:?} ",
        a.value,
        a.value,
        a.partial_cmp(a)
    );
    prop_assert_ne!(a.partial_cmp(a), None);
    Ok(())
}

fn prop_order_antisymm<D: LatticeDef>(
    case: &str,
    a: &LatticeElt<D>,
    b: &LatticeElt<D>,
) -> Result<(), TestCaseError>
where
    D: Debug,
    D::T: Debug,
    LatticeElt<D>: Clone,
{
    if a <= b && b <= a {
        println!("explicit order is antisymmetric for {}", case);
        println!("\tcmp1: {:?} <= {:?} ", a.value, b.value);
        println!("\tcmp2: {:?} <= {:?} ", b.value, a.value);
        println!("\teq: {:?} == {:?} ? {:?}", a.value, b.value, a == b);
        prop_assert_eq!(a.partial_cmp(b), Some(Ordering::Equal));
    } else {
        println!("antisymmetry premise missed for {}", case);
    }
    Ok(())
}

fn prop_order_trans<D: LatticeDef>(
    case: &str,
    a: &LatticeElt<D>,
    b: &LatticeElt<D>,
    c: &LatticeElt<D>,
) -> Result<(), TestCaseError>
where
    D: Debug,
    D::T: Debug,
    LatticeElt<D>: Clone,
{
    if a <= b && b <= c {
        println!("explicit order is transitive for {}", case);
        println!("\tcmp1: {:?} <= {:?} ", a.value, b.value);
        println!("\tcmp2: {:?} <= {:?} ", b.value, c.value);
        println!("\tcmp3: {:?} <= {:?} ? {:?}", a.value, c.value, a <= c);
        prop_assert_ne!(a.partial_cmp(c), Some(Ordering::Greater));
        prop_assert_ne!(a.partial_cmp(c), None);
    } else {
        println!("transitivity premise missed for {}", case);
    }
    Ok(())
}

fn all_props<D: LatticeDef>(
    case: &str,
    a: &LatticeElt<D>,
    b: &LatticeElt<D>,
    c: &LatticeElt<D>,
) -> Result<(), TestCaseError>
where
    D: Debug,
    D::T: Debug,
    LatticeElt<D>: Clone,
{
    prop_assoc(case, &a, &b, &c)?;

    prop_comm(case, &a, &b)?;
    prop_comm(case, &a, &c)?;
    prop_comm(case, &b, &c)?;

    prop_unit(case, &a)?;
    prop_unit(case, &b)?;
    prop_unit(case, &c)?;

    prop_idem(case, &a)?;
    prop_idem(case, &b)?;
    prop_idem(case, &c)?;

    prop_induced_order(case, &a, &b)?;
    prop_induced_order(case, &a, &c)?;
    prop_induced_order(case, &b, &c)?;

    prop_order_refl(case, &a)?;
    prop_order_refl(case, &b)?;
    prop_order_refl(case, &c)?;

    prop_order_antisymm(case, &a, &b)?;
    prop_order_antisymm(case, &a, &c)?;
    prop_order_antisymm(case, &b, &c)?;
    prop_order_antisymm(case, &a, &a)?;
    prop_order_antisymm(case, &b, &b)?;
    prop_order_antisymm(case, &c, &c)?;

    prop_order_trans(case, &a, &b, &c)?;
    prop_order_trans(case, &a, &c, &b)?;
    prop_order_trans(case, &b, &a, &c)?;
    prop_order_trans(case, &b, &c, &a)?;
    prop_order_trans(case, &c, &b, &a)?;
    prop_order_trans(case, &c, &a, &b)?;

    Ok(())
}

#[cfg(feature = "bits")]
prop_compose! {
    // This generates triples of bitsets that have a mix of
    // relationship and non-relationship to one another.
    fn arb_three_bitsets()
        (x in bitset::sampled(2, 0..100).boxed(),
         y in bitset::sampled(2, 0..100).boxed(),
         z in bitset::sampled(2, 0..100).boxed())
        (a in Just(x.clone()).boxed()
         .prop_union(Just(y.clone()).boxed())
         .or(Just(z.clone()).boxed())
         .or(Just(x.union(&y).collect()).boxed())
         .or(Just(x.union(&z).collect()).boxed())
         .or(Just(y.union(&z).collect()).boxed())
         .or(Just(BitSet::new()).boxed()),
         b in Just(x.clone()).boxed()
         .prop_union(Just(y.clone()).boxed())
         .or(Just(z.clone()).boxed())
         .or(Just(x.union(&y).collect()).boxed())
         .or(Just(x.union(&z).collect()).boxed())
         .or(Just(y.union(&z).collect()).boxed())
         .or(Just(BitSet::new()).boxed()),
         c in Just(x.clone()).boxed()
         .prop_union(Just(y.clone()).boxed())
         .or(Just(z.clone()).boxed())
         .or(Just(x.union(&y).collect()).boxed())
         .or(Just(x.union(&z).collect()).boxed())
         .or(Just(y.union(&z).collect()).boxed())
         .or(Just(BitSet::new()).boxed()),
        )
         -> (BitSet, BitSet, BitSet)
    {
        (a, b, c)
    }
}

#[cfg(feature = "bits")]
prop_compose! {
    fn arb_three_bitsets_with_union()
        ((a, b, c) in arb_three_bitsets())
         -> (LatticeElt<BitSetWithUnion>,
             LatticeElt<BitSetWithUnion>,
             LatticeElt<BitSetWithUnion>)
    {
        type E = LatticeElt<BitSetWithUnion>;
        (E::from(a), E::from(b), E::from(c))
    }
}

#[cfg(feature = "bits")]
prop_compose! {
    fn arb_three_bitsets_with_intersection()
        ((x, y, z) in arb_three_bitsets())
        (a in option::weighted(0.9, Just(x)),
         b in option::weighted(0.9, Just(y)),
         c in option::weighted(0.9, Just(z)))
         -> (LatticeElt<BitSetWithIntersection>,
             LatticeElt<BitSetWithIntersection>,
             LatticeElt<BitSetWithIntersection>)
    {
        type E = LatticeElt<BitSetWithIntersection>;
        (E{value:a.map(|x| BitSetWrapper(x))},
         E{value:b.map(|x| BitSetWrapper(x))},
         E{value:c.map(|x| BitSetWrapper(x))})
    }
}

prop_compose! {
    // This generates triples of u32 triples that have
    // some ordering relationships to one another.
    fn arb_u32_triple_triple()
        (x in arb_u32(),
         y in arb_u32(),
         z in arb_u32())
        (a in Just((x, y, z)).boxed()
         .prop_union(Just((0, y, z)).boxed())
         .or(Just((x, 0, z)).boxed())
         .or(Just((x, y, 0)).boxed()),

         b in Just((x+1, y, z)).boxed()
         .prop_union(Just((x, y+1, z)).boxed())
         .or(Just((x, y, z+1)).boxed())
         .or(Just((x+1, y+1, z)).boxed())
         .or(Just((x, y+1, z+1)).boxed())
         .or(Just((x+1, y+1, z+1)).boxed()),

         c in Just((x+2, y, z)).boxed()
         .prop_union(Just((x, y+2, z)).boxed())
         .or(Just((x, y, z+2)).boxed())
         .or(Just((x+2, y+2, z)).boxed())
         .or(Just((x, y+2, z+2)).boxed())
         .or(Just((x+2, y+2, z+2)).boxed()))
         -> ((u32, u32, u32),
             (u32, u32, u32),
             (u32, u32, u32))
    {
        (a, b, c)
    }
}

type IntMap = BTreeMap<u32, LatticeElt<MaxDef<u32>>>;

fn union_intmaps(lhs: &IntMap, rhs: &IntMap) -> IntMap {
    let mut tmp: IntMap = lhs.clone();
    for (k, v) in rhs.iter() {
        tmp.entry(k.clone())
            .and_modify(|e| *e = e.join(v))
            .or_insert(v.clone());
    }
    tmp
}

prop_compose! {
    fn arb_u32()(a in 0_u32..128_u32) -> u32
    {
        a
    }
}

prop_compose! {
    fn arb_i32()(a in -128_i32..128_i32) -> i32
    {
        a
    }
}

prop_compose! {
    fn arb_u32_maxdef()(a in arb_u32()) -> LatticeElt<MaxDef<u32>>
    {
        LatticeElt::<MaxDef<u32>>::new_from(a)
    }
}

prop_compose! {
    fn arb_three_int_maps()
        (x in collection::btree_map(arb_u32(), arb_u32_maxdef(), 3),
         y in collection::btree_map(arb_u32(), arb_u32_maxdef(), 3),
         z in collection::btree_map(arb_u32(), arb_u32_maxdef(), 3))
        (a in Just(x.clone()).boxed()
         .prop_union(Just(y.clone()).boxed())
         .or(Just(z.clone()).boxed())
         .or(Just(union_intmaps(&x, &y)).boxed())
         .or(Just(union_intmaps(&x, &z)).boxed())
         .or(Just(union_intmaps(&y, &z)).boxed()),
         b in Just(x.clone()).boxed()
         .prop_union(Just(y.clone()).boxed())
         .or(Just(z.clone()).boxed())
         .or(Just(union_intmaps(&x, &y)).boxed())
         .or(Just(union_intmaps(&x, &z)).boxed())
         .or(Just(union_intmaps(&y, &z)).boxed()),
         c in Just(x.clone()).boxed()
         .prop_union(Just(y.clone()).boxed())
         .or(Just(z.clone()).boxed())
         .or(Just(union_intmaps(&x, &y)).boxed())
         .or(Just(union_intmaps(&x, &z)).boxed())
         .or(Just(union_intmaps(&y, &z)).boxed()))
         -> (IntMap, IntMap, IntMap)
    {
        (a, b, c)
    }
}

prop_compose! {
    fn arb_three_int_maps_with_union()
        ((a, b, c) in arb_three_int_maps())
         -> (LatticeElt<BTreeMapWithUnion<u32, MaxDef<u32>>>,
             LatticeElt<BTreeMapWithUnion<u32, MaxDef<u32>>>,
             LatticeElt<BTreeMapWithUnion<u32, MaxDef<u32>>>)
    {
        type E = LatticeElt<BTreeMapWithUnion<u32, MaxDef<u32>>>;
        (E{value:a}, E{value:b}, E{value:c})
    }
}

prop_compose! {
    fn arb_three_int_maps_with_intersection()
        ((x, y, z) in arb_three_int_maps())
        (a in option::weighted(0.9, Just(x)),
         b in option::weighted(0.9, Just(y)),
         c in option::weighted(0.9, Just(z)))
         -> (LatticeElt<BTreeMapWithIntersection<u32, MaxDef<u32>>>,
             LatticeElt<BTreeMapWithIntersection<u32, MaxDef<u32>>>,
             LatticeElt<BTreeMapWithIntersection<u32, MaxDef<u32>>>)
    {
        type E = LatticeElt<BTreeMapWithIntersection<u32, MaxDef<u32>>>;
        (E{value:a}, E{value:b}, E{value:c})
    }
}

proptest! {
    #[test]
    fn proptest_bitset_union((a,b,c) in arb_three_bitsets_with_union()) {
        all_props("bitset with union", &a, &b, &c)?;
    }

    #[test]
    fn proptest_bitset_intersection((a,b,c) in arb_three_bitsets_with_intersection()) {
        all_props("bitset with intersection", &a, &b, &c)?;
    }

    #[test]
    fn proptest_u64_maxdef(a in 0_u64..128_u64,
                           b in 0_u64..128_u64,
                           c in 0_u64..128_u64)
    {
        type E = LatticeElt<MaxDef<u64>>;
        all_props("unsigned int with max",
                  &E::new_from(a), &E::new_from(b), &E::new_from(c))?;
    }

    #[test]
    fn proptest_i32_minopt(a in option::weighted(0.9, arb_i32()),
                           b in option::weighted(0.9, arb_i32()),
                           c in option::weighted(0.9, arb_i32()))
    {
        type E = LatticeElt<MinOpt<i32>>;
        all_props("signed int (plus maximal point) with min",
                  &E::new_from(a), &E::new_from(b), &E::new_from(c))?;
    }

    #[test]
    fn proptest_string_maxdef(a in "[a-c]{1,3}",
                              b in "[a-c]{1,3}",
                              c in "[a-c]{1,3}")
    {
        type E = LatticeElt<MaxDef<String>>;
        all_props("string with max",
                  &E::new_from(a), &E::new_from(b), &E::new_from(c))?;
    }

    #[test]
    fn proptest_string_minopt(a in option::weighted(0.9, "[a-c]{1,3}"),
                              b in option::weighted(0.9, "[a-c]{1,3}"),
                              c in option::weighted(0.9, "[a-c]{1,3}"))
    {
        type E = LatticeElt<MinOpt<String>>;
        all_props("string (plus maximal point) with min",
                  &E::new_from(a), &E::new_from(b), &E::new_from(c))?;
    }

    #[test]
    fn proptest_i32_minnum(a in arb_i32(),
                           b in arb_i32(),
                           c in arb_i32())
    {
        type E = LatticeElt<MinNum<i32>>;
        all_props("signed int with min",
                  &E::new_from(a), &E::new_from(b), &E::new_from(c))?;
    }

    #[test]
    fn proptest_tuple3_u32((a, b, c) in arb_u32_triple_triple())
    {
        type E = LatticeElt<Tuple3<MaxDef<u32>,MaxDef<u32>,MaxDef<u32>>>;
        all_props("triples of unsigned ints",
                  &E::new_from(a), &E::new_from(b), &E::new_from(c))?;
    }

    #[test]
    fn proptest_u32_union_map_with_int_maxdef((a, b, c) in arb_three_int_maps_with_union())
    {
        all_props("u32 union-maps with value-max", &a, &b, &c)?;
    }

    #[test]
    fn proptest_u32_intersection_map_with_int_maxdef((a, b, c) in arb_three_int_maps_with_intersection())
    {
        all_props("u32 intersection-maps with value-max", &a, &b, &c)?;
    }
}
