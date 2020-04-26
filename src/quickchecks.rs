// Copyright 2020 Graydon Hoare <graydon@pobox.com>
// Licensed under the MIT and Apache-2.0 licenses.

use super::*;
use quickcheck::{quickcheck, Arbitrary, Gen};
use std::fmt::Debug;
use std::cmp::Ordering;

// Quickcheck stuff
//
// This is broadly less-effective than the proptest stuff because many
// of the sparse lattices (maps, sets) just produce unordered elts. But it's
// cheap to test, and some of the tests are meaningful, so why not?

fn join_assoc<D: LatticeDef>(a: LatticeElt<D>, b: LatticeElt<D>, c: LatticeElt<D>) -> bool
where
    D: Debug,
    D::T: Debug,
{
    println!(
        "\tjoin associativity lhs: {:?} + ({:?} + {:?}) == {:?}",
        a.value,
        b.value,
        c.value,
        (&a + (&b + &c)).value
    );
    println!(
        "\tjoin associativity rhs: ({:?} + {:?}) + {:?} == {:?}",
        a.value,
        b.value,
        c.value,
        ((&a + &b) + &c).value
    );
    (&a + (&b + &c)) == ((&a + &b) + &c)
}

fn join_comm<D: LatticeDef>(a: LatticeElt<D>, b: LatticeElt<D>) -> bool
where
    D: Debug,
    D::T: Debug,
{
    println!(
        "\tjoin commutativity lhs: {:?} + {:?} == {:?}",
        a.value,
        b.value,
        (&a + &b).value
    );
    println!(
        "\tjoin commutativity rhs: {:?} + {:?} == {:?}",
        b.value,
        a.value,
        (&b + &a).value
    );
    (&a + &b) == (&b + &a)
}

fn join_idem<D: LatticeDef>(a: LatticeElt<D>) -> bool
where
    D: Debug,
    D::T: Debug,
{
    println!(
        "\tjoin idempotence: {:?} + {:?} == {:?}",
        a.value,
        a.value,
        (&a + &a).value
    );
    (&a + &a) == a
}

fn join_unit<D: LatticeDef>(a: LatticeElt<D>) -> bool
where
    D: Debug,
    D::T: Debug,
{
    println!(
        "\tjoin unit: {:?} + {:?} == {:?}",
        a.value,
        &LatticeElt::<D>::default(),
        (&a + &LatticeElt::<D>::default())
    );
    (&a + &LatticeElt::<D>::default()) == a
}

fn join_order<D: LatticeDef>(a: LatticeElt<D>, b: LatticeElt<D>) -> bool
where
    D: Debug,
    D::T: Debug,
{
    match a.partial_cmp(&b) {
        None => {
            println!("\tjoin order: {:?} unordered wrt. {:?}", a.value, b.value);
            true
        }
        Some(Ordering::Equal) => {
            println!(
                "\tjoin order: {:?} == {:?} ==> {:?} + {:?} == {:?} == {:?}",
                a.value, b.value, a.value, b.value, a.value, b.value
            );
            (&a + &b) == b && (&a + &b) == a
        }
        Some(Ordering::Less) => {
            println!(
                "\tjoin order: {:?} < {:?} ==> {:?} + {:?} == {:?}",
                a.value, b.value, a.value, b.value, b.value
            );
            (&a + &b) == b
        }
        Some(Ordering::Greater) => {
            println!(
                "\tjoin order: {:?} > {:?} ==> {:?} + {:?} == {:?}",
                a.value, b.value, a.value, b.value, a.value
            );
            (&a + &b) == a
        }
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
