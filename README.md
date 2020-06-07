# Pergola

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

To minimize disruption caused by this two-level wrapping scheme and provide a
degree of compositionality, every `LatticeElt` also implements `LatticeDef` by
delegation to the `LatticeDef` it's parameterized over.

# Examples

## Simple u32-with-max lattice

```rust
use pergola::{MaxDef,LatticeElt};

type Def = MaxDef<u32>;      // lattice def for "u32 with max for join"
type Elt = LatticeElt<Def>;  // element struct, implementing std traits
let v = Elt::new_from(1);
let u = Elt::new_from(2);
let w = v + u;               // calls join(), which calls max()
assert!(v < u);
assert!(v < w);
```

## Trickier union-map-of-union-bitsets lattice

```rust
use pergola::{BTreeMapWithUnion,BitSetWithUnion,LatticeElt};
use bit_set::BitSet;
use std::collections::BTreeMap;

type Def = BTreeMapWithUnion<String,BitSetWithUnion>;
type Elt = LatticeElt<Def>;
let bs_a1 = BitSet::from_bytes(&[0b11110000]);
let bs_a2 = BitSet::from_bytes(&[0b00001111]);
let bs_b = BitSet::from_bytes(&[0b10101010]);
let v = Elt::new_from([(String::from("a"),bs_a1.into()),
                       (String::from("b"),bs_b.into())].iter().cloned().collect());
let u = Elt::new_from([(String::from("a"),bs_a2.into())].iter().cloned().collect());
let w = &v + &u;
assert!(!(v < u));  // bs_a1 is not a subset of bs_a2,
                    // so v["a"] is unordered wrt. u["a"].
assert!(v < w);     // However, w is a join and join unions
                    // the values at common keys, so v["a"] < w["a"].
assert!(u < w);     // And likewise the other input to the join.
assert_eq!(w.value["a"].value.0, BitSet::from_bytes(&[0b11111111]));
assert_eq!(w.value["b"].value.0, BitSet::from_bytes(&[0b10101010]));
```

# Name

Wikipedia:

> A pergola is an outdoor garden feature forming a shaded walkway, passageway, or
> sitting area of vertical posts or pillars that usually support cross-beams and a
> sturdy open lattice, often upon which woody vines are trained.
