// Copyright 2020 Graydon Hoare <graydon@pobox.com>
// Licensed under the MIT and Apache-2.0 licenses.
use crate::*;

#[test]
fn manual_product_order_test() {
    type TestLD = ArcOrdSetWithUnion<u8>;
    type TestLE = LatticeElt<Tuple2<TestLD, TestLD>>;
    let empty = TestLE::default();
    let mut singleton = empty.clone();
    singleton.value.0.value.insert(1);
    assert!(empty <= singleton);
    assert!(!(singleton <= empty));
}
