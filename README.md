# apbf

An implmentation of an Age-Partioned Bloom Filter (ABPF) in rust.
ABPF is a probablistic data structure that operates on top of data streams.
It provides an interface similar to Bloom Filters, i.e., `insert` and
`contains` where older elements are discarded as new elements are inserted.

---
[mbrdg](https://github.com/mbrdg)
