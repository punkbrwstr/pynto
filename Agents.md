# Agent Notes

These notes apply to any AI coding agent working in this repository.

## Pynto Word Composition

- When a Pynto word uses `pt.keep.cols[...]` and will be reused or composed onto other words, wrap it in `(...).local`.
- This prevents the kept-column filter from unintentionally applying to the previous stack during later composition.
- When loading a single known column from a frame, prefer specifying the column in the frame key with `#`, such as `pt.load('lib:fx:spot:usd#JPY')`, instead of loading the whole frame and then applying `pt.keep.cols['JPY']`.
- Be careful with cumulative words such as `pt.cadd` and `pt.cmul` when applying them to a multi-column group, for example `pt.cadd.cols[:]`. These grouped cumulative operations use the whole grouped array, so a `NaN` in any column can produce `NaN` values across the grouped output at that row. If each column should accumulate independently, map the cumulative word per column instead, for example `~pt.cadd + pt.map` or `~pt.cmul + pt.map`.

