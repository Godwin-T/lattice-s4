"""
Arms: one module per approach, all emitting the same hand-in table.

An arm reads the frozen split manifest and the canonical job table, and writes
predictions in the shape `metrics.md` section 7 defines. It never chooses its own
train/test rows, and it never computes a metric.
"""
