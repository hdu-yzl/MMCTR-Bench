# Generated AntM2C data

`canonical-v1/` is the current `sharded-named-npy-v1` training store. It is generated from the
audited legacy TFRecords and shared item arrays; it is not a full replay of the raw event/image
encoders. The tracked manifest is provenance evidence while arrays remain ignored.

Any future complete raw re-encoding belongs in `canonical-v2/` (or a later version) and must not
replace `canonical-v1/` in place. `benchmark-v1/` is only the format comparison slice.
