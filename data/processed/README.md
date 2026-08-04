# Generated datasets

Preprocessing writes versioned, reproducible datasets here. Payload files are ignored; tracked
README files and small manifests describe the interface and provenance.

```text
data/processed/
├── antm2c/canonical-v1/
├── microlens/canonical-v1/
└── tiktok/canonical-v1/
```

Never download provider data directly into this directory. Put user-obtained inputs under
`data/raw/<dataset>/` (or configure an absolute ignored local path), then write generated outputs
here. A semantic preprocessing change must create a new version directory.
