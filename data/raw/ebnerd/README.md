# EB-NeRD raw inputs

Download the official small bundle and embedding archives from the
[provider](https://recsys.eb.dk/dataset/). Keep original ZIPs in `archives/`.

```text
ebnerd_small/
  articles.parquet
  train/behaviors.parquet
  train/history.parquet
  validation/behaviors.parquet
  validation/history.parquet
artifacts/
  Ekstra_Bladet_image_embeddings/image_embeddings.parquet
  google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet
```

All payloads remain ignored. Source URLs, HTTP headers, download dates and archive/extracted
SHA-256 values are recorded locally. macOS archive metadata is not used.
See [the processing protocol](../../../docs/data.md#eb-nerd) for the approved local test
definition, missing images and conversion commands. No dataset redistribution permission
is inferred from the software license.
