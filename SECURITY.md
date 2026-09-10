# Security and data handling

The CLI does not download data, read trading credentials, execute broker orders or make network calls. Model snapshots use JSON, CSV and NumPy arrays loaded with `allow_pickle=False`; no executable pickle model loading is provided.

Run directories include the input data and provenance labels. Keep sensitive runs private. The report escapes text supplied as asset/source labels and bundles its fonts/charts without third-party requests.

Only load artifacts and datasets from sources you trust. Input size and compute budgets are controlled by the caller; this is a local research library, not a hardened multi-tenant service. Do not expose it directly as an unauthenticated server.

Report security problems privately to the repository owner using GitHub's private reporting feature when enabled. If unavailable, request a private contact channel without publishing exploitation details or sensitive data. Supported release line: 0.1.x.
