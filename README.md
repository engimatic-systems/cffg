# cffg

Constrained Forney-style Factor Graphs (CFFG) and Lagrangian Active Inference
(LAIF) in JAX, implementing Koudahl, van de Laar & de Vries — *Realising
Synthetic Active Inference Agents*, Parts I & II (arXiv:2306.08014).

- Ground rules and session protocol: [AGENTS.md](AGENTS.md)
- Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- What this will and won't do: [docs/LIMITATIONS.md](docs/LIMITATIONS.md)

## Development

```sh
uv sync          # install deps
uv run pytest    # run tests
tk ready         # see what's next
```
