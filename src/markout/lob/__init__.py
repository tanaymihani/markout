"""Module D, the microstructure lab (LOBSTER, Nasdaq 2012-06-21), and module G, its C++ core.

lobster     loaders, constants, the message/book consistency replay
facts       spreads, depth, activity, trade-sign memory, tick-size regime
ofi         Cont-Kukanov-Stoikov order-flow imbalance: contemporaneous and lagged
imbalance   queue imbalance -> next mid move (Gould & Bonart 2016)
fills       fill simulator (touch / FIFO queue / trade-through), Python reference + C++ engine
markouts    sampled passive orders, markouts, effective-spread decomposition
postcross   post-or-cross decision rule, scored out of sample
synthetic   a tiny exchange that writes LOBSTER-format data, for tests
report      `python -m markout.lob.report` -> reports/02_microstructure.md
bench       `python -m markout.lob.bench` -> reports/results/cpp_bench.json
"""
