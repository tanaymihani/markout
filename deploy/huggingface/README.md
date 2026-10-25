---
title: Markout Desk
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Paper-trading desk for prediction markets, one sandbox each
---

# Markout Desk

A trading desk for prediction markets, running on fictional markets and paper money.
Every visitor gets a private sandbox: the desk pools reference prices in log-odds,
proposes trades only when the whole uncertainty band clears the ask, sizes them with
Kelly, and trades nothing without a click. Move the clock forward to watch prices
drift, markets resolve and the decision journal score each forecast.

Source, tests and the research behind it: [github.com/tanaymihani/markout](https://github.com/tanaymihani/markout).
This Space is deployed from that repository by GitHub Actions; please open issues there.
