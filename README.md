<div align="center">

# Maximum-Sharpe Portfolios: Monte Carlo vs. SLSQP

### Two ways to find the tangency portfolio for a four-stock universe

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](#requirements)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](#requirements)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](#notebooks)

</div>

---

Which weights across AMD, AAPL, MSFT, and ORCL maximize the Sharpe ratio? This project answers the question twice: by brute force, sampling ten thousand random portfolios, and by direct optimization with Sequential Least Squares Programming. Comparing the two shows what random search can and cannot tell you about the efficient frontier.

## Approach

**Monte Carlo search.** Draw 10,000 random long-only weight vectors that sum to one, compute each portfolio's annualized return, volatility, and Sharpe ratio, and keep the best.

**Direct optimization.** Maximize the Sharpe ratio with `scipy.optimize.minimize` (SLSQP) by minimizing its negative,

$$
\max_{w}\ \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}} \quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\quad 0 \le w_i \le 1.
$$

## Result

The best random portfolio put about 57% in AAPL, 28% in AMD, 15% in MSFT, and almost nothing in ORCL, with a Sharpe ratio of 1.83. SLSQP lands slightly higher, as expected: random sampling maps the shape of the risk-return cloud, while the optimizer finds the corner of it.

## Notebooks

| Notebook | Contents |
|---|---|
| `Case_Study_Portfolio_Optimization.ipynb` | Portfolio construction, performance metrics, and the 10,000-portfolio simulation |
| `Sequential_Least_Squares_Quadratic_Programming.ipynb` | SLSQP optimization and comparison with the simulated optimum |

`Data/` holds historical prices for the four stocks.

## Requirements

```bash
pip install numpy pandas scipy matplotlib jupyter
```
