import sympy as sym

x, a, b, l = sym.symbols("x a b l")

eq = sym.Eq(((a * b) ** 2 / l * x * sym.exp(-(b * x) ** 2 / (2 * l))) /
            sym.sqrt(2 * a ** 2 * (1 - sym.exp(-(b * x) ** 2 / (2 * l)))) - 1)

print(sym.solve(eq, x))
