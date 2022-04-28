import sympy as sym

a, b, c, x, y = sym.symbols("a b c x y")
eq = sym.Eq(a * x ** 2 + b * x + c)
# print(eq)
print(sym.solve(eq, x))
