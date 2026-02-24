
class Dummy:
    pass

try:
    a = iter([1]) + 1.5
except Exception as e:
    print(repr(e))
