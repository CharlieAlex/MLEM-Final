from functools import cached_property

class Test():
    def __init__(self, score):
        self.score = score

    @cached_property
    def add_100(self):
        self.score += 100
        return self.score

x = Test(100)
print('a',x.score) #100
x.add_100 #None
print('b', x.score) #200
print('c', x.add_100) #200
print('d', x.score) #200

print('e', x.add_100) #200