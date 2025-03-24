import random

class Cat:
    def __init__(self, race: int, price: float, length: float):
        self.race = race
        self.price = price
        self.length = length

def get_cats() -> list[Cat]:
    cats: list[Cat] = []
    
    # Generate 1000 cats with correlated properties
    for _ in range(1000):
        # Race from 1 (highest quality) to 5 (common)
        race = random.randint(1, 5)
        
        # Base price depends on race (higher price for lower race number)
        base_price = 1000 - (race - 1) * 150  # race 1: 1000, race 2: 850, race 3: 700, etc.
        
        # Length varies by race (higher quality races tend to be larger)
        min_length = 30 + (5 - race) * 2
        max_length = 45 + (5 - race) * 2
        length = round(random.uniform(min_length, max_length), 1)
        
        # Final price affected by both race and length
        length_bonus = (length - min_length) * 10  # Larger cats are more valuable
        final_price = round(base_price + length_bonus, 2)
        
        cats.append(Cat(race, final_price, length))
    
    return cats
