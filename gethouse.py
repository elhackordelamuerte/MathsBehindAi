from sys import argv
import json
import random

argc = len(argv)

class House:
    def __init__(self, location_quality: int, price: float, num_rooms: int,  size: float):
        self.location_quality = location_quality
        self.price = price
        self.num_rooms = num_rooms
        self.size = size

    def __repr__(self):
        return f"House(\n\tlocation_quality: {self.location_quality}\n\tprice: {self.price}\n\tnum_rooms: {self.num_rooms}\n\tsize: {self.size}\n)"

class Apartment(House):
    def __init__(self, location_quality: int, price: float, num_rooms: int, size: float):
        super().__init__(location_quality, price, num_rooms, size)

    def __repr__(self):
        return f"Apartment(\n\tlocation_quality: {self.location_quality}\n\tprice: {self.price}\n\tnum_rooms: {self.num_rooms}\n\tsize: {self.size}\n)"

class Mansion(House):
    def __init__(self, location_quality: int, price: float, num_rooms: int,  size: float):
        super().__init__(location_quality, price, num_rooms, size)

    def __repr__(self):
        return f"Mansion(\n\tlocation_quality: {self.location_quality}\n\tprice: {self.price}\n\tnum_rooms: {self.num_rooms}\n\tsize: {self.size}\n)"

def get_houses(filename : str = None) -> list[House]:
    houses: list[House] = []
    
    # If filename is provided, load from JSON file
    if filename:
        if filename.endswith(".json"):
            with open(filename, "r") as file:
                data = json.load(file)
                for d in data:
                    houses.append(
                        House(
                            d['location_quality'],
                            d['price'],
                            d['num_rooms'],
                            d['size']
                        )
                    )
        return houses
    
    # Generate 1000 random houses with correlated properties
    for _ in range(1000):
        # Location quality from 1 (best) to 5 (worst)
        location_quality = random.randint(1, 5)
        
        # Base price depends on location quality
        base_price = 500000 - (location_quality - 1) * 75000  # Better locations cost more
        
        # Number of rooms varies by location quality (better locations tend to have more rooms)
        min_rooms = 2 + (5 - location_quality)
        max_rooms = 5 + (5 - location_quality)
        num_rooms = random.randint(min_rooms, max_rooms)
        
        # Size varies by number of rooms
        min_size = num_rooms * 30  # minimum 30 sq meters per room
        max_size = num_rooms * 50  # maximum 50 sq meters per room
        size = round(random.uniform(min_size, max_size), 1)
        
        # Final price affected by location, rooms, and size
        room_bonus = num_rooms * 25000  # Each room adds value
        size_bonus = size * 1000  # Each sq meter adds value
        final_price = round(base_price + room_bonus + size_bonus, 2)
        
        houses.append(House(location_quality, final_price, num_rooms, size))
    
    return houses
