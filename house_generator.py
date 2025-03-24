import random
import json

# Base data
houses = [
    {
        "location_quality": 8,
        "price": 250.0,
        "num_rooms": 4,
        "size": 2.0
    },
    {
        "location_quality": 7,
        "price": 320.0,
        "num_rooms": 5,
        "size": 2500.0
    },
    {
        "location_quality": 5,
        "price": 180.0,
        "num_rooms": 3,
        "size": 1500.0
    },
    {
        "location_quality": 6,
        "price": 275.0,
        "num_rooms": 4,
        "size": 2200.0
    },
    {
        "location_quality": 7,
        "price": 300.0,
        "num_rooms": 4,
        "size": 2400.0
    },
    {
        "location_quality": 8,
        "price": 350.0,
        "num_rooms": 5,
        "size": 2800.0
    },
    {
        "location_quality": 9,
        "price": 400.0,
        "num_rooms": 6,
        "size": 3200.0
    },
    {
        "location_quality": 4,
        "price": 220.0,
        "num_rooms": 3,
        "size": 1800.0
    },
    {
        "location_quality": 6,
        "price": 260.0,
        "num_rooms": 4,
        "size": 2100.0
    },
    {
        "location_quality": 7,
        "price": 290.0,
        "num_rooms": 4,
        "size": 2300.0
    }
]

# Function to generate a random house based on the base data
def generate_house(base_house, index):
    return {
        "location_quality": base_house["location_quality"],
        "price": round(base_house["price"] + random.uniform(-50, 50), 2),
        "num_rooms": base_house["num_rooms"] + random.randint(-1, 1),
        "size": round(base_house["size"] + random.uniform(-500, 500), 2)
    }

# Generate 1000 houses
generated_houses = []
for i in range(1000):
    base_house = houses[i % len(houses)]
    generated_houses.append(generate_house(base_house, i + 1))

# Save to JSON file
with open('houses_1000.json', 'w') as f:
    json.dump(generated_houses, f, indent=4)
