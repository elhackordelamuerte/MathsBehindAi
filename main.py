from gethouse import *
from aihouse import *
from getcat import *
from aicat import *

def main(av, ac) -> int:
     if ac != 2:
        print("Usage: main.py <house_file>.json")
         return 84
     houses: list[House] = get_houses(av[1])
     housing_AI(houses)
    # cat_AI(get_cats())
    return 0

if __name__ == '__main__':
    exit(main(argv, argc))
