# ELEC 844 Assignment # 
**A* and LPA***

## Deps
- SDL2
- SDL2_ttf

## Usage
At the top of the `main()` function, the `world` variable sets the world to start with.
There are currently 5 possible worlds: `eWorld1A`, `eWorld1B`, `eWorld2A`, `eWorld2B`, and `eWorld2C`.

Four lines below the `world` variable is the `search_init` function.
This function sets up the search based on the `eSearchType` parameter.
Currently, only `eSearchA` for A* and `eSearchLPA` for LPA* can be used.

This function also loads the huristic used during the search.
Currently, `hfn_zero`, `hfn_euclidian`, `hfn_manhattan`, and `hfn_inflated` are provided.
Additional heuristics can be used by definining a new function as follows:

```c 
static HFN(my_heuristic){
    (void)s;
    (void)d;
    return 0.0;
}
```

After selecting the world, search type, and heuristic, compile and run the program.
After the search terminates, press the `space` bar to load the next world and continue the search.

For A*, pressing the `space` key will reload all objects and start the search from scratch.
For LPA*, pressing the `space` key will resume the search with the enviroment changes.



## Worlds
Worlds, or search enviroments are defined as 2D arrays in the world file.
`1` is considered invalid, `2` as the source point, `3` as the goal.
All other numbers are considered free space.

## Animation Speed
Near the bottom of the main loop is an `SDL_Delay` function.
The value of this function is the time delay between animation steps in ms.

