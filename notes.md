* TODO: try and combine `add` and `const 1` into a single `add1` component, and
  do the same for other combos, and see if that helps with synthesizing
  popcount. That is what brahma did, iiuc.

* DONE: actually use the narrowest width for lines that we can

* DONE: it seems like attempting to synthesize a too-minimal program takes too
  long, so it would be faster to go longest -> shortest, keeping only the last,
  shortest program we synthesized.

# Notes from reading Souper source

* `include/souper/Infer/InstSynthesis.h`
* `lib/Infer/InstSynthesis.cpp`
* DONE: Souper says the final output is either equal to one of the original inputs, or
  the location of the output is equal to one of the component's output locations.
  * And then lets the `(l_x = l_y) => x = y` handle ensuring that the final
    output is correct
  * this means you could ask for smaller and smaller programs to be synthesized
    by requiring that L[O] be smaller and smaller...
* DONE: Souper does a bunch of work to avoid invalid wirings in the connectivity
  constraint and elsewhere. This should cut down on the number of clauses /
  constraints given to the solver a bunch.
* DONE: souper creates 4 initial concrete inputs for synthesis via asking the
  solver to find some sort of counter examples
  * see `InstSynthesis::getInitialConcreteInputs`
  * but I'm not totally sure what the query is... I think it is the negation of
    the spec
* DONE: after synthesizing a wiring that the solver finds a counter example for,
  the particular wiring is forbidden in new synthesis queries
  * see `InstSynthesis::forbidInvalidCandWiring`
  * but it only does this if there are no synthesized constants in the program!
    if there are synthesized constants, then it only forbids that wiring with
    that particular constant value (and has a limit before forbidding this
    wiring altogether)
* has a loop around the CEGIS for constraining to use fewer components, and then
  more and more until we successfully synthesize a program
* also builds cost/benefit model into loop around CEGIS
