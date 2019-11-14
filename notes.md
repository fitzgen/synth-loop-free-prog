# Notes from reading Souper source

* `include/souper/Infer/InstSynthesis.h`
* `lib/Infer/InstSynthesis.cpp`
* Souper says the final output is either equal to one of the original inputs, or
  the location of the output is equal to one of the component's output locations.
  * And then lets the `(l_x = l_y) => x = y` handle ensuring that the final
    output is correct
  * this means you could ask for smaller and smaller programs to be synthesized
    by requiring that L[O] be smaller and smaller...
* Souper does a bunch of work to avoid invalid wirings in the connectivity
  constraint and elsewhere. This should cut down on the number of clauses /
  constraints given to the solver a bunch.
