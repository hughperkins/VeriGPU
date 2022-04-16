# compute_unit

Some code for testing a compute unit, comprising multiple cores

Initially we will avoid needing barriers and stuff. But we probalby at least need a way to get the thread number.

Maybe we will create a pseudoinstruction for this for now? Where should we store the thread number? In a register? On the stack? Somewhere else? (are we even going to have a stack? :P )


