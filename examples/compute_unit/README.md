# compute_unit

Some code for testing a compute unit, comprising multiple cores

Initially we will avoid needing barriers and stuff. But we probalby at least need a way to get the thread number.

Maybe we will create a pseudoinstruction for this for now? Where should we store the thread number? In a register? On the stack? Somewhere else? (are we even going to have a stack? :P )

We could pass it into the kernel as an addition, unseen, parameter perhaps?

For now, we'r ejust coding the assembly by hand anyway, so we'll just put it into a register perhaps?

Hmmmm. Like register a0, taht we set  .... hmmm... so, the Compute unit is going to have to be the done that gives this to the core, so the SM cant really mess with the assembly too much. Although we could leave a placeholder in the assembly for the compute unit to write to? this would save adding extra wires between the compute unit controller and each of the cores.

We could also learn AXI, adn maybe AXI gives a solution to this?

Maybe lets uses wires for now, and then we can check AXI spec later?


Note: this section is in progress currently :)
