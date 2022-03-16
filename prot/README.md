naming conventions:
- a synthesizable module: no special prefix or postfix
- a adriver module for a synthesize module `X.sv`: `X_test.sv`
    - this keeps the driver adjacent tto the module it is testing
- a self-contained non-synthesizable module to test some specific thing: `test_[something]`
