# Kalman Filter JSON tests

1. The "basic" test runs a kalman filter without a plant submodel
2. The other tests use a plant submodel.

These tests include models built in the UI where the KF parameters are defined
as Model Parameters. This mostly validates that the parameters are properly
parsed and passed around.

This does not test much, does not check numerical results (you can add), but
the integration relies on a lot of tricky parts of the code.
