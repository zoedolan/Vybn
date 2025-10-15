# Protocol: Digital Complex U(1) Loop

ID: protocol_digital_complex_u1_loop_v1

Objective: Measure complex phase response around learning-rate loops with angular leg as rotated-basis regularization/dephasing.

Design:
- r: learning-rate amplitude; θ: cycle phase
- Angular leg: L2/L1 in rotated basis (m) not parallel to weight-update axis (n)
- Orientation flips, zero-area controls

Signatures:
- Area slope ~ 10^3 with r ~ 0.8
- Orientation sensitivity (CCW positive / CW negative)
- Null at zero-area lines
- Independence from micro-schedule
