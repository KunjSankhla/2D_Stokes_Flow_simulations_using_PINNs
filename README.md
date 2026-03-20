# Physics-Informed Neural Networks for solving 2D Stokes flow problems in PyTorch. Two cases implemented.

## Case 1: Channel Flow (Poiseuille)

- Inlet U=1 on left wall, pressure exit on right wall, no-slip top/bottom
- Validated against analytical Poiseuille centerline velocity (U_c = 1.5): 95.45% accuracy at the exit plane
- Architecture: 2 → 64 → 128 → 64 → 3, Tanh, Adam + ReduceLROnPlateau

## Case 2: Lid-Driven Cavity

- Top lid moves at U=1, all other walls no-slip, closed cavity
- No analytical solution — validated qualitatively via velocity contours and quiver plots
- Architecture: 2 → 64 → 64 → 64 → 3, Tanh, Adam fixed LR

## Physics
Both models enforce 2D Stokes momentum equations and continuity via automatic differentiation through the network. BCs imposed as soft constraints weighted 2x in the loss.

## To Run
...
pip install torch numpy matplotlib
jupyter notebook
...

...
## Repository Structure
basics/               # introductory PINN exercise
stokes_flow/          # main simulations
...
