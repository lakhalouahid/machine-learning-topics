# Machine learning topics:

## Regression
I be presenting here, some of least known machine learning algorithms, applied to regression fitting problem,
some are simple enough to implement myself. For others, i used the [scipy machine learning library](https://docs.scipy.org/doc/scipy/reference/index.html) implementations
in my code to show there performance. also i should mention, that in the docs, links to the implementation
reference.

This notebook presentesh the following methods:
  1. First Order methods (using the first order information: gradient), with sub-linear convergence
      1. Full-batch gradient descent.
      2. Mini-batch gradient descent.
      3. Momentum gradient descent.
      4. Nesterov accelerated gradient.
      5. ADAM
      6. RMSProp
      7. Mini-batch gradient descent with backtracking of the best learning rate, using Armijo condition.
  2. Second Order methods (using first, and second order information: gradient, hessian), for super-linear to linear convergence
      1. Newton algorithm.
      2. bfgs algorithm.
      3. lbfgs algorithm.
      4. conjugate gradient algorithm.
      5. newton conjugate gradient algorithm.
      6. levenberg-marquardt algorithm.
