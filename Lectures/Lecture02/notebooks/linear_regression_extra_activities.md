# Extra Activities --- Linear Regression from Scratch

These activities are designed for early finishers and progressively
deepen the understanding of forward propagation, backpropagation,
optimization, and linear models.

------------------------------------------------------------------------

# Activity 1 --- Predict Before Running ⭐

**Objective:** Understand tensor dimensions.

Before running the code, answer:

-   What is the shape of `X`?
-   What is the shape of `w`?
-   What is the shape of `b`?
-   What will be the shape of `X @ w + b`?

Then execute the cell and verify your answers.

------------------------------------------------------------------------

# Activity 2 --- Break the Code ⭐

Modify one dimension intentionally.

Examples:

-   Give `w` one extra row.
-   Make `y` have shape `(n,)` instead of `(n,1)`.

Questions:

1.  What error appears?
2.  Why?
3.  How would you fix it?

------------------------------------------------------------------------

# Activity 3 --- Finite Difference Gradient Check ⭐⭐

Approximate one gradient numerically:

$$
\frac{\partial L}{\partial w_i}
\approx
\frac{L(w_i+\epsilon)-L(w_i-\epsilon)}{2\epsilon}
$$

using

$$
\epsilon = 10^{-5}.
$$

Compare it with the analytical gradient from `backward()`.

------------------------------------------------------------------------

# Activity 4 --- Remove the Factor of 2 ⭐⭐

Replace

$$
\frac{2}{n}X^T(y_{pred}-y)
$$

with

$$
\frac{1}{n}X^T(y_{pred}-y).
$$

Train again and discuss what changes.

------------------------------------------------------------------------

# Activity 5 --- Find the Bug ⭐⭐

Replace

``` python
dw = X.T @ error
```

with

``` python
dw = X @ error
```

Why is this wrong?

------------------------------------------------------------------------

# Activity 6 --- Learning Rate Tournament ⭐⭐

Train with:

-   η = 1
-   η = 0.1
-   η = 0.01
-   η = 0.001
-   η = 0.0001

Compare all loss curves.

------------------------------------------------------------------------

# Activity 7 --- Bad Initialization ⭐⭐

Initialize with

``` python
100 * torch.randn(...)
```

and later

``` python
1000 * torch.randn(...)
```

Observe convergence.

------------------------------------------------------------------------

# Activity 8 --- Feature Importance ⭐⭐⭐

Train separate models using only:

-   TV
-   Radio
-   Newspaper

Compare the test MSE.

------------------------------------------------------------------------

# Activity 9 --- Feature Engineering ⭐⭐⭐

Create:

-   TV²
-   Radio²
-   TV × Radio

Does the model improve?

------------------------------------------------------------------------

# Activity 10 --- Mini-Batch Gradient Descent ⭐⭐⭐

Implement mini-batch training using batch sizes 8, 16, and 32.

------------------------------------------------------------------------

# Activity 11 --- Regularization ⭐⭐⭐

Implement Ridge:

$$
L = MSE + \lambda ||w||_2^2.
$$

Then implement Lasso:

$$
L = MSE + \lambda ||w||_1.
$$

------------------------------------------------------------------------

# Activity 12 --- Become PyTorch ⭐⭐⭐⭐

Using pencil and paper, compute:

1.  Forward pass
2.  Error
3.  MSE
4.  dw
5.  db
6.  One gradient descent update

Verify with the notebook.

------------------------------------------------------------------------

# Activity 13 --- Build Your Own Linear Layer ⭐⭐⭐⭐

Implement:

``` python
class MyLinearRegression:

    def forward(self, X):
        ...

    def backward(self, X, y):
        ...

    def step(self):
        ...
```

without looking at the notebook.

------------------------------------------------------------------------

# Bonus Challenge ⭐⭐⭐⭐⭐

Implement Logistic Regression from Scratch using the same training loop
but replacing the output activation, loss, and gradients.

------------------------------------------------------------------------

# Reflection Questions

1.  Why does gradient descent work?
2.  Why do we use `X.T` in the gradient?
3.  Why is the bias gradient the sum of the errors?
4.  What happens if the learning rate is too large?
5.  Why do we standardize the data?
6.  Why is linear regression a neural network with no hidden layers?
7.  Which parts of this notebook would remain unchanged for deep neural
    networks?
