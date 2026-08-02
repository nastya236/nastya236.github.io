---
title: 'RMSNorm backward beauty'
description: 'The RMSNorm gradient looks like an arbitrary pile of terms until you notice it is just a projection.'
pubDate: 2026-07-30
tags: ['ml', 'autograd', 'math']
---

RMSNorm is the normalization layer that won. It is in Llama, in Gemma, in
basically every transformer trained after 2023, and it won by *deleting* things
from LayerNorm rather than adding them. No mean subtraction, no bias term.

Its backward pass is where that deletion pays off. If you derive it mechanically
you get a small pile of terms that look arbitrary. If you look at the pile for
another minute, it collapses into one geometric statement, and then you can never
un-see it.

## The forward pass

Given a vector $x \in \mathbb{R}^d$ and a learned gain $g$:

$$
s = \operatorname{mean}(x^2) + \varepsilon,
\qquad
r = \frac{1}{\sqrt{s}},
\qquad
\hat{x} = r\,x,
\qquad
y = g \odot \hat{x}
$$

Three operations. Compare LayerNorm, which also subtracts the mean and adds a
bias. The thing to notice, and the thing this whole post rests on, is that
RMSNorm is **scale invariant**: feed it $\lambda x$ instead of $x$ and $r$ shrinks
by $\lambda$, so $\hat{x}$ and therefore $y$ come out identical. The layer
discards the magnitude of $x$ and keeps only its direction.

Hold onto that. It is going to tell us the answer before we compute it.

## Deriving the backward

We are handed $\partial L / \partial y$, which I will write as $dy$, and we want
$\partial L/\partial x$ and $\partial L / \partial g$.

The gain is immediate, since $y_i$ depends only on $g_i$:

$$
\frac{\partial L}{\partial g} = dy \odot \hat{x}
$$

For the input, first push through the gain. Define

$$
v = dy \odot g = \frac{\partial L}{\partial \hat{x}}
$$

Now the annoying part. We have $\hat{x} = r x$, but $r$ is a function of *every*
component of $x$, so $\hat{x}_j$ depends on $x_i$ for all $i$. Chain rule over the
whole vector:

$$
\frac{\partial L}{\partial x_i}
= \sum_j v_j \frac{\partial \hat{x}_j}{\partial x_i},
\qquad
\frac{\partial \hat{x}_j}{\partial x_i}
= \underbrace{\delta_{ij}\, r}_{\text{direct}}
+ \underbrace{x_j \frac{\partial r}{\partial x_i}}_{\text{through } r}
$$

Two paths: the direct one, where $x_i$ scales itself, and the indirect one, where
$x_i$ nudges $r$ and thereby rescales *everything*. The second is where all the
coupling lives.

For that derivative, with $s = \operatorname{mean}(x^2) + \varepsilon$:

$$
\frac{\partial s}{\partial x_i} = \frac{2}{d}\,x_i,
\qquad
\frac{\partial r}{\partial s} = -\tfrac{1}{2}\,s^{-3/2}
$$

$$
\frac{\partial r}{\partial x_i}
= -\tfrac{1}{2}\,s^{-3/2} \cdot \frac{2}{d}\,x_i
= -\frac{1}{d}\,x_i\,r^3
$$

Substituting back:

$$
\frac{\partial L}{\partial x_i}
= r\,v_i - \frac{1}{d}\,x_i\,r^3 \sum_j v_j x_j
= r\left[\,v_i - \frac{1}{d}\,x_i\,r^2 \sum_j v_j x_j\,\right]
$$

Now the small move that makes it readable. There are two spare factors of $r$
inside the bracket and two bare $x$'s; push one onto each, turning them into
$\hat{x}$:

$$
\frac{\partial L}{\partial x_i}
= r\left[\,v_i - \frac{1}{d}\,\hat{x}_i \sum_j v_j \hat{x}_j\,\right]
$$

And $\frac{1}{d}\sum_j v_j \hat{x}_j$ is just a mean. Writing
$c = \operatorname{mean}(v \odot \hat{x})$:

$$
\boxed{\;dx = r\,\bigl(v - c\,\hat{x}\bigr)\;}
$$

That is the entire backward pass. One reduction, one subtraction, one scale.

## The punchline

Look at $c$ again. How long is $\hat{x}$? Setting $\varepsilon = 0$ for a moment:

$$
\|\hat{x}\|^2 = \frac{\sum_i x_i^2}{s}
= \frac{d \cdot \operatorname{mean}(x^2)}{\operatorname{mean}(x^2)} = d
$$

So $\hat{x}$ has squared norm $d$, always, for every input. Which means the
projection of $v$ onto the direction of $\hat{x}$ is

$$
\operatorname{proj}_{\hat{x}}(v)
= \frac{\langle \hat{x}, v\rangle}{\|\hat{x}\|^2}\,\hat{x}
= \frac{\langle \hat{x}, v\rangle}{d}\,\hat{x}
= \operatorname{mean}(v \odot \hat{x})\,\hat{x}
= c\,\hat{x}
$$

That is *exactly* the term being subtracted. So:

$$
dx = r\,\bigl(v - \operatorname{proj}_{\hat{x}}(v)\bigr)
$$

**The RMSNorm backward pass is a projection.** It takes the incoming gradient,
removes the component pointing along $\hat{x}$, and scales what remains by
$1/\mathrm{rms}$. The pile of terms was never arbitrary —
$-\frac{1}{d}\hat{x}\sum_j v_j \hat{x}_j$ is simply what "subtract the radial
component" looks like written out in coordinates.

## Why it had to be this

Now go back to scale invariance. If $y$ does not change when we scale $x$, then
moving $x$ radially — along $\hat{x}$ — cannot change the loss. A direction that
cannot change the loss must have zero gradient. So $dx$ is *required* to be
orthogonal to $x$, and the only way to guarantee that is to project the radial
part out.

We can check it directly. With $\varepsilon = 0$ we have
$\operatorname{mean}(x^2) = 1/r^2$, so

$$
\sum_i \hat{x}_i x_i = r \sum_i x_i^2 = r \cdot \frac{d}{r^2} = \frac{d}{r},
\qquad
\sum_i v_i x_i = \frac{1}{r}\sum_i v_i \hat{x}_i = \frac{d\,c}{r}
$$

and therefore

$$
\langle dx, x\rangle
= r\left[\sum_i v_i x_i - c \sum_i \hat{x}_i x_i\right]
= r\left[\frac{d\,c}{r} - c \cdot \frac{d}{r}\right]
= 0
$$

The gradient always comes out perpendicular to the input. The layer normalizes
away magnitude on the way forward; the backward pass refuses to produce gradient
in the direction it just discarded. Those are the same fact, seen from both
sides.

This is also the honest answer to "why is RMSNorm cheaper than LayerNorm?" It is
not mainly about skipping a mean. LayerNorm is invariant to *two* things, shift
and scale, so its backward projects out a two-dimensional subspace and carries
two correction terms. RMSNorm gives up shift invariance and gets a backward pass
with exactly one.

## Code

The whole thing, batched over a trailing feature axis:

```python
import numpy as np

def rmsnorm_forward(x, g, eps=1e-6):
    """x: (..., d), g: (d,)"""
    s = np.mean(x**2, axis=-1, keepdims=True) + eps
    r = 1.0 / np.sqrt(s)
    xhat = x * r
    return g * xhat, (xhat, g, r)

def rmsnorm_backward(dy, cache):
    xhat, g, r = cache
    v = dy * g
    c = np.mean(v * xhat, axis=-1, keepdims=True)   # the one reduction
    dx = r * (v - xhat * c)                          # project, then scale
    dg = np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))
    return dx, dg
```

Note what the backward needs from the forward: $\hat{x}$ and $r$. Not $x$, not
$\operatorname{mean}(x^2)$. Two saved tensors, one of which is a single scalar per
row — part of why fused RMSNorm kernels come out so tidy.

And because a derivation is worth nothing unless it survives a finite-difference
check:

```python
rng = np.random.default_rng(0)
x = rng.normal(size=(4, 8))
g = rng.normal(size=(8,))
dy = rng.normal(size=(4, 8))

y, cache = rmsnorm_forward(x, g)
dx, dg = rmsnorm_backward(dy, cache)

# numerical gradient of L = sum(dy * y) w.r.t. x
h = 1e-6
num = np.zeros_like(x)
for i in np.ndindex(x.shape):
    xp, xm = x.copy(), x.copy()
    xp[i] += h
    xm[i] -= h
    lp = np.sum(dy * rmsnorm_forward(xp, g)[0])
    lm = np.sum(dy * rmsnorm_forward(xm, g)[0])
    num[i] = (lp - lm) / (2 * h)

print(np.abs(dx - num).max())          # ~1e-9, the finite-difference noise floor

# Orthogonality — but this one is only exact when eps = 0, see below.
_, cache0 = rmsnorm_forward(x, g, eps=0.0)
dx0, _ = rmsnorm_backward(dy, cache0)
print(np.abs((dx0 * x).sum(-1)).max())  # ~1e-15
```

That last line is my favourite assertion in any normalization test suite. It is
not checking arithmetic, it is checking that the geometry is right.

## The $\varepsilon$ asterisk

Everything above quietly set $\varepsilon = 0$. With $\varepsilon > 0$ the layer
is no longer exactly scale invariant — scale $x$ up and $\varepsilon$ matters
relatively less — so

$$
\|\hat{x}\|^2 = \frac{d \cdot \operatorname{mean}(x^2)}{\operatorname{mean}(x^2) + \varepsilon} < d
$$

a hair under $d$, and the radial component is no longer perfectly cancelled.

This is not negligible at the precision a test asserts at. On the example above,
$|\langle dx, x\rangle|$ comes out around $10^{-15}$ with $\varepsilon = 0$ but
around $10^{-5}$ with $\varepsilon = 10^{-6}$ — ten orders of magnitude apart. If
you write that orthogonality check as a unit test, either set $\varepsilon = 0$ or
give it a tolerance that reflects $\varepsilon$, or you will have a red test and a
perfectly correct implementation.

Nobody cares for training purposes: a $10^{-5}$ leak in the radial direction
against gradients of order one changes nothing. But it is worth knowing which
fact is structural and which is approximate. The projection is structural. The
$\|\hat{x}\|^2 = d$ that makes it look so clean is approximate, and it degrades
precisely when activations get small enough for $\varepsilon$ to dominate — which
is also when your normalization layer has other problems.

## What to take away

The mechanical derivation and the geometric one give the same formula, but only
one of them is memorable. $dx = r(v - \hat{x}\operatorname{mean}(v \odot \hat{x}))$
is a string of symbols to memorize. "Project out the radial component, then divide
by the RMS" is a sentence you can reconstruct the algebra from, on a whiteboard, a
year later.

Most of the backward passes worth knowing are like this. Softmax's Jacobian is a
statement about a simplex. Attention's is a statement about a weighted average.
The terms look arbitrary right up until you find the invariance they are
protecting.
