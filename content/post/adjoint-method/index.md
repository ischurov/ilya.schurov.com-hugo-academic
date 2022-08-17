---
title: Adjoint State Method, Backpropagation and Neural ODEs

summary: |-
    You probably heard about Neural ODEs, a neural network architecture based on
    the ordinary differential equations. To train this kind of models, a mysterious
    trick called _adjoint state method_ is used. How does it work, why do we
    need it and how it is related to backpropagation? 

tags:
- machine learning
- differential equations
- NeuralODE

date: "2022-08-05T21:58:00Z"

draft: false

# Optional external URL for project (replaces project detail page).
external_link: ""

# image:
#   caption: Photo by Michael Daniel 
#   focal_point: Smart
# url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
---


You probably heard about Neural ODEs[^1], a neural network architecture based on the ordinary differential equations. When I first read about them, my thoughts were: okay, we have an ODE given by a neural network, now we need to learn weights of that neural network, thus we need gradients, thus we have to find derivates of the solution of an ODE with respect to parameters. And we have a standard tool for that in ODE theory, it's called “variational equations”, so let's just apply them and we're done. Case closed.

However, in the paper, the authors used much more elaborate approach, based on something called _adjoint state method_. It was strange: why we need some cryptic mathematical magic to solve such a standard problem? In various other discussions, I heard from time to time that these mysterious _adjoints_ are somehow related to backpropagation. However, all tutorials on adjoint state method I was able to find used a bunch of sophisticated infinite-dimensional optimization theory, and it was not clear for me, how this can be related to such a simple thing as backpropagation?

It was my surprise when I understood that adjoint state method in fact is based on a very simple idea. And now I want share it with you.

{{< callout note >}}

This post turned out to be rather long. I tried to make it as accessible as possible, so included detailed explanations of all the derivations. This led to inclusion of various equations that sometimes can look scary. Don't be afraid: there are also a lot of illustrations and informal descriptions to guide you through the story.

- In the [first part](#forward-and-backward), I recall briefly some notions from multidimensional calculus and present the main constructions that will be used later. The reader is expected to be familiar with matrix calculations, the notion of a derivative of multidimensional map and the chain rule, whilst the two latter will be recalled.

- The [second part](#backpropagation-in-neural-networks) is an introduction to the backpropagation in the usual dense neural networks. Here I present an exposition that is focused on the effective implementation of the backpropagation using matrix calculations and also has close ties with the adjoint state method in the neural ODEs. Nevertheless, I hope you will find something new and interesting about backpropagation from this part even if you are not interested in the neural ODEs.

- The [last part](#adjoint-state-method-in-neural-odes) is devoted to the adjoint state method. Here I expect some very basic knowledge of the ordinary differential equations. The main results from the ODE theory will be recalled.

That is not an easy journey, but I hope you will find it as exciting as I did. If you have any questions regarding this post, do not hesitate to get in touch on [Twitter](https://twitter.com/ilya_schurov).

Now, let's go!

{{< /callout >}}

## Forward and backward

### How to mupliply matrices

Before we begin with backpropagation and neural ODEs, let's talk about something very simple: about matrix multiplication.

Assume we have two square {{< math >}}$n \times n${{< /math >}} matrices, {{< math >}}$A${{< /math >}} and {{< math >}}$B${{< /math >}}, and {{< math >}}$n${{< /math >}}-dimensional vector (vector-column) {{< math >}}$x${{< /math >}}. Consider the following product:

{{< math >}}
$$ABx$$
{{< /math >}}

As matrix multiplication is associatative, we don't need any brackets in this formula. However, if we try to put them, we'll note that it can be done in two different ways: we can either write it like this:

{{< math >}}
$$(AB)x$$
{{< /math >}} 

or like this:

{{< math >}}
$$A(Bx).$$
{{< /math >}} 

Of course, we'll get the same results, but computationally these two formulas
are different. In the first case, we find matrix {{< math >}}$AB${{< /math >}},
that takes {{< math >}}$O(n^3)${{< /math >}} elementary multiplications, then keep
this new matrix in memory, that is {{< math >}}$O(n^2)${{< /math >}}, and then
multiply it on {{< math >}}$x${{< /math >}}. The last operation is cheep and
only needs {{< math >}}$O(n^2)${{< /math >}} operations.

In the second approach, we first find {{< math >}}$Bx${{< /math >}}, that is
cheap, {{< math >}}$O(n^2)${{< /math >}} operations and {{< math >}}$O(n)${{<
/math >}} memory. Than we multiply {{< math >}}$A${{< /math >}} by the result of
previous computation, that is again cheap. And we're done! So, the difference
between two method is dramatic: {{< math >}}$O(n^3)${{< /math >}} vs. {{< math
>}}$O(n^2)${{< /math >}} in operations and {{< math >}}$O(n^2)${{< /math >}} vs.
{{< math >}}$O(n)${{< /math >}} in memory. The second approach is much more
efficient!

Of course, it works only if we have only one vector {{< math >}}$x${{< /math >}} that should be multiplied by {{< math >}}$AB${{< /math >}}; if there are
many such vectors, it can be more efficient to find the product {{< math >}}$AB${{< /math >}} once and then reuse it. However, as we will see below, in
our problems, including backpropagation, this kind of calculation is effectively
one-time. 

The last thing I want to mention here is that if instead of vector-column $x$ we
consider vector-row {{< math >}}$u${{< /math >}} (mathematically speaking, rather 
covector than vector, if we represent vectors as vector-columns), and we want to
find a product {{< math >}}$uAB${{< /math >}}, this again can be done in two
different ways:

{{< math >}}
$$
\begin{equation}
\label{phiAB}
uAB=(uA)B=u (AB),
\end{equation}
$$
{{< /math >}}

and now the first one is much cheaper.

Does it sounds reasonable? If yes, congratulations: you understood the main idea of the adjoint state method!

### Derivatives and gradients

In what follows, we will be interested in maps from multidimensional spaces to multiminesional spaces (i.e. from $\mathbb R^n$ to $\mathbb R^m$ for some positive integer $n$ and $m$) and their derivatives. In general, we treat a derivative of a map $f\colon \mathbb R^n \to \mathbb R^m$ as the [Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant), i.e. matrix of partial derivatives of components of $f$. We will denote it by $\partial f(x)/\partial x$, sometimes ommitting $(x)$. This matrix has $n$ columns and $m$ rows. By definition of a derivative, for any vector $\Delta x \in \mathbb R^n$ from some neighborhood of $0$,

{{< math >}}
$$
f(x+\Delta x)-f(x)=\frac{\partial f(x)}{\partial x} \Delta x+o(\|v\|),
$$
{{< /math >}}

or, informally,

{{< math >}}
$$ f(x+\Delta x)-f(x) \approx \frac{\partial f(x)}{\partial x} \Delta x.
$$
{{< /math >}}

A useful illustration of this approximation is given on [Figure 1](#figure-ill-deriv): for map from $\mathbb R$ to $\mathbb R$, the length of an image of the small segment is approximately equal to the derivative multiplied by the length of the segment itself. If we replace segments with vectors based at $x$, the same illustration will work for multidimensional case.

{{< figure src="/img/adjoint-state/backprop-14.svg" width="90%" title="Figure 1. Illustration of a derivative" id="ill-deriv" >}}

In a special case when $m=1$, the derivative is a matrix with 1 row, i.e. it is a vector-row. In this case we also call it a *gradient* of $f$ and denote by {{< math >}}$\nabla_{\!x} f(x)${{< /math >}}. (Strictly speaking, one should call this vector-row differential, not gradient, because it is a covector, and gradient is a vector, but we'll discuss the difference between them next time.)

Let us also recall the well-known [chain rule](https://en.wikipedia.org/wiki/Chain_rule) that simply says that derivative of a composition $g\circ f$ is a product (i.e. composition) of the derivatives:

{{< math >}}$$
\begin{equation}
\label{chain-rule}
\frac{\partial (g\circ f(x))}{\partial x} = \frac{\partial g(h)}{\partial
h}\frac{\partial f(x)}{\partial x},
\end{equation}
$${{< /math >}}

where the first derivative is taken at point $h=f(x)$. This formula can be easily illustrated with the following picture.

{{< figure src="/img/adjoint-state/backprop-15.svg" width="90%" title="Figure 2. The chain rule" id="chainrule" >}}

We will also consider functions that depend on additional multidimensional parameter, usually denoted by $\theta \in \mathbb R^p$. Formally, such a function is just a map

{{< math >}}$$
f\colon \mathbb R^n \times \mathbb R^p \to \mathbb R^m,
$${{< /math >}}

i.e. it is a function of two vector arguments $f(x, \theta)$, but we usually write it like $f_\theta(x)$ instead. In this case, $\partial f_\theta/\partial x$ is a derivative with respect to argument $x$ (keeping $\theta$ fixed) and $\partial f_\theta / \partial \theta$ is a derivative with respect to parameter (keeping $x$ fixed). The following approximations take place:

{{< math >}}$$
\begin{gather}
\label{x-approx}
f_\theta(x+\Delta x)-f_\theta(x) \approx \frac{\partial f_\theta(x)}{\partial x} \Delta x,\\
\label{theta-approx}
f_{\theta+\Delta \theta}(x)-f_\theta(x) \approx \frac{\partial f_\theta(x)}{\partial \theta} \Delta \theta.\\
\end{gather}
$${{< /math >}}

### Gradient of composition

Now let us consider a map $G\colon \mathbb R^n \to \mathbb R$ that can be represented as a composition:

{{< math >}}
$$G(x)=g^N\circ g^{N-1} \circ\cdots \circ g^1(x),$$
{{< /math >}}

where $g^1, \ldots, g^N$ are some differentiable maps from multidimensional spaces to multidimensional spaces. (Superscripts do not denote powers here.)

{{< callout note >}}

It is very important for us that codomain of $G$ is one-dimensional. When we will discuss neural networks, $G$ will represent some loss function with values in $\mathbb R$. As $g^N$ is the last applied map, its codomain coincides with the codomain of $G$ and thus it is one-dimensional as well.

{{< /callout >}}

{{< figure src="/img/adjoint-state/backprop-5.svg" width="90%" title="Figure 3. Composition of several functions. All axes except the last one represent multidimensional spaces. The last axis is one-dimensional" id="composition" >}}

If we have a value $x$ and want to find $G(x)$, the algorithm is straightforward: we find $h_1:=g^1(x)$, put it into $g^2$, thus finding $h_2:=g^2(h_1)$, put it into $g^3$ and so on, the last step is $y=g^N(h_{N-1})$. The flow of calculation is forward, from smaller indexes to larger (right-to-left, if we look at the formula, or left-to-right, if we look at the picture). This is what usually called “forward pass” in the neural networks.

Now what if we want to find the gradient {{< math >}}$\nabla_{\! x}$ {{< /math >}} (or, in other terms, the derivative $\partial G / \partial x$)?

The chain rule \eqref{chain-rule} immediately gives us:

{{< math >}}
$$
\nabla_{\! x} G=
\frac{\partial g^N}{\partial h_{N-1}}
\frac{\partial g^{N-1}}{\partial h_{N-2}}\cdots
\frac{\partial g^2}{\partial h_1} \frac{\partial g^1}{\partial x}.
$$
{{< /math >}}

As we already said, $g^N$ maps to $\mathbb R$, and therefore one can denote the first multiplier in this product by {{< math >}}$\nabla_{\!h_{N-1}} g^N${{< /math >}}:

{{< math >}}
$$
\begin{equation}
\label{nablaG}
\nabla_{\! x} G=
\nabla_{\!h_{N-1}} g^N
\frac{\partial g^{N-1}}{\partial h_{N-2}}\cdots
\frac{\partial g^2}{\partial h_1} \frac{\partial g^1}{\partial x}.
\end{equation}
$$
{{< /math >}}

How to use this equation to find the gradient? First of all, we have to find $h_1, \ldots, h_{N-1}$, i.e. perform all the steps of the forward pass (except the last one). Then we have to find all the derivatives of functions $g^N$, $g^{N-1}$, …, $g^2$, $g^1$ at the corresponding points $h_{N-1}$, $h_{N-2}$, …, $h_1$, $x$. Then we have to multiply everything.

As the leftmost multiplier is a vector-row, we are in the situation very similar to equation $\eqref{phiAB}$: we have a vector-row that is multiplied to a product of matrices. Just like we discussed above, the most natural and efficient way is to do it left-to-right: we first find a product

{{< math >}}$$
\nabla_{\!h_{N-1}} g^N
\frac{\partial g^{N-1}}{\partial h_{N-2}},
$${{< /math >}}

obtain a new vector-row, multiply it by the next matrix, and so on. Now the calculation flow goes backward, from the terms with large indexes to the terms with small indexes (left-to-right if we look at the formula, or right-to-left if we look at the picture). This is what is known as _backward pass_.

Theoretically, one _can_ find the product in the right-hand side of equation $\eqref{nablaG}$ in a different order, e.g. right-to-left, but it would not be very efficient: one had to find and store some large intermediate matrices during the calculations. In our approach, we store only the initial matrices and intermediate vector-rows.

{{< callout note >}}

To summarise: consider a function that is given as a composition. There are two natural problems associated with it: to find its value at a particular point and to find its gradient. The flow of calculation of the value is forward, and the flow of calculation of the gradient is backward. To perform backward pass, we need to perform forward pass first to be able to find the derivatives that are needed in the backward pass.

{{< /callout >}}

### Truncated compositions

It is instructive to study intermediate steps of the forward and the backward passes. Let's begin with the forward pass.

For each integer $j$, $0 < j \le N$, consider the following “truncated” composition:

{{< math >}}$$
G^{0:j}(x):=g^j \circ g^{j-1}\circ \cdots \circ g^{1}(x).
$${{< /math >}}



{{< figure src="/img/adjoint-state/backprop-6.svg" width="90%" title="Figure 4. Forward truncated compositions" id="forward-trunc" >}}

Each $G^{0:j}$ shows how $h_j$ depends on $x$. In the forward pass, we find consequently $G^{0:1}(x)$, $G^{0:2}(x)$, and so on. At step $j$ we find $G^{0:j}(x)$ applying $g^j$ to the result of the previous step. At the last step $N$ we find $G^{0:N}(x)=G(x)$. That's literally straightforward.

To consider backward pass, we need a different “truncation”. For each integer $i$, $0 \le i < N$, let

{{< math >}}$$
\begin{equation}
\label{truncated}
G^{i:N}(h_i):=g^N \circ g^{N-1}\circ \cdots \circ g^{i+1}(h_i).
\end{equation}
$${{< /math >}}

{{< figure src="/img/adjoint-state/backprop-7.svg" width="90%" title="Figure 5. Backward truncated compositions" id="back-trunc" >}}

It is a map with codomain $\mathbb R$ that shows how $y$ depends on $h_i$. Its gradient can be found using the chain rule:

{{< math >}}$$
\nabla_{\! h_{i}} G^{i:N}=
\nabla_{\!h_{N-1}} g^N
\frac{\partial g^{N-1}}{\partial h_{N-2}}\cdots
\frac{\partial g^{i+1}}{\partial h_{i}}.
$${{< /math >}}

One can see that the right-hand side of this equation is a truncated version of equation $\eqref{nablaG}$: we only keep the first $(N-i)$ multipliers. And this is exactly what backward pass calculates at each step: for each $i$ decreasing from $(N-1)$ to $0$, we find {{< math >}}$ \nabla_{\! h_{i}} G^{i:N}${{< /math >}} multiplying the result of the previous step to $\partial g^{i+1}/\partial h_i$:

{{< math >}}$$
\begin{equation}
\label{nablastep}
\nabla_{\! h_{i}} G^{i:N}=\nabla_{\! h_{i+1}}G^{i+1:N}\cdot \frac{\partial g^{i+1}}{\partial h_{i}}.
\end{equation}
$${{< /math >}}

Clearly, $G^{0:N}=G$ and at the last step we obtain gradient of $G$.

Here we see that forward and backward passes are very similar in nature, but at the same time has substantial difference. In the forward pass, the domain of each function $G^{0:j}$ we consider is fixed (it's $\mathbb R^n$, the same as the domain of $G$), but codomain shifts in “forward” direction, see [Figure 4](#figure-forward-trunc). In the backward pass, the codomain of the function $G^{i:N}$ is fixed (it's $\mathbb R$, the same as the codomain of $G$), but domain shifts in “backward” direction: at step $i$, argument of $G^{i:N}$ is $h_i$, and $i$ decreases, see [Figure 5](#figure-back-trunc).

{{< callout note >}}

Let us summarise with an informal description. During the calculations, we want to begin with a simple object and transform it to the object we need. In the forward pass, the simple object is just a vector $x$, that “lives” at the “beginning” of the composition. We transform it by application of the corresponding $g^j$'s until we pull it through the whole composition and get $G(x)$. In the backward pass, the simple object we begin with is the gradient $\nabla_{\\! h_{N-1}} g^{N}$. We can be sure it's “simple” (i.e. a vector-row, not a full matrix) because $g^N$'s codomain is guaranteed to be one-dimensional. This gradient “lives“ at the “end” of the composition, and it is natural to transform it by extending “backward”. When we pull it through the whole composition, we get the desired gradient $\nabla_{\\! x} G$.

{{< /callout >}}

{{< spoiler text="Interested in mathematical details? Click here!" >}}

I cannot resist the temptation to discuss a bit more mathematical perspective on equation \eqref{nablastep} and add some rigour to the informal description above. To this end, we have to define formally the spaces where the gradients live.

Let's say that for each $i=1,\ldots, N$, $g^i$ is a map from $\mathcal M_{i-1}=\mathbb R^{n_{i-1}}$ to $\mathcal M_{i}=\mathbb R^{n_i}$, $n_N=1$. As before, $h_i = g^i(h_{i-1})$ and $h_0=x$. The gradient $\nabla_{\\! h_i} G^{i:N}$ is a linear map that acts on vectors $\Delta h_i$. It is natural to think about this vectors as based at point $h_i$. The vector space of all such vectors is called a _tangent space_ of $\mathcal M_{i}$ at point $h_i$; it is denoted by $T_{h_i} \mathcal M_i$. Thus the gradient $\nabla_{\\! h_i} G^{i:N}$ is a linear map from $T_{h_i} \mathcal M_i$ to $\mathbb R$, such linear maps (with codomain $\mathbb R$) also known as _linear functionals_ or _covectors_.

The set of all linear functionals defined on some vector space $V$ is again a vector space: one can add linear functionals to each other and multiply them by real numbers. This space is called _dual space_ to $V$ and denoted by $V^*$. The dual space to the tangent space $T_{h_i} \mathcal M_i$ has a special name: it's called a _cotangent space_ of $\mathcal M_i$ at point $h_i$ and denoted by $T_{h_i}^\* \mathcal M_i$.

So, the gradient $\nabla_{\\! h_i} G^{i:N}$ belongs to the cotangent space $T_{h_i}^\* \mathcal M_i$.

Now let's consider a derivative of $g^i$ at point $h_{i-1}$. It is a linear map that transforms vectors based at point $h_{i-1}$ to vectors based at point $h_i$, so it's a map

{{< math >}}$$
\frac{\partial g^i(h_{i-1})}{\partial h_{i-1}}\colon T_{h_{i-1}} \mathcal M_{i-1}
\to T_{h_i} \mathcal M_i.
$${{< /math >}}

Now I want to consider a very abstract setup that distills the main relations between the objects we introduced so far. We have two vector spaces, denote them by $V$ and $W$, and a linear map

{{< math >}}$$
\mathcal A\colon V \to W.
$${{< /math >}}

Consider the dual spaces $V^\*$ and $W^\*$. Then $\mathcal A$ naturally induces a map

{{< math >}}
$$\mathcal A^\*\colon W^\* \to V^\*$$ 
{{< /math >}}

(Compare this equation with the equation above. You see: $V$ and $W$ are swapped!) For each covector $\psi \in V^*$, we define its image $\mathcal A^\* \psi$ with the following formula:

{{< math >}}$$
(\mathcal A^\* \psi)(v)=\psi (\mathcal A v)\quad \text{for each $v\in V$.}
$${{< /math >}}

What is written here? First, as $\mathcal A^\*$ acts from $W^\*$ to $V^\*$, the image $\mathcal A^\* \psi$ is a covector in $V^\*$, i.e. it is a linear functional defined on $V$. To define this functional, we have to define how it acts on vectors. The value of $\mathcal A^* \psi$ on a vector $v \in V$ is defined in the following way: first, we apply operator $\mathcal A$ to $v$, get a new vector that belongs to $W$, then apply functional $\psi$ (that works on $W$) to this vector. The result is the value of the functional $\mathcal A^\* \psi$ on the vector $v$.

Operator $\mathcal A^\*$ is called an _adjoint_ to $\mathcal A$. If you think about it a little bit, you see that this construction is very-very natural. In fact, it is an example of [contravariant Hom-functor](https://en.wikipedia.org/wiki/Hom_functor) in category theory, but we will not dive into such depths.

Let's return to our derivatives. Now we can consider an adjoint to the derivative $\partial g^i / \partial h_{i-1}$:

{{< math >}}$$
\left(\frac{\partial g^i(h\_{i-1})}{\partial h\_{i-1}}\right)^\* \colon T^\*\_{h_{i}} \mathcal M\_{i}
\to T^*\_{h\_{i-1}} \mathcal M\_{i-1}.
$${{< /math >}}

And equation \eqref{nablastep} takes form:

{{< math >}}$$
\nabla_{\\\! h_{i}} G^{i:N}=\left(\frac{\partial g^{i+1}}{\partial h_{i}}\right)^\*
\nabla_{\\\! h_{i+1}}G^{i+1:N}.
$${{< /math >}}

(Just check from the definition of adjoint that this is indeed equivalent to \eqref{nablastep}.)

So, it is the adjoint to the derivative of $g^i$ that acts on the gradients! And as it is an adjoint, it acts “backwards” relative to the action of the derivative itself (and thus to the map $g^i$). So it solves the mystery of “backwardness” in backpropagation. Mathematically speaking, we are simply applying contravariant Hom-functor and it reverses all the arrows. That's it!

{{< /spoiler >}}

Now let's look how it works in the neural networks.

## Backpropagation in neural networks

The backpropagation algorithm is very much well-known, but I present here an exposition that is specifically designed to stress the relation of the backprop and the adjoint state method in the neural ODEs.

### The usual neural network

For simplicity, assume we have a neural network that consists only of three layers, two of them are hidden. Layer number {{< math >}}$i${{< /math >}}, {{< math >}}$i=1,2,3${{< /math >}}, transforms its input to output using a function

{{< math >}}$$
    f^{i}_\theta\colon \mathbb R^{n_{i-1}} \to \mathbb R^{n_i},
$${{< /math >}} 

where {{< math >}}$\theta\in \mathbb R^p${{< /math >}}
is a vector of all parameters of the neural network (i.e. all weights and
biases), {{< math >}}$n_i${{< /math >}} is the dimensionality of the output of
{{< math >}}$i${{< /math >}}'th layer, {{< math >}}$n_0${{< /math >}} is the input
dimensionality of the network. Usually each layer depends only
on a subset of parameters in {{< math >}}$\theta${{< /math >}} and implements an
affine function in elementwise composition with nonlinear activation function,
but we are not interested in such architecture details now and consider rather
general case. The full network
defines a function

{{< math >}}$$
    f_{\theta}(x) := f^{3}_\theta\circ f^{2}_\theta \circ f^{1}_\theta(x)
$${{< /math >}}

This is a very similar to that discussed in the [previous section](#gradient-of-composition). The main difference is that now all the functions in this composition depend also on the parameter $\theta$.

Our composition can be visualized in the following way:

{{< figure src="/img/adjoint-state/backprop-1.svg" width="90%" title="Figure 6. Three-layer neural network" id="three-layer" >}}

We also have some loss function $L(y, y_{true})$ (e.g. in case of quadratic loss, $L(y, y_{true})=(y-y_{true})^2$). If we put the output of the network into the loss, we obtain an optimization objective

{{< math >}}$$
\mathcal L(\theta) := L(f_\theta(x_{input}), y_{true})
$${{< /math >}}

that should be minimized during the training. For simplicity, we are discussing the loss at one datapoint; in the real settings, we would average this over the batch.

### Gradient of the loss

To perform the optimization of $\mathcal L(\theta)$ with gradient descent, one need to find its gradient. Chain rule immediately gives:

{{< math >}}$$
\begin{equation}
\label{nablamathcalL}
\nabla_{\!\theta} \mathcal L(\theta) = \nabla_{\!y} L \cdot \frac{\partial
f_\theta(x_{input})}{\partial \theta},
\end{equation}
$${{< /math >}}

where the first multiplier is a gradient of {{< math >}}$L${{< /math >}}, i.e. vector-row of dimensionality {{< math >}}$n_3${{< /math >}} (dimensionality of the output layer), and the second multiplier is a {{< math >}}$(n_3 \times p)${{< /math >}}-matrix.

It is easy to find {{< math >}}$\nabla_{\!y} L ${{< /math >}} provided that {{< math >}}$y${{< /math >}} is already calculated (i.e. in the case of quadratic loss, it's just {{< math >}}$(2y-2y_{true})${{< /math >}}). To find the second multiplier, one have to decompose {{< math >}}$f_\theta${{< /math >}} into a composition of subsequent layer maps and again apply the chain rule. In contrast with the [previous part](#gradient-of-composition), each function now depends not only on its argument, but also on the parameter $\theta$. This leads to new phenomena and I'd like to study it with some not-so-rigorous visualization.

Let's fix some small vector {{< math >}}$\Delta \theta \in \mathbb R^p${{< /math >}} and consider a “trajectory” of $x_{input}$ under the action of the “perturbed” maps {{< math >}}$f^{i}_{\theta+\Delta \theta}${{< /math >}}, {{< math >}}$i=1,2,3${{< /math >}}:

{{< figure src="/img/adjoint-state/backprop-2.svg" width="90%" title="Figure 7. What happens with the output of neural network if we slighly change parameters." id="nn-change" >}}

The difference between outputs {{< math >}}$f_{\theta+\Delta \theta}(x_{input})-f_\theta(x_{input})${{< /math >}} is approximately equal to

{{< math >}}$$
\frac{\partial f_\theta}{\partial \theta} \Delta \theta
$${{< /math >}}

provided that {{< math >}}$\Delta \theta${{< /math >}} is small by the definition of the derivative, see equation \eqref{theta-approx}. (Note that on the picture this difference is represented by a segment on a line, but in reality it's a {{< math >}}$n_3${{< /math >}}-dimensional vector.)

### Derivative of the network

Now let's decompose this difference into a sum of three parts in the following way (see [Figure 8](#figure-decomp-net) below). For each of the intermediate points of the unperturbed trajectory (i.e. $f^1_\theta(x_{input})$ and $f^2_\theta\circ f^1_\theta(x_{input})$), we consider a trajectory of the perturbed network that starts from this point. These trajectories split the segment $[f_{\theta}(x_{input}), f_{\theta+\Delta \theta}(x_{input})]$ into three smaller segments denoted (from top to bottom) by {{< math >}}$\Delta^3_1${{< /math >}}, {{< math >}}$\Delta^3_2${{< /math >}} and {{< math >}}$\Delta^3_3${{< /math >}}.

{{< figure src="/img/adjoint-state/backprop-4.svg" width="90%" title="Figure 8. Decomposition of the network's derivative" id="decomp-net" >}}

Here all the red arrows represent the action of the corresponding {{< math >}}$f^i_{\theta+\Delta \theta}${{< /math >}}.

{{< callout note >}}

Of course, this is not an exact figure: in reality, the output space is multidimensional, and we do not split a segment into smaller segments. Nevertheless, the argument is correct: we can represent a vector from $f_\theta(x_{input})$ to $f_{\theta+\Delta \theta}(x_{input})$ as a sum of three vectors given as a difference between the values of the corresponding compositions. So, no cheating here!

{{< /callout >}}

We will approximate each of the smaller parts using the appropriate derivatives. Let's begin with {{< math >}}$\Delta^3_3${{< /math >}}. It measures the difference between the images of some point under action of {{< math >}}$f^3_{\theta+\Delta \theta}${{< /math >}} and {{< math >}}$f^3_{\theta}${{< /math >}}. Again, we use the definition of a derivate (particularly, equation \eqref{theta-approx}) and get the following approximation:

{{< math >}}$$
\Delta^3_3 \approx \frac{\partial f^3_{\theta}}{\partial \theta} \Delta \theta.
$${{< /math >}}

That was easy. Now consider {{< math >}}$\Delta^3_2${{< /math >}}. Here we have two steps. At the first step, we have two functions, {{< math >}}$f^2_\theta${{< /math >}} and {{< math >}}$f^2_{\theta+\Delta \theta}${{< /math >}} that are applied to the same point. The difference between the images is denoted by {{< math >}}$\Delta^2_2${{< /math >}} and like in the previous case is approximately equal to

{{< math >}}$$
\Delta^2_2 \approx \frac{\partial f^2_{\theta}}{\partial \theta} \Delta \theta.
$${{< /math >}}

At the second step, we have one function, {{< math >}}$f^2_{\theta+\Delta \theta}${{< /math >}}, that is applied to two different points. To find the difference between the images now, we have to use the derivative of {{< math >}}$f^3_{\theta+\Delta \theta}(h_2)${{< /math >}} with respect to its argument {{< math >}}$h_2${{< /math >}}, see equation \eqref{x-approx}. Namely:

{{< math >}}$$
\Delta^3_2 \approx 
\frac{\partial f^3_{\theta + \Delta \theta}}{\partial h_2}\Delta^2_2 \approx 
\frac{\partial f^3_{\theta+\Delta \theta}}{\partial h_2} \frac{\partial
f^2_{\theta}}{\partial \theta} \Delta \theta.
$${{< /math >}}

And finally for {{< math >}}$\Delta^3_1${{< /math >}} we have three steps:

{{< math >}}$$
\Delta^3_1 \approx 
\frac{\partial f^3_{\theta + \Delta \theta}}{\partial h_2}\Delta^2_1 \approx 
\frac{\partial f^3_{\theta+\Delta \theta}}{\partial h_2} \frac{\partial
f^2_{\theta+\Delta \theta}}{\partial h_1} \Delta_1^1 \approx
\frac{\partial f^3_{\theta+\Delta \theta}}{\partial h_2} \frac{\partial
f^2_{\theta+\Delta \theta}}{\partial h_1}  \frac{\partial f^1_{\theta}}{\partial
\theta} \Delta \theta.
$${{< /math >}}

Now let's sum up everything:

{{< math >}}$$
\begin{align*}
\frac{\partial f_{\theta}}{\partial \theta}\Delta \theta \approx {} & \Delta^3_3 +
\Delta^3_2 + \Delta^3_1 \approx  \\
& \left(
\frac{\partial f^3_{\theta}}{\partial \theta} + 
\frac{\partial f^3_{\theta+\Delta \theta}}{\partial h_2} \frac{\partial
f^2_{\theta}}{\partial \theta} +
\frac{\partial f^3_{\theta+\Delta \theta}}{\partial h_2} \frac{\partial
f^2_{\theta+\Delta \theta}}{\partial h_1}  \frac{\partial f^1_{\theta}}{\partial
\theta}\right) \Delta \theta.
\end{align*}
$$
{{< /math >}}

As $\Delta \theta$ tends to zero, the approximations become more and more precise, and now one can easily believe that

{{< math >}}$$
\begin{equation}
\label{partialftheta}
\frac{\partial f_{\theta}}{\partial \theta} = 
\frac{\partial f^3_{\theta}}{\partial \theta} + 
\frac{\partial f^3_{\theta}}{\partial h_2} \frac{\partial
f^2_{\theta}}{\partial \theta} +
\frac{\partial f^3_{\theta}}{\partial h_2} \frac{\partial
f^2_{\theta}}{\partial h_1}  \frac{\partial f^1_{\theta}}{\partial
\theta}.
\end{equation}
$${{< /math >}}

We used a lot of informal derivations with “approximate equal” signs that does not count as a rigorous proof. (Do not try to sell it to your Calculus professor, unless it's me!) They can be easily replaced with several applications of the chain rule, but I want to make clear where each term in this formula came from, and it was easier to do that with the informal picture.

{{< callout note >}}

Let's look at the last formula again. We see that to find a derivative of the network with respect to the parameter $\theta$, we have to account for two effects:

1. Change of the parameter $\theta$ affects output of a particular layer.

2. Change of the output of a layer affects outputs of the subsequent layers, even if we ignore change of the parameter for them.

The first effect is addressed by $\partial f_\theta^i / \partial \theta$ multipliers. The second effect is addressed by $\partial f_\theta^i / \partial h^{i-1}$ multipliers. The derivative is a sum of the corresponding effects for each layer.

{{< /callout >}}

### Back to the gradient

Now let's use the equation for the derivative of $f$ to find a gradient of $\mathcal L$. We put $\eqref{partialftheta}$ to $\eqref{nablamathcalL}$ and obtain:

{{< math >}}
$$
\begin{align*}
\nabla_{\!\theta} \mathcal L= {} &\nabla_{\!y} L \cdot \frac{\partial
f_\theta}{\partial \theta}=\\
& \nabla_{\!y} L \cdot \left(
\frac{\partial f^3_{\theta}}{\partial \theta} + 
\frac{\partial f^3_{\theta}}{\partial h_2} \frac{\partial
f^2_{\theta}}{\partial \theta} +
\frac{\partial f^3_{\theta}}{\partial h_2} \frac{\partial
f^2_{\theta}}{\partial h_1}  \frac{\partial f^1_{\theta}}{\partial
\theta}\right)=\\
& 
\nabla_{\!y} L \frac{\partial f^3_{\theta}}{\partial \theta} + 
{\nabla_{\!y} L 
\frac{\partial f^3_{\theta}}{\partial h_2}}
\frac{\partial f^2_{\theta}}{\partial \theta} +
{\nabla_{\!y} L 
\frac{\partial f^3_{\theta}}{\partial h_2}} 
\frac{\partial
f^2_{\theta}}{\partial h_1}  \frac{\partial f^1_{\theta}}{\partial
\theta}.
\end{align*}
$$
{{< /math >}}

Note the familiar pattern? In each term, we have vector-row {{< math >}}$\nabla_{\!y}L${{< /math >}} that is multiplied by a sequence of matrices. That means we need to multiply it left-to-right. Moreover, if we look closer, we see there are common parts in the second and the third summands:

{{< math >}}
$$
\begin{align}
\nonumber
\nabla_{\!\theta} \mathcal L(\theta)= 
\nabla_{\!y} L \frac{\partial f^3_{\theta}}{\partial \theta} & + 
{\color{teal}\left(\nabla_{\!y} L 
\frac{\partial f^3_{\theta}}{\partial h_2}\right)}
\frac{\partial f^2_{\theta}}{\partial \theta} \\
\label{nablamcL}
&+ {\color{teal} \left(\nabla_{\!y} L 
\frac{\partial f^3_{\theta}}{\partial h_2}\right)} 
\frac{\partial
f^2_{\theta}}{\partial h_1}  \frac{\partial f^1_{\theta}}{\partial
\theta}.
\end{align}
$$
{{< /math >}}

It means that we can find this common part {{< math >}}$\nabla_{\!y} L \cdot \partial f^3_{\theta}/\partial h_2${{< /math >}} when calculate the second summand, and then reuse it when calculating the third summand. That's allows us to do the calculations even more efficiently. And this is not a coincidence: the same trick works in deeper networks as well!

### General algorithm for backpropagation

Previously we considered a network with three layers. Now I want to generalize the formula for loss gradient to the general case of the network with $N$ layers.

Note that in each summand on the right-hand side of equation $\eqref{nablamcL}$ only the last multiplier is a derivative with respect to the parameters $\theta$. The beginning part of each product is a gradient of “truncated composition” like in $\eqref{truncated}$ with respect to the output of some of the hidden layer. Indeed, the chain rule implies:

{{< math >}}$$
\nabla_{\!y} L 
\frac{\partial f^3_{\theta}}{\partial h_2} = 
\nabla_{\!h_2}(L\circ f^3_\theta)
$${{< /math >}}

and

{{< math >}}$$
\nabla_{\!y} L 
\frac{\partial f^3_{\theta}}{\partial h_2} \frac{\partial f^2_\theta}{\partial
h_1} = 
\nabla_{\!h_1}(L\circ f^3_\theta \circ f^2_\theta).
$${{< /math >}}

In other words, these multipliers show how the loss function depends on the output value of the second and the first hidden layers correspondingly. To simplify the notation, we will write {{< math >}}$\nabla_{\!h_2} L${{< /math >}} and {{< math >}}$\nabla_{\!h_1} L${{< /math >}} and omit the subsequent compositions with the layer maps.

Let us denote $h_3 \equiv y$. Then the following nice relations take place:

{{< math >}}$$
\begin{align}
\label{nablaLstep1}
\nabla_{\!h_2} L & = \nabla_{\!h_3} L \cdot  
    \frac{\partial f^3_\theta}{\partial h_2}, \\
\label{nablaLstep2}
\nabla_{\!h_1} L & = \nabla_{\!h_2} L \cdot 
    \frac{\partial f^2_\theta}{\partial h_1}.
\end{align}
$${{< /math >}}

This is actually just a restatement of the general equation $\eqref{nablastep}$ for truncated compositions.

With this new notation, we can rewrite the formula for the gradient $\eqref{nablamcL}$ in the following compact way:

{{< math >}}$$
\nabla_{\!\theta} \mathcal L(\theta)=
\nabla_{\!h_3} L \frac{\partial f^3_{\theta}}{\partial \theta} + 
\nabla_{\!h_2} L 
\frac{\partial
f^2_{\theta}}{\partial \theta} +
\nabla_{\!h_1} L 
\frac{\partial f^1_{\theta}}{\partial
\theta}.
$$
{{< /math >}}

And this can be easily generalized to the case of $N$ layers:

{{< math >}}$$
\begin{equation}
\label{nabla-L-sum}
\nabla_{\!\theta} \mathcal L(\theta)=\sum_{i=N}^1 \nabla_{\!h_i}L \frac{\partial
f_\theta^i}{\partial \theta},
\end{equation}
$${{< /math >}}

where $h_N\equiv y$. (I am intentionally start the summation from $i=N$ and then decrease $i$ until it equals $1$ for consistency with the previous equation and the algorithm below.) This equation looks simple, and, moreover, there exists efficient algorithm to calculate its right-hand side. First, note that equations $\eqref{nablaLstep1}$-$\eqref{nablaLstep2}$ are immediately generalized as

{{< math >}}$$
\begin{equation}
\label{nablaLstep}
\nabla_{\!h_i} L  = \nabla_{\!h_{i+1}} L \cdot  
    \frac{\partial f^{i+1}_\theta}{\partial h_i}, \quad i = N, \ldots, 1.
\end{equation}
$${{< /math >}}

(Again, this is just equation $\eqref{nablastep}$ with different notation.)

Then we have the following algorithm:

{{< callout note >}}

1. Do the forward pass to find values $h_1$, $h_2$, …, $h_N\equiv y$.

2. Initialize the accumulator to store the gradient with zero $p$-dimensional vector-row. (Recall that $p$ is the number of parameters.)

3. Find {{< math >}}$\nabla_{\\!y} L(y)${{< /math >}}.

4. Find {{< math >}}$(\nabla_{\\!y} L) (\partial f^N_\theta(h_{N-1}) / \partial \theta)${{< /math >}} and add it to the accumulator.

5. For each $i$ from $(N - 1)$ to $1$:

- Find {{< math >}}$\nabla_{\\!h_i} L${{< /math >}} by multiplication of the previously found {{< math >}}$\nabla_{\\!h_{i+1}} L${{< /math >}} to the derivative {{< math >}}$\partial f^{i+1}_\theta(h\_{i}) / \partial h\_{i}${{< /math >}}

- Find {{< math >}}$(\nabla_{\\! h_i} L) (\partial f^{i}\_\theta(h_{i-1}) / \partial \theta)${{< /math >}} and add it to the accumulator.

6. Return the value of the accumulator.

{{< /callout >}}

That's it. That's how backpropagation allows to efficiently calculate gradients in the usual neural networks. Now let's pass to neural ODEs.

## Adjoint State Method in Neural ODEs

### Neural ODEs: quick recap

As we discussed previously, during the forward pass, the usual neural network transforms its inputs to outputs in a sequence of discrete steps: one step corresponds to one layer. In neural ODEs, this transformation is performed continously. Now we don't have a discrete set of layers, enumerated by natural numbers $i=1, \ldots, N$. Instead, we have a continuum set of “moments of time”, represented as a segment $[0, T]$. At each moment, we specify “infinitesimal transformation” that occurs when the value passes through this moment.

Technically, neural ODEs are obtained as a limit case of so-called _residual networks_ (also known as _ResNets_). In the residual networks, the output value of $i$'th layer is determined as

{{< math >}}$$
h_{i}=h_{i-1} + f^{i}_\theta(h_{i-1}).
$${{< /math >}}

The difference with the usual neural networks is the presence of “$h_{i-1}+{}$” term. It allows the network to learn more efficiently: ResNets can be very deep and still learnable. (Note that to write such an equation we must demand that the dimensionality of each layer be the same and equal to the dimensionality of the input space.) Now we can imagine a very-very deep network: to make sure that the output doesn't tend to infinity, let's add some small coefficient that will decrease as the network depth increases:

{{< math >}}$$
h_{i}=h_{i-1} + \varepsilon f^{i}_\theta(h_{i-1}), \quad i=1,\ldots, N,
$${{< /math >}}

where $\varepsilon \sim 1/N$. And that's an equation of well-known [Euler method](https://en.wikipedia.org/wiki/Euler_method) of the numerical solution of a differential equation! As $N$ tends to infinity, the sequence of values $h_i$ tends to a solution of the corresponding differential equation. That's the rationale.

Now the formal settings. Consider a differential equation

{{< math >}}$$
\dot x(t) = f_\theta(t, x(t)),
$${{< /math >}}

where $x$ is a function, $x(t) \in \mathbb R^n$, $n$ is the dimensionality of the input space, $\dot x(t)$ is a derivative $dx(t)/dt$, $f_\theta$ is a smooth function that depends on the vector of parameters $\theta$. Denote the input of the network by $x_{input} \in \mathbb R^n.$ Consider a point $(t=0, x=x_{input})$: that's our starting point. We can find a solution of the differential equation whose graph passes through this point. Let's denote this solution by $\varphi(t; x_{input}; \theta)$. By construction, $\varphi(0; x_{input}; \theta)=x_{input}$. The output of the network, by definition, is the value of this solution at moment $T$, where $T$ is some fixed positive number. If one changes the initial value $x_{input}$, the solution changes as well and so does the output. In other words, our network defines a map

{{< math >}}$$
\begin{align*}
g^{0:T}_\theta & \colon \mathbb R^n  \to \mathbb R^n, \\
g^{0:T}_\theta & (x_{input}) =\varphi(T; x_{input};\theta),
\end{align*}
$${{< /math >}}

also known as _phase flow map_ of our differential equation.



{{< figure src="/img/adjoint-state/backprop-8.svg" width="90%" title="Figure 9. The usual neural network vs. neural ODE" >}}

{{< callout note >}}

There is a “dictionary” between the usual neural networks and neural ODEs. The moment of time $t$ corresponds to the index of layer $i$. For a given $t$, the function $f_\theta(t, x)$, seen as a function of $x$, corresponds to the layer function $f^{i}\_\theta$. The phase flow map $g^{0:T}\_\theta$ corresponds to the composition of the layer functions $f^N_\theta \circ \cdots \circ f^1\_\theta$ (compare also with the notation in the section [Truncated compositions](#truncated-compositions)).

{{< /callout >}}

Okay, now if we have a differential equation, we have a corresponding map from inputs to outputs. But how we define this differential equation? How to define the function $f_\theta(t, x)$. And that's easy: let's say it is given by another (usual) neural network! In this case, $\theta$ is a vector of parameters of this network (weights and biases), and it determines the function $f_\theta$ and therefore determines the map $g^{0:T}_\theta$.

How to learn $\theta$?

### Derivative of the solution of ODE

As usual, we have some loss function $L(y, y_{true})$ and can define our training objective as

{{< math >}}$$
\mathcal L(\theta)=L(g^{0:T}_\theta(x_{input}), y_{true})
$${{< /math >}}

and its gradient is immediately given by

{{< math >}}$$
\begin{equation}
\label{nablaLode}
\nabla_{\! \theta} \mathcal L(\theta) = \nabla_{\! y}L\cdot \frac{\partial g^{0:T}_\theta(x_{input})}{\partial \theta}.
\end{equation}
$${{< /math >}}

The second multiplier in the right-hand part can be also written as

{{< math >}}$$
\frac{\partial g^{0:T}_\theta(x_{input})}{\partial \theta}=\frac{\partial
\varphi(x_{input}; T; \theta)}{\partial \theta},
$${{< /math >}}

so we are interested in the derivative of the solution of a differential equation with respect to the parameter.

It is a well-studied topic in the theory of ODE. We have a theorem here.

{{< callout note >}}

**Theorem.** For each $t\in [0, T]$ and some fixed $\theta$, let us denote

{{< math >}}$$
v(t) := \frac{\partial \varphi(x\_{input}; t; \theta)}{\partial \theta}.
$${{< /math >}}

So $v(t)$ measures how the solution of our equation at point $t$ depends on the parameters $\theta.$ Then (under reasonable assumptions) $v$ satisfies the following differential equation:

{{< math >}}$$
\begin{equation}
\label{var-u-theta}
\dot v = \frac{\partial f\_\theta(t, x(t))}{\partial \theta} + \frac{\partial
f_\theta(t, x(t))}{\partial x} v,
\end{equation}
$${{< /math >}}

where $x(t)=\varphi(t; x\_{input}; \theta)$.

{{< /callout >}}

{{< spoiler text="What initial condition on $v(0)$ should we impose? How do you think?" >}}

We know by definition, that $\varphi(x_{input}; 0; \theta)=x_{input}$ and therefore does not depend on $\theta$. So the initial condition is $v(0)=0$ (zero matrix of appropriate size).

{{< /spoiler >}}

Now we can solve equation $\eqref{var-u-theta}$, find $v(T)$ and that's all. What's the problem with this approach?

In fact, $v(t)$ is a large $(n\times p)$-matrix, it contains the derivatives of all $n$ components of the solution with respect to every parameter. It can be expensive to calculate this matrix, especially if $n$ is large. And we don't need it by itself: what we are interested in is the product of this matrix to $\nabla_{\\! y}L$. Basically, to solve this equation is like to find a product in equation $\eqref{nablaG}$ from right to left: possible, but not optimal. How to do it in a smart way? Of course, with backpropagation!

### Perturbed trajectories

We will follow the ideas discussed in the section [Derivative of the network](#derivative-of-the-network). First, let us consider a network with perturbed vector of parameters, i.e. instead of $f(t, x, \theta)$ we consider an equation given by $f(t, x, \theta+\Delta \theta)$ for some small value of $\Delta \theta$. Consider two solutions that pass through the same point $(0, x_{input})$: the initial solution $x(t; \theta)$ and the perturbed solution $x(t; \theta+\Delta \theta)$. (From now on, we fixed the input value and skip $x_{input}$ in the notation.) We also introduce pertubed phase flow $g^{0:T}_{\theta+\Delta \theta}$.

The difference between the solutions at moment $T$ can be approximated using the derivative of the phase flow map with respect to $\theta$:

{{< math >}}$$
\varphi(T; \theta+\Delta \theta)-\varphi(T; \theta)  = 
g^{0:T}_{\theta+\Delta \theta}(x_{input}) -
g^{0:T}_{\theta}(x_{input}) \approx 
\frac{\partial g^{0:T}_\theta}{\partial \theta} \Delta \theta.
$${{< /math >}}

{{< figure src="/img/adjoint-state/backprop-9.svg" width="90%" title="Figure 10. The initial and the perturbed solutions" id='initial-pert' >}}

Our idea is to decompose this difference into sum of smaller differences like we did in the section [Derivative of the network](#derivative-of-the-network) (see [Figure 8](#figure-decomp)). For this, we need to introduce a bit more notation.

Consider arbitrary moment $t^* \in [0, T]$ and a point {{< math >}}$(t^*, x(t^*; \theta))${{< /math >}} that lies on the graph of the unperturbed solution (also called _unperturbed trajectory_). Sometimes for brevity we will skip $\theta$ in the notation for the unperturbed solutions and write simply $x(t^\*)$. Now let's consider a solution of _perturbed_ system through this point, see [Figure 11](#figure-additional-perturbed) below. The value of this solution at point $T$ is denoted by

{{< math >}}$$
g^{t^*:T}_{\theta+\Delta \theta}(x(t^*)),
$${{< /math >}}

thus giving us a flow map that transforms points over $t^*$ to points over $T$ under the action of the perturbed system. (Compare the notation with equation $\eqref{truncated}$ and Figure 3 at the section [Truncated compositions](#truncated-compositions).)

{{< figure src="/img/adjoint-state/backprop-10.svg" width="90%" title="Figure 11. Additional perturbed trajectory" id='additional-perturbed' >}}

Now we are ready to construct decomposition like in [Figure 6](#figure-decomp-net). Let's do it!

### Derivative decomposition

Let us divide the segment $[0, T]$ into $K$ smaller segments of equal length and denote the endpoints of these segments by $t_0$, $t_1$, …, $t_K$, where $t_0=0$ and $t_K=T$.

{{< figure src="/img/adjoint-state/backprop-11.svg" width="90%" title="Figure 12. Many perturbed trajectories" id="many-perturbed" >}}

For each point $(t_j, x(t_j))$, we consider a solution of the perturbed system through this point. The corresponding trajectories divide the segment {{< math >}}$[g^{0:T}_\theta(x_{input}), g^{0:T}_{\theta+\Delta \theta}(x_{input})]${{< /math >}} into $K$ smaller segments. Let us denote them by $\Delta\_1, \ldots, \Delta\_K$. They are direct counterparts of the segments $\Delta^3_1$, $\Delta^3_2$, and $\Delta^3_3$ defined [above](#derivative-of-the-network), see [Figure 8](#figure-decomp-net). For each integer $j=1, 2, \ldots, K-1$, we are also interested in the segment of the line $t=t_j$ between point $x(t_j)$ lying on the unperturbed trajectory and the perturbed trajectory through the point $(t_{j-1}, x(t_{j-1}))$. Denote them by $\tilde \Delta_j$. They are direct counterparts of segments $\Delta_1^1$ and $\Delta_2^2$ on [Figure 8](#figure-decomp-net). Note that each $\Delta_j$ is an image of $\tilde \Delta_j$ under the map $g^{t_j:T}_{\theta+\Delta \theta}$.

{{< callout note >}}

Again, there's a catch: $\Delta_j$'s are not segments, they are vectors in the multidimensional space, and strictly speaking they cannot “split” the segment $[g^{0:T}\_{\theta }(x\_{input}), g^{0:T}\_{\theta+\Delta \theta}(x\_{input})]$. Nevertheless, their sum is equal to the vector $g^{0:T}\_{\theta+\Delta \theta}(x\_{input}) - g^{0:T}_{\theta}(x\_{input})$, and this is actual assertion we will use. However, it is easier to think about “segments”, so I'll keep this terminology. The same holds for $\tilde \Delta_j$.

{{< /callout >}}

Now we can write:

{{< math >}}$$
\begin{equation}
\label{dgdtheta}
\frac{\partial g^{0:T}_\theta(x_{input})}{\partial \theta} \Delta \theta \approx
\sum_{j=1}^{K} \Delta_j.
\end{equation}
$${{< /math >}}

The rest is to investigate $\Delta_j$'s.

### Smaller segments



As it was noted, each $\Delta_j$ is an image of $\tilde \Delta_j$ under the phase flow map $g^{t_j:T}_{\theta+\Delta \theta}$. If $K$ is large, these segments are small, and the phase flow map can be approximated by its derivative. Therefore,

{{< math >}}$$
\begin{equation}
\label{Delta-star}
\Delta_j \approx \frac{\partial g^{t_j: T}_{\theta+\Delta \theta}(x(t_j))}{\partial x} \tilde \Delta_j.
\end{equation}
$${{< /math >}}

To find $\tilde \Delta_{j}$, let us approximate the unperturbed and perturbed trajectories throught the point $(t_{j-1}, x(t_{j-1}))$ by their respective tangent lines at this point, see [Figure 13](#figure-approx-traj).

{{< figure src="/img/adjoint-state/backprop-13.svg" width="90%" title="Figure 13. Approximation trajectories with tangents" id="approx-traj" >}}

The trajectories are graphs of the solutions of the equations with right-hand side given by $f_{\theta+\Delta \theta}$ and $f_\theta$. The slopes of the tangent lines thus are equal to the value of these functions: $f_{\theta+\Delta \theta}(t_{j-1}, x(t_{j-1}))$ and $f_{\theta}(t_{j-1}, x(t_{j-1}))$. Therefore, the length of the segment over $t_j$ cut by tangents is equal to

{{< math >}}$$
\bar \Delta_j = (f_{\theta+\Delta \theta}(t_{j-1}, x(t_{j-1})) -
f_{\theta}(t_{j-1},
x(t_{j-1}))) \Delta t_j,
$${{< /math >}}

where $\Delta t_j=t_j-t_{j-1}$. For large $K$, $\Delta t_j$ is small and the actual trajectories lie close to the respective tangent lines, and thus $\tilde \Delta_j \approx \bar \Delta_j$. Now for small $\Delta \theta$, we can approximate $(f_{\theta+\Delta \theta}(t_{j-1}, x(t_{j-1})) - f_{\theta}(t_{j-1}, x(t_{j-1})))$ by the corresponding derivative, and obtain:

{{< math >}}$$
\tilde \Delta_j \approx \frac{\partial f_\theta(t_{j-1}, x(t_{j-1}))}{\partial
\theta} \Delta \theta \Delta t_j.
$${{< /math >}}

Put it into equation $\eqref{Delta-star}$ and obtain:

{{< math >}}$$
\Delta_j \approx 
\frac{\partial g^{t_j: T}_{\theta+\Delta \theta}(x(t_j))}{\partial x} 
\frac{\partial f_\theta(t_{j-1}, x(t_{j-1}))}{\partial \theta} \Delta \theta\, \Delta t_j.
$${{< /math >}}

Now return to equation $\eqref{dgdtheta}$. We have:

{{< math >}}$$
\frac{\partial g^{0:T}_\theta(x_{input})}{\partial \theta} \Delta \theta \approx
\sum_{j=1}^K 
\frac{\partial g^{t_j: T}_{\theta+\Delta \theta}(x(t_j))}{\partial x} 
\frac{\partial f_\theta(t_{j-1}, x(t_{j-1}))}{\partial \theta} \Delta \theta\, \Delta t_j
$${{< /math >}}

Clearly, it looks pretty much like an integral sum! Thus it is easy to believe in the following approximation:

{{< math >}}$$
\begin{align*}
\frac{\partial g^{0:T}_\theta(x_{input})}{\partial \theta} \Delta \theta \approx &
\int_{0}^T 
\frac{\partial g^{t: T}_{\theta+\Delta \theta}(x(t))}{\partial x} 
\frac{\partial f_\theta(t, x(t))}{\partial \theta} \Delta \theta \, dt= \\
& \left(\int_0^T \frac{\partial g^{t: T}_{\theta+\Delta \theta}(x(t))}{\partial x} 
\frac{\partial f_\theta(t, x(t))}{\partial \theta} \, dt\right) \Delta \theta.
\end{align*}
$${{< /math >}}

As $\Delta \theta$ becomes small, this equality becomes more and more exact. It holds for any small $\Delta \theta$, therefore, the following (now exact) equality on the derivative takes place:

{{< math >}}$$
\begin{equation}
\label{dgdtheta-int}
\frac{\partial g^{0:T}_\theta(x_{input})}{\partial \theta} =
\int_0^T \frac{\partial g^{t: T}_{\theta}(x(t))}{\partial x} 
\frac{\partial f_\theta(t, x(t))}{\partial \theta} \, dt.
\end{equation}
$${{< /math >}}

This equation is a continuous counterpart of equation \eqref{partialftheta}. Indeed, the first multiplier corresponds to a derivative of the output of the network with respect to the output of some intermediate layer. The second multiplier gives the dependency of the output of intermediate layer with respect to the parameter.

{{< callout note >}}

The derivation above is even less rigorous and more risky than the previous one. We deal with two limits here, $\Delta t \to 0$ and $\Delta \theta \to 0$, and this is a red flag for everyone who studied Calculus: intuition can easily fool us here. I present this handwaving only because I know the actual proof and absolutely sure everything is OK. At the same time, I believe that this kind of approximate derivations and plots like [Figure 12](#figure-many-perturbed) allows us to _understand_ what is really going on, and the formal proof is just a check that our intuition still works correctly.

{{< /callout >}}

Okay, you may ask, we obtained a new formula for the derivative of the output of the network with respect to the parameters. But we discussed previously that it can be expensive to find it, and we don't actually need it. How this new formula helps us?

Glad you asked! We are ready for an answer.

### Back to the gradient revisited

Let us put \eqref{dgdtheta-int} into \eqref{nablaLode}:

{{< math >}}
$$
\begin{align*}
\nabla_{\! \theta} \mathcal L(\theta) & =   \nabla_{\! y}L \cdot \int_0^T \frac{\partial g^{t: T}_{\theta}(x(t))}{\partial x} 
\frac{\partial f_\theta(t, x(t))}{\partial \theta} \, dt  \\
& = \int_0^T \left(  \nabla_{\! y}L \frac{\partial g^{t: T}_{\theta}(x(t))}{\partial x} \right)
\frac{\partial f_\theta(t, x(t))}{\partial \theta} \, dt.
\end{align*}
$$
{{< /math >}}

Let us consider the first multiplier. Note that $\nabla_{\\! y}L$ is the gradient of function $L$ calculated at point $y=x(T)$. It measures how $L$ depends on the output of the network. At the same time, $\partial g_{\theta}^{t:T}/\partial x$ measures how output of the network depends on $x(t)$, i.e. “output of layer $t$”. Thus, by the chain rule, the product of these derivatives measures how $L$ depends on $x(t)$. One may write:

{{< math >}}$$
\nabla_{\! y}L \frac{\partial g^{t: T}_{\theta}(x(t))}{\partial x} =
\nabla_{\! x}(L\circ g^{t:T}_\theta(x)),
$${{< /math >}}

where the gradient is taken at point $x=x(t)$. Let us introduce a bit informal notation:

{{< math >}}$$
\nabla_{\! x}(L\circ g^{t:T}_\theta(x))=:\nabla_{\! x(t)} L.
$${{< /math >}}

It is a counterpart of $\nabla_{\\! h_i} L$ in the section [General algorithm for backpropagation](general-algorithm-for-backpropagation). With this notation, the integral above can be written in the following form:

{{< math >}}$$
\begin{equation}
\label{nabla-L-int}
\nabla_{\! \theta} \mathcal L(\theta) = \int_0^T  \nabla_{\! x(t)}L \frac{\partial f_\theta(t, x(t))}{\partial \theta} \, dt.
\end{equation}
$${{< /math >}}

And it looks very similar to equation \eqref{nabla-L-sum}, isn't it? Note that we don't have the large matrix derivative $\partial g_\theta^{t:T}(x) / \partial x$ in the formula anymore: it was “swallowed“ by the gradient $\nabla_{\\! x(t)} L$. Looks like a good news! But are there any good ways to find this gradient?

In the usual backpropagation, we used recurrent equation \eqref{nablaLstep} to find $\nabla_{\\! h_i} L$ one by one (the backward pass). In the continous setting, we don't have such a recurrence. Looks like a bad news. But don't worry, a bit of ODE magic will help us!

### The adjoint equation

We are so close! Just a couple of steps ahead.

Let's denote $\nabla_{\\! x(t)} L$ by $a(t)$. This is a vector-row (covector). It is called _adjoint state_ or simply _adjoint_. It depends on $t$, and I feel it should satisfy some differential equation. How to find this equation?

From now on, we omit the dependence on $\theta$ in the notation. For any $t \in [0, T]$ and for any input value $x_0$, let us consider the following decomposition:

{{< math >}}$$
g^{0:T}(x_0)=g^{t:T} \circ g^{0:t}(x_0).
$${{< /math >}}

This formula says that to find the output of the network for the input $x_0$, we have to find the output value of the intermediate layer $t$ for input $x_0$ (that's $g^{0:t}(x_0)$), and then pass it to the rest of the network ($g^{t:T}$). So that's a trivial identity that holds for any $t$.

Let's add $L$ to the left:

{{< math >}}$$
L \circ g^{0:T}(x_0)=L \circ g^{t:T} \circ g^{0:t}(x_0).
$${{< /math >}}

Then take a gradient with respect to $x_0$ and use the chain rule:

{{< math >}}$$
\nabla_{\! x_0} (L \circ g^{0:T}(x_0)) = \nabla_{\! x} (L \circ g^{t:T})
\frac{\partial g^{0: t}(x_0)}{\partial x_0}.
$${{< /math >}}

Clearly, the left-hand side is $\nabla_{\\! x(0)} L$, i.e. $a(0)$, and the first multiplier of the right-hand side is $\nabla_{\\! x(t)} L=a(t)$. Therefore, one have:

{{< math >}}$$
a(0) = a(t)\frac{\partial g^{0: t}(x_0)}{\partial x_0}.
$${{< /math >}}

Now let's take a derivative with respect to $t$. The left-hand side does not depend on $t$, so the derivative is 0. At the right-hand side, one have a product of two time-dependent matrices, so [Leibniz product rule](https://en.wikipedia.org/wiki/Product_rule) should be applied:

{{< math >}}$$
0 = \dot a(t)\frac{\partial g^{0: t}(x_0)}{\partial x_0}+a(t) \frac{d}{dt} \frac{\partial g^{0: t}(x_0)}{\partial x_0}.
$${{< /math >}}

Note that $g^{0:t}(x_0)$, seen as a function of $t$, is just a solution of our equation with the initial condition $x(0)=x_0$:

{{< math >}}$$
g^{0:t}(x_0)=\varphi(t; x_0).
$${{< /math >}}

So the derivative $\partial g^{0:t}(x_0)/\partial x_0$ is well-known as a ”derivative of the solution with respect to the initial conditions”. And we have a theorem here.

{{< callout note >}}

**Theorem.** Consider differential equation

{{< math >}}$$
\dot x = f(t, x), \quad x(t) \in \mathbb R^n,
$${{< /math >}}

and let $x=\varphi(t; x_0)$ be its solution with initial condition $x(0)=x_0$. Let us denote

{{< math >}}$$
w(t):=\frac{\partial \varphi(t; x_0)}{\partial x_0},
$${{< /math >}}

where $w$ is $(n\times n)$-matrix. Then $w$ satisfies the following linear equation:

{{< math >}}$$
\dot w = \frac{\partial f(t, x)}{\partial x} w,
$${{< /math >}}

where derivative is taken at point $x=\varphi(t; x_0)$.

{{< /callout >}}

{{< spoiler text="Wanna proof? Click here!" >}}

**Proof.** It is rather simple if we believe that $\varphi$ depends smoothly on $t$ and $x_0.$

Let's find a derivative of $w(t)$ with respect to $t$. Note that $w$ is itself a derivative, and we can change the order of differentiation (if we believe in smoothness):

{{< math >}}$$
\dot w(t) = \frac{\partial}{\partial t}\frac{\partial \varphi(t; x_0)}{\partial x_0}=
\frac{\partial}{\partial x_0} \frac{\partial \varphi(t; x_0)}{\partial t}.
$${{< /math >}}

Now we use the fact that $\varphi$ is a solution of our equation, so its derivative with respect to time equal to the right-hand side:

{{< math >}}$$
\dot w(t) = \frac{\partial}{\partial x_0} f(t, \varphi(t; x_0)).
$${{< /math >}}

Function $f$ does not depend on $x_0$ directly, but it depends on the solution $\varphi(t; x_0)$ that depends on $x_0$; thus, the chain rule should be applied:

{{< math >}}$$
\dot w(t)=\left.\frac{\partial f(t, x)}{\partial x}\right|_{x=\varphi(t; x_0)}
\frac{\partial \varphi(t; x_0)}{\partial x_0}.
$${{< /math >}}

But now the second multiplier in the right-hand part is just a $w(t)$, so we obtained the desired equation. End of proof.

{{< /spoiler >}}

The derivative $\partial g^{0:t}(x_0)/\partial x_0$ is the same thing as $w(t)$ in the theorem, therefore

{{< math >}}$$
\frac{d}{dt} \frac{\partial g^{0: t}(x_0)}{\partial x_0} = 
\frac{\partial f(t, x)}{\partial x}\frac{\partial g^{0: t}(x_0)}{\partial x_0}
$${{< /math >}}

and we have the following equation for $a(t)$:

{{< math >}}$$
0 = \dot a(t)\frac{\partial g^{0: t}(x_0)}{\partial x_0}+a(t) \frac{\partial f(t, x)}{\partial x}\frac{\partial g^{0: t}(x_0)}{\partial x_0}.
$${{< /math >}}

It can be shown from theory of linear differential equations that matrix $\partial g^{0:t}(x_0) / \partial x_0$ is nondegenerate for all $t$. Therefore, we can multiply the equation by the inverse matrix and obtain:

{{< math >}}$$
\begin{equation}
\label{adjoint}
\dot a(t) = - a(t) \frac{\partial f(t, x)}{\partial x}.
\end{equation}
$${{< /math >}}

This! This is the adjoint equation we are looking for! In the right-hand part the derivative is taken at point $x=x(t)$, i.e. along the solution of the initial differential equation.

So, we have an equation on $a$, but we know that in ODE the equation alone is not enough to specify a solution. What about the initial condition? Look at $a(T)$. It is a derivative of $L$ with respect to the value of the layer $T$. But the layer $T$ is the output layer of the network. Therefore, $a(T)$ is just a derivative of $L(y, y_{true})$ with respect to $y$, where $y=x(T)$. For a given $x(T)$, the derivative does not depend on the network! And therefore it's an appropriate initial condition for the adjoint equation:

{{< math >}}$$
\begin{equation}
\label{adjoint-initial}
a(T)=\nabla_{\! y} L(y, y_{true}), \quad y=x(T).
\end{equation}
$${{< /math >}}

### Backpropagation in neural ODEs

Let's summarize the algorithm to find loss gradient for Neural ODEs. We have an equation

{{< math >}}$$
\begin{equation}
\label{eq}
\dot x=f_\theta(t, x),
\end{equation}
$${{< /math >}}

the input value $x_{input}$, the true output $y_{true}$ and some value of the parameter vector $\theta$. We want to find the gradient of the training objective

$$\mathcal L(\theta) = L(x(T;\theta), y_{true})$$

with respect to $\theta$. Here $x(t; \theta)$ is a solution of equation \eqref{eq} with the initial condition $x(0; \theta)=x_{input}$. As before, we will omit the dependence on $\theta$ in the following equations.

First, we do the forward pass, i.e. solve the equation \eqref{eq} numerically and find $x(T)$. In the usual neural network, we store all the outputs of intermediate layers $h_1, \ldots, h_N$ to use them in the backward pass. In the neural ODE, strictly speaking, it's impossible to store all the intermediate outputs, because there are infinite number of them. We can theoretically store intermediate outputs at some time sequence, i.e. store $x(t_j)$ for some moments $t_j$, that can be used to approximate the full trajectory. However, it appears that we don't need it and can make our algorithms memory-efficient. So, just store $x_{output}=x(T)$.

Now backward pass. We need to solve the adjoint equation \eqref{adjoint} and find integral \eqref{nabla-L-int}. It can be a bit tricky.

First, as before, we want to be memory-efficient and thus don't want to store the trajectories. So we need to solve the adjoint equation and integrate at the same time. Moreover, the right-hand side of the adjoint equation \eqref{adjoint} depends on the solution of the initial equation $x(t)$, that we didn't store. So we have to reconstruct it together with solving adjoint equation and integrating.

A lot of things to do! However, it appears we can do all together by combining everything into one system of ODEs:

{{< math >}}$$
\begin{equation}
\label{final}
\begin{cases}
\dot x=f_\theta(t, x),\\
\dot a = - a \cdot \frac{\partial f_\theta(t, x)}{\partial x},\\
\dot u = - a \cdot \frac{\partial f_\theta(t, x)}{\partial \theta}.
\end{cases}
\end{equation}
$${{< /math >}}

that should be solved backward in time with the initial conditions

{{< math >}}$$
\begin{cases}
x(T)=x_{output} & \text{(found at the forward pass)}\\
a(T)=\nabla_{\! y} L(y, y_{true}), & y=x_{output}\\
u(T)=0
\end{cases}
$${{< /math >}}

The first two equations of the system \eqref{final} are just equations \eqref{eq} and \eqref{adjoint}. What about the third one? We see that its right-hand side doesn't depend on the unkown variable $u$, so its solution (provided that we know the solutions of two other equations) is just an integral:

{{< math >}}$$
\begin{align*}
u(t)&=-\int_{T}^{t} a(\tau) \cdot \frac{\partial f_\theta(\tau, x(\tau))}{\partial \theta}
d\tau\\
&=\int_{t}^T a(\tau) \cdot \frac{\partial f_\theta(\tau, x(\tau))}{\partial \theta}
d\tau.
\end{align*}
$${{< /math >}}

The limits of integration were chosen in such a way to satisfy the initial condition $u(T)=0$. Now recall that $a(t)=\nabla_{\\! x(t)} L$. Put it into the integral above and let $t=0$: voila, we have integral \eqref{nabla-L-int}! Thus

{{< math >}}$$
u(0)=\nabla_{\! \theta} \mathcal L(\theta),
$${{< /math >}}

and this is exactly the value we are interested in!

So, in the backward pass we just solve system \eqref{final} with the given initial conditions over the segment $[0, T]$ and return $u(0)$. That's all!

{{< spoiler text="Interested in the rigorous derivation of system \eqref{final}? I have one!" >}}

The derivation is short but a bit cryptic. It uses the following well-known trick: include the parameter $\theta$ as a phase variable. E.g. instead of equation \eqref{eq}, consider the following system

{{< math >}}$$
\dot x=f_\theta(t, x), \quad \dot \theta=0.
$${{< /math >}}

Then consider an extended adjoint

{{< math >}}$$
\vec{a}(t):=\nabla_{\\! (x(t),\\, \theta(t))} L = 
(\nabla_{\\! x(t)}L,
\nabla_{\\! \theta(t)} L)=(\nabla_{\\! x} (L\circ g^{t:T}\_\theta), \nabla\_{\\! \theta}
(L\circ g^{t:T}\_\theta)) = (a(t), u(t)),
$${{< /math >}}

i.e. $\vec{a}$ is just a concatenation of vectors $a$ and $u$, the first component, as before, measures how $L$ depends on the “output of the layer $t$” (i.e. $x(t)$), and the second component measures how $L$ depends on the parameter $\theta$. Both gradients are found at point $x(t)$ of the unperturbed solution. In other words, while calculating $u(t)$, one considers a system that works like the following: on the segment $[0, t]$, it uses the original value of the parameter $\theta$, and on the segment $[t, T]$ is uses the perturbed value of the parameter, i.e. $\theta + \Delta \theta$. Then $u(t)$ measures the effect of $\Delta \theta$ on the output $x(T; \theta + \Delta \theta)$.

Then $\vec{a}$ should satisfy the adjoint equation \eqref{adjoint}, where $x$ is replaced with $(x, \theta)$ and $f$ is replaced with $(f_{\theta}(t, x), 0)$. One have:

{{< math >}}$$
(\dot a, \dot u) =
-(a, u) 
\begin{pmatrix}
\frac{\partial f_{\theta}(t, x)}{\partial x} & 
\frac{\partial f_{\theta}(t, x)}{\partial \theta} \\\\
0 & 0
\end{pmatrix}.
$${{< /math >}}

The second row of the matrix is $0$ because the second component of our extended system is $0$ and therefore its derivatives with respect to $x$ and $\theta$ are zeroes. Making matrix multiplication, one obtains \eqref{adjoint}.

Hooray!

{{< /spoiler >}}

## Concluding remarks

That was a long story, and it's time to conclude. Let me reiterate several main ideas:

- The goal of backpropagation and adjoint state method is to find a gradient of the loss function with respect to the parameters in a computationally efficient way. We don't want to waste resources calculating more than needed, so the order of operations matters.

- These methods are based on a simple idea: when you have a composition of several functions such that the last function in the composition takes values in one-dimensional space (and therefore the full composition do the same), the derivative of the output of such a composition with respect to any intermediate output is just a vector-row (covector), and not a full matrix.

- This idea can be naturally extended to continuous settings, where instead of a long composition we have a differential equation.

- Both in discrete and continuous settings there are effective algorithms to calculate the derivatives of the one-dimensional output with respect to intermediate values. These algorithms work “backward”: from the last intermediate “layers” to the first ones. In discrete settings, it's the reccurrence \eqref{nablastep} (also known as \eqref{nablaLstep}). In continous settings, it's the adjoint equation \eqref{adjoint}.

- These algorithms efficiently reuse the derivative they found at the previous “steps” and do not waste time calculating things that don't needed.

- It is possible to adapt these algorithms to settings when you need a derivative of the output with respect to the parameters, as we have in the neural networks.

- To do that efficiently, we have to disentangle two effects: 1. Change of the output of some intermediate layer due to change of the parameters; 2. Change of the output of the subsequent layers due to change of the output of the intermediate layer. Then we have to integrate over all intermediate layers, and that leads to the solution.

That's all!

Did you enjoy this post? Follow me on [Twitter](https://twitter.com/ilya_schurov) and let's stay in touch!

[^1]: Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. [Neural Ordinary Differential Equations](https://proceedings.neurips.cc/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf). 32nd Conference on Neural Information Processing Systems (NeurIPS 2018).
