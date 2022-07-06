---
title: Adjoint State Method, Backpropagation and Neural ODE

summary: Gently introduction to adjoint state method in Neural ODEs

tags:
- differential equations
- machine learning
date: "2022-07-02T00:00:00Z"

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

You probably heard of Neural ODEs, a neural network architecture based on
ordinary differential equations. When I first read about them, my thoughts were:
okay, we have an ODE given by a neural network, now we need to learn weights of
that neural network, thus we need gradients, thus we have to find derivates of
the solution of an ODE with respect to parameters. And we have a standard tool
for that in ODE theory, it's called "equations in variations", so let's just
apply them and we're done. Case closed.

However, in the paper, the authors used much more elaborate approach, based
on something called _adjoint state method_. It was strange: why we need some cryptic
mathematical magic to solve such a standard problem? In various other
discussions, I heard from time to time that these mysterious _adjoints_
are somehow related to backpropagation. However, all tutorials on adjoint state
method I was able to find used a bunch of infinite-dimensional optimization
theory, and it was not clear, how this theory can be related to such a simple
thing as backpropagation? 

It was my surprise when I understood that adjoint state method in fact is based
on a very simple idea. And now I want share it with you.

### How to multiply matrices
Before we begin with backpropagation and neural ODEs, let's talk about something
very simple: about matrix multiplication.

Assume we have two square {{< math >}}$n \times n${{< /math >}} matrices, {{< math >}}$A${{< /math >}} and {{< math >}}$B${{<
/math >}}, and an {{< math >}}$n${{< /math >}}-dimensional vector (vector-column) {{< math >}}$x${{< /math >}}. Consider the
following product:

{{< math >}}
$$ABx$$
{{< /math >}}

As matrix multiplication is associatative, we don't need any brackets in this
formula. However, if we try to add them, we'll note that it can be done in two
different ways: we can either write it like this:
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
only need {{< math >}}$O(n^2)${{< /math >}} operations.

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

The last thing I want to mention here is that if {{< math >}}$x${{< /math >}} is
not a vector-column but vector-row (mathematically speaking, not a vector but
covector), and we want to find a product {{< math >}}$xAB${{< /math >}}, this
again can be done in two different ways:
{{< math >}}$$xAB=(xA)B=x(AB)$$,{{< /math >}}
and now the first one is much cheaper.

Does it sounds reasonable? If yes, congratulations: you understood the main idea
of the adjoint state method!

### Backpropagation
Now consider backpropagation algorithm. It is very much well-known, but I
present here an exposition that is specifically designed to stress the relation
of backprop to the adjoint state method in neural ODEs.

Assume we have a neural network that consists of three layers, two of them are
hidden. Layer number {{< math >}}$i${{< /math >}}, {{< math >}}$i=1,2,3${{<
/math >}}, is given by a function 

{{< math >}}$$f^({i)}_\theta\colon \mathbb R^{n_{i-1}} \to \mathbb R^{n_i}$${{< /math >}}, 
where {{< math >}}$\theta\in \mathbb R^p${{< /math >}}
is a vector of all parameters of the neural network (i.e. all weights and
biases), {{< math >}}$n_i${{< /math >}} is the dimensionality of the output of
{{< math >}}$i${{< /math >}}'th layer, {{< math >}}$n_0${{< math >}} is the input
dimensionality. It is clear that each layer depends only
on a subset of parameters in {{< math >}}$\theta${{< /math >}}, but we don't
need to take care about it now. So, the full network defines a function

{{< math >}}$$
f_{\theta}(x) := f^{(3)}_\theta\circ f^{(2)}_\theta \circ f^{(1)}_\theta(x)
$${{< /math >}}

This can be visualized in the following way:

{{< figure src="/img/adjoint-state/network1.svg" width="90%" >}}

We also have some loss function $L(y, y_{true})$ (e.g. in case of quadratic
loss, $L(y, y_{true})=(y-y_{true})^2$). If we put the output of the network into
the loss, we obtain a function
{{< math >}}$$
\mathcal L(\theta) := L(f_\theta(x), y_{true})
$${{< /math >}}
For simplicity, we are discussing the loss at one datapoint; in the real
settings, we would average this over the batch.

To perform gradient descent, we need to find a gradient of {{< math >}}$\mathcal
L${{< /math >}} with respect to parameter vector {{< math >}}$\theta${{< /math >}}. Chain rule immediately gives:

{{< math >}}$$\nabla_\theta \mathcal L(\theta) = \nabla_y L \frac{\partial
f_\theta(x)}{\partial \theta},$${{< /math >}}
where the first multiplier is a gradient of {{< math >}}$L${{< /math >}}, i.e.
vector-row of dimensionality {{< math >}}$n_3${{< /math >}} (dimensionality of
output layer), and the second multiplier is a 
{{< math >}}$n_3 \times p${{< /math >}}-matrix.

It is easy to find {{< math >}}$\nabla_y L ${{< /math >}} provided that {{< math
>}}$y${{< /math >}} is already calculated (i.e. in the case of quadratic loss,
it's just {{< math >}}$(2y-2y_{true})${{< /math >}}). To find the second
multiplier, one have to decompose {{< math >}}$f_\theta${{< /math >}} into a
composition of subsequent layer maps and again apply the chain rule. However, I
prefer to do it visually.

Let's fix some small vector {{< math >}}$\Delta \theta \in \mathbb R^p${{< /math >}} and consider “trajectory” of $x_{input}$ under action of “perturbed” maps
{{< math >}}$f^i_{\theta+\Delta \theta}${{< /math >}}, {{< math >}}$i=1,2,3${{<
/math >}}:

{{< figure src="/img/adjoint-state/network2.svg" width="90%" >}}

The difference between outputs {{< math >}}$f_{\theta+\Delta
\theta}(x_{\input})-f_\theta(x_{input})${{< /math >}} is approximately equal to

{{< math >}}$$
\frac{\partial f_\theta}{\partial \theta} \Delta \theta
$${{< /math >}}
provided that {{< math >}}$\Delta \theta${{< /math >}} is small: that's
basically a definition of a derivative. (Note that on
the picture this difference is represented by a segment on a line, but in
reality it's {{< math >}}$n_3${{< /math >}}-dimensional vector.) Now let's
decompose this difference in a sum of three parts, denoted as {{< math >}}$\Delta^3_1${{< /math >}},
{{< math >}}$\Delta^3_2${{< /math >}} and {{< math >}}$\Delta^3_3${{< /math >}}, according to the following figure.

{{< figure src="/img/adjoint-state/network2.svg" width="90%" >}}

Here all the red arrows represent the action of the corresponding {{< math
>}}$f^i_{\theta+\Delta \theta}${{< /math >}}.

Let's begin with {{< math >}}$\Delta^3_3${{< /math >}}. It measures the
difference between the images of some point under action of {{< math
>}}$f^3_{\theta+\Delta \theta}${{< /math >}} and {{< math >}}$f^3_{\theta}${{<
/math >}}. Again, we use definition of a derivate and get the following
approximation:

{{< math >}}$$
\Delta^3_3 \approx \frac{\partial f^3_{\theta}}{\partial \theta} \Delta \theta.
$${{< /math >}}

That was easy. Now consider {{< math >}}$\Delta^3_2${{< /math >}}. Here we have
two steps. At the first step, we have two functions, {{< math >}}$f^2_\theta${{<
/math >}} and {{< math >}}$f^2_{\theta+\Delta \theta}${{< /math >}} that are applied to the same point. The difference between the images is denoted by {{< math >}}$\Delta^2_2${{< /math >}} and is approximately equal to
{{< math >}}$$
\Delta^2_2 \approx \frac{\partial f^2_{\theta}}{\partial \theta} \Delta \theta
$${{< /math >}}
At the second step, we have 
one function, {{< math >}}$f^2_{\theta+\Delta \theta}${{< /math >}}, that is
applied to two different points. To find the difference between the images now,
we have to use the derivative of {{< math >}}$f^3_{\theta+\Delta \theta}(h_2)${{< /math >}} with
respect to its argument {{< math >}}$h_2${{< /math >}}. Namely:
{{< math >}}$$
\Delta^3_2 \approx \frac{\partial f^3_{\theta + \Delta \theta}(h_2)\Delta^2_2
\approx \frac{\partial f^3_{\theta+\Delta \theta}}{\partial h_2} \frac{\partial
f^2_{\theta}}{\partial \theta} \Delta \theta
$${{< /math >}}

And finally for {{< math >}}$\Delta^3_1${{< /math >}} we have three steps:

{{< math >}}$$
\Delta^3_1 \approx \frac{\partial f^3_{\theta + \Delta \theta}(\partial h_2)\Delta^2_1
\approx \frac{\partial f^3_{\theta+\Delta \theta}{h_2) \frac{\partial
f^2_{\theta}}{\partial \theta} \Delta \theta
$${{< /math >}}
