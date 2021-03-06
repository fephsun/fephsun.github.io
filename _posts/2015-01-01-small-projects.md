---
title: Smaller Projects
---

These are some smaller projects I worked on at MIT for classes and fun.

<h3> 6.852 (distributed algorithms) final project - bitcoin and Byzantine fault-tolerance </h3>

<p>
We put the bitcoin currency in the framework of Byzantine fault-tolerance, one of the most fundamental problems in distributed algorithms.  Byzantine fault-tolerance refers to correct behavior when some processes in the network are controlled by a malicious adversary.  We present a formalized version of the bitcoin algorithm, and prove that it maintains correctness in the face of Byzantine failures.  In fact, bitcoin can be used to implement a Byzantine consensus algorithm that has high probability of correctness as long as less than half of the processes are malicious.  All previous Byzantine consensus algorithms could tolerate less than one-third failures.  
</p>

<p>
<a href="/projects/bitcoin.pdf">Final report here.</a>
</p>

<h3> 6.867 (machine learning) final project - probabilistic programming </h3>

<p>
I worked on making inference on probabilistic programs faster.  Probabilistic programming is a way to specify probability models using the syntax of a programming language (as opposed to a graphical model, or an equation).  If you want to play with probabilistic programming, check out <a href="http://projects.csail.mit.edu/church/wiki/Church">Church</a>, a simple probabilistic programming language built on top of Lisp.
</p>

<p>
I specifically tried to determine when the inference algorithm for <a href="http://probcomp.csail.mit.edu/venture/">Venture</a>, a more sophisticated probabilistic programming language, was stuck in a local maximum.  To do this, I modeled the log score of the Venture inference algorithm as a hidden Markov model, and wrote code to determine the average "jumping time" of the log score.  <a href="/projects/6.867_final.pdf">Report here</a>.
</p> 

<h3> 9.77 (computer/human vision) final project - jigsaw puzzles </h3>

<p>
I wrote a program that solves jigsaw puzzles using computer vision techniques.  It's not perfect (especially when compared to human performance), but it approaches state-of-the-art accuracy.  What's more, I use a purely greedy approximation, resulting in substantially faster runtime.  This is significant because jigsaw puzzle solving is an NP-complete problem, and has been traditionally tackled with global relaxation techniques.  A good greedy solution at the local level makes relaxation at the global level easier.
</p>

<p>
Watch it in action here:
<br />
<iframe width="420" height="315" src="https://youtube.com/embed/-oG4REJcXg8" frameborder="0" allowfullscreen></iframe>
</p>

<p>
<a href="/projects/jigsaw.pdf"> Write-up </a>.
</p>

<h3> Carbonate (6.470 project) </h3>
<img src="/carbonate.png" />
<p>
A website that generates organic chemistry problems, based on a machine understanding of the laws of chemistry.  Try to synthesize molecules using a drag-and-drop interface.  Because problems are generated on the spot, you will never run out of practice material.  Hosted at <a href="http://orgo.mit.edu"> orgo.mit.edu </a>.
</p>
