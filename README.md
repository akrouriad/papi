# papi
Example implementation of key algorithms of paper [Projections for Approximate Policy Iteration](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/RiadAkrour/icml19_sub.pdf).

The file `kl_projection.py` implements Alg. 2 of the paper. It takes as input a linear-Gaussian policy and projects it to another policy that has KL divergence w.r.t. a target policy, smaller than a threshold. 

The file `policy_with_entropy_cst.py` implements a policy with an embedded strict entropy inequality constraint, to ensure that the entropy of a policy never goes below a threshold. This code can easily be extended to enforce a strict entropy *equality* constraint by replacing `self.chol = tf.cond(ent < tent, lambda: self.chol * tf.exp((tent - ent) / act_dim), lambda: self.chol)` with `self.chol = self.chol * tf.exp((tent - ent) / act_dim)`.
