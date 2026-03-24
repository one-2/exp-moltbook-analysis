# Moltbook Analysis

**Status**: Having reflected on the evidence that MoltBook is a highly contaminated dataset (Li, 2026), the value of further analysis here is unclear. I'm now spending time on the broader problem MoltBook speaks to: safety risks in very-large systems of AI. Follow our work on this problem at [Gigascale Labs](https://www.gigascale-labs.org).

Repo for analysis of moltbook data.

## Motivation

Multi-agent AI systems (MAS) are a possible path to superintelligence, (Bostrom, 2014) and present unique risks, like emergent agency, not encountered in the single-agent setting. Our understanding of these risks is relatively poor. (Hammond et al., 2025) MAS also offer benefits from increased coordination and autonomous workforces. (Tomasev et al., 2025) Progress on MAS safety is important to realising these benefits.

[Moltbook](https://www.moltbook.com/) is a social media site for AI agents, where agent can interact with each other on a reddit-style forum free of human oversight. 

As the first  is the first natural MAS-only social environment, Moltbook extends the possibilities for study MAS safety beyond simulated environments like (UNSW Institute for Cyber Security, 2025). 

Treating Moltbook as an early case study in MAS safety, I aim to explore a few research questions:

- How do agents behave on Moltbook?
- Which MAS safety risks manifest in Moltbook, and how commonly?
- Do individual agents have power on Moltbook?
- How do subnetworks on Moltbook affect agent behaviour?
- Does the system as a whole develop trends, and to what extent can these be seen as emergent behaviour?
- How reliable is Moltbook as a case study?

I will not articulate hypotheses yet as I am presently (1 Feb 2026) aiming to characterise the system in broadstroke and investigate anything interesting, rather than conduct experiments or answer a specific question.

## The analysis

You can find my analysis in the Jupyter notebooks in the source file.

### Other work on Moltbook

You can find some other analysis on Moltbook here:

- [36,000 AI Agents Are Now Speedrunning Civilization](https://www.lesswrong.com/posts/jDeggMA22t3jGbTw6/36-000-ai-agents-are-now-speedrunning-civilization)
- [Moltbook shitposts are actually really funny](https://www.lesswrong.com/posts/LT7cxegQn4FLGFQR6/moltbook-shitposts-are-actually-really-funny)
- [Karpathy's analysis](https://www.reddit.com/r/accelerate/comments/1qrv90f/andrej_karpathy_on_moltbook/)
- [Humans can post on Moltbook](https://www.lesswrong.com/posts/XtnmhHL4tjL5MeM2z/humans-can-post-on-moltbook)
- [Inflated user counts](https://www.forbes.com/sites/guneyyildiz/2026/01/31/inside-moltbook-the-social-network-where-14-million-ai-agents-talk-and-humans-just-watch/)
- [Database vulnerability](https://www.binance.com/sv/square/post/02-01-2026-moltbook-database-vulnerability-exposes-sensitive-information-35862945061314)


## Tools

Thank you to:

- (Newman and Rimey, 2026) for the [Moltbook data source](https://www.lesswrong.com/posts/WyrxmTwYbrwsT72sD/moltbook-data-repository).
- (Perez et al., 2022) for the traits dataset.

References are at the bottom of the page

## Open problems

### Exploratory data analysis v/

### Self-improvement

- [ ] Further analyse content tagged with this highly prevalent trait

### Methodological improvements
- [ ] Estimator stability under resampling or probe (trait) definition variations
- [ ] Population-level analysis

### Alternative lenses
- [ ] Influence networks
- [ ] Clustered traits to see if there's a post population structure, as suggested by correlations
- [ ] Selection pressures for post success (Hammond et al., 2025) and effect on agent lifetimes
- [ ] Emergent agency: emergent capabilities, emergent goals. (Hammond et al., 2025)
- [ ] Toxic content -  hate speech, calls for violence, etc   
- [ ] System dynamics

## Acknowledgements

This work was undertaken as part of the [Sydney AI Safety Fellowship 2026](https://sasf26.com/).

## References

Bostrom, N. (2014). *Superintelligence: Paths, dangers, strategies*. Oxford University Press.

Hammond, L., Chan, A., Clifton, J., Hoelscher-Obermaier, J., Khan, A., McLean, E., Smith, C., Barfuss, W., Foerster, J., Gavenčiak, T., Han, T. A., Hughes, E., Kovařík, V., Kulveit, J., Leibo, J. Z., & Oesterheld, C. (2025). *Multi-agent risks from advanced AI (Technical Report No. 1)*. Cooperative AI Foundation. https://doi.org/10.48550/arXiv.2502.14143

Li, N. (2026). The Moltbook illusion: Separating human influence from emergent behavior in AI agent societies. arXiv. https://arxiv.org/abs/2602.07432

Newman, E. and Rimey, K. (2026). *Moltbook Data*. GitHub. https://github.com/ExtraE113/moltbook_data

Perez, E., Ringer, S., Lukošiūtė, K., Nguyen, K., Chen, E., Heiner, S., Pettit, C., Olsson, C., Kundu, S., Kadavath, S., Jones, A., Chen, A., Mann, B., Israel, B., Seethor, B., McKinnon, C., Olah, C., Yan, D., Amodei, D., . . . Kaplan, J. (2022). *Discovering language model behaviors with model-written evaluations*. arXiv. https://doi.org/10.48550/arXiv.2212.09251

Tomasev, N., Franklin, M., Leibo, J. Z., Jacobs, J., Cunningham, W. A., Gabriel, I., & Osindero, S. (2025). *Virtual agent economies*. arXiv. https://doi.org/10.48550/arXiv.2509.10147

UNSW Institute for Cyber Security. (2025). *Capture the Narrative*. https://capturethenarrative.com/
